import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"


import logging
import math
import numpy as np
import re
import pickle
import copy
import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

from typing import List, Callable, Tuple, Dict, Iterator, Iterable, Union
# from itertools import takewhile, repeat
# from PIL import Image
import re
import requests
import sys 
import json
from tqdm import tqdm
import copy

import random
import psutil
import argparse
import warnings
import logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
sys.path.append('./')
import json5
from transformers import AutoTokenizer, AutoProcessor
# from transformers.cache_utils import DynamicCache
# from qwen_vl_utils import process_vision_info
# from vllm import SamplingParams
# from inference.utils import PigaiModel, generate_pigai_prompt_tokens
# from inference.utils import MathAccuracyORM
from inference import math_verifier
from config.prompt import *
from multiprocessing import Process, synchronize, Lock, Manager, Pool, set_start_method
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from six.moves import queue

from config import prompt

def split_list(lst, n=4):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def expand_list(input_list, times=3):
    return [item for item in input_list for _ in range(times)]


def get_chunk(lst, n, k):
    chunk_size = math.ceil(len(lst) / n)  # integer division
    chunks = [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    return chunks[k]


def check_possible_fragment_index(policy_tokenizer, think_token_ids):
    """check possible fragment index split by \n"""
    think_tokens = policy_tokenizer.batch_decode(think_token_ids)
    possible_index = [index for index in range(len(think_tokens)) if "\n" in think_tokens[index]]
    return possible_index

def chat_with_Qwen3_policy_request(input_text, max_length, top_p, top_k, temperature, model_name, eos_token):
    data = {
        "model": model_name,
        "prompt": input_text,
        "max_tokens": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "include_stop_str_in_output": True,
    }
    headers = {"Authorization": "Bearer token-123456"}
    res = requests.post("http://localhost:8000/v1/completions", headers=headers, json=data)
    try:
        output = res.json()["choices"][0]
        stop_reason = output["stop_reason"]
        finish_reason = output["finish_reason"]
        output_text = output["text"]
        if stop_reason == None and finish_reason == 'stop':
            output_text += eos_token

    except Exception as e:
        print(res.json())
        print(e)
        raise ValueError
    return output_text
    

def interaction_sample(rank, test_dataset, model, tokenizer, qwen2vl_infer_batch, args, use_api=False):
    # test_dataset = expand_list(test_dataset, args.best_of_n)
    test_dataset_batch = split_list(test_dataset, 1)
    
    batch_num = 0
    max_response_len = args.max_len
    top_p = args.top_p
    top_k = args.top_k
    temperature = args.temperature

    # sampling_params = SamplingParams(
    #     temperature=temperature,
    #     top_p=top_p,
    #     top_k=top_k,
    #     max_tokens=max_response_len,
    #     skip_special_tokens=False,
    #     include_stop_str_in_output=False
    # )
    sampling_params = None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.use_api == 'True' or args.use_api == True:
        use_api = True
    else:
        use_api = False


    with torch.no_grad():
         for batch in test_dataset_batch: # 分batch推理
            batch_num += 1
            logger.info(f"rank: {rank}, {batch_num}/{len(test_dataset_batch)}, {batch_num/len(test_dataset_batch)}")
            

            batch_id_list = [item['id'] for item in batch]
            if args.chat_template == 'qwen_math':
                interaction_prompt = [prompt.qwen_math_interaction_prompt_en.replace("[PROMPT]", item['question']) for item in batch]
            elif args.chat_template == 'qwen3_slow_thinking':
                interaction_prompt = [prompt.qwen3_tir_interaction_prompt_en.replace("[PROMPT]", item['question']) for item in batch]
            elif args.chat_template == 'qwen3_fast_thinking':
                interaction_prompt = [prompt.qwen_math_template.replace("[PROMPT]", item['question']) for item in batch]
            elif args.chat_template == 'qwen_math_text_only':
                interaction_prompt = [prompt.qwen_math_template.replace("[PROMPT]", item['question']) for item in batch]
            elif args.chat_template == 'r1_distill_qwen':
                interaction_prompt = [prompt.r1_distill_qwen_prompt.replace("[PROMPT]", item['question']) for item in batch]
            else:
                raise ValueError(f"Unsupported chat template {args.chat_template}")
            
            messages = [
                    {"role": "user", "content": interaction_prompt[0]}
                ]
            if args.chat_template == 'qwen3_fast_thinking':
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            if args.chat_template == 'r1_distill_qwen':
                input_text = input_text + "<think>\n"

            if not use_api:
                raise ValueError
            else:
                policy_response = chat_with_Qwen3_policy_request(input_text, max_length=max_response_len, top_p=top_p, top_k=top_k, temperature=temperature, model_name=args.model_path, eos_token=tokenizer.eos_token)

            predict_length = len(tokenizer.encode(policy_response))
            
            # 保存临时结果
            cur_output_text = {
                'id': batch_id_list[0],
                'question': batch[0]['question'],
                'predict': input_text + policy_response,
                'answer': batch[0]['answer'],
                'predict_length': predict_length
            }
            cur_output_text = [cur_output_text]
            with open(os.path.join(args.output_dir, f'tmp_result_rank_{rank}.jsonl'), "a", encoding="utf-8") as f:
                for _item in cur_output_text:
                    f.write(json.dumps(_item, ensure_ascii=False) + "\n")




def chat_with_api(input_text, max_length, model_name, stop, temperature=0.6, top_p=1.0, top_k=20):
    data = {
        "model": model_name,
        "prompt": input_text,
        "max_tokens": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stop": stop,
        "include_stop_str_in_output": True,
    }
    headers = {"Authorization": "Bearer token-123456"}
    try:
        res = requests.post(f"http://localhost:8000/v1/completions", headers=headers, json=data)
        outputs = res.json()["choices"]
    except Exception as e:
        print(res.json())
        raise ValueError

    return outputs


def do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    line = params["line"]
    line_id = params["line_id"]
    policy_tokenizer = params["policy_tokenizer"]
    max_try_times = params["max_try_times"]
    infer_batch_size = args.qwen2vl_infer_batch
    test_dataset = [line]
    
    rank = line_id
    
    interaction_sample(
        rank=rank,
        test_dataset=test_dataset,
        model=None,
        tokenizer=policy_tokenizer,
        qwen2vl_infer_batch=infer_batch_size,
        args=args,
        use_api=True
    )
    


def try_do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    try:
        do_single_task(params, args, records_queue, records_queue_lock, shared_params)
    except Exception as e:
        print(f"Error {e}")
    if records_queue_lock is not None:
        with records_queue_lock:
            records_queue.put(1)



def main(args):
    rank = 0 
    world_size = 1
    local_rank = 0
    
    # 读取数据集
    test_dataset = json.load(open(args.dataset_path)) # 读取数据集
    if args.exist_file != 'None' and os.path.exists(args.exist_file):
        logger.info(f'load exist dataset file: {args.exist_file}')
        exist_file_dataset = json.load(open(args.exist_file))
        exist_file_id = [item['id'] for item in exist_file_dataset]
        exist_file_id = set(exist_file_id) # 去重
        new_test_dataset = []
        for item in tqdm(test_dataset):
            if item['id'] in exist_file_id:
                continue
            else:
                new_test_dataset.append(item)
        test_dataset = new_test_dataset
    test_dataset = expand_list(test_dataset, args.best_of_n)
    logger.info(f'len_test_datasets: {len(test_dataset)}')

    # 加载policy model, judge model
    # policy model 
    policy_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if not args.use_api:
        # policy_model = LLM(
        #     model=args.model_path,
        #     device=f'cuda:{local_rank}',
        #     gpu_memory_utilization=0.25,
        #     enforce_eager=True,
        #     enable_chunked_prefill=True,
        # )
        pass
        policy_model = None
    else:
        policy_model = None
    

    # init metrics
    manager = Manager()
    records_queue = manager.Queue()
    records_queue_lock = manager.Lock()
    max_try_times = args.max_try_times

    shared_params = {}
    all_tasks = []
    line_id = -1
    
    
    lines = test_dataset
    if args.num_workers <= 1:
        for line in tqdm(lines):
            line_id += 1
            params = {}
            params["line"] = line
            params["line_id"] = line_id
            params["policy_tokenizer"] = policy_tokenizer
            params["max_try_times"] = max_try_times

            cur_task = (params, args, records_queue, records_queue_lock, shared_params)
            do_single_task(*cur_task)
            if line_id > 10:
                break
    else:
        for line in tqdm(lines):
            line_id += 1
            params = {}
            params["line"] = line
            params["line_id"] = line_id
            params["policy_tokenizer"] = policy_tokenizer
            params["max_try_times"] = max_try_times

            cur_task = (params, args, records_queue, records_queue_lock, shared_params)
            all_tasks.append(cur_task)
            # if line_id > 100:
            #     break
        def print_error(error):
            print("error:", error)

        poolSize = args.num_workers
        pool = Pool(poolSize)
        pool.starmap_async(try_do_single_task, all_tasks, error_callback=print_error)
        pool.close()
        tq = tqdm(total=len(all_tasks))
        count = 0
        print("begin")
        #try:
        while count < len(all_tasks):
            try:
                c = records_queue.get_nowait()
            except queue.Empty:
                continue
            count += 1
            tq.update(1)

        pool.join()
    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./experiments/infer_result/policy_result/')
    parser.add_argument("--model_path", type=str, default='./pretrained_models/Qwen2.5-1.5B-Instruct') # policy model
    parser.add_argument("--pigai_model_path", type=str, default='./pretrained_models/Qwen2.5-1.5B-Instruct') # pigai model
    parser.add_argument("--dataset_path", type=str, default='./dataset/Text_Only/benchmark/MATH-500/MATH_500.json')
    parser.add_argument("--qwen2vl_infer_batch", type=int, default=1)
    parser.add_argument("--best_of_n", type=int, default=1) # BoN
    parser.add_argument("--max_try_times", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.9) # BoN
    parser.add_argument("--top_k", type=int, default=50) # BoN
    parser.add_argument("--temperature", type=float, default=0.7) # BoN
    parser.add_argument("--max_len", type=int, default=16000)
    parser.add_argument("--chat_template", type=str, default="qwen_math_text_only") # qwen3_fast_thinking
    parser.add_argument("--use_thinking_budget", type=str, default='False')
    parser.add_argument("--thinking_budget", type=int, default=10000)
    parser.add_argument("--max_interaction_round", type=int, default=3)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--exist_file", type=str, default='None')
    parser.add_argument("--is_debug", type=str, default='False')
    parser.add_argument("--use_api", type=str, default='False')
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    

    set_start_method("spawn")

    rank = 0
    logging.basicConfig(
        format="Node[{}] %(asctime)s - %(levelname)s - %(message)s".format(rank),
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
            ]
    )
    logger.setLevel(logging.INFO)

    def handle_exception(exc_type, exc_value, exc_tb):
        if exc_type == SystemExit:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.error("未捕获的异常", exc_info=(exc_type, exc_value, exc_tb))
    sys.excepthook = handle_exception


    from datetime import datetime
    now = datetime.now()
    current_time_str = now.strftime("%Y%m%d%H")
    dataset_name = args.dataset_path.split('/')[-1].replace('.json', '')
    exp_name = f'BoN_{args.best_of_n}_[{dataset_name}]_{current_time_str}'
    par_dir = args.output_dir
    args.output_dir = os.path.join(args.output_dir, exp_name)
    if os.path.exists(args.output_dir):
        dirs = os.listdir(par_dir)
        num = [item for item in dirs if exp_name in item]
        num_len = len(num)
        args.output_dir = args.output_dir + '_' + str(num_len+1)


    if args.use_api == 'True':
        args.use_api = True
    else:
        args.use_api = False
     

    main(args)



