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

def chat_with_Qwen3_policy_request(input_text, max_length, model_name, eos_token):
    data = {
        "model": model_name,
        "prompt": input_text,
        "max_tokens": max_length,
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": 30,
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
        print(res)
        raise ValueError
    return output_text


def check_abnormal_boxed_response(tokenizer, response, last_token_num=100):
    '''
    Args:
        response: string
    
    Returns: 
        bool: if the response contain boxed{} response
    '''
    
    # breakpoint()
    pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    response_mask = (response != pad_token_id)
    response_length = response_mask.sum(dim=-1)
    response = response[:response_length] # no pad token
    response = response[-last_token_num:] # the last 100 tokens
    response_str = tokenizer.decode(response)
    
    # 在最后100个token中没有boxed{}框的话, 就把结果过滤掉
    boxed_answer = re.findall(r"\\boxed{(.*)}", response_str, re.DOTALL)
    if len(boxed_answer) > 0 and len(boxed_answer[-1]) > 0:
        return True
    else:
        return False



def interaction_sample(rank, test_dataset, model, tokenizer, qwen2vl_infer_batch, args, use_api=False):
    # test_dataset = expand_list(test_dataset, args.best_of_n)
    test_dataset_batch = split_list(test_dataset, 1)
    
    batch_num = 0


    max_response_len = args.max_len
    max_interaction_round = args.max_interaction_round
    max_prompt_length = 1000

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


    total_output_texts = []
    total_execution_passes = []
    total_response_length = []
    with torch.no_grad():
         for batch in test_dataset_batch: # 分batch推理
            batch_num += 1
            logger.info(f"rank: {rank}, {batch_num}/{len(test_dataset_batch)}, {batch_num/len(test_dataset_batch)}")
            

            batch_id_list = [item['id'] for item in batch]
            if args.chat_template == 'qwen_math':
                interaction_prompt = [prompt.qwen_math_interaction_prompt_en.replace("[PROMPT]", item['question']) for item in batch]
            elif args.chat_template == 'qwen3_slow_thinking':
                interaction_prompt = [prompt.qwen3_tir_interaction_prompt_en.replace("[PROMPT]", item['question']) for item in batch]
            else:
                raise ValueError(f"Unsupported chat template {args.chat_template}")
            
            
            messages = [[{"role": "user", "content": _prompt}] for _prompt in interaction_prompt]
            code_and_result = []
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            cur_input_text = input_text
            
            vllm_inputs = [{'prompt_token_ids': tokenizer.encode(_text)} for _text in input_text]
            
            responses_list, real_response_length, execution_passes = _tir_generate(model, vllm_inputs, sampling_params=sampling_params, use_tqdm=False, tokenizer=tokenizer, max_interaction_round=max_interaction_round, prompt_length=max_prompt_length, response_length=max_response_len, use_api=use_api, model_name=args.model_path, args=args)
            # real_response_length, 真实的reasponse length, 通过统计vllm的输出来计算


            _response_length = [len(res) for res in responses_list]
            code_and_result = [[] for _ in range(len(cur_input_text))]
            responses_list = [tokenizer.decode(item) for item in responses_list]
            responses_list_with_prompt = [x+y for x,y in zip(cur_input_text, responses_list)]
            cur_output_text = [{'id': cur_id, 'predict': cur_predict, 'code_and_result': cur_code} for cur_id, cur_predict, cur_code in zip(batch_id_list, responses_list_with_prompt, code_and_result)]
            

            total_output_texts += cur_output_text
            total_execution_passes += execution_passes.tolist()
            total_response_length += _response_length
            
            filter_code = True
            filter_abnormal_length = True
            filter_abnormal_boxed_string = True
            filter_ratio = 0.0
            min_response_length = 150
            if filter_code or filter_abnormal_length or filter_abnormal_boxed_string:
                # print("begin filter code output result.")
                group_num = args.best_of_n
                batch_size = len(cur_output_text)
                thred_num = math.ceil(filter_ratio * group_num)

                execution_passes_list = execution_passes
                execution_passes_per_query = [execution_passes_list[i:i + group_num] for i in range(0, batch_size, group_num)]
                response_length_list = _response_length

                kept_index = []
                for query_id, query_execution_pass in enumerate(execution_passes_per_query):
                    # no code failed and true response length
                    prompt_kept_flag = []
                    for i, score in enumerate(query_execution_pass):
                        global_index = group_num * query_id + i
                        flag_code = True
                        flag_length = True
                        flag_boxed_string = True
                        if filter_code:
                            flag_code = execution_passes_list[global_index] == 1
                        if filter_abnormal_length:
                            flag_length = response_length_list[global_index] > min_response_length
                        if filter_abnormal_boxed_string:
                            response_ids = torch.tensor(tokenizer.encode(cur_output_text[global_index]['predict']))
                            flag_boxed_string = check_abnormal_boxed_response(tokenizer, response_ids)
                            
                        if flag_code and flag_length and flag_boxed_string:
                            prompt_kept_flag.append(1)
                        else:
                            prompt_kept_flag.append(0)
                    kept_num = len([flag for flag in prompt_kept_flag if flag==1])
                    if kept_num < thred_num:
                        prompt_kept_flag = [0 for _ in range(len(prompt_kept_flag))]
                    kept_index += prompt_kept_flag
            else:
                # kept_index = list(range(len(response)))
                kept_index = [1 for _ in range(len(cur_output_text))]

            for i in range(len(cur_output_text)):
                cur_output_text[i]['kept_flag'] = kept_index[i]




            # 保存临时结果
            for idx in range(len(cur_output_text)):
                cur_output_text[idx]['question'] = batch[idx]['question']
                cur_output_text[idx]['answer'] = batch[idx]['answer']
                cur_output_text[idx]['real_response_length'] = real_response_length
            with open(os.path.join(args.output_dir, f'tmp_result_rank_{rank}.jsonl'), "a", encoding="utf-8") as f:
                for _item in cur_output_text:
                    f.write(json.dumps(_item, ensure_ascii=False) + "\n")




def _get_prompts_and_indices(samples_info, validate=True):
    prompts, indices=[], []
    for index, info in enumerate(samples_info): 
        # 如果存在代码执行失败的结果, 不继续往下推理
        # [item['execution_pass'] for item in samples_info]
        if not validate:
            if 0 in info['execution_pass']: # 推理时即使有执行失败的结果也推理完
                continue 
        if not info['stop']:
            prompts.append(info['sequence'])
            indices.append(info['index'])
    return prompts, indices


def extract_program(result: str, last_only=True):
    """
    extract the program after "```python", and before "```"
    """
    program = ''
    start = False
    for line in result.split('\n'):
        if line.startswith('```python') or line.endswith('```python') or line.startswith('python') or line.endswith('python'):
            if last_only:
                program = ''  # only extract the last program
            else:
                program += '\n# ========\n'
            start = True
        elif line.startswith('```'):
            start = False
        elif start:
            program += line + '\n'
    if start:
        # the code is incomplete
        program = ''
    return program

def _detect_tool(text: str) -> Tuple[bool, str, str, str]:
    program = extract_program(text)
    if program:
        program = json.dumps({'code': program}, ensure_ascii=False)
        try:
            _x = json5.loads(program) # 过滤到无法被json解析的code
        except:
            program = ''

    return (program != ''), 'python_executor', program, text


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

def send_request(json_data):
    try:
        url = 'http://127.0.0.1:8080/run_code/'
        response = requests.post(url, json=json_data, timeout=30)
        return response.json()  # 返回响应的 JSON 数据
    except:
        # print("sanbox timeout")
        return {"error": "unknown"}


def code_interpreter_batch_call(tool_inputs, timeout=30):
    tool_inputs=[{'code': tool_input, 'language': 'python', "run_timeout": timeout, "compile_timeout": timeout} for tool_input in tool_inputs]
    results = [None] * len(tool_inputs)
    max_workers = 6
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(send_request, input): i for i, input in enumerate(tool_inputs)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result(timeout=timeout)
                results[index] = result
            except:
                results[index] = {"run_result": {"stdout": "Error", "stderr": "TimeoutError"}}
    
    def postproc(output):
        # breakpoint()
        try:
            if str(output['run_result']['return_code'])=='0' and len(str(output['run_result']['stdout'])) != 0:
                return output['run_result']['stdout'], "Done"
            else:
                return output['run_result']['stdout'], output['run_result']['stderr'].strip()
        except Exception:
            return "Error", "UnknownError"
    results=[postproc(result) for result in results]
    return results



def _tokenize_and_find_mask_token_indices(sample_info, tokenizer):
    response=sample_info['response']
    mask_str_ranges=sample_info['mask_info']

    encoding=tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    
    response_token_ids=encoding['input_ids']

    offset_mapping_tensor=torch.tensor(encoding['offset_mapping'], dtype=torch.long)
    token_starts = offset_mapping_tensor[:,0]
    token_ends = offset_mapping_tensor[:,1]

    mask_tensor=torch.ones(len(response_token_ids))
    for mask_str_range in mask_str_ranges:
        start_index, end_index=mask_str_range[0], mask_str_range[1]
        mask = (token_starts < end_index) & (token_ends > start_index) & (token_starts >= start_index)
        mask_tensor[mask]=0 

    return response_token_ids, mask_tensor




def _tir_generate(inference_engine, prompts=None, sampling_params=None, use_tqdm=False, tokenizer=None, max_interaction_round=3, prompt_length=1000, response_length=3000, use_api=False, model_name=None, args=None):
    # sampling_params = copy.deepcopy(sampling_params)
    # prompts=self.tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=True)
    prompts=[tokenizer.decode(prompt['prompt_token_ids'], skip_special_tokens=False) for prompt in prompts]
    prompts=[prompt for prompt in prompts for _ in range(1) ]
    # sampling_params.n=1
    # sampling_params.detokenize=True
    # sampling_params.include_stop_str_in_output=True
    
    top_p = args.top_p
    top_k = args.top_k
    temperature = args.temperature

    OBS_START = '```output'
    OBS_END = '\n```\n'
    stop_list = ["```"]
    stop = stop_list
    samples_info = [{"prompt": prompt, "sequence": prompt, "response": "", "stop": False, "finish_reason": None, "index": index, "mask_info": [], "execution_pass": []} for index, prompt in enumerate(prompts)] # save sample info in dict
    # program2output=[]
    num_llm_calls_available = copy.deepcopy(max_interaction_round)
    
    if args.use_thinking_budget == 'True':
        response_length = args.thinking_budget
    self_try_id = 0 
    real_response_length = 0
    while num_llm_calls_available >= 0:

        if num_llm_calls_available<=0: stop=None
        
        # llm generate response, stop at eos token or stop_token
        input_prompts, indices = _get_prompts_and_indices(samples_info, validate=True)
        
        max_len = 0
        for i, x in enumerate(input_prompts):
            prompt_token_ids = tokenizer.encode(x, add_special_tokens=False)
            max_len = max(max_len, len(prompt_token_ids))
            input_prompts[i] = {'prompt_token_ids': prompt_token_ids[:prompt_length + response_length]}
        if max_len > response_length: # 注意, 只适用于batch_size为1的情况!
            assert len(input_prompts) == 1
            break

        
        # input_prompts = [{'prompt_token_ids': tokenizer.encode(x, add_special_tokens=False)[:prompt_length + response_length]} for x in input_prompts]
        if not use_api:
            raise ValueError
            # outputs = inference_engine.generate(prompts=input_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
            # sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
            # responses=[x.outputs[0].text for x in sorted_outputs]
            # finish_reason=[x.outputs[0].finish_reason for x in sorted_outputs]
            # stop_reason=[x.outputs[0].stop_reason for x in sorted_outputs]
        else:
            sample_len = response_length + prompt_length - len(input_prompts[0]['prompt_token_ids'])
            input_text = tokenizer.decode(input_prompts[0]['prompt_token_ids'])
            outputs = chat_with_api(input_text, max_length=sample_len, model_name=model_name, stop=stop, temperature=temperature, top_p=top_p, top_k=top_k)
            responses = [x['text'] for x in outputs]
            finish_reason = [x['finish_reason'] for x in outputs]
            stop_reason = [x['stop_reason'] for x in outputs]
            
            if args.self_retry:
                real_response_length += len(tokenizer.encode(responses[0]))


        if num_llm_calls_available == -1:
            for i ,index in enumerate(indices):
                samples_info[index]['response']+=responses[i]
                samples_info[index]['sequence']+=responses[i]
                samples_info[index]['stop']=True
                samples_info[index]['finish_reason']=finish_reason[i]
            break

        def _python_execution(finish_reason, stop_reason):
            if finish_reason=='stop' and stop_reason == None: return False
            if finish_reason=='stop' and stop_reason in stop_list: return True
            if finish_reason=='length': False
            return False
        is_execution = [_python_execution(finish_reason[i], stop_reason[i]) for i in range(len(finish_reason))]
        # check if all samples are finished
        if all([not x for x in is_execution]):
            for i, index in enumerate(indices):
                samples_info[index]['response'] += responses[i]
                samples_info[index]['sequence'] += responses[i]
                samples_info[index]['stop'] = True
                samples_info[index]['finish_reason'] = finish_reason[i]
            break
        
        # prepare for python execution
        tool_infos = [ _detect_tool(response) for response in responses]
        tool_indices=[]
        tool_inputs=[]
        for i, tool_info in enumerate(tool_infos):
            if tool_info[0] and is_execution[i]:
                tool_indices.append(i)
                tool_inputs.append(tool_info[2])
       
        
        def postproc_observation(observation):
            execution_pass=0
            try:
                observation_list=observation
                if observation_list[-1] == 'Done':
                    observation = observation_list[0]
                    execution_pass=1
                else:
                    observation = observation_list[-1]
            except Exception:
                observation="Error"
            if "Error" in observation: observation=observation.strip().split("\n")[-1]
            if len(observation.strip())==0: observation="timeout_decorator.timeout_decorator.TimeoutError: 'Timed Out'"
            observation = observation.strip()
            if len(observation)>=256:
                observation = observation[:128]+"..."+observation[-128:]
            observation = f'\n{OBS_START}\n{observation}{OBS_END}'
            return observation, execution_pass

        # execute python code
        if len(tool_inputs) > 0:
            num_llm_calls_available-=1
            # execute python code
            use_remote_code = True
            call_list = []
            for x in tool_inputs:
                call_list.append(json5.loads(x)['code'])

            if use_remote_code:
                observations = code_interpreter_batch_call(call_list)
            else:
                # observations = executor.batch_apply(call_list)
                raise ValueError
        else:
            observations = []
        
        # construction responses from observations
        # responses = [response+"\n" if not response.endswith('\n') else response for response in responses]
        responses_w_res=copy.deepcopy(responses)
        execution_passes=[[] for _ in range(len(responses))]
        for i, index in enumerate(tool_indices):
            processed_observation=postproc_observation(observations[i])
            responses_w_res[index]+=processed_observation[0]
            execution_passes[index].append(processed_observation[1])

        # program2output.append([{"code": tool_input, "answer": postproc_observation(observations[idx])} for idx, tool_input in enumerate(tool_inputs)])
        # update samples_info
        def backtrack_response(tokenizer, response, num_token):
            if "<|im_start|>assistant\n" in response:
                response_split = response.split("<|im_start|>assistant\n")
                prompt = response_split[0]
                response = response_split[1]
            else:
                prompt = ""
                response = response
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)[:-num_token]
            response_ids = prompt_ids + response_ids
            return tokenizer.decode(response_ids)

        if args.self_retry:
            # 最多尝试n次
            backtrack_token = args.backtrack_token
            self_retry_num = args.self_retry_num
            for i ,index in enumerate(indices):
                if i in tool_indices:
                    if 0 in execution_passes[i] and self_try_id < self_retry_num:
                        # 执行失败, 忽略代码并向前回溯若干token, 重新推理
                        self_try_id += 1
                        samples_info[index]['response'] = backtrack_response(tokenizer, samples_info[index]['response'], num_token=backtrack_token)
                        samples_info[index]['sequence'] = backtrack_response(tokenizer, samples_info[index]['sequence'], num_token=backtrack_token)
                        samples_info[index]['stop']=not is_execution[i]
                        samples_info[index]['finish_reason']=finish_reason[i]
                    else:
                        # 执行成功, 或者超过重试次数
                        self_try_id = 0
                        mask = [ len(responses[i]) + len('```output'), len(responses_w_res[i]) ]
                        samples_info[index]['mask_info'].append(mask)
                        samples_info[index]['execution_pass']+=execution_passes[i]
                        samples_info[index]['response']+=responses_w_res[i]
                        samples_info[index]['sequence']+=responses_w_res[i]
                        samples_info[index]['stop']=not is_execution[i]
                        samples_info[index]['finish_reason']=finish_reason[i]
                else:
                    samples_info[index]['response']+=responses_w_res[i]
                    samples_info[index]['sequence']+=responses_w_res[i]
                    samples_info[index]['stop']=not is_execution[i]
                    samples_info[index]['finish_reason']=finish_reason[i]
        else:
            for i ,index in enumerate(indices):
                if i in tool_indices:
                    mask = [ len(responses[i]) + len('```output'), len(responses_w_res[i]) ]
                    samples_info[index]['mask_info'].append(mask)
                    samples_info[index]['execution_pass']+=execution_passes[i]
                samples_info[index]['response']+=responses_w_res[i]
                samples_info[index]['sequence']+=responses_w_res[i]
                samples_info[index]['stop']=not is_execution[i]
                samples_info[index]['finish_reason']=finish_reason[i]
    
    for i, line in enumerate(samples_info):
        if samples_info[i]['finish_reason']!='length': samples_info[i]['response']+=tokenizer.eos_token

    responses_ids=[]
    tool_output_masks=[]
    execution_passes=[]
    for idx, sample_info in enumerate(samples_info):
        response_id, tool_output_mask = _tokenize_and_find_mask_token_indices(sample_info, tokenizer)
        # responses_ids.append(response_id[:response_length])
        # tool_output_masks.append(tool_output_mask[:response_length])
        responses_ids.append(response_id[:])
        tool_output_masks.append(tool_output_mask[:])

        if len(sample_info['execution_pass']) == 0:
            execution_passes.append(1.0)
        else:
            execution_passes.append(sum(sample_info['execution_pass']) / len(sample_info['execution_pass']))
        # save id and mask to check correctness
    #     samples_info[idx]['responses_id']=response_id[:self.config.response_length]
    #     samples_info[idx]['tool_output_mask']=tool_output_mask[:self.config.response_length].tolist()
    

    # thinking_budget
    # 检查think是否结束
    if args.use_thinking_budget == 'True':
        input_text = tokenizer.decode(responses_ids[0])
        if input_text.endswith(tokenizer.eos_token):
            pass # end
        else:
            if "</think>" not in input_text: # early stop
                early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                input_text += early_stopping_text
            else:
                input_text = input_text
            answer_length = args.max_len - args.thinking_budget
            outputs = chat_with_api(input_text, max_length=answer_length, model_name=model_name, stop=None, temperature=temperature, top_p=top_p, top_k=top_k)
            responses = [input_text + x['text'] for x in outputs]
            responses_ids = [tokenizer.encode(responses[0])]


    return responses_ids, real_response_length, torch.tensor(execution_passes, dtype=torch.long)






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
    parser.add_argument("--chat_template", type=str, default="qwen3_slow_thinking")
    parser.add_argument("--use_thinking_budget", type=str, default='False')
    parser.add_argument("--thinking_budget", type=int, default=10000)
    parser.add_argument("--max_interaction_round", type=int, default=3)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--self_retry", type=str, default='False')
    parser.add_argument("--backtrack_token", type=int, default=500)
    parser.add_argument("--self_retry_num", type=int, default=4)
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
        # 统计同名的数量
        num = [item for item in dirs if exp_name in item]
        num_len = len(num)
        args.output_dir = args.output_dir + '_' + str(num_len+1)


    if args.use_api == 'True':
        args.use_api = True
    else:
        args.use_api = False
    
    if args.self_retry == 'True':
        args.self_retry = True
    else:
        args.self_retry = False

    main(args)



