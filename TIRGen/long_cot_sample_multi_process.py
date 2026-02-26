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
from vllm import SamplingParams
# from inference.utils import PigaiModel, generate_pigai_prompt_tokens
# from inference.utils import MathAccuracyORM
from inference import math_verifier
from config.prompt import *
from multiprocessing import Process, synchronize, Lock, Manager, Pool, set_start_method
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from six.moves import queue


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
        print(res.json())
        raise ValueError
    return output_text
    

def interaction_sample(rank, test_dataset, policy_model, advisor_model, policy_tokenizer, advisor_tokenizer, qwen2vl_infer_batch, try_id, args, use_api=False):
    # test_dataset_batch = split_list(test_dataset, qwen2vl_infer_batch) # 分batch
    total_output_texts = []
    total_predict_answers = []
    batch_num = 0

    top_p = args.top_p
    top_k = args.top_k
    temperature = args.temperature

    total_max_length = args.max_len
    step_len = args.step_len
    window_size = args.window_size
    short_cot = args.short_cot
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    policy_sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=step_len,
        skip_special_tokens=False,
        include_stop_str_in_output=True
    )

    advisor_sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=step_len,
        skip_special_tokens=False,
        include_stop_str_in_output=False
    )

    for i, item in enumerate(test_dataset):
        try:
            success = False
            advisor_response = None
            last_input_text = None
            end_think = False
            start_index = 0
            end_index = 0
            execution_passes = 0.0
            while True:
                # 将advisor_response拼到policy_inputs中
                if last_input_text is not None:
                    # 修正的思维链
                    if (advisor_response is not None) and (advisor_response not in last_input_text):
                        advised_policy_response = advisor_response
                    else:
                        advised_policy_response = policy_response
                    
                    if short_cot:
                        policy_input_text = last_input_text + advised_policy_response
                        end_think = True
                    else:
                        if "<think>" not in last_input_text and not advised_policy_response.startswith("<think>"):
                            policy_input_text = last_input_text + f"<think>\n{advised_policy_response}"
                        elif "<think>" not in last_input_text and advised_policy_response.startswith("<think>"):
                            policy_input_text = last_input_text + f"{advised_policy_response}"
                        else:
                            # 多次修正的input_text
                            policy_input_text = last_input_text + f"{advised_policy_response}"
                    
                        if "</think>" in policy_input_text:
                            end_think = True
                else:
                    # 第一次推理
                    policy_messages = [
                            {"role": "user", "content": policy_prompt.replace('[question]', item['question']) }
                        ]
                    policy_input_text = advisor_tokenizer.apply_chat_template(
                                policy_messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=True
                            )
                if policy_input_text.endswith("<|im_end|>"):
                    break
                if len(policy_tokenizer.encode(policy_input_text)) > total_max_length:
                    break
                last_input_text = policy_input_text
                
                if not use_api:
                    policy_outputs = policy_model.generate(policy_input_text, policy_sampling_params, use_tqdm=False)
                    policy_response = policy_outputs[0].outputs[0].text
                else:
                    # policy_outputs = chat_with_Qwen3_4B(policy_messages, max_len=policy_sampling_params.max_tokens, think=True)
                    # policy_response = policy_outputs.message.content
                    if args.short_cot:
                        query_length = policy_sampling_params.max_tokens - len(policy_tokenizer.encode(policy_input_text))
                        policy_response = chat_with_Qwen3_policy_request(policy_input_text, max_length=query_length, model_name=args.model_path, eos_token=policy_tokenizer.eos_token)
                    else:
                        policy_response = chat_with_Qwen3_policy_request(policy_input_text, max_length=policy_sampling_params.max_tokens, model_name=args.model_path, eos_token=policy_tokenizer.eos_token)
                
                # print(policy_response)
                total_response = policy_input_text + policy_response
                
                long_cot_response = total_response
                
                if short_cot:
                    think_part = long_cot_response
                else:
                    if "</think>" in long_cot_response: # 有完整的思维链
                        pattern_non_greedy = r'<think>(.*?)</think>'
                        match = re.search(pattern_non_greedy, long_cot_response, re.DOTALL) 
                        if match:
                            think_part = match.group(1)
                        else:
                            raise ValueError(f"{long_cot_response}")
                    else: # 没有完整的思维链
                        pattern_non_greedy = r'<think>(.*)'
                        match = re.search(pattern_non_greedy, long_cot_response, re.DOTALL) 
                        if match:
                            think_part = match.group(1)
                        else:
                            raise ValueError(f"{long_cot_response}")

                if end_think:
                    continue # 如果结束think, 就不再和advisor交互, 持续推理直到<|im_end|>


                think_token_ids = advisor_tokenizer.encode(think_part)
                # 滑动窗口对think过程处理
                token_fragments = []
                L = len(think_token_ids)
                end_index = start_index + window_size
                # 确保片段不会超出 token_ids 的长度
                optional_fragments = check_possible_fragment_index(policy_tokenizer, think_token_ids)

                if end_index > L:
                    # 如果剩余的 token 不足一个完整的窗口，可以根据需求选择处理方式：
                    # 1. 忽略剩余的 token
                    # 2. 截取到 L (end_index = L)
                    # 3. 填充 (例如用特殊 token) 使其达到 window_size
                    
                    end_index = L
                if len(optional_fragments) > 0:
                    # find the nearest fragments index
                    distance = [abs(frag - end_index) for frag in optional_fragments]
                    min_distance_index = distance.index(min(distance))
                    end_index = optional_fragments[min_distance_index]


                if short_cot:
                    current_fragment_ids = think_token_ids[:-1]
                else:
                    current_fragment_ids = think_token_ids[start_index: end_index]
                    # 如果节选的片段过短, 就没必要使用advisor推理
                    if len(current_fragment_ids) < 2048:
                        continue
                
                current_fragment = advisor_tokenizer.decode(current_fragment_ids)
                prompts = reasoning_step_to_python_code_en.replace("[long_cot]", current_fragment)
                # print(prompts)
                
                advisor_messages = [
                    {"role": "user", "content": prompts}
                ]
                advisor_input_text = advisor_tokenizer.apply_chat_template(
                            advisor_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                
                # 与代码工具交互
                vllm_inputs = [{'prompt_token_ids': advisor_tokenizer.encode(advisor_input_text)}]
                advisor_response_ids, _, execution_passes = _tir_generate(advisor_model, vllm_inputs, advisor_sampling_params, tokenizer=advisor_tokenizer, max_interaction_round=5, prompt_length=10000, response_length=step_len, use_api=use_api, model_name=args.advisor_model_path)
                advisor_response = advisor_tokenizer.decode(advisor_response_ids[0]).replace(advisor_tokenizer.eos_token, "")
                execution_passes = execution_passes.tolist()[0]
                if execution_passes < 1.0:
                    break # 执行失败的response丢弃
                
                # 直接vllm生成
                # advisor_outputs = advisor_model.generate(advisor_input_text, advisor_sampling_params, use_tqdm=False)
                # advisor_response = advisor_outputs[0].outputs[0].text

                # 调用API
                # print(advisor_response)
                # advisor_outputs = chat_with_Qwen3_32B(advisor_messages)
                # advisor_response = advisor_outputs.message.content
                # print(advisor_response)

                # 下一次修正从advisor输出的结束位置开始, 因为advisor的输出会拼接到思维链上
                advisor_response_ids = advisor_tokenizer.encode(advisor_response)
                start_index += len(advisor_response_ids) + 1

            # 保存结果
            item["execution_passes"] = execution_passes
            item["predict"] = total_response
            item["predict_length"] = len(policy_tokenizer.encode(total_response))
            item["try_id"] = try_id
            try:
                item["math_verify_score"] = math_verifier.compute_score(total_response, item["answer"])
            except:
                item["math_verify_score"] = -1.0
            
            output_file = os.path.join(args.output_dir, f'tmp_result_rank_{rank}.jsonl')
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # 检查query是否成功回答
            success = filter_dataset(sample_data=item)
        except Exception as e:
            print(e)

    
    return success



def filter_dataset(sample_data):
    def check_repetition(completion: str, repetition_ngram_size=50, repetition_max_num=30) -> bool:
        """
        检测当前回复是否出现重复解码
        Returns:
            flags: True/False
        """
        def get_repetition_penalty(str_: str, ngram_size: int) -> float:
            n_garm_counter = dict()
            max_ = 0
            most_freq_n_gram = ''
            tokens = str_
            for i in range(0, len(tokens) - ngram_size):
                tmp_n_gram = tokens[i: i+ngram_size]
                if n_garm_counter.get(tmp_n_gram) is None:
                    n_garm_counter[tmp_n_gram] = 0
                n_garm_counter[tmp_n_gram] += 1
                if max_ < n_garm_counter[tmp_n_gram]:
                    max_ = n_garm_counter[tmp_n_gram]
                    most_freq_n_gram = tmp_n_gram
            return max_

        max_repetition_num = get_repetition_penalty(completion, repetition_ngram_size)

        if max_repetition_num > repetition_max_num:
            # exist repetition
            return True
        else:
            # not exist repetition
            return False

    def check_think(predict: str):
        """
        包含完整的think字段, 且think字段中含有python代码
        Returns:
            flags: True/False
        """
        think_part_match = re.findall(r"<think>(.*?)</think>", predict, re.DOTALL)
        code_num = 0
        if think_part_match and len(think_part_match) > 0:
            assert len(think_part_match) == 1
            think_part_str = think_part_match[0]
            assert len(think_part_str) > 0
            code_match = re.findall(r"```python(.*?)```", think_part_str, re.DOTALL)
            if code_match and len(code_match) > 0:
                code_num = len(code_match)
                return True, code_num
        return False, code_num

    def check_answer(predict: str):
        """
        包含完整的answer字段, 且answer字段不含代码
        Returns:
            flags: True/False
        """
        if "</think>" in predict:
            answer_part_str = predict.split("</think>")[1]
            if len(answer_part_str) > 0:
                code_match = re.findall(r"```python(.*?)```", answer_part_str, re.DOTALL)
                # if len(code_match) == 0:
                #     return True
                if answer_part_str.endswith("<|im_end|>"):
                    return True
                else:
                    x = 1
                    return False
        return False


    item = sample_data
    if item["math_verify_score"] != 1.0:
        return False

    # 代码执行正确
    if item["execution_passes"] != 1.0:
        return False
    
    predict = item["predict"]
    # 检查think字段
    think_flag, code_num = check_think(predict)
    if not think_flag:
        return False

    # 检查answer字段
    answer_flag = check_answer(predict)
    if not answer_flag:
        return False

    # 过滤重复解码
    repetition_flag = check_repetition(predict)
    if repetition_flag:
        return False

    return True






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


def chat_with_api(input_text, max_length, model_name, stop):
    data = {
        "model": model_name,
        "prompt": input_text,
        "max_tokens": max_length,
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": 30,
        "stop": stop,
        "include_stop_str_in_output": True,
    }
    headers = {"Authorization": "Bearer token-123456"}
    res = requests.post(f"http://localhost:8001/v1/completions", headers=headers, json=data)
    outputs = res.json()["choices"]
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




def _tir_generate(inference_engine, prompts=None, sampling_params=None, use_tqdm=False, tokenizer=None, max_interaction_round=3, prompt_length=1000, response_length=3000, use_api=False, model_name=None):
    group_num = sampling_params.n
    sampling_params = copy.deepcopy(sampling_params)
    # prompts=self.tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=True)
    prompts=[tokenizer.decode(prompt['prompt_token_ids'], skip_special_tokens=False) for prompt in prompts]
    prompts=[prompt for prompt in prompts for _ in range(sampling_params.n) ]
    sampling_params.n=1
    sampling_params.detokenize=True
    sampling_params.include_stop_str_in_output=True

    OBS_START = '```output'
    OBS_END = '\n```\n'
    stop_list = ["```"]
    sampling_params.stop = stop_list
    samples_info = [{"prompt": prompt, "sequence": prompt, "response": "", "stop": False, "finish_reason": None, "index": index, "mask_info": [], "execution_pass": []} for index, prompt in enumerate(prompts)] # save sample info in dict
    # program2output=[]
    num_llm_calls_available = copy.deepcopy(max_interaction_round)
    

    while num_llm_calls_available >= 0:

        if num_llm_calls_available<=0: sampling_params.stop=None
        
        # llm generate response, stop at eos token or stop_token
        input_prompts, indices = _get_prompts_and_indices(samples_info, validate=True)
        input_prompts = [{
            'prompt_token_ids': tokenizer.encode(x, add_special_tokens=False)[:prompt_length + response_length]} for x in input_prompts]
        if not use_api:
            outputs = inference_engine.generate(prompts=input_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
            sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
            responses=[x.outputs[0].text for x in sorted_outputs]
            finish_reason=[x.outputs[0].finish_reason for x in sorted_outputs]
            stop_reason=[x.outputs[0].stop_reason for x in sorted_outputs]
        else:
            # outputs = inference_engine.generate(prompts=input_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
            input_text = tokenizer.decode(input_prompts[0]['prompt_token_ids'])
            outputs = chat_with_api(input_text, max_length=sampling_params.max_tokens, model_name=model_name, stop=sampling_params.stop)
            responses = [x['text'] for x in outputs]
            finish_reason = [x['finish_reason'] for x in outputs]
            stop_reason = [x['stop_reason'] for x in outputs]
            

        if num_llm_calls_available==-1:
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
                observations=code_interpreter_batch_call(call_list)
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
        responses_ids.append(response_id[:response_length])
        tool_output_masks.append(tool_output_mask[:response_length])
        if len(sample_info['execution_pass']) == 0:
            execution_passes.append(1.0)
        else:
            execution_passes.append(sum(sample_info['execution_pass']) / len(sample_info['execution_pass']))
        # save id and mask to check correctness
    #     samples_info[idx]['responses_id']=response_id[:self.config.response_length]
    #     samples_info[idx]['tool_output_mask']=tool_output_mask[:self.config.response_length].tolist()
    

    return responses_ids, tool_output_masks, torch.tensor(execution_passes, dtype=torch.long)






def do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    line = params["line"]
    line_id = params["line_id"]
    policy_tokenizer = params["policy_tokenizer"]
    advisor_tokenizer = params["advisor_tokenizer"]
    max_try_times = params["max_try_times"]
    infer_batch_size = args.qwen2vl_infer_batch
    test_dataset = [line]
    
    rank = line_id
    
    for try_id in range(max_try_times):
        success = interaction_sample(rank, test_dataset, None, None, policy_tokenizer, advisor_tokenizer, infer_batch_size, try_id, args, use_api=True)
        # if success:
        #     break
    


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
    # test_dataset = expand_list(test_dataset, args.best_of_n) # 不需要重复数据集了, 在do_single_task中通过max_try_times来控制.
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
    

    advisor_tokenizer = AutoTokenizer.from_pretrained(args.advisor_model_path)
    if not args.use_api:
        if args.advisor_model_path == args.model_path:
            advisor_model = policy_model
        else:
            # advisor_model = LLM(
            #     model=args.advisor_model_path,
            #     device=f'cuda:{local_rank}',
            #     gpu_memory_utilization=0.7,
            #     enforce_eager=True,
            #     enable_chunked_prefill=True,
            # )
            advisor_model = None
            pass
    else:
        advisor_model = None




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
            params["advisor_tokenizer"] = advisor_tokenizer
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
            params["advisor_tokenizer"] = advisor_tokenizer
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
    parser.add_argument("--advisor_model_path", type=str, default='./pretrained_models/Qwen2.5-1.5B-Instruct') # policy model
    parser.add_argument("--pigai_model_path", type=str, default='./pretrained_models/Qwen2.5-1.5B-Instruct') # pigai model
    parser.add_argument("--dataset_path", type=str, default='./R1_Zero/dataset/Text_Only/benchmark/MATH-500/MATH_500.json')
    parser.add_argument("--qwen2vl_infer_batch", type=int, default=1)
    parser.add_argument("--best_of_n", type=int, default=1) # BoN
    parser.add_argument("--max_try_times", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.9) # BoN
    parser.add_argument("--top_k", type=int, default=50) # BoN
    parser.add_argument("--temperature", type=float, default=0.7) # BoN
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--window_size", type=int, default=4096)
    parser.add_argument("--step_len", type=int, default=4096)
    parser.add_argument("--window_step", type=int, default=2048)
    parser.add_argument("--max_interaction_round", type=int, default=3)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--exist_file", type=str, default='None')
    parser.add_argument("--short_cot", type=str, default='False')
    parser.add_argument("--is_debug", type=str, default='False')
    parser.add_argument("--use_api", type=str, default='True')
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
        # 如果是系统退出异常，忽略
        if exc_type == SystemExit:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        # 记录异常到日志
        logging.error("未捕获的异常", exc_info=(exc_type, exc_value, exc_tb))
    # 设置 sys.excepthook 捕获未处理的异常
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
     
    if args.short_cot == 'True':
        args.short_cot = True
    else:
        args.short_cot = False
    

    main(args)



