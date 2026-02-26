import json
import os
import pandas as pd
from tqdm import tqdm
import re
import random
from typing import List
# from transformers import AutoTokenizer
import math
import opencc
from multiprocessing import Process, synchronize, Lock, Manager, Pool
from six.moves import queue
import signal
import argparse




# ========================= repetition detection =========================
def handler(signum, frame):
    raise TimeoutError("Function execution has timed out.")

def long_running_find_repeating(gpt_target_str, time_out=100):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_out)  # 设置超时时间为5秒
    try:
        repeating_flag, reason = find_repeating(gpt_target_str)
    except TimeoutError:
        repeating_flag = True
        reason = "Function execution has timed out."
        return repeating_flag, reason
    finally:
        signal.alarm(0)  # 取消闹钟（定时信号）
    return repeating_flag, reason



def find_consecutive_repeating_patterns(s):
    # 简单
    # pattern = re.compile(r'(\S+?)\1+')
    pattern = re.compile(r'(.+)\1+')
    results = []
    while s:
        match = pattern.search(s)
        if match:
            repeating_pattern = match.group(1)
            full_match = match.group(0)
            count = len(full_match) // len(repeating_pattern)
            results.append((repeating_pattern, count))
            s = s[match.end() :]
        else:
            break
    return results


def find_consecutive_repeating_patterns_correctly_optimized(s: str):
    """
    高效地查找连续重复模式，并确保逻辑与原版（寻找最长基本单元）一致。
    该方法通过手动从长到短迭代基本单元长度来避免正则表达式回溯。
    """
    results = []
    str_len = len(s)
    current_pos = 0

    while current_pos < str_len:
        # 从当前位置开始的子串
        substring = s[current_pos:]
        sub_len = len(substring)
        
        match_found = False
        # 1. 迭代所有可能的“基本单元”长度，从最长(len/2)到最短(1)
        for p_len in range(sub_len // 2, 0, -1):
            pattern_candidate = substring[:p_len]
            
            # 2. 检查这个基本单元重复了多少次
            # `startswith` 在这里比切片和比较更高效
            if substring.startswith(pattern_candidate, p_len):
                # 至少重复了一次 (总共至少两个)
                count = 1
                # 从基本单元的末尾开始检查，看能连续匹配多少次
                check_pos = p_len
                while substring.startswith(pattern_candidate, check_pos):
                    count += 1
                    check_pos += p_len
                
                # 3. 既然是从最长的 p_len 开始找，第一个找到的就符合“最长基本单元”的逻辑
                results.append((pattern_candidate, count))
                
                # 4. 将主指针跳过整个匹配块
                current_pos += p_len * count
                match_found = True
                break # 找到了最长的，无需再找更短的了

        if not match_found:
            # 如果在当前位置找不到任何重复模式，则指针前进一步，从下一个字符开始搜索
            current_pos += 1
            
    return results

def find_repeating(target_str):
    flag = False
    find_pattern = ''
    target_str = target_str.replace("\n", "").replace("<ret>", "")
    # patterns = find_consecutive_repeating_patterns(target_str)
    patterns = find_consecutive_repeating_patterns_correctly_optimized(target_str)
    patterns = list(set(patterns))
    if patterns:
        for pattern, count in patterns:
            if len(pattern) >= 10 and count >= 30:
                flag = True
                find_pattern = pattern
                break
            elif len(pattern) >= 30 and count >= 25:
                flag = True
                find_pattern = pattern
                break
            elif count >= 20:
                flag = True
                find_pattern = pattern
                break
    return flag, find_pattern
# ========================================================================

def chk_dollar_pair(target_str):
    # 查看$是否成对
    flag = False
    dollar_latex_list = re.findall(r'(?<!\\)\$', target_str, flags=re.DOTALL)
    dollar_num = len(dollar_latex_list)
    if dollar_num % 2 == 1 and dollar_num != 1:
       flag = True
    return flag, dollar_num

def find_traditional_chars(s):
    """找出字符串中的繁体字"""
    converter = opencc.OpenCC('t2s')  # 繁体到简体的转换配置
    for char in s:
        # 将字符转换为简体，若转换后不同则为繁体字
        if converter.convert(char) != char:
            return True
    return False




def read_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def check_code_blocks(code_blocks: List[str]) -> bool:
    thresh_ratio = 0.4
    code_num = len(code_blocks)
    thresh_num = math.ceil(thresh_ratio * code_num)
    # 如果低质量代码超过thresh_num
    def is_low_quality_detection(code_string):
        lines = code_string.split("\n")
        valid_lines = []
        for line in lines:
            if line == '' or line.startswith('#'):
                continue
            else:
                valid_lines.append(line)
        
        lines = valid_lines
        code_lines = len(lines)
        # 代码行数特别短, 小于等于5行
        if code_lines <= 5:
            return False

        return True
        # # import包说明相对有效
        # is_import_packages = False
        # for line in lines:
        #     if 'import' in line:
        #         is_import_packages = True
        #         break
        # if is_import_packages:
        #     return True
        
        # 定义函数说明相对有效
    
    low_quality_num = 0
    for block in code_blocks:
        if not is_low_quality_detection(block):
            low_quality_num += 1
    
    if low_quality_num > thresh_num:
        return False
    else:
        return True
    


def check_think(predict: str):
    """包含完整的think字段, 且think字段中含有python代码, 筛选掉代码过于简单的回复"""
    low_quality_blocks = []
    think_part_match = re.findall(r"<think>(.*?)</think>", predict, re.DOTALL)
    code_num = 0
    if think_part_match and len(think_part_match) > 0:
        assert len(think_part_match) == 1
        think_part_str = think_part_match[0]
        assert len(think_part_str) > 0
        code_match = re.findall(r"```python(.*?)```", think_part_str, re.DOTALL)
        if code_match and len(code_match) > 0:
            code_blocks_flag = check_code_blocks(code_match)
            code_num = len(code_match)
            if code_blocks_flag:
                return True, code_num
            else:
                low_quality_blocks.append(code_match)
                # print(code_match)
                return False, code_num
    return False, code_num


def check_answer(predict: str):
    """包含完整的answer字段, 且answer字段不含代码"""
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


def check_repetition(completion: str, repetition_ngram_size=30, repetition_max_num=20) -> bool:
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

def is_no_boxed(target_str):
    if 'boxed' not in target_str:
        return True
    elif re.search(r'\boxed\{\s*\}', target_str, re.DOTALL):
        return True
    else:
        return False


def filter_dataset(merged_dataset):
    new_dataset = []
    repetition_list = []
    need_complete_dataset = []
    for item in merged_dataset:
        predict = item["predict"]

        # 回复正确
        if item["math_verify_score"] != 1.0:
            continue
        # 代码执行正确
        if item["execution_passes"] != 1.0:
            continue
        # 回复长度过短
        if item['predict_length'] < 3000:
            continue
        # 没有boxed{}的数据
        if is_no_boxed(predict):
            continue
        # 查看$是否成对
        dollar_flag, dollar_num = chk_dollar_pair(predict)
        if dollar_flag:
            # print(f"dollar_flag error: {predict}, dollar_num: {dollar_num}")
            continue
        # 检查是否有繁体字
        traditional_chars = find_traditional_chars(predict)
        if traditional_chars:
            continue
        
        # 检查think字段
        think_flag, code_num = check_think(predict)
        if not think_flag:
            continue
        
        # 检查answer字段
        answer_flag = check_answer(predict)
        if not answer_flag:
            need_complete_dataset.append(item)
            continue
        
        # 过滤重复解码
        repetition_flag = check_repetition(predict, repetition_ngram_size=20, repetition_max_num=20)
        if repetition_flag:
            repetition_list.append(item)
            continue
        
        # 通过过滤
        item["code_num"] = code_num
        new_dataset.append(item)

    # print(f"completition过滤之后的数据: {len(new_dataset)}")
    # 过滤query, 同一个query只保留一个
    dataset_per_query = {}
    for item in new_dataset:
        if item['id'] not in dataset_per_query.keys():
            dataset_per_query[item['id']] = [item]
        else:
            dataset_per_query[item['id']].append(item)
    for key, value in dataset_per_query.items():
        dataset_per_query[key] = random.choice(value)
    new_dataset = list(dataset_per_query.values())



    return new_dataset, repetition_list





def do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    line = params["line"]
    line_id = params["line_id"]
    test_dataset = [line]
    
    dataset, dataset_repetition = filter_dataset(test_dataset)
    if len(dataset) > 0:
        out_dataset = shared_params["out_dataset"]
        out_dataset_lock = shared_params["out_dataset_lock"]
        with out_dataset_lock:
            out_dataset.put(dataset)
    
    if len(dataset_repetition) > 0:
        out_repetition = shared_params["out_repetition"]
        out_repetition_lock = shared_params["out_repetition_lock"]
        with out_repetition_lock:
            out_repetition.put(dataset_repetition)



def try_do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    try:
        do_single_task(params, args, records_queue, records_queue_lock, shared_params)
    except Exception as e:
        print(f"Error {e}")
    if records_queue_lock is not None:
        with records_queue_lock:
            records_queue.put(1)





def main(args):
    output_dir = args.output_dir
    filter_dataset_path = os.path.join(output_dir, "sample_dataset_total_filter.json")
    merged_data = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        merged_data += json.load(f)

    print(f"原始数据集: {len(merged_data)}")
    # dataset = filter_dataset(merged_data, output_dir)
    
    
    # init metrics
    manager = Manager()
    records_queue = manager.Queue()
    records_queue_lock = manager.Lock()

    shared_params = {}
    shared_params["out_dataset"] = manager.Queue()
    shared_params["out_dataset_lock"] = manager.Lock()
    shared_params["out_repetition"] = manager.Queue()
    shared_params["out_repetition_lock"] = manager.Lock()

    all_tasks = []
    line_id = -1
    
    lines = merged_data
    if args.num_workers <= 1:
        for line in tqdm(lines):
            line_id += 1
            params = {}
            params["line"] = line
            params["line_id"] = line_id

            cur_task = (params, args, records_queue, records_queue_lock, shared_params)
            do_single_task(*cur_task)
            if line_id > 300:
                break
    else:
        for line in tqdm(lines):
            line_id += 1
            params = {}
            params["line"] = line
            params["line_id"] = line_id

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


    total_filter_dataset =[]
    out_dataset = shared_params["out_dataset"]
    while not out_dataset.empty():
        line = out_dataset.get()
        total_filter_dataset += line
    
    repetition_dataset = []
    out_repetition = shared_params["out_repetition"]
    while not out_repetition.empty():
        repetition_dataset += out_repetition.get()


    dataset = total_filter_dataset
    print(f"query过滤之后的数据集: {len(dataset)}")



    # 统计数据集中不同代码调用比例的占用
    code_interaction_num = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
    for item in dataset:
        code_num = item['code_num']
        if code_num < 7:
            code_interaction_num[code_num].append(item)
        else:
            code_interaction_num[7].append(item)
    
    # 对调用次数1, 下采样50%
    code_interaction_num[1] = random.sample(code_interaction_num[1], k=int(len(code_interaction_num[1]) * 0.5))

    for key, value in code_interaction_num.items():
        print(f"Interaction num:   {key}: {len(value)}")
    new_dataset = []
    for value in code_interaction_num.values():
        new_dataset += list(value)
    dataset = new_dataset

    # 统计数据集的回复长度比例
    response_length_num = {'<2000': [], '2000~5000': [], '5000~8000': [], '8000~11000': [], '11000~14000': [], '>14000': []}
    for item in dataset:
        predict_length = item['predict_length']
        if predict_length < 2000:
            response_length_num['<2000'].append(item)
        elif 2000 <= predict_length and predict_length < 5000:
            response_length_num['2000~5000'].append(item)
        elif 5000 <= predict_length and predict_length < 8000:
            response_length_num['5000~8000'].append(item)
        elif 8000 <= predict_length and predict_length < 11000:
            response_length_num['8000~11000'].append(item)
        elif 11000 <= predict_length and predict_length < 14000:
            response_length_num['11000~14000'].append(item)
        else:
            response_length_num['>14000'].append(item)
    for key, value in response_length_num.items():
        print(f"Response length:   {key}: {len(value)}")

    print(f"最终数据集长度: {len(dataset)}")
    
    print(filter_dataset_path)
    with open(filter_dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    


    repetition_output_path = os.path.join(output_dir, "sample_dataset_repetition.json")
    print(repetition_output_path)
    with open(repetition_output_path, "w", encoding="utf-8") as f:
        json.dump(repetition_dataset, f, indent=4, ensure_ascii=False)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./[202507241521]_sample_dataset_total_[116637].json")
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()
    
    main(args)


