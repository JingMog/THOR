import argparse
import os
import json
from tqdm import tqdm


def split_dataset_sequential(data, num_splits):
    """
    将数据集按顺序分成num_splits份。
    Args:
        data (list): 要划分的数据集，可以是任何可迭代对象。
        num_splits (int): 要划分成的份数。
    Returns:
        list: 包含num_splits个子列表的列表。
    """
    total_size = len(data)
    chunk_size = total_size // num_splits
    remainder = total_size % num_splits

    splits = []
    current_index = 0
    for i in range(num_splits):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_index = current_index + current_chunk_size
        splits.append(data[current_index:end_index])
        current_index = end_index
    return splits


def main(args):
    num_node = args.num_node
    # 读取数据集
    test_dataset = json.load(open(args.dataset_path)) # 读取数据集
    if args.exist_file != 'None' and os.path.exists(args.exist_file):
        print(f'load exist dataset file: {args.exist_file}')
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
    print(f'len_test_datasets: {len(test_dataset)}')

    # 将数据集划分成num_node份, 并分别存储
    split_dataset = split_dataset_sequential(test_dataset, num_node)

    for i, dataset in enumerate(split_dataset):
        save_path = os.path.join(args.output_dir, f"sample_dataset_{i}.json")
        print(save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./TIRGen/long_cot_sample/sample_multi_process_multi_node/data')
    parser.add_argument("--dataset_path", type=str, default='')
    parser.add_argument("--exist_file", type=str, default='None')
    parser.add_argument("--num_node", type=int, default=10)
    args = parser.parse_args()


    main(args)
