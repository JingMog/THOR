#! /bin/bash
export dir_name=$(basename "$PWD")
export parent_dir_name=$(basename "$(dirname "$(pwd)")")


dir_name=$(basename "$PWD")
NNODES=1
current_dir=$(pwd)
dataset_output_dir=${current_dir}/data
mkdir ${dataset_output_dirs}
dataset=./DeepScaleR-Preview-Dataset/data/deepscaler.json
exist_file=None


python ./TIRGen/split_dataset.py \
     --output_dir ${dataset_output_dir} \
     --dataset_path ${dataset} \
     --exist_file ${exist_file} \
     --num_node ${NNODES}


for (( i=0; i<${NNODES}; i++ )); do
bash_path=construct.sh
log_path=RANK_[${i}].log
dataset=${dataset_output_dir}/sample_dataset_${i}.json
output_dir=./infer_result/Long_CoT_sample/${parent_dir_name}/${dir_name}/${i}/

${bash_path} ${dataset} ${output_dir} ${i}

done

