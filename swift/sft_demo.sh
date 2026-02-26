export script_dir=$(pwd)
export dir_name=$(basename "$PWD")

set -x 
export MODELSCOPE_CACHE=./dataset/tmp/${dir_name}
export HF_DATASETS_CACHE=./dataset/tmp/${dir_name}
export PYTHONPATH=$(pwd)
export TRITON_CACHE_DIR=./experiment/cache/triton_cache/${RANK}
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_NTHREADS=4
export NCCL_TIMEOUT=600
export TORCHELASTIC_AGENT_EXIT_BARRIER_TIMEOUT=60000



model_path=./pretrained_models/Qwen3-8B/
dataset_path=./your/dataset/
output_dir=./your/dir/
max_length=20000
deepspeed=zero2
gradient_accumulation_steps=8
per_device_train_batch_size=1
lr=2e-6
dataset_num_proc=8
epoch=3


NGPUS=8
NNODES=2
master_port=12345

# total_batch_size = 2 * 8 * 1 * 8 = 128
# step_per_epoch = 27652 / 128 = 216


if [[ $NNODES -gt 1 ]]; then
    torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$master_port \
        ./swift/cli/sft.py \
        --model ${model_path} \
        --train_type full \
        --dataset ${dataset_path} \
        --torch_dtype bfloat16 \
        --num_train_epochs ${epoch} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size 1 \
        --eval_strategy no \
        --learning_rate ${lr} \
        --lr_scheduler_type constant \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --save_strategy epoch \
        --save_total_limit 5 \
        --dataset_num_proc ${dataset_num_proc} \
        --deepspeed ${deepspeed} \
        --logging_steps 5 \
        --max_length ${max_length} \
        --output_dir ${output_dir} \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --attn_impl flash_attn

else
    torchrun --nproc_per_node=$NGPUS --master_port=$master_port \
        ./swift/cli/sft.py \
        --model ${model_path} \
        --train_type full \
        --dataset ${dataset_path} \
        --torch_dtype bfloat16 \
        --num_train_epochs ${epoch} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size 1 \
        --eval_strategy no \
        --learning_rate ${lr} \
        --lr_scheduler_type constant \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --save_strategy epoch \
        --save_total_limit 5 \
        --dataset_num_proc ${dataset_num_proc} \
        --deepspeed ${deepspeed} \
        --logging_steps 5 \
        --max_length ${max_length} \
        --output_dir ${output_dir} \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --attn_impl flash_attn



fi