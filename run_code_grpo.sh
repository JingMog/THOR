#!/bin/bash
source ./.bashrc
cd ../../
source /dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/bin/activate verl_qk



export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eno1  # 多机训练配置为eno1或eno2.100，单机配置务必改成eth0
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=105
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NET_GDR_READ=1
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_P2P_LEVEL=NVL
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)



export VLLM_ATTENTION_BACKEND=XFORMERS
# export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

export SWANLAB_MODE=local     # cloud、cloud-only、local、disabled
set -x


train_files=/dmx-csy-mix01/cog3/permanent/qkchang/R1_Zero/dataset/Text_Only/Open_source_RL_data/verl_dataset/tool_rl_v4/data/code_rl_train.parquet
test_files=/dmx-csy-mix01/cog3/permanent/qkchang/R1_Zero/dataset/Text_Only/Open_source_RL_data/verl_dataset/tool_rl_v4/data/aime24_25_amc.parquet
model_path=/dmx-csy-mix01/cog3/permanent/qkchang/R1_Zero/experiment/ray/toolrl/train_v1.4_filter_maxcall_1_rftloss_offpolicy/checkpoints/global_step_36/actor/huggingface
project_name=$1
experiment_name=$2
export EXP_DIR=/dmx-csy-mix01/cog3/permanent/qkchang/R1_Zero/experiment/ray/${project_name}/${experiment_name}/checkpoints
export rollout_dir=/dmx-csy-mix01/cog3/permanent/qkchang/R1_Zero/experiment/ray/${project_name}/${experiment_name}/rollout
mkdir -p $EXP_DIR
mkdir -p $rollout_dir



# ============ debug 参数 ============
val_before_train=False
# ====================================
update_steps_per_round=40

filter_groups=True # True
filter_code=True # True
filter_abnormal_length=True # True
filter_abnormal_boxed_string=True # 相当于会保留重复解码的结果
pass_rate_optimization=True # 代码通过率优化
del_last_token_num=100
use_remote_code=True
# train_batch_size=128
train_batch_size=$((128 * $update_steps_per_round))
gen_prompt_bsz=$(awk "BEGIN {print $train_batch_size * 1}")
group_num=16
pass_rate_group_num=${group_num}
episode=20
temperature=1.0
ppo_mini_batch_size=16 # update_batch_size_per_gpu, num completion; 如果设置了update_steps_per_round, 这个选项就不起作用
lr=1e-6
kl_loss_coef=0.0
kl_coef=0.0
entropy_coeff=0
grpo_select_alpha=0.01
max_gen_length=3200
max_prompt_length=800
ppo_max_token_len_per_gpu=49152 # 32768, 65536
max_interaction_round=5
sandbox_url="http://127.0.0.1:8080/run_code"
template_type=tir_base_0309

NGPUS=8
NNODES=4


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.train_batch_size=${train_batch_size} \
    +data.gen_batch_size=${gen_prompt_bsz} \
    +algorithm.filter_groups.enable=${filter_groups} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_gen_length} \
    data.template_type=${template_type} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    +actor_rollout_ref.actor.update_steps_per_round=${update_steps_per_round} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    +actor_rollout_ref.actor.grpo_select_alpha=${grpo_select_alpha} \
    +actor_rollout_ref.actor.clip_ratio_high=0.28 \
    +actor_rollout_ref.actor.clip_ratio_low=0.2 \
    +actor_rollout_ref.actor.clip_ratio_c=3.0 \
    +actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.filter_code=${filter_code} \
    +actor_rollout_ref.rollout.pass_rate_optimization=${pass_rate_optimization} \
    +actor_rollout_ref.rollout.del_last_token_num=${del_last_token_num} \
    +actor_rollout_ref.rollout.pass_rate_n=${pass_rate_group_num} \
    +actor_rollout_ref.rollout.filter_abnormal_length=${filter_abnormal_length} \
    +actor_rollout_ref.rollout.filter_abnormal_boxed_string=${filter_abnormal_boxed_string} \
    +actor_rollout_ref.rollout.max_interaction_round=${max_interaction_round} \
    +actor_rollout_ref.rollout.sandbox_url=${sandbox_url} \
    +actor_rollout_ref.rollout.use_remote_code=${use_remote_code} \
    actor_rollout_ref.rollout.num_llm_calls_available=${max_interaction_round} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.n=${group_num} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.val_before_train=${val_before_train} \
    +trainer.pass_rate_optimization=${pass_rate_optimization} \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.default_local_dir=${EXP_DIR} \
    trainer.resume_mode=auto \
    +trainer.samples_save_path=${rollout_dir} \
    trainer.total_epochs=${episode}



