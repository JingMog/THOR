export script_dir=$(pwd)
export dir_name=$(basename "$PWD")
export parent_dir_name=$(basename "$(dirname "$(pwd)")")

dataset_path=$1
output_dir=$2
RANK=$3

# ============= start remote sandbox =============
nohup bash ./SandboxFusion/run_sandbox.sh > sandbox_[$RANK].log 2>&1 &
# ================================================


model_path=your_model_path


# ================ start vllm server =============
# begin policy model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup vllm serve ${model_path} --tensor-parallel-size 8 --api-key token-123456 --enable-prefix-caching --gpu-memory-utilization 0.95 --port 8000 > vllm_server_8B_[${RANK}].log 2>&1 &
sleep 200s

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x
VLLM_PORT="8000"
VLLM_HOST="localhost"
CHECK_URL="http://localhost:${VLLM_PORT}/v1/models"
echo "Checking vLLM server on port ${VLLM_PORT}..."
while true; do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer token-123456" "${CHECK_URL}")
    echo ${RESPONSE}
    if [ "$RESPONSE" -eq 200 ]; then
        echo "vLLM server on port ${VLLM_PORT} is up and running (HTTP 200 OK)."
        break
    else
        echo "vLLM server on port ${VLLM_PORT} returned unexpected HTTP code: $RESPONSE. Waiting..."
        sleep 10s # 等待一段时间再重试
    fi
done
# ================================================



export PYTHONPATH=$(pwd)

pigai_model_path=your_judge_model
qwen2vl_infer_batch=1
chat_template=qwen3_slow_thinking

best_of_n=4
top_p=1.0
top_k=-1
temperature=0.6
max_len=16000
use_thinking_budget=False
max_interaction_round=10
gpu_memory_utilization=0.8
num_workers=64


export NGPUS=8
export NNODES=1
export master_port=10025


python ./inference/BoN_policy_qwen3_multi_process.py \
    --output_dir ${output_dir} \
    --model_path ${model_path} \
    --pigai_model_path ${pigai_model_path} \
    --dataset_path ${dataset_path} \
    --qwen2vl_infer_batch ${qwen2vl_infer_batch} \
    --best_of_n ${best_of_n} \
    --top_p ${top_p} \
    --top_k ${top_k} \
    --temperature ${temperature} \
    --max_len ${max_len} \
    --use_thinking_budget ${use_thinking_budget} \
    --max_interaction_round ${max_interaction_round} \
    --gpu_memory_utilization ${gpu_memory_utilization} \
    --is_debug False \
    --chat_template ${chat_template} \
    --use_api True \
    --num_workers ${num_workers}
 


ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs kill





