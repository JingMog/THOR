#!/bin/bash
export dir_name=$(basename "$PWD")
source ./.bashrc

# ============= start remote sandbox =============
nohup bash /dmx-csy-mix01/cog3/permanent/qkchang/R1_Zero/code/SandboxFusion-main/z_sand_box/run_sandbox.sh > sandbox_[$RANK].log 2>&1 &
# ================================================

source /dmx-csy-mix01/cog3/permanent/qkchang/miniconda3/bin/activate verl_qk
source /etc/profile.d/modules.sh


env
ulimit -u 16384
ulimit -a
# set -x

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
cd $SCRIPT_DIR

rm /tmp/ray


######################################

project_name=toolrl
experiment_name=${dir_name}
bash_path=/dmx-csy-mix01/cog3/permanent/qkchang/R1_Zero/code/ToRL/scripts/train_v1.6_Qwen_Math_7B_pass_optim_filter_maxcall_5_rftloss_offpolicy_codefilter_testfilter_back100/run_code_grpo.sh
SCRIPTS="${bash_path} ${project_name} ${experiment_name}"

######################################

ray stop


echo $RANK
if [$RANK == '']; then
    RANK='0'
fi

if [ $RANK == '0' ]; then
    echo start create ray head...
    RAY_memory_monitor_refresh_ms=0 ray start --head --num-cpus 80 --num-gpus 8 --dashboard-host=0.0.0.0
    sleep 30
    ray status
    
    cp /dmx-csy-mix01/cog3/permanent/qkchang/R1_Zero/code/verl_0609/mk_log_name.sh .
    bash mk_log_name.sh
    rm mk_log_name.sh

    bash $SCRIPTS 2>&1 | tee ray_train.log
    
    ###### copy ray session from /tmp ######
    echo copy ray_session to workdir...
    mkdir -p ray_sessions
    real_path=$(readlink -f /tmp/ray/session_latest)
    dir_name=$(basename "$real_path")
    cp -r --preserve=links "$real_path" "./ray_sessions/$dir_name"
    echo "done backup sessions."
    if [ -d 'outputs' ]; then
        mv outputs ./ray_sessions/$dir_name
    fi
    #######################################
    
    
    exit 1
    
else
    sleep 10
    echo sleep for 10 sec, start create ray workers...
    RAY_memory_monitor_refresh_ms=0 ray start --address=$MASTER_ADDR:6379 --num-cpus 80 --num-gpus 8

    for i in {1..1000}; do
        sleep 10h
        # echo "ray worker node-$RANK has slept for $i*10 min..."
    done
fi



sleep 6000