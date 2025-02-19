#!/bin/bash
# args
model_path=$1
dataset_path=$2
request_rate=$3
num_requests=$4
pressure_test=$5
max_concurrent_requests=$6

# parameter combination
tp_size=$7
pp_size=$8
max_num_seqs=$9
max_num_batched_tokens=${10}
scheduler_delay_factor=${11}
block_size=${12}
port=${13}
device_group=${14}
model=${15}
enable_chunked_prefill=${16}
dataset_name=${17}
enable_prefix_caching=${18}
disable_custom_all_reduce=${19}
use_v2_block_manager=${20}
OLD_IFS="$IFS"
IFS=','
read -ra ADDR <<< "$port"
IFS="$OLD_IFS"

OLD_IFS="$IFS"
IFS='#'
read -ra GPU_ADDR <<< "$device_group"
IFS="$OLD_IFS"

echo benchmark_pipeine.sh
echo model_path=$1 ${model_path}
echo dataset_path=$2 ${dataset_path}
echo request_rate=$3 ${request_rate}
echo num_requests=$4 ${num_requests}
echo pressure_test=$5 ${pressure_test}
echo max_concurrent_requests=$6 ${max_concurrent_requests}
echo tp_size=$7 ${tp_size}
echo pp_size=$8 ${pp_size}
echo max_num_seqs=$9 ${max_num_seqs}
echo max_num_batched_tokens=${10} ${max_num_batched_tokens}
echo scheduler_delay_factor=${11} ${scheduler_delay_factor}
echo block_size=${12} ${block_size}
echo port=${13} ${port}
echo device_group=${14} ${device_group}
echo model=${15} ${model}
echo enable_chunked_prefill=${16} ${enable_chunked_prefill}
echo dataset_name=${17} ${dataset_name}
echo enable_prefix_caching=${18} ${enable_prefix_caching}
echo disable_custom_all_reduce=${19} ${disable_custom_all_reduce}
echo use_v2_block_manager=${20} ${use_v2_block_manager}

echo "server start!"
for ((i=0; i<${#ADDR[@]}; i++)); do
    device=${GPU_ADDR[i]}
    echo device:${device}
    echo addr:${ADDR[i]}
    CUDA_VISIBLE_DEVICES=${device} bash run_server.sh ${model_path} ${ADDR[i]} ${tp_size} ${pp_size} ${max_num_seqs} ${max_num_batched_tokens} ${scheduler_delay_factor} ${block_size} ${enable_chunked_prefill} ${enable_prefix_caching} ${disable_custom_all_reduce} ${use_v2_block_manager}&
done
echo "finish server start!"

echo "client start!"
bash run_client.sh ${model_path} ${dataset_path} ${request_rate} ${num_requests} ${pressure_test} ${max_concurrent_requests} ${tp_size} ${pp_size} ${max_num_seqs} ${max_num_batched_tokens} ${scheduler_delay_factor} ${block_size} ${port} ${model} ${dataset_name} ${enable_chunked_prefill} ${enable_prefix_caching} ${disable_custom_all_reduce} ${use_v2_block_manager}
echo "finish client start!"

# Kill the whole process and then kill each port to ensure that the engine process has been fully killed
pgrep -f "vllm.entrypoints.api_server" | xargs kill -9
for i in "${ADDR[@]}"; do
    int_port=$((i-0))
    echo int_point=${int_port}
    lsof -t -i:${int_port} | xargs kill -9
done
