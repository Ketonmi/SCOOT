model_path=$1
port=$2
tp_size=$3
pp_size=$4
max_num_seqs=$5
max_num_batched_tokens=$6
scheduler_delay_factor=$7
block_size=$8
enable_chunked_prefill=$9
enable_prefix_caching=${10}
disable_custom_all_reduce=${11}
use_v2_block_manager=${12}
int_port=$((port + 0))
additional_options=""

if [ "${enable_chunked_prefill}" == "True" ]; then
    additional_options="--enable-chunked-prefill "
fi

if [ "${enable_prefix_caching}" == "True" ]; then
    additional_options+="--enable-prefix-caching "
fi

if [ "${disable_custom_all_reduce}" == "True" ]; then
    additional_options+="--disable-custom-all-reduce "
fi

if [ "${use_v2_block_manager}" == "True" ]; then
    additional_options+="--use-v2-block-manager "
fi

echo run_server.sh
echo tp_size ${tp_size}
echo pp_size ${pp_size}
echo enable_chunked_prefill ${enable_chunked_prefill}
echo enable_prefix_caching ${enable_prefix_caching}
echo disable_custom_all_reduce ${disable_custom_all_reduce}
echo use_v2_block_manager ${use_v2_block_manager}
echo  python -m vllm.entrypoints.api_server \
    --model ${model_path} \
    --disable-log-requests \
    --max-num-batched-tokens ${max_num_batched_tokens} \
    --max-num-seqs ${max_num_seqs} \
    --scheduler-delay-factor ${scheduler_delay_factor} \
    --port ${int_port} \
    --tensor-parallel-size ${tp_size} \
    --pipeline-parallel-size ${pp_size} \
    --block-size ${block_size} \
    --gpu-memory-utilization 0.9\
    --trust-remote-code\
    $additional_options
#     

python -m vllm.entrypoints.api_server \
    --model ${model_path} \
    --disable-log-requests \
    --max-num-batched-tokens ${max_num_batched_tokens} \
    --max-num-seqs ${max_num_seqs} \
    --scheduler-delay-factor ${scheduler_delay_factor} \
    --port ${int_port} \
    --tensor-parallel-size ${tp_size} \
    --pipeline-parallel-size ${pp_size} \
    --block-size ${block_size} \
    --gpu-memory-utilization 0.9\
    --trust-remote-code\
    $additional_options