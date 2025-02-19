model_path=$1
dataset_path=$2
request_rate=$3
num_requests=$4
pressure_test=$5
max_concurrent_requests=$6

tp_size=$7
pp_size=$8
max_num_seqs=$9
max_num_batched_tokens=${10}
scheduler_delay_factor=${11}
block_size=${12}
port=${13}
model=${14}
dataset_name=${15}
enable_chunked_prefill=${16}
enable_prefix_caching=${17}
disable_custom_all_reduce=${18}
use_v2_block_manager=${19}

if [ "${pressure_test}" == "True" ]; then
    additional_options="--pressure-test --max-concurrent-requests ${max_concurrent_requests}"
fi

echo run_client.sh
echo python -m clients.benchmark_serving \
            --backend vllm \
            --tokenizer ${model_path}\
            --dataset-name ${dataset_name} \
            --dataset-path ${dataset_path} \
            --request-rate ${request_rate}\
            --model ${model}\
            --num-prompts ${num_requests}\
            --save-result \
            --max-num-batched-tokens ${max_num_batched_tokens}\
            --max-num-seqs ${max_num_seqs}\
            --scheduler-delay-factor ${scheduler_delay_factor}\
            --enable-chunked-prefill ${enable_chunked_prefill}\
            --tensor-parallel-size ${tp_size}\
            --pipeline-parallel-size ${pp_size} \
            --block-size ${block_size}\
            --port ${port}\
            --enable-prefix-caching ${enable_prefix_caching}\
            --disable-custom-all-reduce ${disable_custom_all_reduce}\
            --use-v2-block-manager ${use_v2_block_manager}\
            --trust-remote-code\
            --disable-tqdm\
            --seed 42\
            $additional_options
            
python -m clients.benchmark_serving \
            --backend vllm \
            --tokenizer ${model_path}\
            --dataset-name ${dataset_name} \
            --dataset-path ${dataset_path} \
            --request-rate ${request_rate}\
            --model ${model}\
            --num-prompts ${num_requests}\
            --save-result \
            --max-num-batched-tokens ${max_num_batched_tokens}\
            --max-num-seqs ${max_num_seqs}\
            --scheduler-delay-factor ${scheduler_delay_factor}\
            --enable-chunked-prefill ${enable_chunked_prefill}\
            --tensor-parallel-size ${tp_size}\
            --pipeline-parallel-size ${pp_size} \
            --block-size ${block_size}\
            --port ${port}\
            --enable-prefix-caching ${enable_prefix_caching}\
            --disable-custom-all-reduce ${disable_custom_all_reduce}\
            --use-v2-block-manager ${use_v2_block_manager}\
            --trust-remote-code\
            --disable-tqdm\
            --seed 42\
            $additional_options
