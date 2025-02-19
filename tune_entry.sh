yes | cp ./clients/api_server.py {path to vllm/entrypoints/}
bash run_entry_bo_scoot.sh {path to model weights and configs} {model_name} {path to datasets} {dataset name} {request rate IN client} {number of requests} {GPU num} {gpu gpu_type}
# bash run_entry_bo_scoot.sh ./LLaMA2-7B-fp16 llama2_7b_scoot ./sharegpt.json sharegpt 20 1000 2 L20