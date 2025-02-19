import json
import os.path

from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

import logging
import argparse
import torch
import math
import gc

DEFAULT_SEQUENCE_LENGTH = 4096


def get_other_model_size(model_path):
    file_list = os.listdir(model_path)
    file_names = [file_name for file_name in file_list if file_name.endswith('.safetensors')]
    from safetensors import safe_open
    total_param_size = 0
    for file_name in file_names:
        file_path = os.path.join(model_path, file_name)
        with safe_open(file_path, framework="pt") as f:
            for name in f.keys():
                total_param_size += f.get_tensor(name).nbytes  # LLM parameters are generally stored in fp16, each parameter takes 2 bytes
    res = total_param_size / (1024 ** 3)
    return res


def calc_min_world_size(model_path, max_token_num):
    device = torch.device("cuda")
    gpu_capacity = 0.9 * torch.cuda.get_device_properties(device).total_memory / (
            1024 ** 3)  # 0.9 is set to prevent (Out-of-Memory) OOM errors

    try:
        try:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        except ValueError:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        model_param_size = model.num_parameters() * 2 / (1024 ** 3)  # Unit: GB
        del model
        gc.collect()
    except Exception as e:
        model_param_size = get_other_model_size(model_path)  # Unit: GB
        print(model_param_size, e)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hidden_size = config.hidden_size
    try:
        layer_num = config.num_hidden_layers
    except AttributeError:
        layer_num = config.num_layers
    kv_cache_size = max_token_num * 2 * layer_num * hidden_size * 2 / (1024 ** 3)  # Unit: GB

    total_memory_consumption = model_param_size + kv_cache_size # It has to be able to accommodate at least one request
    min_world_size = math.ceil(total_memory_consumption / gpu_capacity)
    if min_world_size <= 1:
        min_world_size = 1
    elif min_world_size <= 2:
        min_world_size = 2
    elif min_world_size <= 4:
        min_world_size = 4
    else:
        min_world_size = 8
    print('model size:', model_param_size, 'kv cache param size:', kv_cache_size,
          'total memory consumption:', total_memory_consumption, 'available gpu memory capacity', gpu_capacity,
          'min_world_size:', min_world_size)
    return min_world_size


def obtain_max_sequence_length(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_sequence_length = getattr(config, 'max_sequence_length',
                                  getattr(config, 'model_max_length',
                                          getattr(config, 'seq_length', DEFAULT_SEQUENCE_LENGTH)))
    return max_sequence_length


def main(args):
    model_path = args.model_path

    print(f"tuner_conf: model_path={model_path}")
    conf_file_path = 'conf.json'

    max_sequence_length = obtain_max_sequence_length(model_path)
    min_world_size = calc_min_world_size(model_path, max_sequence_length)

    if os.path.exists(conf_file_path):
        with open(conf_file_path, 'r') as f:
            conf = json.load(f)
    else:
        conf = {}

    conf['max_sequence_length'] = max_sequence_length
    conf['min_world_size'] = min_world_size

    with open(conf_file_path, 'w') as f:
        json.dump(conf, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, required=True, help='Path of the model file')

    arguments = parser.parse_args()
    main(arguments)
