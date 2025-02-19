import torch
import argparse
import os
import logging
import tqdm
import traceback
import json
import time
import math

from typing import Union, Tuple
from utils import gen_res_dir_path, check_port, get_ref_config, read_historical_data
import pandas as pd
import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.hebo_constr import HEBOConstr
from transformers import AutoConfig
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('expand_frame_repr', False)

RES_DIR_PREFIX = 'scoot'
RES_DIR = './tune_res'
LOG_DIR = os.path.join(RES_DIR,'logs')
RAW_DIR = os.path.join(RES_DIR,'raw')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--request_rate", type=int, default=20)
    parser.add_argument("--num_requests", type=int, default=1000)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--total_resource",
        type=str,
        required=True,
        help="Total Resources",
    )
    parser.add_argument("--bo_loop", type=int, default=50)
    parser.add_argument("--bo_batch_size", type=int, default=1)
    parser.add_argument(
        "--exp_num",
        type=int,
        default=1,
        help="Total experiment run time",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sharegpt"
    )

    parser.add_argument(
        "--num_obj",
        type=int,
        default=2)

    parser.add_argument(
        "--pressure_test",
        action='store_true')
    return parser


def run_benchmark_pipeline(combination: Tuple, gpu_nums, args):
    tp_size = combination[0]
    ports = ",".join([str(8000 + i) for i in range(int(gpu_nums / tp_size))])
    gpus = [str(i) for i in range(gpu_nums)]
    grouped_gpus = [','.join(gpus[i:i + tp_size]) for i in range(0, gpu_nums, tp_size)]
    grouped_gpus_string = '#'.join(grouped_gpus)
    raw_file_path = os.path.join(RAW_DIR, f'benchmark_tp_{combination[0]}_mns_{combination[1]}_mnbt_{combination[2]}_bs_{combination[3]}.txt')
    for i in range(3):
        # retry 3 times in case of failure
        try:
            logging.info(
                f"bash benchmark_pipeline.sh {args.model_path} {args.dataset_path} {args.request_rate} {args.num_requests} {args.pressure_test} {0} {combination[0]} {1} {combination[1]} {combination[2]} {combination[5]} {combination[3]} {ports} {grouped_gpus_string} {args.model} {combination[4]} {args.dataset_name} {combination[6]} {combination[7]} {combination[8]} "
                f"2>&1 | tee {raw_file_path}")
            os.system(
                f"bash benchmark_pipeline.sh {args.model_path} {args.dataset_path} {args.request_rate} {args.num_requests} {args.pressure_test} {0} {combination[0]} {1} {combination[1]} {combination[2]} {combination[5]} {combination[3]} {ports} {grouped_gpus_string} {args.model} {combination[4]} {args.dataset_name} {combination[6]} {combination[7]} {combination[8]} "
                f"2>&1 | tee {raw_file_path}")
            break
        except Exception:
            logging.error(f'init error: {traceback.format_exc()}')


def obj(rec: pd.DataFrame, gpu_nums, res_dir_path, args, min_world_size: int = 1) -> Union[None, np.ndarray]:
    combination = (
        int(rec['tp'].tolist()[0]),
        int(rec['max_num_seqs'].tolist()[0]),
        int(rec['max_num_batched_tokens'].tolist()[0]),
        int(rec['block_size'].tolist()[0]),
        str(rec['enable_chunked_prefill'].tolist()[0]),
        float(rec['scheduler_delay_factor'].tolist()[0] / 10),
        str(rec['enable_prefix_caching'].tolist()[0]),
        str(rec['disable_custom_all_reduce'].tolist()[0]),
        str(rec['use_v2_block_manager'].tolist()[0])
    )

    # clear processes and ports sequentially
    try:
        os.system(f'pgrep -f "vllm.entrypoints.api_server" | xargs kill -9')
    except Exception as e:
        print("Kill Whole Process Error:", e)
    # check the ports in the last round is cleared. If not, close them
    ports = ",".join([str(8000 + i) for i in range(int(gpu_nums / min_world_size))])
    for port in ports.split(','):
        for _ in range(3):
            if check_port(int(port)):
                logging.info(f'port {int(port)} is not cleared. Closing it!!!')
                try:
                    os.system(f"lsof -t -i:{int(port)} | xargs kill -9")
                except Exception as e:
                    print("Kill Port Process Error:", e)
            else:
                break
        assert not check_port(
            int(port)), "For some reason, the ports are not cleared! Experiments cannot continue!!"
    time.sleep(5)  
    run_benchmark_pipeline(combination, gpu_nums, args)

    # We utilize the hack way: get the latest file as the result in thie bo loop.
    for root, _, files in os.walk(res_dir_path):
        assert len(files) >= 1
        file_path = os.path.join(root, files[len(files) - 1])
        assert file_path.split('/')[-1].startswith("vllm"), f"in the {res_dir_path} only resutls files exist"
        with open(file_path, 'r') as f:
            act_result = json.load(f)

    # check wether the result from the latest file ("act_result") is indeed the output of actual env from the "rec"
    if combination == (int(act_result['tp']),
                       int(act_result['max_num_seqs']),
                       int(act_result['max_num_batched_tokens']),
                       int(act_result['block_size']),
                       str(act_result['enable_chunked_prefill']),
                       float(act_result['scheduler_delay_factor']),
                       str(act_result['enable_prefix_caching']),
                       str(act_result['disable_custom_all_reduce']),
                       str(act_result['use_v2_block_manager'])):
        return np.array(
            [[
                -1 * act_result["request_throughput"],
                act_result["mean_ttft_ms"],
                act_result["mean_tpot_ms"]
            ]]
        )
    else:
        return None


def read_rec_history(rec_history_file_path):
    if os.path.exists(rec_history_file_path):
        with open(rec_history_file_path, 'r') as fp:
            rec_history_data = json.load(fp)
            failed_num = len([item for item in rec_history_data if item['obj'] is None])
            return failed_num, rec_history_data
    else:
        return 0, []


def obtain_random_forest_train_set(rec_history):
    x = []
    y = []
    for data_item in rec_history:
        x.append([
            data_item['rec'][0]['tp'],
            data_item['rec'][0]['max_num_seqs'],
            data_item['rec'][0]['max_num_batched_tokens'],
            data_item['rec'][0]['block_size'],
            1 if data_item['rec'][0]['enable_chunked_prefill'] else 0,
            data_item['rec'][0]['scheduler_delay_factor'],
            1 if data_item['rec'][0]['enable_prefix_caching'] else 0,
            1 if data_item['rec'][0]['disable_custom_all_reduce'] else 0,
            1 if data_item['rec'][0]['use_v2_block_manager'] else 0
        ])
        y.append(0 if data_item['obj'] is None else 1)
    return {'train_x': x, 'train_y': y}


def random_forest_regressor(train_set):
    x = train_set['train_x']
    y = train_set['train_y']
    if x and y and len(x) >= (len(x[0]) - 1):
        rfr = RandomForestRegressor()
        rfr.fit(x, y)
        return rfr
    else:
        return None


def compute_delta_and_continuous_right(rec_history):
    delta = 0.5
    continuous_right = 0
    if rec_history:
        init_train_size = len(rec_history[0]['rec'][0])
        if len(rec_history) >= init_train_size:  
            for i in range(init_train_size, len(rec_history)):
                if rec_history[i]['obj'] is None:
                    delta = round(min(0.75, max(delta + 0.05, 0.5)), 3)
                    continuous_right = 0
                else:
                    continuous_right += 1
                    if continuous_right > 5:
                        continuous_right = continuous_right - 5
                        delta = round(max(0.25, delta - 0.05), 3)
    return delta, continuous_right


def main(args):
    print(f'Input Tuning Arguments: {args}')
    gpu_nums = torch.cuda.device_count()
    assert gpu_nums % 2 == 0 or gpu_nums == 1 
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, f'bo_{args.bo_batch_size}_{args.model}_{args.total_resource}_num_requests{args.num_requests}_request_rate{args.request_rate}_{args.dataset_name}.log'),
        level=logging.INFO)

    for e in range(args.exp_num):
        res_dir_path = gen_res_dir_path(args.model, args.request_rate, args.num_requests, args.total_resource,
                                        args.dataset_name, RES_DIR, exp=e, bo=True, dir_prefix=RES_DIR_PREFIX)
        print(res_dir_path)
        os.environ["RES_DIR_PATH"] = res_dir_path

        logging.info(f"New BO loop of experiment {e} begins!!!")

        # 1. observe historical data
        xx, yy = read_historical_data(res_dir_path)

        # 2. DesignSpace
        min_world_size = get_ref_config('min_world_size')  # world_size = tp * pp
        max_sequence_length = get_ref_config('max_sequence_length')

        logging.info(f"Min World Size: {min_world_size}, "
                     f"Max World Size: {gpu_nums}"
                    )
        para_dict = [
                    {"name": "tp", "type": "int_exponent", "lb": min_world_size, "ub": gpu_nums, "base": 2},
                     {"name": "max_num_seqs", "type": "int_exponent", "lb": 64, "ub": 8192, "base": 2},     # 8192 for 8卡单实例调优, 2048 for 4卡单实例调优 或 多实例调优
                     {"name": "max_num_batched_tokens", "type": "pow_int",  # "ub": int(2 ** (8 + gpu_nums))
                      "lb": 64, 'ub': max(8192, max_sequence_length * 2), "base": 2},
                     {"name": "block_size", "type": "int_exponent", "lb": 8, "ub": 32, "base": 2},
                     {"name": "enable_chunked_prefill", "type": "bool"},
                     {"name": "scheduler_delay_factor", 'type': "step_int", "lb": 0, "ub": 20, "step": 2},
                     {"name": "enable_prefix_caching", "type": "bool"},
                     {"name": "disable_custom_all_reduce", "type": "bool"},
                     {"name": "use_v2_block_manager", "type": "bool"},
                    ]

        space = DesignSpace().parse(para_dict)

        # 3. BO optimizer
        # 3.1 sample ref point
        if len(xx) > 0 and len(yy) > 0:
            # The first run in the history is ref point
            ref_point = np.array(yy[0])
            logging.info(f'input for ref point is {xx[0]} ')
        else: 
            ref_rec = pd.DataFrame.from_dict(
                {
                "tp": [min_world_size],
                "max_num_seqs": [256],
                "max_num_batched_tokens": [max(4096, max_sequence_length)],
                "block_size": [16],
                "enable_chunked_prefill": [False],
                "scheduler_delay_factor": [0.0],
                "enable_prefix_caching": [False],
                "disable_custom_all_reduce": [False],
                "use_v2_block_manager": [False],
                 })
            ref_point = obj(ref_rec, gpu_nums, res_dir_path, args, min_world_size=min_world_size)
            logging.info(f'ref config: {ref_rec}')
            logging.info(f'ref obj: {ref_point}')
            xx, yy = read_historical_data(res_dir_path)

        ref_max_seq_len = math.ceil(math.log2(max_sequence_length))
        if args.num_obj > 1:
            # multi objective optimization
            opt = GeneralBO(space=space,
                            num_obj=args.num_obj,
                            num_model_constr=0,
                            num_hard_constr=3,
                            num_hidden_constr=1,
                            ref_point=ref_point,
                            model_config={"optimizer": "adam", "base_model_name": "gpy", "space": space},
                            use_noise=True,
                            max_sequence_length=ref_max_seq_len
                            )
        elif args.num_obj == 1:

            opt = HEBOConstr( 
                space=space,
                num_model_constr=0,
                num_hard_constr=3,
                num_hidden_constr=1,
                max_sequence_length=ref_max_seq_len,
            )
        else:
            raise ValueError('Work with Positive Numbers Only')

        pbar = tqdm.tqdm(total=args.bo_loop)

        # 3.3 merge history data
        if len(xx) > 0 and len(yy) > 0:
            print(f'Tuning History Configurations: {xx}')
            print(f'Tuning History Objectives: {yy}')
            explored_x = pd.DataFrame.from_dict(xx)
            explored_y = np.array(yy)
            opt.observe(explored_x, explored_y)

        # 4. bo loop execution
        succeed_num = len(xx)
        rec_history_file_path = os.path.join(res_dir_path, 'rec_history.json')
        failed_num, rec_history_list = read_rec_history(rec_history_file_path)

        train_set = obtain_random_forest_train_set(rec_history_list)
        random_forest = random_forest_regressor(train_set)
        delta, continuous_right = compute_delta_and_continuous_right(rec_history_list)
        i = failed_num + succeed_num  
        pbar.update(i)
        while i < args.bo_loop: 
            s_time = time.time()
            rec = opt.suggest(n_suggestions=1, rf_with_thres=(random_forest, delta))
            e_time = time.time()
            rec_time = e_time - s_time

            s_time = time.time()
            y = obj(rec, gpu_nums, res_dir_path, args, min_world_size=min_world_size)
            e_time = time.time()
            run_time = e_time - s_time

            y_none = False
            if y is None:
                y_none = True
                failed_num += 1
                if i > len(para_dict): 
                    delta = round(min(0.75, max(delta + 0.05, 0.5)), 3)  # round operation is to avoid the error caused by float calculation
                    continuous_right = 0
            else:
                opt.observe(rec, y)  # update surrogate model
                succeed_num += 1
                if i > len(para_dict):  # i = len(rec_history)+1
                    continuous_right += 1
                    if continuous_right >= 5:
                        continuous_right = continuous_right - 5
                        delta = round(max(0.25, delta - 0.05), 3) 

            pbar.update(1)
            rec_history = {
                'rec': rec.to_dict(orient='records'),
                'obj': None if y_none else y.tolist(),
                'rec_time': rec_time,
                'run_time': run_time
            }
            rec_history_list.append(rec_history)
            random_forest = random_forest_regressor(obtain_random_forest_train_set(rec_history_list))

            with open(rec_history_file_path, 'w') as fp:
                json.dump(rec_history_list, fp)

            i += 1
            logging.info(f'total_tune_num: {i}, succeed_num: {succeed_num}, failed_num: {failed_num}')
            logging.info(f'For the {i}th iteration, BO time cost is {rec_time}')
            logging.info(f'For the {i}th iteration, rec config is {rec}')
            logging.info(f'For the {i}th iteration, rec obj is {y}')
            logging.info(f'After {i} iterations, best config is {opt.best_x}')
            logging.info(f'After {i} iterations, best obj is {opt.best_y}')
            logging.info(f'After {i} iterations, the threshold delta is {delta}')
            logging.info(f'After {i} iterations, the continuous right number is {continuous_right}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for raw data generation !!')
    parser = add_args(parser)
    arguments = parser.parse_args()
    main(arguments)
