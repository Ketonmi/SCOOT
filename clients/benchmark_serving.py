# From vllm benchmarks
import argparse
import asyncio
import json
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
from clients.backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                                RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase, AutoTokenizer


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    p95_latency_ms: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float


def sample_sharegpt_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def sample_sonnet_requests(
        dataset_path: str,
        num_requests: int,
        input_len: int,
        output_len: int,
        prefix_len: int,
        tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int]]:
    assert (
            input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
            input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
            prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(
            prefix_lines +
            random.sample(poem_lines, num_input_lines - num_prefix_lines))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len))

    return sampled_requests


def sample_default_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase):
    # Load the dataset.
    with open(dataset_path) as f:
        items = json.load(f)

    if num_requests > len(items):
        raise ValueError(f"num_requests must be less than the dataset volume {len(items)}")

    items = items[:num_requests]
    random.shuffle(items)
    sampled_prompts = []
    for i in range(num_requests):
        try:
            if 'input_len' not in items[i]:
                items[i]['input_len'] = len(tokenizer(items[i]['prompt']).input_ids)
            if 'output_len' not in items[i]:
                items[i]['output_len'] = len(tokenizer(items[i]['response']).input_ids)
            sampled_prompts.append((items[i]['prompt'], items[i]['input_len'],
                                    items[i]['output_len']))
        except Exception:
            raise ValueError(f"Item key error! {items[0].keys()}, "
                             f"each request item mush have 'prompt' "
                             f"and have at least one of 'response' and 'output_len'.")
    return sampled_prompts


async def get_request(
        input_requests: List[Tuple[str, int, int]],
        request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
        input_requests: List[Tuple[str, int, int]],
        outputs: List[RequestFuncOutput],
        dur_s: float,
        tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens = []
    total_input = 0
    completed = 0
    tpots = []
    ttfts = []
    latencies = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(tokenizer(outputs[i].generated_text).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            ttfts.append(outputs[i].ttft)
            completed += 1
            latencies.append(outputs[i].latency)
        else:
            actual_output_lens.append(0)

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        p95_latency_ms=np.percentile(latencies, 95) * 1000,
        mean_ttft_ms=np.mean(ttfts or 0) *
                     1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots) * 1000,
        median_tpot_ms=np.median(tpots) * 1000,
        p99_tpot_ms=np.percentile(tpots, 99) * 1000,
    )

    return metrics, actual_output_lens

async def benchmark(
        backend: str,
        api_url: str,
        model_id: str,
        tokenizer: PreTrainedTokenizerBase,
        input_requests: List[Tuple[str, int, int]],
        best_of: int,
        use_beam_search: bool,
        request_rate: float,
        disable_tqdm: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("P95 Latency (ms):", metrics.p95_latency_ms))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                    metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                               n=50,
                               c='-'))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                    metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "p95_latency_ms": metrics.p95_latency_ms,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        'gpu_hit_rates': [output.gpu_hit_rate for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    return result


async def benchmark_pressure_test(
        backend: str,
        api_url: str,
        model_id: str,
        tokenizer: PreTrainedTokenizerBase,
        input_requests: List[Tuple[str, int, int]],
        best_of: int,
        use_beam_search: bool,
        request_rate: float,
        disable_tqdm: bool,
        max_concurrent_requests: int,  # 新增的最大并发请求数
        ignore_eos: bool = True
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(f"Traffic request rate: {request_rate}")
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    benchmark_start_time = time.perf_counter()

    # 创建信号量以限制并发请求数
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    # 存储所有任务的列表
    tasks = []
    outputs = []

    async def sem_task(request):
        async with semaphore:
            prompt, prompt_len, output_len = request
            request_func_input = RequestFuncInput(
                model=model_id,
                prompt=prompt,
                api_url=api_url,
                prompt_len=prompt_len,
                output_len=output_len,
                best_of=best_of,
                use_beam_search=use_beam_search,
            )

            output = await request_func(request_func_input=request_func_input, 
                                        pbar=pbar,
                                        tokenizer=tokenizer,
                                        ignore_eos=ignore_eos)
            return output


    async for request in get_request(input_requests, request_rate):
        task = asyncio.create_task(sem_task(request))
        tasks.append(task)

    # 使用 asyncio.gather 等待所有任务完成
    outputs = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )
    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("P95 Latency (ms):", metrics.p95_latency_ms))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                    metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                               n=50,
                               c='-'))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                    metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("=" * 50)
    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "p95_latency_ms": metrics.p95_latency_ms,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        'gpu_hit_rates': [output.gpu_hit_rate for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    return result



def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    ports_list = args.port.split(',')

    api_url_list = []
    for port in ports_list:
        if args.base_url is not None:
            api_url_list.append(f"{args.base_url}{args.endpoint}")
        else:
            api_url_list.append(f"http://{args.host}:{int(port)}{args.endpoint}")

    api_url = ','.join(api_url_list)

    def check_port(port):

        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((f'{args.host}', port))
        sock.close()
        return result == 0

    # wait for the model server is connected
    for i in range(1000):
        if all([check_port(int(port)) for port in ports_list]):
            break
        else:
            time.sleep(1)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id,
                                                 trust_remote_code=args.trust_remote_code)

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2)
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt, prompt_len, output_len)
                              for prompt, prompt_formatted, prompt_len,
                                  output_len in input_requests]
        else:
            assert (
                    tokenizer.chat_template or tokenizer.default_chat_template
            ), "Tokenizer/model must have chat template for sonnet dataset."
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt_formatted, prompt_len, output_len)
                              for prompt, prompt_formatted, prompt_len,
                                  output_len in input_requests]

    else:
        input_requests = sample_default_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer
        )
        print(f'Sample Default Dataset! Dataset Name:{args.dataset_name}')


    if not args.pressure_test:
        benchmark_result = asyncio.run(
            benchmark(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                tokenizer=tokenizer,
                input_requests=input_requests,
                best_of=args.best_of,
                use_beam_search=args.use_beam_search,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm,
            ))
    else:
        assert args.max_concurrent_requests, 'In the pressure_test mode, please set max_concurrent_requests'
        benchmark_result = asyncio.run(
            benchmark_pressure_test(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                tokenizer=tokenizer,
                input_requests=input_requests,
                best_of=args.best_of,
                use_beam_search=args.use_beam_search,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm,
                max_concurrent_requests=args.max_concurrent_requests,
            ))

    # Save config and results to json
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = (datetime.now() + timedelta(seconds=random.randint(1, 5))).strftime("%Y%m%d-%H%M%S-%f")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts
        result_json["tp"] = args.tensor_parallel_size
        result_json["pp"] = args.pipeline_parallel_size
        result_json["max_num_batched_tokens"] = args.max_num_batched_tokens
        result_json["max_num_seqs"] = args.max_num_seqs
        result_json["enable_chunked_prefill"] = args.enable_chunked_prefill
        result_json["enable_prefix_caching"] = args.enable_prefix_caching
        result_json["disable_custom_all_reduce"] = args.disable_custom_all_reduce
        result_json["use_v2_block_manager"] = args.use_v2_block_manager
        result_json["block_size"] = args.block_size
        result_json["scheduler_delay_factor"] = args.scheduler_delay_factor
        result_json["max_concurrent_requests"] = args.max_concurrent_requests
        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  # noqa
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)

        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )

    parser.add_argument(
        "--pressure-test",
        action="store_true",
        help="whether using pressure_test mode"
    )

    parser.add_argument("--max-concurrent-requests", 
                        type=int, 
                        default= None,
                        help="The coccurent number in the pressure_test mode",)

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default=8000, help="API port: multiple ports is '8000,8001'.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/generate",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
             "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
             "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
             "from the ShareGPT dataset.")
    parser.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
             "then all the requests are sent at time 0. "
             "Otherwise, we use Poisson process to synthesize "
             "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
             "for metadata of this run to be saved in the result JSON file "
             "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
             "If not specified, results are saved in the current directory.",
    )
    parser.add_argument('--tensor-parallel-size',
                        '-tp',
                        type=int,
                        help='number of tensor parallel replicas')
    parser.add_argument('--pipeline-parallel-size',
                    '-pp',
                    type=int,
                    help='number of pipeline parallel replicas')
    parser.add_argument('--block-size',
                        type=int,
                        choices=[8, 16, 32, 64, 128],
                        help='token block size')
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        help='maximum number of batched tokens per '
                             'iteration')
    parser.add_argument('--max-num-seqs',
                        type=int,
                        help='maximum number of sequences per iteration')
    parser.add_argument(
        '--scheduler-delay-factor',
        type=float,
        help='Apply a delay (of delay factor multiplied by previous'
             'prompt latency) before scheduling next prompt.')
    parser.add_argument(
        '--enable-chunked-prefill',
        type=str,
        choices=["True", "False"],
        help="""enable-chunked-prefill to the benchmark_pipline.sh""")
    parser.add_argument(
        '--enable-prefix-caching',
        type=str,
        choices=["True", "False"],
        help="""enable--prefix-caching to the benchmark_pipline.sh""")
    parser.add_argument(
        '--disable-custom-all-reduce',
        type=str,
        choices=["True", "False"],
        help="""disable-custom-all-reduce to the benchmark_pipline.sh""")
    parser.add_argument(
        '--use-v2-block-manager',
        type=str,
        choices=["True", "False"],
        help="""use-v2-block-manager to the benchmark_pipline.sh""")
    args = parser.parse_args()

    res_dir_path = os.environ.get("RES_DIR_PATH", None)
    assert res_dir_path, 'please set environment variable RES_DIR_PATH'
    args.result_dir = res_dir_path
    print(f'Benchmarking Args: {args}')
    main(args)
