import json
import os
import sys
import time
import traceback
import asyncio
import random
import aiohttp

from dataclasses import dataclass, field
from typing import List, Optional
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60)

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    gpu_hit_rate: List[float] = field(
        default_factory=list)  
    error: str = ""

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
async def fetch_stats(session, url):
    url=url.replace('generate','metrics')
    async with session.get(url) as response:
        if response.status == 200:
            
            return await response.text()
        else:
            raise Exception(f"Failed to fetch GPU usage from {url}")

async def async_request_vllm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    ignore_eos: bool = True,
    **kwargs
) -> RequestFuncOutput:
    api_url_list = request_func_input.api_url.split(',')
    
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": request_func_input.prompt,
            "n": 1,
            "best_of": request_func_input.best_of,
            "use_beam_search": request_func_input.use_beam_search,
            "temperature": 0.0 if request_func_input.use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "ignore_eos": ignore_eos,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            
            if len(api_url_list)==1:
                api_url=api_url_list[0]
            else:
                gpu_usages_waiting_len = await asyncio.gather(*[fetch_stats(session, url) for url in api_url_list])
                min_pending_queue, min_gpu_usage, min_index = sorted([(json.loads(metric)["pending_queue_length"], json.loads(metric)["gpu_cache_usage"], idx) for idx, metric in enumerate(gpu_usages_waiting_len)], key=lambda x: (x[0],x[1]))[0]
                api_url = api_url_list[min_index]
            assert api_url.endswith("generate")
            
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp
                    output.latency = time.perf_counter() - st

                    # When streaming, '\0' is appended to the end of the response.
                    body = data.decode("utf-8").strip("\0")
                    
                    # body=data.decode("utf-8").split("\0")[0].strip("\0")
                    try:
                        output.generated_text = json.loads(
                        body)["text"][0][len(request_func_input.prompt):]
                        
                        output.success = True
                    except:
                        output.success = False

                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

        gpu_usages_waiting_len = await asyncio.gather(*[fetch_stats(session, url) for url in api_url_list])
        output.gpu_hit_rate = [json.loads(metric)['gpu_hit_rate'] for metric in gpu_usages_waiting_len]

        if pbar:
            pbar.update(1)
        return output

ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_vllm
}
