# -*- coding: utf-8 -*-
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, wait, TimeoutError
import threading
from utils import get_default_config, ask_chatgpt, read_prompts, ask_chatgpt_async_send, ask_chatgpt_async_fetch
import csv
import random
import os
import select
import sys
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='批量请求 GPT4')

# Add arguments
parser.add_argument('--is_async', type=bool, default=True, help='异步调用')
parser.add_argument('--worker_num', type=int, default=50, help='并发度，调太高容易超 token/min 限制')
parser.add_argument('--async_worker_num', type=int, default=600, help='并发度，异步模式下自动进行并发度的调整')
parser.add_argument('--max_retries', type=int, default=50, help='最大重试次数')
parser.add_argument('--max_timeout_seconds', type=int, default=90000, help='最长总等待时间，超过了程序会存储已经成功的，自动结束')
parser.add_argument('--prompt_csv_file', type=str, default="prompts.csv", help='并发度，调太高容易超 token/min 限制')
parser.add_argument('--output_file_name', type=str, default="gpt_response.csv", help='并发度，调太高容易超 token/min 限制')
parser.add_argument('--is_retry', type=bool, default=False, help='如果设置成 true，会先加载 output file，然后讲其中 timeout 和空值进行更新，需要确保 output 和 prompt 顺序能对上.')
parser.add_argument('--use_gpt4', type=bool, default=False, help='如果设置成 true，会使用 gpt4，不然使用 gpt3.5')
parser.add_argument('--temperature', type=str, default="0.2", help='如果设置成 true，会使用 gpt4，不然使用 gpt3.5')

DEBUG = False
# Parse the command-line arguments
args = parser.parse_args()
IS_ASYNC = args.is_async
WORKER_NUM = args.async_worker_num if args.is_async else args.worker_num  # 并发数
MAX_RETRIES = args.max_retries  # 最大重试次数
MAX_TIMEOUT_SECONDS = args.max_timeout_seconds  # 最多等待时长，单位秒，建议设置值为：条数/2
PROMPT_CSV_FILE = args.prompt_csv_file
OUTPUT_FILE_NAME = args.output_file_name
IS_RETRY = args.is_retry
temperature = args.temperature

USE_GPT4 = args.use_gpt4
MODEL = "gpt-3.5-turbo" if not USE_GPT4 else "gpt-4-turbo"

# 所有的可选集合
# doc:  https://yuque.antfin-inc.com/sjsn/biz_interface/pdxkcxwic4kdarrf
assert MODEL in ("gpt-3.5-turbo",
                 "gpt-3.5-turbo-16k",
                 "gpt-3.5-turbo-instruct", # 一个新模型，似乎是从 gpt3.5 重新训的
                 "gpt-4",
                 "gpt-4-32k",
                 "gpt-4-turbo", # 推荐的 gpt4 接口，128k，而且比前面两个旧接口更便宜
                 )

prompts = read_prompts(PROMPT_CSV_FILE)

if IS_RETRY:
    responses = []
    with open(OUTPUT_FILE_NAME, 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            responses.append(row[2])

def request_model(msg, temp = "0.2"):    
    param = get_default_config(model=MODEL) 
    param["queryConditions"]["messages"][0]["content"] = msg
    param["queryConditions"]["temperature"] = temp
    if IS_ASYNC:
        ask_chatgpt_async_send(param)
        time.sleep(120)
        return ask_chatgpt_async_fetch(param) # 如果请求失败会直接 throw
    else:
        return ask_chatgpt(param) # 同步模式


def request_with_try(idx, prompt, cancel_flag, temp="0.2"):
    retry_count = 0
    result = None
    if IS_RETRY:
        if idx < len(responses) and responses[idx] not in [None, "", "Timeout"]:
            return responses[idx]
    time.sleep(random.randint(0, 120)) # 防止所有人一起请求，因为后面 sleep 了 120 秒，所以让大家均匀的散布在 120s 内开始执行
    while retry_count < MAX_RETRIES:
        try:
            if cancel_flag.is_set():
                # task cancelled
                return None
            result = request_model(prompt, temp=temp)
            break
        except Exception as e:
            if DEBUG:
                print (f"Exception at request with retry: {e}, idx:{idx}, prompt:{prompt[1:30]}...")
            if retry_count == MAX_RETRIES - 1:
                print(
                    f"MAX_RETRIES reached for prompt: {prompt[:20]}..., \nerror {e}, \nresult:{result}")
            # 当前接口分钟级更新流量限制，连续请求容易导致大量报错，加入随机数防止线程都挤到一起。
            time.sleep(random.random() * 15 + 3)
            retry_count += 1

    # print(f'---Prompt---\n{prompt}\n---Response---\n{result}')
    return result


# 创建线程池
with ThreadPoolExecutor(max_workers=WORKER_NUM) as executor:
    start_time = time.time()
    print(f"Start at: {datetime.datetime.now().strftime('%H:%M:%S')}")

    # 使用线程池并行调用函数
    futures = [executor.submit(request_with_try, idx, prompt, threading.Event(
    ), temperature) for idx, prompt in enumerate(prompts)]

    total_waited_second = 0
    TIMEOUT_SECONDS_PER_WAIT = 5
    LOG_INTERVAL = 60  # need to be a multiple of TIMEOUT_SECONDS_PER_WAIT

    try:
        while total_waited_second < MAX_TIMEOUT_SECONDS:
            any_input, _, _ = select.select([sys.stdin], [], [], 0)

            if any_input:
                user_input = sys.stdin.readline().strip()
                print(
                    f"User input:{user_input}. Enter 'quit' or 'exit' to terminate the program and write to output file.\nIt takes up to {TIMEOUT_SECONDS_PER_WAIT}s to terminate.")
                if user_input in ["quit", "exit"]:
                    raise TimeoutError

            # Wait for all futures to complete with a maximum timeout of 10 seconds
            _, _ = wait(futures, timeout=TIMEOUT_SECONDS_PER_WAIT)
            total_waited_second += TIMEOUT_SECONDS_PER_WAIT
            if total_waited_second % LOG_INTERVAL == 0:
                finished_count = sum(future.done() for future in futures)
                if finished_count == len(futures):
                    break
                print(f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}, Number of finished process: {finished_count}. Enter 'quit' or 'exit' to terminate the program and write to output file.")
    except TimeoutError:
        print("Timeout occurred for at least one future.")

    finished_cnt = 0

    with open(OUTPUT_FILE_NAME, 'w', encoding="utf-8-sig", newline='') as file:
        # Create a CSV writer
        writer = csv.writer(file)

        # Write header row
        writer.writerow(['Index', 'Prompt', 'Response'])

        for idx, future in enumerate(futures):
            if future.done():
                writer.writerow([idx, prompts[idx], future.result()])
                finished_cnt += 1
            elif IS_RETRY and idx < len(responses) and responses[idx] not in [None, "", "Timeout"]:
                # 有可能任务还未启动，但可以命中缓存
                writer.writerow([idx, prompts[idx], responses[idx]])
                finished_cnt += 1
            else:
                writer.writerow([idx, prompts[idx], "Timeout"])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Finished num: {finished_cnt}, timeout num:{len(prompts) - finished_cnt}, finished rate:{round(finished_cnt / len(prompts), 2)}")
    print(
        f"Total time: {round(elapsed_time, 2)} seconds, process speed: {round(finished_cnt / elapsed_time, 2)} prompt/sec")

    # forcily terminate the process, to end all running futures
    os.kill(os.getpid(), 9)