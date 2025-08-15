# -*- coding: utf-8 -*-
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, wait, TimeoutError
import threading
from utils import (
    get_asyn_config,
    get_fetch_config,
    read_prompts,
    send_request,
    parse_response,
    parse_fetch_result
)
import csv
import random
import os
import select
import sys
import hashlib
import argparse
from tqdm import tqdm

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='批量请求 GPT4')

# Add arguments
parser.add_argument('--prompt_csv_file', type=str, default="prompts.csv", help='并发度，调太高容易超 token/min 限制')
parser.add_argument('--output_file_name', type=str, default="gpt_response.csv", help='并发度，调太高容易超 token/min 限制')
parser.add_argument('--is_retry', type=bool, default=False, help='如果设置成 true，会先加载 output file，然后讲其中 timeout 和空值进行更新，需要确保 output 和 prompt 顺序能对上.')
parser.add_argument('--use_gpt4', type=bool, default=True, help='如果设置成 true，会使用 gpt4，不然使用 gpt3.5')
parser.add_argument('--temperature', type=str, default="0.0", help='如果设置成 true，会使用 gpt4，不然使用 gpt3.5')
parser.add_argument('--mode', type=str, default="send_then_fetch", help='send 来发送异步请求，fetch 来获取结果, send_then_fetch 来同步等待结果，同步等待时，使用 wait_time 设定等待时长')
parser.add_argument('--wait_time', type=int, default=60, help='仅在 mode=send_then_fetch 时有作用，同步等待结果')
parser.add_argument('--max_retries', type=int, default=50, help='最大重试次数')

# Parse the command-line arguments
args = parser.parse_args()
PROMPT_CSV_FILE = args.prompt_csv_file
OUTPUT_FILE_NAME = args.output_file_name
IS_RETRY = args.is_retry
temperature = args.temperature
MAX_RETRIES = args.max_retries  # 最大重试次数

USE_GPT4 = args.use_gpt4
MODEL = "gpt-3.5-turbo" if not USE_GPT4 else "gpt-4-turbo"
MODE = args.mode
WAIT_TIME = args.wait_time

# 所有的可选集合
# doc:  https://yuque.antfin-inc.com/sjsn/biz_interface/pdxkcxwic4kdarrf
assert MODEL in ("gpt-3.5-turbo",
                 "gpt-3.5-turbo-16k",
                 "gpt-3.5-turbo-instruct", # 一个新模型，似乎是从 gpt3.5 重新训的
                 "gpt-4",
                 "gpt-4-32k",
                 "gpt-4-turbo", # 推荐的 gpt4 接口，128k，而且比前面两个旧接口更便宜
                 )
assert MODE in ("send", "fetch", "send_then_fetch")

prompts = read_prompts(PROMPT_CSV_FILE)

if IS_RETRY:
    responses = []
    with open(OUTPUT_FILE_NAME, 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            responses.append(row[-1])

def send_asyn_request(msg, msg_key, temp=0.2):
    param = get_asyn_config(model=MODEL) 
    param["queryConditions"]["messages"][0]["content"] = msg
    param["queryConditions"]["messageKey"] = msg_key
    param["queryConditions"]["temperature"] = temp
    response = send_request(param)
    try:
        result = parse_response(response)
        return result["data"]["success"]
    except Exception as e:
        print(e)
        return False


def fetch_asyn_result(msg_key):
    param = get_fetch_config(model=MODEL)
    param["queryConditions"]["messageKey"] = msg_key
    response = send_request(param)
    try:
        return parse_fetch_result(response)
    except Exception as e:
        print(e)
        return None

msg_keys = [hashlib.sha256(str(p+temperature).encode('utf-8')).hexdigest() for p in prompts] 

if MODE in ("send", "send_then_fetch"):
    for idx, msg_key in tqdm(enumerate(msg_keys)):
        if IS_RETRY and responses[idx] not in [None, "", "Timeout"]:
            continue
        msg = prompts[idx]
        retry_count = 0
        while retry_count < MAX_RETRIES:
            result = send_asyn_request(msg, msg_key, temperature)
        # print (f"{msg_key=}")
        # print (f"{result=}")
            if result:
                break
            retry_count = retry_count + 1
            time.sleep(random.random() * 10 + 3)
        if not result:
            print(f"Send request failed for prompt starts with: {msg[1:30]}...")

if MODE == "send_then_fetch":
    time.sleep(WAIT_TIME)

if MODE in ("fetch", "send_then_fetch"):
 
    with open(OUTPUT_FILE_NAME, 'w', encoding="utf-8-sig", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index','hashkey', 'prompt', 'response'])

        for idx, msg_key in tqdm(enumerate(msg_keys)):
            if IS_RETRY and responses[idx] not in [None, "", "Timeout"]:
                writer.writerow([idx, msg_key, prompts[idx], responses[idx]])
            else:
                result = fetch_asyn_result(msg_key)
                writer.writerow([idx, msg_key, prompts[idx], result])

