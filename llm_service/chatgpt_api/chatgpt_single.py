# -*- coding: utf-8 -*-
import time
from utils import get_default_config, ask_chatgpt



prompt='''explain what is chatgpt?'''

USE_GPT4 = False
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

def request_model(msg):    
    param = get_default_config(model=MODEL) 
    param["queryConditions"]["messages"][0]["content"] = msg
    
    return ask_chatgpt(param)

max_retries = 5
retry_count = 0
result=None
while retry_count < max_retries:
    try:
        result = request_model(prompt)
        break
    except Exception as e:
        if retry_count == max_retries - 1:
            print(f"MAX_RETRIES reached for prompt: {prompt}, \nerror {e}, \nresult:{result}")
        time.sleep(1)
        retry_count += 1

print(f'---Prompt---\n{prompt}\n---Response---\n{result}')

