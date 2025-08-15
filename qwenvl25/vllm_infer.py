import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from PIL import Image
from transformers import Qwen2Tokenizer
from vllm import LLM, SamplingParams
from transformers import Qwen2_5_VLProcessor

# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation,
using the chat template defined by the model.
"""
import os
from argparse import Namespace
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from PIL.Image import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None

def load_qwen2_vl(model_path, question: str, image_urls: list[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    model_name = model_path

    # Tested on L40
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768 if process_vision_info is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
            {
                "type": "text",
                "text": question
            },
        ],
    }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:
        image_data, _ = process_vision_info(messages)

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )

def load_qwen2_5_vl(model_path, question: str, image_urls: list[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    model_name = model_path

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=32768 if process_vision_info is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
            {
                "type": "text",
                "text": question
            },
        ],
    }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:
        image_data, _ = process_vision_info(messages,
                                            return_video_kwargs=False)

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


model_example_map = {
    "qwen2_vl": load_qwen2_vl,
    "qwen2_5_vl": load_qwen2_5_vl,
}

def run_generate(model, model_path, question: str, image_urls: list[str], seed: Optional[int]):
    req_data = model_example_map[model](model_path, question, image_urls)
    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=4096,
                                     stop_token_ids=req_data.stop_token_ids)

    outputs = llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {
                "image": req_data.image_data
            },
        },
        sampling_params=sampling_params,
        lora_request=req_data.lora_requests,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)

if __name__ == '__main__':
    if u.get_os() == 'mac': NAS_PATH = f'{u.get_home()}/data/'
    else: NAS_PATH = '/mnt/agent-s1/common/public/kevin/'
    model_rel_path = 'output'
    model_name = '20250610153514_Qwen2.5-VL-7B-Instruct-SCOT_sft_sw0_pt/checkpoint-100'
    model_path = f'{NAS_PATH}/{model_rel_path}/{model_name}'
    img_file = '/ossfs/workspace/kevin_git/qwenvl25/qwen-vl-finetune/demo/images/COCO_train2014_000000580957.jpg'

    QUESTION = "图片里有什么"
    image_urls = [img_file]
    run_generate('qwen2_5_vl', model_path, QUESTION, image_urls, 0)
