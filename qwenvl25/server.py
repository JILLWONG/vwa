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
from http.server import BaseHTTPRequestHandler, HTTPServer
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch
from flask import Flask, request
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class WebService:
    def __init__(self, config):
        self.app = Flask(__name__)
        self.config = config
        model_path = self.config['model_path']

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self._initialize()

    def _initialize(self):
        @self.app.route('/', methods=['GET', 'POST'])
        def home():
            if request.method == 'POST':
                data = request.get_json()
                sys_prompt = data.get('sys_prompt', '')
                prompt = data.get('prompt', '')
                image_file = data.get('image_file', '')
                
                image = Image.open(image_file)
                image_local_path = "file://" + image_file
                messages = [
                    {
                        "role": "system", 
                        "content": sys_prompt
                    },
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt
                            },
                            {
                                "image": image_local_path
                            },
                        ]
                    },
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
                inputs = inputs.to('cuda')

                max_new_tokens=4096

                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                return output_text[0]
            else:
                # 处理 GET 请求
                return f"Error"

    def run(self):
        self.app.run(host='0.0.0.0', port=self.config.get('port'))


if __name__ == '__main__':
    config = {
        'port': 8888,
        'model_path': f'{u.get_home()}/model/Qwen2.5-VL-72B-Instruct',
    }
    server = WebService(config)
    server.run()


