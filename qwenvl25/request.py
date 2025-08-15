import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
import requests

def request_qwen(sys_prompt, prompt, image_file):
    url = 'http://localhost:8888/'
    data = {
        'sys_prompt': sys_prompt,
        'prompt': prompt,
        'image_file': image_file,
    }
    response = requests.post(url, json=data)
    return response.text

if __name__ == '__main__':
    sys_prompt = 'You are a helpful assistant.',
    prompt = "What kind of bird is this? Please give its name in Chinese and English.",
    image_file = u.get_home() + '/kevin_git/qwenvl25/cookbooks/assets/universal_recognition/unireco_bird_example.jpg'

    response = request_qwen(sys_prompt, prompt, image_file)
    DEBUG(response)




