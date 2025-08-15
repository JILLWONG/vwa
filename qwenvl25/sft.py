import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u
from transformers import logging as tlogging
tlogging.set_verbosity_info()  # 设置为 INFO 级别（默认是 WARNING）
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)

if __name__ == '__main__':
    job_name = sys.argv[1]
    MASTER_ADDR = "127.0.0.1" # [Required] Master node IP for multi-GPU training
    MASTER_PORT = 7777 # Random port to avoid conflicts
    NPROC_PER_NODE = 1 # Automatically detects available GPUs
    # DEVICE = 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7'
    DEVICE = 'CUDA_VISIBLE_DEVICES=0'
    home = u.get_home()
    model_name = 'Qwen2.5-VL-3B-Instruct'
    MODEL_PATH = f"{home}/model/{model_name}/"
    OUTPUT_DIR = f"{home}/data/checkpoints/"
    u.mkdir(OUTPUT_DIR)
    LOG_DIR = f'{home}/data/log/'
    u.mkdir(LOG_DIR)
    CACHE_DIR  = f"{home}/data/cache"
    u.mkdir(CACHE_DIR)
    DATASETS = "demo_dataset%100" # [DataArguments] Dataset with sampling rate
    CODE_PATH = f'{u.get_home()}/kevin_git/qwenvl25/'
    # DEEP_SPEED = ''
    DEEP_SPEED = f'{CODE_PATH}/qwen-vl-finetune/scripts/zero1.json'
    SAVE_DIR = f'{u.get_time()}_{model_name}_{DATASETS}_{job_name}'

    MAIN_CMD = \
f'''{CODE_PATH}/qwen-vl-finetune/qwenvl/train/train_qwen.py \
--model_name_or_path {MODEL_PATH} \
--tune_mm_llm True \
--tune_mm_vision False \
--tune_mm_mlp False \
--bf16 \
--dataset_use {DATASETS} \
--output_dir {OUTPUT_DIR + SAVE_DIR} \
--cache_dir {CACHE_DIR} \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--learning_rate 1e-5 \
--mm_projector_lr 1e-5 \
--vision_tower_lr 1e-6 \
--optim adamw_torch \
--model_max_length 8192 \
--log_level info \
--data_flatten True \
--max_pixels {2560*1440} \
--min_pixels {16*28*28} \
--base_interval 2 \
--video_max_frames 8 \
--video_min_frames 4 \
--video_max_frame_pixels {1664*28*28} \
--video_min_frame_pixels {256*28*28} \
--num_train_epochs 800000 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--logging_steps 1 \
--logging_dir {LOG_DIR} \
--save_steps 10000000 \
--save_total_limit 1 \
--deepspeed {DEEP_SPEED} \
--report_to none \
--save_strategy steps'''

    command = \
f'''{DEVICE} torchrun --nproc_per_node {NPROC_PER_NODE} --master_addr {MASTER_ADDR} --master_port {MASTER_PORT} {MAIN_CMD}'''

    u.execute(command)