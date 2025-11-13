import torch
import os
import sys
import argparse
import logging
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.image as image
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import matplotlib.text as text
import matplotlib.font_manager as font_manager
import matplotlib.colors as colors

current_dir = os.path.dirname(os.path.abspath(__file__))

# pretrained model path
INPAINTING_MODEL_PATH = "/home/liying/Documents/stable-diffusion-inpainting"
INPAINTING_2_MODEL_PATH = "/home/liying/Documents/stable-diffusion-2-inpainting"

STABLE_DIFFUSION_V1_5_MODEL_PATH = "/home/liying/Documents/stable-diffusion-v1-5"

QWEN2_5_VL_7B_INSTUCT_MODEL_PATH = "/home/liying/Documents/Qwen2.5-VL-7B-Instruct"

HALLUSEGBENCH_DATASET_PATH = "/home/liying/Desktop/IMAGE_EDITE-CVPR-2025/Qwen2.5-VL-7B-Instruct_Based/HalluSegBench/HalluSegBench"
PROMPTS_OUTPUT_DIR = os.path.join(HALLUSEGBENCH_DATASET_PATH, "prompts_output_new")
os.makedirs(PROMPTS_OUTPUT_DIR, exist_ok=True)


HALLUSEGGOOD_DATASET_PATH = "/home/liying/Desktop/IMAGE_EDITE-CVPR-2025/Qwen2.5-VL-7B-Instruct_Based/dataset/HalluSegGood"
PROMPTS_OUTPUT_HALLUSEGGOOD_DIR = os.path.join(HALLUSEGGOOD_DATASET_PATH, "prompts_output_new")
os.makedirs(PROMPTS_OUTPUT_HALLUSEGGOOD_DIR, exist_ok=True)

EditBench_DATASET_PATH = "/home/liying/Desktop/IMAGE_EDITE-CVPR-2025/Qwen2.5-VL-7B-Instruct_Based/dataset/EditBench"
PROMPTS_OUTPUT_EDITBENCH_DIR = os.path.join(EditBench_DATASET_PATH, "prompts_output_new")
os.makedirs(PROMPTS_OUTPUT_EDITBENCH_DIR, exist_ok=True)


# dataset path
CHECKPOINTS_DIR = os.path.join(current_dir, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# result path
OUTPUT_DIR = os.path.join(current_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
