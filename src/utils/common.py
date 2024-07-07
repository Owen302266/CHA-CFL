import os
import torch
import numpy as np
import random
import json


# 根目录
project_root = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)


def set_random_seed(random_seed):
    """
       random seed setting
    """
    torch.manual_seed(random_seed)  # if use cpu
    torch.cuda.manual_seed(random_seed)  # if use single-GPU
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = (
        True  # only use when using GPU, the convolutional algorithm returns the same
    )
    torch.backends.cudnn.benchmark = (
        False  # whether use cuDNN to accelerate the training process
    )

    np.random.seed(random_seed)
    random.seed(random_seed)


def load_config(path):
    """
        load json files for parameters configuration
    """
    with open(path, mode='r', encoding='utf-8') as f:
        return json.load(f)
