import random
import numpy as np
import torch
import os
from loguru import logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
