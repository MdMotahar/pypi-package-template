"""
Bangla Named Entity Recognition System
"""

import os
import torch
from . import inference
from .inference import Infer

VERSION = '0.0.1'
ROOT_DIR = os.path.dirname(__file__) 
CACHE_DIR = os.path.join(ROOT_DIR,'.cache')

device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

__all__ = ['inference','utils']
