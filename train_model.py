import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/cas6d_train/')
flags = parser.parse_args()

trainer = Trainer(load_cfg(flags.cfg),)
trainer.run()