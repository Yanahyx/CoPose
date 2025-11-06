import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/cas6d_train/')
flags = parser.parse_args()

trainer = Trainer(load_cfg(flags.cfg),)
trainer.run()