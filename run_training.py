
# Basic Libraries
import os 
import argparse
import yaml

# Deep Learning Libraries
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything 

# set reproducible 
torch.backends.cudnn.deterministc = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('medium')

seed = 0
seed_everything(seed, workers=True)

if __name__ == "__main__": 
    pass