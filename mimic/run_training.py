# Basic Libraries
import os 
import argparse
import yaml

# Deep Learning Libraries
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything 
from torch.utils.data import DataLoader

# internal files
from mimic.get_data import get_data, train_sampler
from mimic import get_model
from utils.run_trainer import run_trainer
from utils.setup_configs import setup_configs

# set reproducible 
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('medium')

def run_training(): 
    """
    Each batch will return: 
    - Modality 1: [B, 5] 
    - Modality 2: [B, 24, 12]
    - Class Labels: [B], 6 classes
    """
    # manage configs and set reproducibility
    args = setup_configs()

    # datasets
    train_dataset, val_dataset, test_dataset = get_data(task=args.task_num, imputed_path=args.data_path) # -1 indicates mortality 6 class task

    # get dataloaders
    # TODO: add WeightedRandomSampler
    sampler = train_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus, 
        persistent_workers=True,
        prefetch_factor = 4,
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_cpus, 
        persistent_workers=True, 
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_cpus, 
        persistent_workers=True, 
        prefetch_factor=4,
    )

    # get model
    model = get_model(args)

    # start training 
    run_trainer(args, model, train_loader, val_loader, test_loader)