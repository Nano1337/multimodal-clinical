
# Basic Libraries
import os 
import argparse
from regex import R
import yaml

# Deep Learning Libraries
from torch.utils.data import DataLoader

# internal files
from enrico.get_data import get_data
from enrico import get_model
from utils.run_trainer import run_trainer
from utils.setup_configs import setup_configs

# set reproducible 
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('medium')

def run_training(): 
    """
    Data: 
    - batch[0] is (B, 3, 256, 128) screenshot, modality x1
    - batch[1] is (B, 3, 256, 128) wireframe, modality x2
    - batch[2] is [B] labels, 20 classes of design topics
    """

    # manage configs and set reproducibility
    args = setup_configs()

    # datasets
    train_dataset, val_dataset, test_dataset, sampler = get_data(args)
    setattr(args, "num_samples", len(train_dataset))

    # get dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus, 
        persistent_workers=True,
        prefetch_factor = 4,
        sampler=sampler,
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

    model = get_model(args)

    run_trainer(args, model, train_loader, val_loader, test_loader)