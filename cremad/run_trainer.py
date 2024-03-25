
# Basic Libraries
import os 
import argparse
import wandb
import yaml
from importlib import resources 

# Deep Learning Libraries
from torch.utils.data import DataLoader

# internal files
from cremad import get_model
from cremad.get_data import get_data, make_balanced_sampler
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
    - batch[0] is (B, 257, 1004) audio spectrogram, modality x1
    - batch[1] is (B, 3, 3, 224, 224) sample of 3 images from a video, modality x2
    - batch[2] is [B] labels, 6 sentiment classes
    """

    # manage configs and set reproducibility
    args = setup_configs()

    # datasets
    train_dataset, val_dataset, test_dataset = get_data(args)
    setattr(args, "num_samples", len(train_dataset))

    # get dataloaders
    train_sampler = make_balanced_sampler(train_dataset.label)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus, 
        persistent_workers=True,
        prefetch_factor = 4,
        collate_fn=train_dataset.custom_collate,
        sampler=train_sampler,
    )

    val_sampler = make_balanced_sampler(val_dataset.label)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_cpus, 
        persistent_workers=True, 
        prefetch_factor=4,
        collate_fn=train_dataset.custom_collate, 
        sampler=val_sampler,
    )

    # test_sampler = make_balanced_sampler(test_dataset.label) # don't change test distribution
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_cpus, 
        persistent_workers=True, 
        prefetch_factor=4,
        collate_fn=train_dataset.custom_collate, 
    )

    model = get_model(args)

    # start training
    run_trainer(args, model, train_loader, val_loader, test_loader)