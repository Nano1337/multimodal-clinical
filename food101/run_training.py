
# Basic Libraries
import os 
import argparse
import wandb
import yaml
from importlib import resources 

# Deep Learning Libraries
from torch.utils.data import DataLoader

# internal files
from food101.get_data import get_data
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
    - batch[0] is (B, 3, 224, 224) image, modality x1
    - batch[1] is (B, S) text, modality x2
    - batch[2] is [B] labels, 101 food classes
    """

    args = setup_configs()

    # datasets
    train_dataset, val_dataset, test_dataset = get_data(args)

    # get dataloaders
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

    # model training type
    if args.model_type == "jlogits":
        from food101.joint_model import MultimodalFoodModel
    elif args.model_type == "ensemble":
        from food101.ensemble_model import MultimodalFoodModel
    elif args.model_type == "jprobas":
        from food101.joint_model_proba import MultimodalFoodModel
    elif args.model_type == "jprobas_jlogits": 
        from food101.joint_model_proba_logits import MultimodalFoodModel
    else:   
        raise NotImplementedError("Model type not implemented")

    # get model
    model = MultimodalFoodModel(args)

    run_trainer(args, model, train_loader, val_loader, test_loader)