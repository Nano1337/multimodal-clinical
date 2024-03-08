
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
from get_data import get_data, make_balanced_sampler

# set reproducible 
import torch
torch.backends.cudnn.deterministc = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('medium')

DEFAULT_GPUS = [0]

if __name__ == "__main__": 
    torch.multiprocessing.set_start_method('spawn')

    # load configs into args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--configs", type=str, default=None) 
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as yaml_file:
            cfg = yaml.safe_load(yaml_file)
    else:
        raise NotImplementedError
    for key, val in cfg.items():
        setattr(args, key, val)


    seed_everything(args.seed, workers=True) 

    # model training type
    if args.model_type == "jlogits":
        from joint_model import *
    elif args.model_type == "ensemble":
        from ensemble_model import *
    elif args.model_type == "jprobas":
        from joint_model_proba import *
    else:   
        raise NotImplementedError("Model type not implemented")

    """
    batch[0] is (B, 129, 626) audio spectrogram, modality x1
    batch[1] is (B, S, 3, 224, 224) sample of S images from a video, modality x2
    batch[1] is (B, 3, 3, 224, 224) sample of 3 images from a video, modality x2
    batch[2] is [B] labels, 309 classes
    """
    # datasets
    train_dataset, val_dataset, test_dataset = get_data(args)

    # get dataloaders
    train_sampler = make_balanced_sampler(train_dataset.label)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus, 
        persistent_workers=True,
        prefetch_factor = 4,
        sampler=train_sampler,
        collate_fn=train_dataset.custom_collate,
    )

    val_sampler = make_balanced_sampler(val_dataset.label)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_cpus, 
        persistent_workers=True, 
        prefetch_factor=4,
        sampler=val_sampler,
        collate_fn=val_dataset.custom_collate,
    )

    test_sampler = make_balanced_sampler(test_dataset.label)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_cpus, 
        persistent_workers=True, 
        prefetch_factor=4,
        collate_fn=test_dataset.custom_collate,
    )

    # get model
    model = MultimodalVGGSoundModel(args) 

    # define trainer
    trainer = None
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    wandb_logger = WandbLogger(
        group=args.group_name,
        )
    if torch.cuda.is_available(): 
        # call pytorch lightning trainer 
        trainer = pl.Trainer(
            strategy="auto",
            max_epochs=args.num_epochs, 
            logger = wandb_logger if args.use_wandb else None,
            deterministic=True, 
            default_root_dir="ckpts/",  
            precision="bf16-mixed", # "bf16-mixed",
            num_sanity_val_steps=0, # check validation 
            log_every_n_steps=30,  
            callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='epoch')],
            # overfit_batches=1,
        )
    else: 
        raise NotImplementedError("It is not advised to train without a GPU")

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader, 
    )

    trainer.test(
        model, 
        dataloaders=test_loader
    )



