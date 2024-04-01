import torch 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime

from torch.utils.data import DataLoader

import os
import numpy as np
def run_trainer(args, model, train_loader, val_loader, test_loader, overfit_batches=0, train_dataset=None):
    """
    Run the pytorch lightning trainer with the given model and data loaders
    """
    
    # define trainer
    trainer = None
    if args.use_wandb:
        wandb_logger = WandbLogger(
            group=args.group_name,
        )
        wandb_run_id = wandb_logger.experiment.name

    # define callbacks
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    if args.use_wandb:
        file_name = wandb_run_id + "_best"
    else:
        file_name = str(datetime.now().strftime("%Y%m%d_%H%M%S")) + "_best"

    checkpoint_logger = pl.callbacks.ModelCheckpoint(
        dirpath=args.data_path + "_ckpts/" + args.group_name + "/",
        filename=file_name,
        save_top_k=1,
        monitor="val_epoch/val_avg_acc",
        mode="max",
    )

    # log config to wandb
    if args.use_wandb:
        wandb_logger.log_hyperparams(args)

    if torch.cuda.is_available():
        # call pytorch lightning trainer 
        trainer = pl.Trainer(
            strategy="auto",
            max_epochs=args.warmup_epochs, 
            logger = wandb_logger if args.use_wandb else None,
            deterministic=True, 
            default_root_dir="ckpts/",  
            precision="bf16-mixed", # "bf16-mixed",
            num_sanity_val_steps=0, # check validation 
            log_every_n_steps=30,  
            callbacks=[
                lr_logger, 
                checkpoint_logger,
                ],
            overfit_batches=overfit_batches, # use 1.0 to check if model is working
        )
    else: 
        raise NotImplementedError("It is not advised to train without a GPU")

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader, 
    )

    model.load_state_dict(torch.load(checkpoint_logger.best_model_path)["state_dict"])
    print("Starting second phase of training")

    # TODO: add a flag to enable/disable weighted training
    n=10
    for epoch in range(args.warmup_epochs, args.num_epochs, n):  # n is the step size for each epoch
        print(f"Epoch {epoch}: Loading saved sample weights")
        save_path = os.path.join(args.data_path, f"importance_scores_epoch_{epoch}.npy")
        importance_scores = np.load(save_path)
        print("Sample check: ", importance_scores[:50])
        # Create a weighted sampler using the importance scores for every sample
        weighted_sampler = torch.utils.data.WeightedRandomSampler(importance_scores, len(importance_scores))
        # Update the train_loader to use the new sampler
        new_train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_cpus, 
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=train_dataset.custom_collate,
            sampler=weighted_sampler,  # Use the weighted sampler
        )
        trainer = pl.Trainer(
            strategy="auto",
            max_epochs=epoch + n,  # Adjust max_epochs for the current loop iteration
            logger=wandb_logger if args.use_wandb else None,
            deterministic=True, 
            default_root_dir="ckpts/",  
            precision="bf16-mixed",  # "bf16-mixed",
            num_sanity_val_steps=0,  # check validation 
            log_every_n_steps=30,  
            callbacks=[
                lr_logger, 
                checkpoint_logger,
            ],
            overfit_batches=overfit_batches,  # use 1.0 to check if model is working
        )
        print(f"Begin weighted training for epoch {epoch}")
        trainer.fit(
            model, 
            train_dataloaders=new_train_loader, 
            val_dataloaders=val_loader, 
            ckpt_path=checkpoint_logger.best_model_path
        )

    trainer.test(
        model, 
        dataloaders=test_loader
    )


