import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from cremad.backbone import resnet18

from torch.optim.lr_scheduler import StepLR
from utils.BaseModel import EnsembleBaseModel

from existing_algos.famo import FAMO

class FusionNet(nn.Module):
    def __init__(
            self, 
            num_classes, 
            loss_fn
            ):
        super(FusionNet, self).__init__()
        self.x1_model = resnet18(modality='audio')
        self.x1_classifier = nn.Linear(512, num_classes)
        self.x2_model = resnet18(modality='visual')
        self.x2_classifier = nn.Linear(512, num_classes)

        self.num_classes = num_classes
        self.loss_fn = loss_fn


    def forward(self, x1_data, x2_data, label):
        """ Forward pass for the FusionNet model. Fuses at logit level.
        
        Args:
            x1_data (torch.Tensor): Input data for modality 1
            x2_data (torch.Tensor): Input data for modality 2
            label (torch.Tensor): Ground truth label

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the logits for modality 1, modality 2, average logits, and loss
        """

        a = self.x1_model(x1_data)
        v = self.x2_model(x2_data)
        
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)
        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        x1_logits = self.x1_classifier(a)
        x2_logits = self.x2_classifier(v)

        x1_loss = self.loss_fn(x1_logits, label) * 3.0 
        x2_loss = self.loss_fn(x2_logits, label) * 3.0 

        return (x1_logits, x2_logits, x1_loss, x2_loss)

class MultimodalCremadModel(EnsembleBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalCremadModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """
        super(MultimodalCremadModel, self).__init__(args)

        self.automatic_optimization = False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.famo = FAMO(n_tasks=2, device=device)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, x1_loss, x2_loss = self.model(x1, x2, label)

        # Calculate acc, unimodal acc not uncalibrated
        x1_acc_cal = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc_cal = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
        preds = torch.argmax(avg_logits, dim=1)
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) / 2

        # Log loss and accuracy
        self.log("train_step/train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_acc", x1_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_acc", x2_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # accumulate accuracies and losses
        self.train_metrics["train_loss"].append(avg_loss)
        self.train_metrics["train_acc"].append(joint_acc)
        self.train_metrics["train_x1_acc"].append(x1_acc_cal)
        self.train_metrics["train_x2_acc"].append(x2_acc_cal)

        opt = self.optimizers()
        opt.zero_grad()
        losses = torch.stack([x1_loss, x2_loss])
        self.famo.backward(losses)
        opt.step()

        with torch.no_grad():
            self.famo.update(curr_loss=losses)


        # Return the loss
        return avg_loss
    
    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        avg_loss = torch.stack(self.train_metrics["train_loss"]).mean()
        avg_acc = torch.stack(self.train_metrics["train_acc"]).mean()
        x1_acc = torch.mean(torch.stack(self.train_metrics["train_x1_acc"]))
        x2_acc = torch.mean(torch.stack(self.train_metrics["train_x2_acc"]))

        self.log("train_epoch/train_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.train_metrics["train_loss"].clear()
        self.train_metrics["train_acc"].clear()
        self.train_metrics["train_x1_acc"].clear()
        self.train_metrics["train_x2_acc"].clear()

        if self.args.use_scheduler:
            schedulers = self.lr_schedulers()
            
            # handle single scheduler and step schedulers per epoch
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            for scheduler in schedulers:
                scheduler.step()
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=70, gamma=0.1),
                'interval': 'epoch',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
            
        return optimizer
    
    def _build_model(self):
        return FusionNet(
            num_classes=self.args.num_classes, 
            loss_fn=nn.CrossEntropyLoss()
        )