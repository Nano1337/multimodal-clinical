from matplotlib.pyplot import step
import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

from utils.BaseModel import EnsembleBaseModel

from torch.optim.lr_scheduler import StepLR

class VGG11Slim(nn.Module): 
    """Extends VGG11 with a fewer layers in the classifier.
    
    Slimmer version of vgg11 model with fewer layers in classifier.
    """
    
    def __init__(self, hiddim, dropout=True, dropoutp=0.2, pretrained=True, freeze_features=True):
        """Initialize VGG11Slim Object.

        Args:
            hiddim (int): Hidden dimension size
            dropout (bool, optional): Whether to apply dropout to output of ReLU. Defaults to True.
            dropoutp (float, optional): Dropout probability. Defaults to 0.2.
            pretrained (bool, optional): Whether to instantiate VGG11 from Pretrained. Defaults to True.
            freeze_features (bool, optional): Whether to keep VGG11 features frozen. Defaults to True.
        """
        super(VGG11Slim, self).__init__()
        self.hiddim = hiddim
        self.model = tmodels.vgg11_bn(pretrained=pretrained)
        self.model.classifier = nn.Linear(512 * 7 * 7, hiddim)
        if dropout:
            feats_list = list(self.model.features)
            new_feats_list = []
            for feat in feats_list:
                new_feats_list.append(feat)
                if isinstance(feat, nn.ReLU):
                    new_feats_list.append(nn.Dropout(p=dropoutp))

            self.model.features = nn.Sequential(*new_feats_list)
        for p in self.model.features.parameters():
            p.requires_grad = (not freeze_features)

    def forward(self, x):
        """Apply VGG11Slim to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return self.model(x)

class FusionNet(nn.Module):
    def __init__(
            self, 
            num_classes, 
            loss_fn
            ):
        super(FusionNet, self).__init__()
        self.x1_model = VGG11Slim(num_classes)
        self.x2_model = VGG11Slim(num_classes)

        self.num_classes = num_classes
        self.loss_fn = loss_fn

    def forward(self, x1_data, x2_data, label):
        x1_logits = self.x1_model(x1_data)
        x2_logits = self.x2_model(x2_data)

        x1_loss = self.loss_fn(x1_logits, label) 
        x2_loss = self.loss_fn(x2_logits, label) 

        return (x1_logits, x2_logits, x1_loss, x2_loss)

class MultimodalEnricoModel(EnsembleBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalEnricoModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalEnricoModel, self).__init__(args)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.count_dict = {
            "train": torch.tensor([0, 0, 0], device=device),
            "val": torch.tensor([0, 0, 0], device=device),
            "test": torch.tensor([0, 0, 0], device=device)
        }
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    def update_counts(self, joint_loss, x1_loss, x2_loss, mode="train"): 
        joint_loss_min = torch.minimum(joint_loss, x1_loss).to(self.device)
        joint_loss_min = torch.minimum(joint_loss_min, x2_loss).to(self.device)
        joint_loss = joint_loss.to(self.device)
        x1_loss = x1_loss.to(self.device)
        x2_loss = x2_loss.to(self.device)
        self.count_dict[mode] += torch.sum(torch.stack([
                joint_loss == joint_loss_min,
                x1_loss == joint_loss_min, 
                x2_loss == joint_loss_min
            ], dim=1), dim=0).int()
        
    def log_counts(self, mode="train"): 
        self.log(f"{mode}_epoch/joint_count", self.count_dict[mode][0], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{mode}_epoch/x1_count", self.count_dict[mode][1], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{mode}_epoch/x2_count", self.count_dict[mode][2], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.count_dict[mode][0] = 0
        self.count_dict[mode][1] = 0
        self.count_dict[mode][2] = 0

    def forward(self, x1, x2, label): 
        return self.model(x1, x2, label)

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
        joint_loss = self.loss_fn(avg_logits, label)
        self.update_counts(joint_loss, x1_loss, x2_loss, mode="train")
        preds = torch.argmax(avg_logits, dim=1)
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        x1_loss = torch.mean(x1_loss)
        x2_loss = torch.mean(x2_loss)
        avg_loss = (x1_loss + x2_loss) 


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

        # Return the loss
        return avg_loss
    
    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        self.log_counts("train")

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

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

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

        # Calculate accuracy
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
        joint_loss = self.loss_fn(avg_logits, label)
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        self.update_counts(joint_loss, x1_loss, x2_loss, mode="val")
        x1_loss = torch.mean(x1_loss)
        x2_loss = torch.mean(x2_loss)
        avg_loss = (x1_loss + x2_loss) / 2
        # Log loss and accuracy
        self.log("val_step/val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_loss"].append(avg_loss)
        self.val_metrics["val_acc"].append(joint_acc)
        self.val_metrics["val_x1_acc"].append(x1_acc)
        self.val_metrics["val_x2_acc"].append(x2_acc)

        # Return the loss
        return avg_loss

    def on_validation_epoch_end(self) -> None:
        """ Called at the end of the validation epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        self.log_counts("val")

        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()
        x1_acc = torch.mean(torch.stack(self.val_metrics["val_x1_acc"]))
        x2_acc = torch.mean(torch.stack(self.val_metrics["val_x2_acc"]))

        self.log("val_epoch/val_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_x1_acc"].clear()
        self.val_metrics["val_x2_acc"].clear()
    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

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

     
        avg_logits = (x1_logits + x2_logits) / 2
        joint_loss = self.loss_fn(avg_logits, label)

        # update counts
        self.update_counts(joint_loss, x1_loss, x2_loss)

        x1_loss = torch.mean(x1_loss)
        x2_loss = torch.mean(x2_loss)

        # Calculate accuracy
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())

        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) / 2

        # Log loss and accuracy
        self.log("test_step/test_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_loss"].append(avg_loss)
        self.test_metrics["test_acc"].append(joint_acc)
        self.test_metrics["test_x1_acc"].append(x1_acc)
        self.test_metrics["test_x2_acc"].append(x2_acc)

        # Return the loss
        return avg_loss
    
    def on_test_epoch_end(self):
        """ Called at the end of the test epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        self.log_counts("test")
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_acc = torch.stack(self.test_metrics["test_acc"]).mean()
        x1_acc = torch.mean(torch.stack(self.test_metrics["test_x1_acc"]))
        x2_acc = torch.mean(torch.stack(self.test_metrics["test_x2_acc"]))

        self.log("test_epoch/test_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_x1_acc"].clear()
        self.test_metrics["test_x2_acc"].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=70, gamma=0.5),
                'interval': 'epoch',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
            
        return optimizer
    
    def _build_model(self):
        return FusionNet(
            num_classes=self.args.num_classes, 
            loss_fn=nn.CrossEntropyLoss(reduction="none")
        )