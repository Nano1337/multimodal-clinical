from matplotlib.pyplot import step
import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

from utils.BaseModel import EnsembleBaseModel

from torch.optim.lr_scheduler import StepLR

def get_vicreg_loss(z_a, z_b): 
    """Calculate VicReg Loss.

    Args:
        x1_embedding (torch.Tensor): Embedding from first model
        x2_embedding (torch.Tensor): Embedding from second model

    Returns:
        torch.Tensor: VicReg Loss
    """
    eps = 1e-8

    # variance loss
    std_z_a = torch.sqrt(z_a.var(dim=0) + eps)
    std_z_b = torch.sqrt(z_b.var(dim=0) + eps)
    loss_v_a = torch.mean(F.relu(1 - std_z_a))
    loss_v_b = torch.mean(F.relu(1 - std_z_b))
    loss_var = loss_v_a + loss_v_b

    # invariance loss
    loss_inv = F.mse_loss(z_a, z_b)

    # covariance loss
    N, D = z_a.shape
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
    cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
    loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
    loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D   
    loss_cov = loss_c_a + loss_c_b

    return loss_var + loss_inv + loss_cov


class ResNet18Slim(nn.Module):
    """Extends ResNet18 with a separate embedding and classifier layers.
    
    Slimmer version of ResNet18 model with separate embedding and classifier layers.
    """
    
    def __init__(self, hiddim, pretrained=True, freeze_features=True):
        """Initialize ResNet18Slim Object.

        Args:
            hiddim (int): Hidden dimension size
            pretrained (bool, optional): Whether to instantiate ResNet18 from Pretrained. Defaults to True.
            freeze_features (bool, optional): Whether to keep ResNet18 features frozen. Defaults to True.
        """
        super(ResNet18Slim, self).__init__()
        self.hiddim = hiddim
        self.model = tmodels.resnet18(pretrained=pretrained)
        
        # Remove the last fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.embedding = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, hiddim)
        
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Apply ResNet18Slim to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            tuple: (embedding, logits)
        """
        features = self.model(x)
        embedding = self.embedding(features).view(features.size(0), -1)
        logits = self.classifier(embedding)
        return embedding, logits

class FusionNet(nn.Module):
    def __init__(
            self, 
            num_classes, 
            loss_fn
            ):
        super(FusionNet, self).__init__()
        self.x1_model = ResNet18Slim(num_classes, freeze_features=False)
        self.x2_model = ResNet18Slim(num_classes, freeze_features=False)

        self.num_classes = num_classes
        self.loss_fn = loss_fn

    def forward(self, x1_data, x2_data, label):
        x1_embedding, x1_logits = self.x1_model(x1_data)
        x2_embedding, x2_logits = self.x2_model(x2_data)

        x1_loss = self.loss_fn(x1_logits, label) 
        x2_loss = self.loss_fn(x2_logits, label) 
        vicreg_loss = get_vicreg_loss(x1_embedding, x2_embedding)

        return (x1_logits, x2_logits, x1_loss, x2_loss, vicreg_loss)

class MultimodalEnricoModel(EnsembleBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalEnricoModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalEnricoModel, self).__init__(args)

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
        x1_logits, x2_logits, x1_loss, x2_loss, vicreg_loss = self.model(x1, x2, label)

        # Calculate acc, unimodal acc not uncalibrated
        x1_acc_cal = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc_cal = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
        preds = torch.argmax(avg_logits, dim=1)
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) + vicreg_loss * 0.1

        # Log loss and accuracy
        self.log("train_step/train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_vicreg_loss", vicreg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
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
        x1_logits, x2_logits, x1_loss, x2_loss, vicreg_loss = self.model(x1, x2, label)

        # Calculate accuracy
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = x1_loss + x2_loss + vicreg_loss * 0.1

        # Log loss and accuracy
        self.log("val_step/val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_vicreg_loss", vicreg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

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
        x1_logits, x2_logits, x1_loss, x2_loss, vicreg_loss = self.model(x1, x2, label)

        # Calculate accuracy
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) + vicreg_loss * 0.1

        # Log loss and accuracy
        self.log("test_step/test_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/test_vicreg_loss", vicreg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
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
            loss_fn=nn.CrossEntropyLoss()
        )