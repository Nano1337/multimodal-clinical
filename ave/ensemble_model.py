import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from backbone import resnet18

from torch.optim.lr_scheduler import StepLR


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

        x1_loss = self.loss_fn(x1_logits, label)
        x2_loss = self.loss_fn(x2_logits, label)

        return (x1_logits, x2_logits, x1_loss, x2_loss)

class MultimodalAveModel(pl.LightningModule): 

    def __init__(self, args): 
        """Initialize MultimodalAveModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalAveModel, self).__init__()

        self.args = args
        self.model = self._build_model()

        self.val_metrics = {
            "val_loss": [], 
            "val_acc": [],
            "val_x1_acc": [], 
            "val_x2_acc": [],
        }

        self.test_metrics = {
            "test_loss": [], 
            "test_acc": [], 
            "test_x1_acc": [],
            "test_x2_acc": [],
        }

    def forward(self, x1, x2, label): 
        return self.model(x1, x2, label)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing screenshot, wireframe, and label
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
        preds = torch.argmax(avg_logits, dim=1)
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) / 2

        # Log loss and accuracy
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # Return the loss
        return avg_loss

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing screenshot, wireframe, and label
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
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) / 2

        # Log loss and accuracy
        self.log("val_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

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

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x1_val_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x2_val_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_x1_acc"].clear()
        self.val_metrics["val_x2_acc"].clear()

    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing screenshot, wireframe, and label
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
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) / 2

        # Log loss and accuracy
        self.log("test_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

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

        self.log("test_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x1_test_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x2_test_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_x1_acc"].clear()
        self.test_metrics["test_x2_acc"].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=10, gamma=0.5),
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