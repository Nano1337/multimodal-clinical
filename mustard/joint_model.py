import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

class GRUNet(nn.Module):
    def __init__(self, input_features, hidden_dim, num_layers, num_classes):
        """ Initialize a simple GRU.
        
        Args:
            input_features (int): Input dimension
            hidden_dim (int): Hidden dimension
            num_layers (int): Number of layers
            num_classes (int): Number of classes
        """
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_features, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)  
        self.fc2 = nn.Linear(64, 32)        
        self.fc3 = nn.Linear(32, num_classes)         

    def forward(self, x):
        """ Apply the forward pass of the GRU.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        _, h = self.gru(x)  # We only need the hidden state
        x = F.relu(self.fc1(h[-1]))  # Take the last layer's hidden state
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

class FusionNet(nn.Module):
    def __init__(
            self, 
            num_classes, 
            loss_fn
            ):
        super(FusionNet, self).__init__()
        self.x1_model = GRUNet(371, 512, 1, num_classes)
        self.x2_model = GRUNet(81, 256, 1, num_classes)
        self.x3_model = GRUNet(300, 600, 1, num_classes)

        self.num_classes = num_classes
        self.loss_fn = loss_fn

    def forward(self, x1_data, x2_data, x3_data, label):
        """ Forward pass for the FusionNet model. Fuses at logit level.
        
        Args:
            x1_data (torch.Tensor): Input data for modality 1
            x2_data (torch.Tensor): Input data for modality 2
            x3_data(torch.Tensor): Input data for modality 3
            label (torch.Tensor): Ground truth label

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the logits for modality 1, modality 2, 
                                                                            modality 3, average logits, and loss
        """
        x1_logits = self.x1_model(x1_data)
        x2_logits = self.x2_model(x2_data)
        x3_logits = self.x3_model(x3_data)

        # fuse at logit level
        avg_logits = (x1_logits + x2_logits + x3_logits) / 2

        label = label.flatten()
        loss = self.loss_fn(avg_logits, label)

        return (x1_logits, x2_logits, x3_logits, avg_logits, loss)

class MultimodalMustardModel(pl.LightningModule): 

    def __init__(self, args): 
        """Initialize MultimodalMustardModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalMustardModel, self).__init__()

        self.args = args
        self.model = self._build_model()

        self.val_metrics = {
            "val_loss": [], 
            "val_acc": [],
            "val_logits": [],
            "val_labels": [],
        }

        self.test_metrics = {
            "test_loss": [], 
            "test_acc": [], 
            "test_logits": [],
            "test_labels": [],
        }

    def forward(self, x1, x2, x3, label): 
        return self.model(x1, x2, x3, label)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing screenshot, wireframe, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, x3, label = batch

        # Get predictions and loss from model
        _, _, _, avg_logits, loss = self.model(x1, x2, x3, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # Return the loss
        return loss

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing screenshot, wireframe, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, x3, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, x3_logits, avg_logits, loss = self.model(x1, x2, x3, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_logits"].append(torch.stack((x1_logits, x2_logits, x3_logits), dim=1))
        self.val_metrics["val_labels"].append(label)
        self.val_metrics["val_loss"].append(loss)
        self.val_metrics["val_acc"].append(joint_acc)
 
        return loss

    def on_validation_epoch_end(self) -> None:
        """ Called at the end of the validation epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.val_metrics["val_labels"], dim=0) # (N)
        logits = torch.cat(self.val_metrics["val_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]
        x3_logits = corrected_logits[:, 2, :]

        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        x3_acc = torch.mean((torch.argmax(x3_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x1_val_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x2_val_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x3_val_acc", x3_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_logits"].clear()
        self.val_metrics["val_labels"].clear()


    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing screenshot, wireframe, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, x3, label = batch 

        # Get predictions and loss from model
        x1_logits, x2_logits, x3_logits, avg_logits, loss = self.model(x1, x2, x3, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_logits"].append(torch.stack((x1_logits, x2_logits, x3_logits), dim=1))
        self.test_metrics["test_labels"].append(label)
        self.test_metrics["test_loss"].append(loss)
        self.test_metrics["test_acc"].append(joint_acc)

        # Return the loss
        return loss
    
    def on_test_epoch_end(self):
        """ Called at the end of the test epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.test_metrics["test_labels"], dim=0) # (N)
        logits = torch.cat(self.test_metrics["test_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]
        x3_logits = corrected_logits[:, 2, :]

        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        x3_acc = torch.mean((torch.argmax(x3_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_accuracy = torch.stack(self.test_metrics["test_acc"]).mean()

        self.log("avg_test_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("avg_test_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x1_test_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x2_test_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x3_test_acc", x3_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_logits"].clear()
        self.test_metrics["test_labels"].clear()

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def _build_model(self):
        return FusionNet(
            num_classes=self.args.num_classes, 
            loss_fn=nn.CrossEntropyLoss()
        )