import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch.optim.lr_scheduler import StepLR

from utils.BaseModel import EnsembleBaseModel  

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        """Initialize a simple MLP.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of classes
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) 
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 32) 
        self.fc4 = nn.Linear(32, num_classes) 

    def forward(self, x):
        """ Apply the forward pass of the MLP.
        
        Args: 
            x (torch.Tensor): Input tensor  

        Returns:
            torch.Tensor: Output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) 
        return x

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
            mlp_input_dim, 
            gru_input_features, 
            gru_hidden_dim, 
            num_layers_gru, 
            num_classes, 
            loss_fn
            ):
        """ Initialize a simple FusionNet.
        
        Args:
            mlp_input_dim (int): Input dimension for the MLP
            gru_input_features (int): Input dimension for the GRU
            gru_hidden_dim (int): Hidden dimension for the GRU
            num_layers_gru (int): Number of layers for the GRU
            num_classes (int): Number of classes
            loss_fn (torch.nn.Module): Loss function
        """
        super(FusionNet, self).__init__()
        self.mlp = MLP(mlp_input_dim, num_classes)
        self.gru = GRUNet(gru_input_features, gru_hidden_dim, num_layers_gru, num_classes)

        self.num_classes = num_classes
        self.loss_fn = loss_fn

    def forward(self, x_static, x_time_series, label):
        """ Don't fuse and evaluate independent losses

        Args:
            x_static (torch.Tensor): Static patient data
            x_time_series (torch.Tensor): Time series data
            label (torch.Tensor): Label

        Returns:
            tuple: Tuple containing x1_logits, x2_logits, avg_logits, loss
        """
        x1_logits = self.mlp(x_static)
        x2_logits = self.gru(x_time_series)

        x1_loss = self.loss_fn(x1_logits, label)
        x2_loss = self.loss_fn(x2_logits, label)

        return (x1_logits, x2_logits, x1_loss, x2_loss)

class MultimodalMimicModel(EnsembleBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalEnricoModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """
        super(MultimodalMimicModel, self).__init__(args)

        self.w1 = 0.8
        self.w2 = 1.5

    def cast_dtype(self, batch): 
        x1, x2, label = batch
        x1 = x1.to(self.dtype)
        x2 = x2.to(self.dtype)
        return x1, x2, label

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = self.cast_dtype(batch)

        # Get predictions and loss from model
        x1_logits, x2_logits, x1_loss, x2_loss = self.model(x1, x2, label)

        # Calculate acc, unimodal acc not uncalibrated
        x1_acc_cal = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc_cal = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = self.w1*x1_logits + self.w2*x2_logits
        preds = torch.argmax(avg_logits, dim=1)
        joint_acc = torch.mean((preds == label).float())
        avg_loss = self.w1*x1_loss + self.w2*x2_loss

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
    

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = self.cast_dtype(batch)

        # Get predictions and loss from model
        x1_logits, x2_logits, x1_loss, x2_loss = self.model(x1, x2, label)

        # Calculate accuracy
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
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
    
    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = self.cast_dtype(batch)

        # Get predictions and loss from model
        x1_logits, x2_logits, x1_loss, x2_loss = self.model(x1, x2, label)

        # Calculate accuracy
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
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

    # Required for pl.LightningModule
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
            mlp_input_dim=5, 
            gru_input_features=12, 
            gru_hidden_dim=32, 
            num_layers_gru=1, 
            num_classes=self.args.num_classes, 
            loss_fn=nn.CrossEntropyLoss()
        )