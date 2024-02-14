import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

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

class MultimodalMimicModel(pl.LightningModule): 

    def __init__(self, args): 
        """Initialize MultimodalEnricoModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """
        super(MultimodalMimicModel, self).__init__()

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

    def _convert_type(self, batch):
        x1, x2, label = batch

        x1, x2 = x1.to(self.dtype), x2.to(self.dtype)

        return (x1, x2, label)

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

        # Extract static info, timeseries, and label from batch
        x1, x2, label = self._convert_type(batch)

        # Get predictions and loss from model
        x1_logits, x2_logits, x1_loss, x2_loss = self.model(x1, x2, label)

        # Calculate accuracy
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
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

        # Extract static info, timeseries, and label from batch
        x1, x2, label = self._convert_type(batch)

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

        # Extract static info, timeseries, and label from batch
        x1, x2, label = self._convert_type(batch)

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

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
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