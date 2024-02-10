import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

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

        # fuse at logit level
        avg_logits = (x1_logits + x2_logits) / 2

        loss = self.loss_fn(avg_logits, label)

        return (x1_logits, x2_logits, avg_logits, loss)

class MultimodalEnricoModel(pl.LightningModule): 

    def __init__(self, args): 
        super(MultimodalEnricoModel, self).__init__()

        self.args = args
        self.model = self._build_model()

        self.train_logits_sum = None # should be dim (M, C)
        self.total_samples = 0

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

    def forward(self, x1, x2, label): 
        return self.model(x1, x2, label)

    def training_step(self, batch, batch_idx): 

        # Extract static info, timeseries, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # Return the loss
        return loss

    def validation_step(self, batch, batch_idx): 

        # Extract static info, timeseries, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.val_metrics["val_labels"].append(label)
        self.val_metrics["val_loss"].append(loss)
        self.val_metrics["val_acc"].append(joint_acc)
 
        return loss

    def on_validation_epoch_end(self) -> None:
        labels = torch.cat(self.val_metrics["val_labels"], dim=0) # (N)
        logits = torch.cat(self.val_metrics["val_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x1_val_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x2_val_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_logits"].clear()
        self.val_metrics["val_labels"].clear()


    def test_step(self, batch, batch_idx):

        # Extract static info, timeseries, and label from batch
        x1, x2, label = batch 

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.test_metrics["test_labels"].append(label)
        self.test_metrics["test_loss"].append(loss)
        self.test_metrics["test_acc"].append(joint_acc)


        # Return the loss
        return loss
    
    def on_test_epoch_end(self):
        labels = torch.cat(self.test_metrics["test_labels"], dim=0) # (N)
        logits = torch.cat(self.test_metrics["test_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_accuracy = torch.stack(self.test_metrics["test_acc"]).mean()

        self.log("avg_test_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("avg_test_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x1_test_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x2_test_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

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