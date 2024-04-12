import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

from cremad.backbone import resnet18

from utils.BaseModel import JointLogitsBaseModel
from torch.optim.lr_scheduler import StepLR
# from torch_kmeans import SoftKMeans
# from cuml.cluster import KMeans
from torch_kmeans import SoftKMeans
from sklearn.metrics import pairwise_distances_argmin_min

class tokenized_resnet18(nn.Module):
    def __init__(self, modality='audio', patch_size=3):
        super(tokenized_resnet18, self).__init__()
        self.model = resnet18(modality=modality)
        self.modality = modality
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(512, 512, kernel_size=patch_size, stride=patch_size)
    


class FusionNet(nn.Module):
    def __init__(
            self, 
            args,
            num_classes, 
            loss_fn
            ):
        super(FusionNet, self).__init__()
        self.args = args
        self.hiddim = 512
        self.x1_model = resnet18(modality='audio') # 33 tokens per modality
        self.x1_classifier = nn.Linear(self.hiddim, num_classes)
        self.x2_model = resnet18(modality='visual')
        self.x2_classifier = nn.Linear(self.hiddim, num_classes)
        self.patch_size = 3
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.patch_embedding_a = nn.Conv2d(512, 512, kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_embedding_v = nn.Conv2d(512, 512, kernel_size=self.patch_size, stride=self.patch_size)
            
        self.k = 5 # number of centroids
        self.kmeans = SoftKMeans(
            n_clusters=self.k, 
            n_init=1, 
            max_iter=30, 
            random_state=self.args.seed, 
            verbose=False, 
            )
        
        self.classifier = nn.Linear(self.hiddim * self.k, self.hiddim)

    def pad_input(self, out):
        pad_height = (self.patch_size - out.shape[2] % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - out.shape[3] % self.patch_size) % self.patch_size
        return F.pad(out, (0, pad_width, 0, pad_height))
    
    def forward(self, x1_data, x2_data, label):
        """ Forward pass for the FusionNet model. Fuses at logit level.
        
        Args:
            x1_data (torch.Tensor): Input data for modality 1
            x2_data (torch.Tensor): Input data for modality 2
            label (torch.Tensor): Ground truth label

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the logits for modality 1, modality 2, average logits, and loss
        """

        # put through unimodal models 
        # TODO: refactor this to be more modular
        a = self.x1_model(x1_data)
        v = self.x2_model(x2_data)
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = v.view(B, C, 3, -1)
        a = self.pad_input(a)
        v = self.pad_input(v)
        a = self.patch_embedding_a(a)
        v = self.patch_embedding_v(v)
        a = a.flatten(2).transpose(1, 2) # 33 tokens
        v = v.flatten(2).transpose(1, 2) # 17 tokens

        # get centroids
        fused_tokens = torch.cat((a, v), dim=1)
        
        centroids = self.kmeans(fused_tokens).centers
        centroids_flattened = centroids.flatten(start_dim=1)

        # classify
        fused_logits = self.classifier(centroids_flattened)
        
        loss = self.loss_fn(fused_logits, label)

        return (None, None, fused_logits, loss)

class MultimodalCremadModel(JointLogitsBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalCremadModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalCremadModel, self).__init__(args)

        self.args = args
        self.model = self._build_model()

        self.num_modality = 2

        self.train_metrics = {
            "train_loss": [],
            "train_acc": [],
            "train_logits": [],
        }

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

        path = "/home/haoli/Documents/multimodal-clinical/data/cremad/_ckpts/cremad_cls6_ensemble_optimal_double/distinctive-snowball-399_best.ckpt"
        state_dict = torch.load(path)["state_dict"]
        for key in list(state_dict.keys()):
            if "model." in key:
                state_dict[key.replace("model.", "", 1)] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)

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
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("train_step/train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # accumulate accuracies and losses
        self.train_metrics["train_acc"].append(joint_acc)
        self.train_metrics["train_loss"].append(loss)

        # Return the loss
        return loss
    
    
    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        avg_loss = torch.stack(self.train_metrics["train_loss"]).mean()
        avg_acc = torch.stack(self.train_metrics["train_acc"]).mean()

        self.log("train_epoch/train_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.train_metrics["train_loss"].clear()
        self.train_metrics["train_acc"].clear()


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
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("val_step/val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_labels"].append(label)
        self.val_metrics["val_loss"].append(loss)
        self.val_metrics["val_acc"].append(joint_acc)
 
        return loss

    def on_validation_epoch_end(self) -> None:
        """ Called at the end of the validation epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.val_metrics["val_labels"], dim=0) # (N)

        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()

        self.log("val_epoch/val_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_labels"].clear()


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
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("test_step/test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

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

        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_accuracy = torch.stack(self.test_metrics["test_acc"]).mean()
        
        self.log("test_epoch/test_avg_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True) 

        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_labels"].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=50, gamma=0.5),
                'interval': 'epoch',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
            
        return optimizer
    
    def _build_model(self):
        return FusionNet(
            args=self.args,
            num_classes=self.args.num_classes, 
            loss_fn=nn.CrossEntropyLoss()
        )