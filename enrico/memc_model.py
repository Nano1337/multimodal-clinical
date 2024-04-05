from gc import freeze
import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

from utils.BaseModel import JointLogitsBaseModel

from torch.optim.lr_scheduler import StepLR

from torch.nn import TransformerEncoderLayer
from cuml.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class ResNet18Slim(nn.Module):
    """Extends ResNet18 with a separate patch embedding layer.
    
    Slimmer version of ResNet18 model with a patch embedding layer.
    """
    
    def __init__(self, hiddim, patch_size=2, pretrained=True, freeze_features=True):
        """Initialize ResNet18Slim Object.

        Args:
            hiddim (int): Hidden dimension size
            patch_size (int): Size of each patch (default: 16)
            pretrained (bool, optional): Whether to instantiate ResNet18 from Pretrained. Defaults to True.
            freeze_features (bool, optional): Whether to keep ResNet18 features frozen. Defaults to True.
        """
        super(ResNet18Slim, self).__init__()
        self.hiddim = hiddim
        self.model = tmodels.resnet18(pretrained=pretrained)
        
        # Remove the last fully connected layer and adaptive average pooling layer
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        
        # Calculate the output size of the ResNet18 feature extractor
        self.feature_size = self.model[-1][-1].bn2.num_features
        
        # Calculate the number of patches
        self.num_patches = (256 // patch_size) * (128 // patch_size) 
        
        # Create the patch embedding layer
        self.patch_embedding = nn.Conv2d(self.feature_size, hiddim, kernel_size=patch_size, stride=patch_size)
        
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Apply ResNet18Slim to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Patch tokens
        """
        features = self.model(x)
        patch_tokens = self.patch_embedding(features)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        return patch_tokens

class FusionNet(nn.Module):
    def __init__(
            self, 
            num_classes, 
            loss_fn,
            hiddim=512,
            num_layers=3,
            nhead=8,
            ):
        super(FusionNet, self).__init__()
        self.x1_model = ResNet18Slim(hiddim, freeze_features=False)
        self.x2_model = ResNet18Slim(hiddim, freeze_features=False)
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        
        # Create transformer encoder layers
        encoder_layer = TransformerEncoderLayer(d_model=hiddim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.num_random = 4

        # Adjust the final classifier to accept the flattened centroids
        self.classifier = nn.Linear(hiddim * self.num_random, num_classes)

    def forward(self, x1_data, x2_data, label):
        """ Forward pass for the FusionNet model. Fuses at token level using a transformer encoder.
        
        Args:
            x1_data (torch.Tensor): Input data for modality 1
            x2_data (torch.Tensor): Input data for modality 2
            label (torch.Tensor): Ground truth label

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the logits for modality 1, modality 2, fused logits, and loss
        """
        x1_tokens = self.x1_model(x1_data)
        x2_tokens = self.x2_model(x2_data)

        # Concatenate tokens from both modalities
        fused_tokens = torch.cat((x1_tokens, x2_tokens), dim=1)
        
        # Apply transformer encoder
        fused_tokens = self.transformer_encoder(fused_tokens)

        # Initialize centroids indices list
        centroids_indices = []
        
        # Run KMeans individually for each sample to find centroid indices
        for sample_idx, sample in enumerate(fused_tokens):
            sample = sample.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=self.num_random, n_init=1, max_iter=5, random_state=42)
            kmeans.fit(sample)
            
            # Find the indices of the closest points to the centroids
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sample)

            # Append indices of centroids for the current sample
            centroids_indices.append(closest)

        # Use indices to gather original tensor centroids
        centroids = torch.stack([fused_tokens[sample_idx, idx] for sample_idx, idx in enumerate(centroids_indices)])
    
        # Flatten centroids to pass through the linear classifier
        centroids_flattened = centroids.flatten(start_dim=1)
        
        # Final classification
        fused_logits = self.classifier(centroids_flattened)
        
        loss = self.loss_fn(fused_logits, label)

        return (None, None, fused_logits, loss)

class MultimodalEnricoModel(JointLogitsBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalEnricoModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalEnricoModel, self).__init__(args)

        self.args = args
        self.model = self._build_model()
        
        # ckpt_path = "/home/haoli/Documents/multimodal-clinical/data/enrico/_ckpts/enrico_cls20_ensemble_lr=0.006/ferengi-directive-458_best.ckpt"
        # state_dict = torch.load(ckpt_path)["state_dict"]
        # for key in list(state_dict.keys()):
        #     if "model." in key:
        #         state_dict[key.replace("model.", "", 1)] = state_dict.pop(key)
        # self.model.load_state_dict(state_dict, strict=False)
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