import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from utils.BaseModel import JointLogitsBaseModel
from torch.optim.lr_scheduler import StepLR

class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=101):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class FusionNet(nn.Module):
    def __init__(
            self, 
            num_classes, 
            loss_fn
            ):
        super(FusionNet, self).__init__()

        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        for param in self.model.parameters():
            param.requires_grad = True
        self.x1_model = MLP(input_dim=768, hidden_dim=512, num_classes=num_classes)
        self.x2_model = MLP(input_dim=768, hidden_dim=512, num_classes=num_classes)

        self.alpha = 0.001

    def entropy(self, logits):
        p = torch.nn.functional.softmax(logits, dim=1)
        log_p = torch.nn.functional.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1)
        return entropy
    
    def forward(self, x1_data, x2_data, label, weights, istrain=False):
        """ Forward pass for the FusionNet model. Fuses at logit level.
        
        Args:
            x1_data (torch.Tensor): Input data for modality 1
            x2_data (torch.Tensor): Input data for modality 2
            label (torch.Tensor): Ground truth label

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the logits for modality 1, modality 2, average logits, and loss
        """

        output = self.model(x1_data, x2_data)
    
        x1_logits = self.x1_model(output['text_embeds'])
        x2_logits = self.x2_model(output['image_embeds'])

        # fuse at logit level
        avg_logits = (x1_logits + x2_logits) / 2

        loss = self.loss_fn(avg_logits, label)

        # sample-weighted loss
        if istrain: 
            x1_entropy = self.entropy(x1_logits)
            x2_entropy = self.entropy(x2_logits)
            x1_loss = x1_entropy @ weights[:, 0]
            x2_loss = x2_entropy @ weights[:, 1]
            loss += self.alpha * (x1_loss + x2_loss)

            return (x1_logits, x2_logits, avg_logits, loss, x1_loss, x2_loss)
        else:
            return (x1_logits, x2_logits, avg_logits, loss)

class MultimodalFoodModel(JointLogitsBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalFoodModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalFoodModel, self).__init__(args)

        self.train_metrics.update({"train_x1_loss": [], "train_x2_loss": []})

    def forward(self, x1, x2, label, weights, istrain=False): 
        return self.model(x1, x2, label, weights, istrain)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label, weights = batch

        weights = weights.to(self.dtype)

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, x1_loss, x2_loss = self.model(x1, x2, label, weights, istrain=True)

        # Calculate uncalibrated accuracy for x1 and x2
        x1_acc_uncal = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())

        # calibrate unimodal logits
        logits_stack = torch.stack([x1_logits, x2_logits])
        self.ema_offset.update(torch.mean(logits_stack, dim=1))
        x1_logits_cal = x1_logits + self.ema_offset.offset[0].to(x1_logits.get_device())
        x2_logits_cal = x2_logits + self.ema_offset.offset[1].to(x2_logits.get_device())

        # Calculate calibrated accuracy for x1 and x2
        x1_acc_cal = torch.mean((torch.argmax(x1_logits_cal, dim=1) == label).float())
        x2_acc_cal = torch.mean((torch.argmax(x2_logits_cal, dim=1) == label).float())

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("train_step/train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_loss", x1_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_loss", x2_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_acc", x1_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_acc", x2_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_uncal_acc", x1_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_uncal_acc", x2_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # accumulate accuracies and losses
        self.train_metrics["train_acc"].append(joint_acc)
        self.train_metrics["train_loss"].append(loss)
        self.train_metrics["train_x1_loss"].append(x1_loss)
        self.train_metrics["train_x2_loss"].append(x2_loss)
        self.train_metrics["train_x1_acc_uncal"].append(x1_acc_uncal.item())
        self.train_metrics["train_x2_acc_uncal"].append(x2_acc_uncal.item())
        self.train_metrics["train_x1_acc"].append(x1_acc_cal.item())
        self.train_metrics["train_x2_acc"].append(x2_acc_cal.item())


        # Return the loss
        return loss
    
    
    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        avg_loss = torch.stack(self.train_metrics["train_loss"]).mean()
        avg_acc = torch.stack(self.train_metrics["train_acc"]).mean()
        avg_x1_loss = torch.stack(self.train_metrics["train_x1_loss"]).mean()
        avg_x2_loss = torch.stack(self.train_metrics["train_x2_loss"]).mean()

        self.log("train_epoch/train_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_loss", avg_x1_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_loss", avg_x2_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc_uncal", np.mean(np.array(self.train_metrics["train_x1_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc_uncal", np.mean(np.array(self.train_metrics["train_x2_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc", np.mean(np.array(self.train_metrics["train_x1_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc", np.mean(np.array(self.train_metrics["train_x2_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.train_metrics["train_loss"].clear()
        self.train_metrics["train_x1_loss"].clear()
        self.train_metrics["train_x2_loss"].clear()
        self.train_metrics["train_acc"].clear()
        self.train_metrics["train_x1_acc_uncal"].clear()
        self.train_metrics["train_x2_acc_uncal"].clear()
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
        x1, x2, label, weights = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label, weights)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("val_step/val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
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

        x1_logits_uncal = logits[:, 0, :]
        x2_logits_uncal = logits[:, 1, :]
        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc_uncal = torch.mean((torch.argmax(x1_logits_uncal, dim=1) == labels).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits_uncal, dim=1) == labels).float())
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()

        self.log("val_epoch/val_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_logits"].clear()
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

        self.test_metrics["test_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
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

        x1_logits_uncal = logits[:, 0, :]
        x2_logits_uncal = logits[:, 1, :]
        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc_uncal = torch.mean((torch.argmax(x1_logits_uncal, dim=1) == labels).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits_uncal, dim=1) == labels).float())
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_accuracy = torch.stack(self.test_metrics["test_acc"]).mean()
        
        self.log("test_epoch/test_avg_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True) 
        self.log("test_epoch/test_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_logits"].clear()
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
            num_classes=self.args.num_classes, 
            loss_fn=nn.CrossEntropyLoss()
        )