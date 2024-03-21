from sympy import Idx
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cremad.backbone import resnet18

from utils.BaseModel import JointLogitsBaseModel
from existing_algos.QMF import QMF


class FusionNet(nn.Module):
    def __init__(
            self, 
            args,
            loss_fn
            ):
        super(FusionNet, self).__init__()

        self.args = args
        self.num_modality = 2
        self.qmf = QMF(self.num_modality, self.args.num_samples)

        self.x1_model = resnet18(modality='audio')
        self.x1_classifier = nn.Linear(512, self.args.num_classes)
        self.x2_model = resnet18(modality='visual')
        self.x2_classifier = nn.Linear(512, self.args.num_classes)

        self.num_classes = self.args.num_classes
        self.loss_fn = loss_fn

    def forward(self, x1_data, x2_data, label, idx):
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

        out = torch.stack([x1_logits, x2_logits])
        logits_df, conf = self.qmf.df(out) # logits_df is (B, C), conf is (M, B)
        loss_uni = []
        for n in range(self.num_modality):
            loss_uni.append(self.loss_fn(out[n], label))
            self.qmf.history[n].correctness_update(idx, loss_uni[n], conf[n].squeeze())

        loss_reg = self.qmf.reg_loss(conf, idx.squeeze())
        loss_joint = self.loss_fn(logits_df, label)

        loss = loss_joint + torch.sum(torch.stack(loss_uni)) + loss_reg

        # fuse at logit level
        avg_logits = (x1_logits + x2_logits) / 2

        return (x1_logits, x2_logits, avg_logits, loss, logits_df)

class MultimodalCremadModel(JointLogitsBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalCremadModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalCremadModel, self).__init__(args)

        self.train_metrics.update({"train_df_acc": []})
        self.val_metrics.update({"val_df_acc": []})
        self.test_metrics.update({"test_df_acc": []})

    # include additional logging

    def forward(self, x1, x2, label, idx): 
        return self.model(x1, x2, label, idx)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label, idx = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, logits_df = self.model(x1, x2, label, idx)

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
        logits_df_acc = torch.mean((torch.argmax(logits_df, dim=1) == label).float())

        # Log loss and accuracy
        self.log("train_step/train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_acc", x1_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_acc", x2_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_uncal_acc", x1_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_uncal_acc", x2_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_df_acc", logits_df_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # accumulate accuracies and losses
        self.train_metrics["train_acc"].append(joint_acc)
        self.train_metrics["train_loss"].append(loss)
        self.train_metrics["train_x1_acc_uncal"].append(x1_acc_uncal.item())
        self.train_metrics["train_x2_acc_uncal"].append(x2_acc_uncal.item())
        self.train_metrics["train_x1_acc"].append(x1_acc_cal.item())
        self.train_metrics["train_x2_acc"].append(x2_acc_cal.item())
        self.train_metrics["train_df_acc"].append(logits_df_acc)

        # Return the loss
        return loss
    
    
    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        avg_loss = torch.stack(self.train_metrics["train_loss"]).mean()
        avg_acc = torch.stack(self.train_metrics["train_acc"]).mean()
        avg_df_acc = torch.mean(torch.stack(self.train_metrics["train_df_acc"]))

        self.log("train_epoch/train_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc_uncal", np.mean(np.array(self.train_metrics["train_x1_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc_uncal", np.mean(np.array(self.train_metrics["train_x2_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc", np.mean(np.array(self.train_metrics["train_x1_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc", np.mean(np.array(self.train_metrics["train_x2_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_df_acc", avg_df_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.train_metrics["train_loss"].clear()
        self.train_metrics["train_acc"].clear()
        self.train_metrics["train_x1_acc_uncal"].clear()
        self.train_metrics["train_x2_acc_uncal"].clear()
        self.train_metrics["train_x1_acc"].clear()
        self.train_metrics["train_x2_acc"].clear()
        self.train_metrics["train_df_acc"].clear()

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label, idx = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, logits_df = self.model(x1, x2, label, idx)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        logits_df_acc = torch.mean((torch.argmax(logits_df, dim=1) == label).float())

        # Log loss and accuracy
        self.log("val_step/val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/logits_df_acc", logits_df_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.val_metrics["val_labels"].append(label)
        self.val_metrics["val_loss"].append(loss)
        self.val_metrics["val_acc"].append(joint_acc)
        self.val_metrics["val_df_acc"].append(logits_df_acc)
 
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
        avg_df_acc = torch.mean(torch.stack(self.val_metrics["val_df_acc"]))

        self.log("val_epoch/val_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_df_acc", avg_df_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_logits"].clear()
        self.val_metrics["val_labels"].clear()
        self.val_metrics["val_df_acc"].clear()


    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label, idx = batch 

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, logits_df = self.model(x1, x2, label, idx)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        logits_df_acc = torch.mean((torch.argmax(logits_df, dim=1) == label).float())

        # Log loss and accuracy
        self.log("test_step/test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/logits_df_acc", logits_df_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.test_metrics["test_labels"].append(label)
        self.test_metrics["test_loss"].append(loss)
        self.test_metrics["test_acc"].append(joint_acc)
        self.test_metrics["test_df_acc"].append(logits_df_acc)

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
        avg_df_acc = torch.mean(torch.stack(self.test_metrics["test_df_acc"]))

        self.log("test_epoch/test_avg_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True) 
        self.log("test_epoch/test_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_df_acc", avg_df_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_logits"].clear()
        self.test_metrics["test_labels"].clear()
        self.test_metrics["test_df_acc"].clear()

    def _build_model(self):
        return FusionNet(
            args=self.args,
            loss_fn=nn.CrossEntropyLoss()
        )