import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

from backbone import resnet18


from torch.optim.lr_scheduler import StepLR

def ogm_ge(model, out_1, out_2, label, alpha=0.5, modulation='OGM_GE'):
    '''
        Example usage:
        In init:
            self.automatic_optimization = False
            self.ogm_modulation = ogm_modulation (None, 'OGM_GE', 'OGM', 'noise')
            self.ogm_alpha = ogm_alpha (float between 0 and 1)
        In training_step:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            if self.ogm_modulation:
                ogm_ge(self, x1_logits, x2_logits, label, modulation=self.ogm_modulation, alpha=self.ogm_alpha)
            opt.step()

        Note: change if/continue below for different models so that only norms are skipped
    '''
    score_v = sum([F.softmax(out_1, dim=-1)[i][label[i]] for i in range(out_1.size(0))])
    score_a = sum([F.softmax(out_2, dim=-1)[i][label[i]] for i in range(out_2.size(0))])

    ratio_v = score_v / score_a
    ratio_a = 1 / ratio_v

    """
    Below is the Eq.(10) in our CVPR paper:
            1 - tanh(alpha * rho_t_u), if rho_t_u > 1
    k_t_u =
            1,                         else
    coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """

    if ratio_v > 1:
        coeff_v = 1 - F.tanh(alpha * F.relu(ratio_v, inplace=True))
        coeff_a = 1
    else:
        coeff_a = 1 - F.tanh(alpha * F.relu(ratio_a, inplace=True))
        coeff_v = 1

    def add_factor(model, coeff):
        for name, parms in model.named_parameters():
            if len(parms.grad.size()) != 4:
                # TODO: update this! Just want to exclude batch norms
                continue
            layer = str(name).split('.')[1]
            if modulation == 'OGM_GE':  # bug fixed
                parms.grad = parms.grad * coeff + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            elif modulation == 'OGM':
                parms.grad *= coeff
            elif modulation == 'noise':
                parms.grad = parms.grad + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

    add_factor(model.x1_model, coeff_v)
    add_factor(model.x2_model, coeff_a)


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

        # fuse at logit level
        avg_logits = (x1_logits + x2_logits) / 2

        loss = self.loss_fn(avg_logits, label)

        return (x1_logits, x2_logits, avg_logits, loss)

class MultimodalCremadModel(pl.LightningModule): 

    def __init__(self, args): 
        """Initialize MultimodalCremadModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """


        super(MultimodalCremadModel, self).__init__()

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
        _, _, avg_logits, loss = self.model(x1, x2, label)

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