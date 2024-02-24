import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels


from torch.optim.lr_scheduler import StepLR
from transformers import BertForTokenClassification

class BertClassifier(nn.Module): 
    def __init__(self, num_classes): 
        super(BertClassifier, self).__init__()
        
        # freeze bert backbone
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        for param in self.model.parameters(): 
            param.requires_grad = False

        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, inputs): 
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        cls_token = last_hidden_states[:, 0, :]
        return self.classifier(cls_token)


class FusionNet(nn.Module):
    def __init__(
            self, 
            num_classes, 
            loss_fn
            ):
        super(FusionNet, self).__init__()
        self.x1_model = tmodels.resnet50(weights="IMAGENET1K_V2")
        for param in self.x1_model.parameters(): 
            param.requires_grad = False
        self.x1_model.fc = nn.Linear(self.x1_model.fc.in_features, num_classes)
        for param in self.x1_model.fc.parameters(): 
            param.requires_grad = True
        self.x2_model = BertClassifier(num_classes)

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

        x1_logits = self.x1_model(x1_data)
        x2_logits = self.x2_model(x2_data)
        x1_probs = self.softmax(x1_logits)
        x2_probs = self.softmax(x2_logits)

        avg_probs = (x1_probs + x2_probs) / 2

        avg_logprobs = torch.log(avg_probs + self.epsilon)
        x1_logprobs = torch.log(x1_probs + self.epsilon)
        x2_logprobs = torch.log(x2_probs + self.epsilon)

        loss = self.loss_fn(avg_logprobs, label)

        return (x1_logprobs, x2_logprobs, avg_logprobs, loss)

class MultimodalFoodModel(pl.LightningModule): 

    def __init__(self, args): 
        """Initialize MultimodalFoodModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """


        super(MultimodalFoodModel, self).__init__()

        self.args = args
        self.model = self._build_model()

        self.val_metrics = {
            "val_loss": [], 
            "val_acc": [],
            "val_logprobs": [],
            "val_labels": [],   
        }

        self.test_metrics = {
            "test_loss": [], 
            "test_acc": [], 
            "test_logprobs": [],
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
        _, _, avg_logprobs, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logprobs, dim=1) == label).float())

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
        x1_logprobs, x2_logprobs, avg_logprobs, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logprobs, dim=1) == label).float())

        # Log loss and accuracy
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_metrics["val_logprobs"].append(torch.stack((x1_logprobs, x2_logprobs), dim=1))
        self.val_metrics["val_labels"].append(label)
        self.val_metrics["val_loss"].append(loss)
        self.val_metrics["val_acc"].append(joint_acc)

        # Return the loss
        return loss

    def on_validation_epoch_end(self) -> None:
        """ Called at the end of the validation epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.val_metrics["val_labels"], dim=0) # (N)
        logprobs = torch.cat(self.val_metrics["val_logprobs"], dim=0) # (N, M, C)
        m_out = torch.mean(logprobs, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logprobs = logprobs + offset

        x1_logprobs = corrected_logprobs[:, 0, :]
        x2_logprobs = corrected_logprobs[:, 1, :]
        
        x1_acc = torch.mean((torch.argmax(x1_logprobs, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logprobs, dim=1) == labels).float())  
        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x1_val_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x2_val_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_logprobs"].clear()
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
        x1_logprobs, x2_logprobs, avg_logprobs, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logprobs, dim=1) == label).float())

        # Log loss and accuracy
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_logprobs"].append(torch.stack((x1_logprobs, x2_logprobs), dim=1))
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
        logprobs = torch.cat(self.test_metrics["test_logprobs"], dim=0) # (N, M, C)
        m_out = torch.mean(logprobs, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logprobs = logprobs + offset

        x1_logprobs = corrected_logprobs[:, 0, :]
        x2_logprobs = corrected_logprobs[:, 1, :]
        
        x1_acc = torch.mean((torch.argmax(x1_logprobs, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logprobs, dim=1) == labels).float())  
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_acc = torch.stack(self.test_metrics["test_acc"]).mean()

        self.log("test_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x1_test_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("x2_test_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_logprobs"].clear()
        self.test_metrics["test_labels"].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=500, gamma=0.75),
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
            
        return optimizer

    def _build_model(self):
        return FusionNet(
            num_classes=self.args.num_classes, 
            loss_fn=nn.NLLLoss()
        )