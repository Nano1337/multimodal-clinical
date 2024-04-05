from matplotlib.pyplot import step
import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

from utils.BaseModel import EnsembleBaseModel

from torch.optim.lr_scheduler import StepLR

class ResNet18Slim(nn.Module):
    """Extends ResNet18 with a separate embedding and classifier layers.
    
    Slimmer version of ResNet18 model with separate embedding and classifier layers.
    """
    
    def __init__(self, hiddim, pretrained=True, freeze_features=True):
        """Initialize ResNet18Slim Object.

        Args:
            hiddim (int): Hidden dimension size
            pretrained (bool, optional): Whether to instantiate ResNet18 from Pretrained. Defaults to True.
            freeze_features (bool, optional): Whether to keep ResNet18 features frozen. Defaults to True.
        """
        super(ResNet18Slim, self).__init__()
        self.hiddim = hiddim
        self.model = tmodels.resnet18(pretrained=pretrained)
        
        # Remove the last fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.embedding = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, hiddim)
        
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Apply ResNet18Slim to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            tuple: (embedding, logits)
        """
        features = self.model(x)
        embedding = self.embedding(features).view(features.size(0), -1)
        logits = self.classifier(embedding)
        return embedding, logits

class FusionNet(nn.Module):
    def __init__(
            self, 
            num_classes, 
            loss_fn
            ):
        super(FusionNet, self).__init__()
        self.x1_model = ResNet18Slim(num_classes, freeze_features=False)
        self.x2_model = ResNet18Slim(num_classes, freeze_features=False)

        self.num_classes = num_classes
        self.loss_fn = loss_fn

    def forward(self, x1_data, x2_data, label):
        x1_embedding, x1_logits = self.x1_model(x1_data)
        x2_embedding, x2_logits = self.x2_model(x2_data)

        x1_loss = self.loss_fn(x1_logits, label) 
        x2_loss = self.loss_fn(x2_logits, label) 

        return (x1_logits, x2_logits, x1_loss, x2_loss)


class MultimodalEnricoModel(EnsembleBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalEnricoModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalEnricoModel, self).__init__(args)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=70, gamma=0.5),
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