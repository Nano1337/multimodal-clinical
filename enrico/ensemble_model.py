from matplotlib.pyplot import step
import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

from utils.BaseModel import EnsembleBaseModel

from torch.optim.lr_scheduler import StepLR

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