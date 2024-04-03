import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from cremad.backbone import resnet18

from torch.optim.lr_scheduler import StepLR
from utils.BaseModel import EnsembleBaseModel
from existing_algos.QMF import QMF

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
        self.num_modality = 2
        self.qmf = QMF(self.num_modality, self.args.num_samples) # TODO: figure out where I created args.num_samples

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

        lam = torch.rand(1).item()
        a_mixed = lam * a + (1 - lam) * v  
        v_mixed = lam * v + (1 - lam) * a
        a = a_mixed
        v = v_mixed

        x1_logits = self.x1_classifier(a)
        x2_logits = self.x2_classifier(v)

        x1_loss = self.loss_fn(x1_logits, label) * 3.0 
        x2_loss = self.loss_fn(x2_logits, label) * 3.0 

        return (x1_logits, x2_logits, x1_loss, x2_loss)

class MultimodalCremadModel(EnsembleBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalCremadModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """
        super(MultimodalCremadModel, self).__init__(args)

    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=70, gamma=0.1),
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