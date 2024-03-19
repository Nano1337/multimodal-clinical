import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from cremad.backbone import resnet18

from utils.BaseModel import JointProbLogitsBaseModel

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

        self.softmax = nn.Softmax(dim=1)
        self.epsilon = 1e-9

    def forward(self, x1_data, x2_data, label, istrain=True):
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

        if istrain: 
            x1_probs = self.softmax(x1_logits)
            x2_probs = self.softmax(x2_logits)

            avg_probs = (x1_probs + x2_probs) / 2

            avg_logprobs = torch.log(avg_probs + self.epsilon)
            x1_logprobs = torch.log(x1_probs + self.epsilon)
            x2_logprobs = torch.log(x2_probs + self.epsilon)

            loss = self.loss_fn(avg_logprobs, label)

            return (x1_logprobs, x2_logprobs, avg_logprobs, loss)
            
        else: 

            # fuse at logit level
            avg_logits = (x1_logits + x2_logits) / 2

            loss = self.loss_fn(avg_logits, label)

            return (x1_logits, x2_logits, avg_logits, loss)
            

class MultimodalCremadModel(JointProbLogitsBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalCremadModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """
        super(MultimodalCremadModel, self).__init__(args)

    def _build_model(self):
        return FusionNet(
            num_classes=self.args.num_classes, 
            loss_fn=nn.CrossEntropyLoss()
        )