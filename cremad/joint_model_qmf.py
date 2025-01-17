from sympy import Idx
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cremad.backbone import resnet18

from utils.BaseModel import QMFBaseModel
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

class MultimodalCremadModel(QMFBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalCremadModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalCremadModel, self).__init__(args)

    def _build_model(self):
        return FusionNet(
            args=self.args,
            loss_fn=nn.CrossEntropyLoss()
        )