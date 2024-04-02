import torch 
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision import models as tmodels

from cremad.backbone import resnet18
from functools import partial
from torch.optim.lr_scheduler import StepLR

from utils.BaseModel import JointLogitsBaseModel
from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora, LoRAParametrization

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

class MultimodalCremadModel(JointLogitsBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalCremadModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalCremadModel, self).__init__(args)
        path = "/home/haoli/Documents/multimodal-clinical/data/cremad/_ckpts/cremad_cls6_ensemble_optimal_double/distinctive-snowball-399_best.ckpt"
        state_dict = torch.load(path)["state_dict"]
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        rank = 4
        self.lora_config = {
            nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=2*rank)
            }, 
            nn.Conv2d: {
                "weight": partial(LoRAParametrization.from_conv2d, rank=rank, lora_alpha=2*rank)
            },
        }
        for name, param in self.model.x2_model.named_parameters():
            if not name_is_lora(name):
                param.requires_grad = False
        for name, param in self.model.x1_model.named_parameters():
            if not name_is_lora(name):
                param.requires_grad = False
        add_lora(self.model.x1_model, self.lora_config)
        add_lora(self.model.x2_model, self.lora_config)


    def configure_optimizers(self):
        parameters = [
            {"params": list(get_lora_params(self.model))},
            {"params": self.model.x1_classifier.parameters()},
            {"params": self.model.x2_classifier.parameters()}
        ]
        optimizer = torch.optim.SGD(parameters, lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
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