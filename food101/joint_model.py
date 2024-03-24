import torch 
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from utils.BaseModel import JointLogitsBaseModel

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


    def forward(self, x1_data, x2_data, label):
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

        return (x1_logits, x2_logits, avg_logits, loss)

class MultimodalFoodModel(JointLogitsBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalFoodModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """


        super(MultimodalFoodModel, self).__init__(args)

    def _build_model(self):
        return FusionNet(
            num_classes=self.args.num_classes, 
            loss_fn=nn.CrossEntropyLoss()
        )