from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch, torch.nn.functional as F

# -------- Model: ResNet18 trunk + masked global average pooling + regressor --------
class CountNetMasked(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)

        self.features = nn.Sequential(*list(backbone.children())[:-2])  # conv -> layer4
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Softplus()
        )

    @staticmethod
    def masked_gap(feat, mask):
        # feat: [B,C,Hf,Wf], mask: [B,1,H,W] -> downsample mask to feat map, then weighted avg
        mask_f = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")  # [B,1,Hf,Wf]
        feat_sum = (feat * mask_f).sum(dim=(2,3))                           # [B,C]
        denom = mask_f.sum(dim=(2,3)).clamp_min(1e-6)                       # [B,1]
        return feat_sum / denom                                             # [B,C]

    def forward(self, x, mask):
        f = self.features(x)
        p = self.masked_gap(f, mask)
        return self.head(p)