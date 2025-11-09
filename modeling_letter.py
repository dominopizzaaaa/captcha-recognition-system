from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch

class LetterCNN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_ch: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_upto: int = 0,
        freeze_bn_stats: bool = False,
    ):
        """
        n_classes:     number of output classes
        in_ch:         input channels (1 for grayscale)
        pretrained:    load ImageNet weights for backbone
        freeze_backbone:
            if True and freeze_upto == 0 -> freeze entire backbone
        freeze_upto:
            freeze first N modules of backbone in order:
            [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
        freeze_bn_stats:
            if True, BatchNorms in frozen modules are put in eval() mode.
        """
        super().__init__()

        # ----- Load base ResNet18 -----
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # ----- Adapt first conv for custom input channels -----
        if in_ch != 3:
            old_conv = base.conv1
            base.conv1 = nn.Conv2d(
                in_ch,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                if pretrained:
                    if in_ch == 1:
                        # Average RGB -> 1 channel
                        base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                    elif in_ch > 3:
                        # Repeat & trim
                        repeat_factor = (in_ch + 2) // 3
                        w = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :in_ch]
                        base.conv1.weight[:] = w
                    else:  # e.g. in_ch == 2
                        w = old_conv.weight[:, :in_ch]
                        base.conv1.weight[:] = w

        # ----- Backbone as a sequential (no original avgpool/fc) -----
        self.features = nn.Sequential(
            base.conv1,   # 0
            base.relu,    # 2
            base.bn1,     # 1
            nn.Dropout2d(p=0.2),
            base.maxpool, # 3
            base.layer1,  # 4
            nn.Dropout2d(p=0.5),
            base.layer2,  # 5
            nn.Dropout2d(p=0.5),
            base.layer3,  # 6
            # base.layer4,  # 7
        )

        # Regularize high-level conv features
        self.dropout_feat = nn.Dropout2d(p=0.5)

        # Global pooling for arbitrary H, W
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # print()

        # feat_dim = base.fc.in_features  # 512 for ResNet18
        feat_dim = 256

        # Beefed-up classifier head
        self.head = nn.Sequential(
            # nn.Linear(feat_dim, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),

            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.25),

            nn.Linear(feat_dim, n_classes),
        )

        # ----- Freezing logic -----
        # If freeze_backbone is True and freeze_upto not specified, freeze all backbone
        if freeze_backbone and freeze_upto == 0:
            freeze_upto = len(list(self.features.children()))

        if freeze_upto > 0:
            self._freeze_backbone_upto(freeze_upto, freeze_bn_stats)

    def _freeze_backbone_upto(self, n: int, freeze_bn_stats: bool):
        """
        Freeze parameters (and optionally BN stats) in the first n modules
        of self.features.
        """
        children = list(self.features.children())
        n = max(0, min(n, len(children)))
        modules_to_freeze = children[1:n]

        for m in modules_to_freeze:
            for p in m.parameters():
                p.requires_grad = False

            if freeze_bn_stats:
                for sub in m.modules():
                    if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        sub.eval()  # stop updating running stats
                        for p in sub.parameters():
                            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)            # (B, C, h', w')
        x = self.dropout_feat(x)        # only active in train()
        x = self.pool(x).flatten(1)     # (B, feat_dim)
        x = self.head(x)                # (B, n_classes)
        return x