import torch, torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from kmeans import kmeans, getSegmentedImages
import torchvision.transforms.functional as TF
import cv2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        print(x.device, mask.device)
        f = self.features(x)
        p = self.masked_gap(f, mask)
        return self.head(p)
 
class LetterCNN(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 1,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()

        # Load base ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # Adapt first conv for custom input channels
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
                        # Average RGB weights -> 1 channel
                        base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                    elif in_ch > 3:
                        # For >3 channels, repeat and trim (simple heuristic)
                        repeat_factor = (in_ch + 2) // 3
                        w = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :in_ch]
                        base.conv1.weight[:] = w
                    else:  # in_ch == 2, etc.
                        w = old_conv.weight[:, :in_ch]
                        base.conv1.weight[:] = w

        # Take everything except the original avgpool & fc
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        # Global pooling for arbitrary H, W
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base.fc.in_features, n_classes)

        # Optionally freeze backbone
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)          # (B, C, h', w')
        x = self.pool(x).flatten(1)   # (B, C)
        x = self.fc(x)                # (B, n_classes)
        return x   

class CaptchaSolver(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.letterCNN = LetterCNN(n_classes, 1)
        self.counterNN = CountNetMasked(False)

    def forward(self, x, mask):
        letterCount = self.counterNN(x, mask)
        x = TF.normalize(x, mean=[0.5], std=[0.5]).to(DEVICE)
        # print(letterCount)
        letterCount = np.rint(letterCount.cpu()) + 1
        letterCount = int(letterCount.item())
        segmentedImages = torch.tensor(getSegmentedImages(x, letterCount)).unsqueeze(3)
        # for i, im in enumerate(segmentedImages):
        #     cv2.imwrite(f"{i}.png", np.array(im))
        # segmentedImages = segmentedImages.unsqueeze(1).to(dtype=torch.float32).to(DEVICE)
        # print(segmentedImages.shape)
        # segmentedImages = TF.to_tensor(segmentedImages)
        segmentedImages = segmentedImages.numpy()
        # segmentedImages = TF.to_tensor(segmentedImages)
        # print(segmentedImages.shape)
        im_t = []
        for i in range(len(segmentedImages)):
            im_t.append(TF.to_tensor(segmentedImages[i]).unsqueeze(0))

        im_t = torch.cat(im_t, dim=0)
        print(im_t.shape)

        y = self.letterCNN(im_t)
    
        return y

letter_ckpt = torch.load("letter_padheight_best.pt", map_location=DEVICE)
cnt_ckpt = torch.load("letter_counter.pt", map_location=DEVICE)
model = CaptchaSolver(36)
model.letterCNN.load_state_dict(letter_ckpt["model"])
model.counterNN.load_state_dict(cnt_ckpt["model"])
model.to(DEVICE)

@torch.no_grad()
def predict_image(img):
    model.eval()
    x = TF.to_tensor(img)

    mask = torch.ones(1, 1, x.shape[1], x.shape[2], dtype=torch.float32).to(DEVICE)
    # x = TF.normalize(x, mean=[0.5], std=[0.5]).unsqueeze(0).to(DEVICE)
    x = x.unsqueeze(0).to(DEVICE)
    logits = model(x, mask)
    idx = logits.argmax(1)
    print(idx)
    return "".join(np.array(letter_ckpt["classes"])[idx.cpu().numpy()])

img = cv2.imread("./data/test/ui7l-0.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(predict_image(img))