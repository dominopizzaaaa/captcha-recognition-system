import torch, torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from kmeans import kmeans, getSegmentedImages
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from train_counter import collate_varsize
from modeling_letter import LetterCNN
from modeling_counter import CountNetMasked

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 

# ===== Image utils =====
def pad_to_height(img: Image.Image, target_h: int, fill=0) -> Image.Image:
    """Pad (not resize) to the target height by adding equal top/bottom padding."""
    w, h = img.size
    if h == target_h:
        return img
    if h < target_h:
        pad_total = target_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        new_img = Image.new(img.mode, (w, target_h), color=fill)
        new_img.paste(img, (0, pad_top))
        return new_img
    else:
        # h > target_h: by default, center-crop to target height (no stretching)
        top = (h - target_h) // 2
        return img.crop((0, top, w, top + target_h))

class CaptchaSolver(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.letterCNN = LetterCNN(n_classes, 1)
        self.counterNN = CountNetMasked(False)
        self.max_w = 512
        self.max_h = 80

    def _maybe_resize(self, img):
        w, h = img.size
        # If height exceeds max_h (shouldn't, but safe-guard), scale down keeping aspect
        if self.max_h and h > self.max_h:
            scale = self.max_h / h
            w = max(8, int(round(w * scale)))
            h = self.max_h
            img = img.resize((w, h), Image.BILINEAR)
        # Clamp very long widths to keep memory in check (optional)
        if self.max_w and w > self.max_w:
            scale = self.max_w / w
            w2 = int(round(w * scale)); h2 = max(8, int(round(h * scale)))
            img = img.resize((w2, h2), Image.BILINEAR)
        return img

    def forward(self, img):

        x = img
        x = self._maybe_resize(x)
        x = transforms.ToTensor()(x)
        x = transforms.Normalize([0.5] * 3, [0.5] * 3)(x)
        mask = torch.ones(1, x.shape[1], x.shape[2], dtype=torch.float32).to(DEVICE)
        # x = x.unsqueeze(0).to(DEVICE)

        # bogus = torch.randn(3, 80, 512).to(DEVICE)
        # print(x.shape)
        # x = collate_varsize([x, bogus])
        xpad = torch.zeros(1, 3, 80, 512)
        mpad = torch.zeros(1, 1, 80, 512)
        c, h, w = x.shape
        xpad[0, :, :h, :w] = x
        mpad[0, :, :h,  :w] = mask
        
        x = xpad.to(DEVICE)
        mask = mpad.to(DEVICE)
        # x = torch.cat([x, bogus])
        # bogus[0] = x

        letterCount = self.counterNN(x, mask)

        # print(letterCount)
        letterCount = np.rint(letterCount.detach().cpu().numpy().ravel()).astype(int) + 1
        letterCount = letterCount[0]

        x = img
        x = pad_to_height(x, 80, fill=255)
        x = np.array(x)

        segmentedImages = getSegmentedImages(x, letterCount)
        # for i, im in enumerate(segmentedImages):
        #     cv2.imwrite(f"{i}.png", np.array(im))
        segmentedImages = torch.tensor(segmentedImages).unsqueeze(3)
        # segmentedImages = segmentedImages.unsqueeze(1).to(dtype=torch.float32).to(DEVICE)
        # print(segmentedImages.shape)
        # segmentedImages = TF.to_tensor(segmentedImages)
        segmentedImages = segmentedImages.numpy()
        # segmentedImages = TF.to_tensor(segmentedImages)
        # print(segmentedImages.shape)
        im_t = []
        for i in range(len(segmentedImages)):
            im_t.append(TF.normalize(TF.to_tensor(segmentedImages[i]).unsqueeze(0), mean=[0.5], std=[0.5]))

        im_t = torch.cat(im_t, dim=0).to(DEVICE)
        # im_t = TF.normalize(im_t, mean=[0.5], std=[0.5]).to(DEVICE)

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
    logits = model(img)
    # return logits
    idx = logits.argmax(1)
    return "".join(np.array(letter_ckpt["classes"])[idx.cpu().numpy()])

csv_path = "./data/test/labels.csv"

samples = pd.read_csv(csv_path)

true = 0
false = 0
length_correct = 0
length_incorrect = 0
for row in samples["path"]:
    # img = cv2.imread(f"./data/test/{row}")
    img = Image.open(f"./data/test/{row}").convert("RGB")
    predicted = predict_image(img)
    truth = row.split('-0')[0]
    print(predicted, truth)
    if predicted == truth:
        true += 1
        length_correct += 1
    else:
        false += 1
        if len(predicted) == len(truth):
            length_correct += 1
        else:
            length_incorrect += 1

print(true, false, length_correct, length_incorrect)

# img = cv2.imread("./data/test/rv3io-0.png")
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(predict_image(img))

# img = Image.open(f"./data/test/ve47-0.png").convert("RGB")
# print(predict_image(img).astype(int))