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
from modeling_counter import CountNetMasked
from ultralytics import YOLO  # <-- added YOLO import

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
        # h > target_h: center-crop to target height
        top = (h - target_h) // 2
        return img.crop((0, top, w, top + target_h))

# ===== Model wrapper =====
class CaptchaSolver(nn.Module):
    def __init__(self, yolo_path):
        super().__init__()
        self.counterNN = CountNetMasked(False)
        self.max_w = 512
        self.max_h = 80
        # Load YOLOv8 classification model
        self.yolo_model = YOLO(yolo_path).to(DEVICE)

    def _maybe_resize(self, img):
        w, h = img.size
        if self.max_h and h > self.max_h:
            scale = self.max_h / h
            w = max(8, int(round(w * scale)))
            h = self.max_h
            img = img.resize((w, h), Image.BILINEAR)
        if self.max_w and w > self.max_w:
            scale = self.max_w / w
            w2 = int(round(w * scale)); h2 = max(8, int(round(h * scale)))
            img = img.resize((w2, h2), Image.BILINEAR)
        return img

    def forward(self, img):
        # ===== Predict letter count =====
        x = img
        x = self._maybe_resize(x)
        x = transforms.ToTensor()(x)
        x = transforms.Normalize([0.5] * 3, [0.5] * 3)(x)
        mask = torch.ones(1, x.shape[1], x.shape[2], dtype=torch.float32).to(DEVICE)

        xpad = torch.zeros(1, 3, 80, 512)
        mpad = torch.zeros(1, 1, 80, 512)
        c, h, w = x.shape
        xpad[0, :, :h, :w] = x
        mpad[0, :, :h, :w] = mask
        
        x = xpad.to(DEVICE)
        mask = mpad.to(DEVICE)

        letterCount = self.counterNN(x, mask)
        letterCount = np.rint(letterCount.detach().cpu().numpy().ravel()).astype(int) + 1
        letterCount = letterCount[0]

        # ===== Segment image into characters =====
        x = img
        x = pad_to_height(x, 80, fill=255)
        x = np.array(x)
        segmentedImages = getSegmentedImages(x, letterCount)

        predicted_chars = []
        for i, seg in enumerate(segmentedImages):
            seg = Image.fromarray(seg).convert("RGB")
            seg = seg.resize((80, 80))
            result = self.yolo_model.predict(seg, imgsz=80, verbose=False, device=DEVICE)
            cls_id = result[0].probs.top1
            char = self.yolo_model.names[cls_id]
            predicted_chars.append(char.upper())

        return predicted_chars


# ===== Load models =====
cnt_ckpt = torch.load("letter_counter.pt", map_location=DEVICE)
model = CaptchaSolver("runs/detect/train5/weights/best.pt")
model.counterNN.load_state_dict(cnt_ckpt["model"])
model.to(DEVICE)
model.eval()

# ===== Prediction function =====
@torch.no_grad()
def predict_image(img):
    chars = model(img)
    return "".join(chars)

# ===== Evaluation loop =====
csv_path = "./data/test/labels.csv"
samples = pd.read_csv(csv_path)

true = 0
false = 0
length_correct = 0
length_incorrect = 0

for row in samples["path"]:
    img = Image.open(f"./data/test/{row}").convert("RGB")
    predicted = predict_image(img)
    truth = row.split('-0')[0].upper()
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
