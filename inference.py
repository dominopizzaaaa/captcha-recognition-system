"""
inference_yolo_cls_pipeline.py
----------------------------------
Hybrid CAPTCHA recognizer:
1. Predict number of letters using CountNetMasked
2. K-means segmentation by color+position
3. Crop 80×80 per letter
4. Classify each crop using trained YOLOv8 classification model
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from torchvision import transforms
from ultralytics import YOLO

# === local imports from your repo ===
from kmeans import getSegmentedImages
from modeling_counter import CountNetMasked

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= LOAD MODELS =========
print("Loading YOLO classification model and CountNetMasked...")

# load your trained YOLOv8 classification weights
yolo_model = YOLO("runs/classify/train/weights/best.pt").to(DEVICE)

# load your trained counter model (same as before)
cnt_ckpt = torch.load("letter_counter.pt", map_location=DEVICE)
counterNN = CountNetMasked(False)
counterNN.load_state_dict(cnt_ckpt["model"])
counterNN.to(DEVICE)
counterNN.eval()

# transform for resizing and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def pad_to_height(img, target_h: int, fill=0):
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
        top = (h - target_h) // 2
        return img.crop((0, top, w, top + target_h))


# ========= MAIN PREDICTION =========
@torch.no_grad()
def predict_image(img: Image.Image):
    """
    1. Use counterNN to get letter count
    2. Segment via KMeans
    3. Use YOLO classifier on each segment
    4. Return predicted text string
    """

    # ---- 1. Predict letter count ----
    x = transform(img).unsqueeze(0).to(DEVICE)
    mask = torch.ones_like(x[:, :1, :, :], dtype=torch.float32).to(DEVICE)

    letterCount = counterNN(x, mask)
    letterCount = np.rint(letterCount.detach().cpu().numpy().ravel()).astype(int) + 1
    letterCount = max(1, letterCount[0])

    # ---- 2. Segment image into characters ----
    x_np = np.array(pad_to_height(img, 80, fill=255))
    segmentedImages = getSegmentedImages(x_np, letterCount)

    predicted_chars = []

    for seg in segmentedImages:
        seg = Image.fromarray(seg).convert("RGB")
        seg = seg.resize((64, 64))  # match YOLO cls training size
        pred = yolo_model.predict(seg, imgsz=64, verbose=False, device=DEVICE)
        cls_id = pred[0].probs.top1
        char = yolo_model.names[cls_id]
        predicted_chars.append(char.upper())

    return "".join(predicted_chars)

# ========= EVALUATION =========
csv_path = "./data/test/labels.csv"
samples = pd.read_csv(csv_path)

true = 0
false = 0
char_correct = 0
char_total = 0

for rel_path in samples["path"]:
    img = Image.open(f"./data/test/{rel_path}").convert("RGB")
    truth = rel_path.split("-0")[0].upper()
    predicted = predict_image(img)
    print(f"{rel_path} → Pred: {predicted} | GT: {truth}")

    if predicted == truth:
        true += 1
    else:
        false += 1

    # per-character match
    for a, b in zip(predicted, truth):
        if a == b:
            char_correct += 1
    char_total += len(truth)

captcha_acc = true / (true + false)
char_acc = char_correct / char_total
print(f"\n✅ CAPTCHA accuracy: {captcha_acc*100:.2f}%")
print(f"✅ Character accuracy: {char_acc*100:.2f}%")
