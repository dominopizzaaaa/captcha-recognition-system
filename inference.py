"""
inference_yolo.py — End-to-End CAPTCHA recognition using trained YOLOv8 model
Loads your best.pt detector and converts detections to predicted text.
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# ========= CONFIG =========
MODEL_PATH = "runs/detect/train5/weights/best.pt"   # path to your trained YOLO weights
TEST_DIR = "./data/test"                            # folder with test captchas
DEVICE = "cpu"                                     # or "cpu"
# ==========================

# ====== LOAD MODEL ========
print(f"Loading YOLO model from {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
model.to(DEVICE)

# ====== INFERENCE FUNCTION ======
def predict_captcha(img_path):
    """
    Runs YOLO on one image and returns the predicted text string.
    """
    results = model.predict(img_path, conf=0.25, verbose=False)
    boxes = results[0].boxes
    names = model.names

    if boxes is None or len(boxes) == 0:
        return ""  # no detection

    # Extract x-centers and class indices
    detections = []
    for b in boxes:
        x_center = b.xywh[0][0].item()
        cls_id = int(b.cls[0].item())
        letter = names[cls_id]
        detections.append((x_center, letter))

    # Sort left-to-right and join
    detections.sort(key=lambda x: x[0])
    predicted_text = "".join([ch for _, ch in detections])
    return predicted_text


# ====== EVALUATION ON TEST SET ======
import pandas as pd

csv_path = os.path.join(TEST_DIR, "labels.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Ensure your test labels.csv exists.")

samples = pd.read_csv(csv_path)

true = 0
false = 0
char_correct = 0
char_total = 0

for rel_path in samples["path"]:
    img_path = os.path.join(TEST_DIR, rel_path)
    truth = os.path.basename(rel_path).split("-0")[0].upper()
    predicted = predict_captcha(img_path)
    print(f"{os.path.basename(img_path)} → Pred: {predicted} | GT: {truth}")

    if predicted == truth:
        true += 1
    else:
        false += 1

    # per-character accuracy
    minlen = min(len(predicted), len(truth))
    for i in range(minlen):
        if predicted[i] == truth[i]:
            char_correct += 1
    char_total += len(truth)

captcha_acc = true / (true + false)
char_acc = char_correct / char_total
print(f"\n✅ CAPTCHA accuracy: {captcha_acc*100:.2f}%")
print(f"✅ Character accuracy: {char_acc*100:.2f}%")
