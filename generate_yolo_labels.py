import os, csv, pathlib
import cv2
import numpy as np
from kmeans import kmeans

# Path setup
split = "train"
DATA_DIR = f"./data/{split}"
DEST_IMG_DIR = f"./data_yolo/{split}/images"
DEST_LABEL_DIR = f"./data_yolo/{split}/labels"

os.makedirs(DEST_IMG_DIR, exist_ok=True)
os.makedirs(DEST_LABEL_DIR, exist_ok=True)

# class list: 0–9 + A–Z
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if not f.lower().endswith(".png"):
            continue

        # ---- Load image ----
        img_path = os.path.join(root, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape

        # Filename like "abcd-0.png" → "abcd"
        text = pathlib.Path(f).stem.split("-")[0]
        k = len(text)

        # ---- Segment via kmeans ----
        labels, centers = kmeans(img, k)

        # For each cluster (roughly 1 letter)
        cols = []
        for col in range(k):
            idxs = np.where(labels == col)[1]
            if len(idxs) == 0:
                continue
            x_mean = np.mean(idxs)
            cols.append((x_mean, col))
        cols.sort()

        yolo_lines = []

        for i, (x_mean, col) in enumerate(cols):
            if i >= len(text):  # safety
                break
            letter = text[i].upper()
            if letter not in class_to_idx:
                continue
            cls_id = class_to_idx[letter]

            # derive bounding box
            l = max(0, round(x_mean) - 40)
            r = min(w - 1, l + 80)
            l = r - 80
            box_w = r - l
            box_h = h
            x_center = (l + box_w / 2) / w
            y_center = 0.5  # assume full height
            box_w /= w
            box_h /= h

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        # ---- Save image and label file ----
        new_img_path = os.path.join(DEST_IMG_DIR, f)
        cv2.imwrite(new_img_path, img)
        label_path = os.path.join(DEST_LABEL_DIR, f.replace(".png", ".txt"))
        with open(label_path, "w") as fp:
            fp.write("\n".join(yolo_lines))

print("✅ YOLO dataset created!")
