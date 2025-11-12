"""
prepare_yolo_cls_dataset.py — reorganize sliced character dataset
into YOLO classification format (train/val by class folders).
"""

import os, csv, shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SRC_TRAIN = "./data_letter/train/labels.csv"
SRC_TEST = "./data_letter/test/labels.csv"
DEST_DIR = "./data_letter_yolo"

# Create folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

def copy_images_from_csv(csv_path, split_name):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc=f"Processing {split_name}"):
            if len(row) < 2:
                continue
            img_path, label = row
            label = label.strip().upper()   # normalize class name
            if label == "":
                continue
            dest_class_dir = os.path.join(DEST_DIR, split_name, label)
            os.makedirs(dest_class_dir, exist_ok=True)
            # Copy image to that class folder
            try:
                shutil.copy(img_path, dest_class_dir)
            except FileNotFoundError:
                continue

# Split train/test logically (we'll just use test as val)
copy_images_from_csv(SRC_TRAIN, "train")
copy_images_from_csv(SRC_TEST, "val")

print(f"✅ Dataset prepared in {DEST_DIR}")
