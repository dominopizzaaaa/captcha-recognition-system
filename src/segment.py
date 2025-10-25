"""
Segmentation (tokenization) for CAPTCHA recognition.
Uses contour + vertical projection splitting to isolate characters.
Skips filename suffixes like "-0" from labels.
"""

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

PROC_TRAIN = "../data/processed/train"
PROC_TEST  = "../data/processed/test"
OUT_TRAIN  = "../data/chars/train"
OUT_TEST   = "../data/chars/test"
DBG_DIR    = "../figures/seg_debug"

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST, exist_ok=True)
os.makedirs(DBG_DIR, exist_ok=True)


# -------------------------------------------------------
# 1️⃣ Helpers
# -------------------------------------------------------

def read_label_from_filename(path):
    """Extract label from filename (remove trailing '-0' etc.)."""
    base = os.path.basename(path)
    label = os.path.splitext(base)[0].lower()
    # Remove anything after a dash (like '-0')
    if "-" in label:
        label = label.split("-")[0]
    return label


def crop_vertical_whitespace(bin_img):
    """Trim empty top/bottom rows."""
    white_pixels = np.sum(bin_img == 255)
    black_pixels = np.sum(bin_img == 0)
    if white_pixels > black_pixels:
        bin_img = cv2.bitwise_not(bin_img)

    rows_ink = (bin_img == 255).sum(axis=1)
    ys = np.where(rows_ink > 0)[0]
    if len(ys) == 0:
        return bin_img
    y1, y2 = ys[0], ys[-1]
    return bin_img[y1:y2 + 1, :]


def vertical_projection_split(img):
    """Split horizontally merged characters using white-pixel projection."""
    H, W = img.shape
    projection = np.sum(img == 255, axis=0)
    projection = cv2.blur(projection.reshape(1, -1), (1, 5)).flatten()

    threshold = 0.15 * H  # higher = more sensitive
    split_points = np.where(projection < threshold)[0]

    if len(split_points) == 0:
        return [img]

    # find continuous zero regions
    cuts, segments = [], []
    start = split_points[0]
    for i in range(1, len(split_points)):
        if split_points[i] != split_points[i - 1] + 1:
            cuts.append((start, split_points[i - 1]))
            start = split_points[i]
    cuts.append((start, split_points[-1]))

    prev_end = 0
    for s, e in cuts:
        if s - prev_end > 3:
            segments.append(img[:, prev_end:s])
        prev_end = e
    if W - prev_end > 3:
        segments.append(img[:, prev_end:W])
    return segments


def find_candidate_boxes(bin_img):
    """Detect potential characters using contours."""
    # Ensure binary and white text
    _, bin_img = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(bin_img == 255)
    black_pixels = np.sum(bin_img == 0)
    if white_pixels < black_pixels:
        bin_img = cv2.bitwise_not(bin_img)

    # Erode slightly to disconnect touching letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.erode(bin_img, kernel, iterations=1)

    # Optional projection split (to help separate blobs)
    segments = vertical_projection_split(bin_img)
    if len(segments) > 1:
        new_img = np.zeros_like(bin_img)
        offset = 0
        for seg in segments:
            w = seg.shape[1]
            new_img[:, offset:offset + w] = seg
            offset += w
        bin_img = new_img

    # --- Find contours ---
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Total raw contours:", len(contours))

    H, W = bin_img.shape[:2]
    area_img = H * W

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Relaxed thresholds
        if area < 0.0005 * area_img or area > 0.4 * area_img:
            continue
        ar = w / (h + 1e-6)
        if ar < 0.08 or ar > 3.0:
            continue
        patch = bin_img[y:y + h, x:x + w]
        density = (patch == 255).mean()
        if density < 0.05 or density > 0.95:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    return boxes, bin_img


def draw_debug(bin_img, boxes, out_path):
    dbg = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite(out_path, dbg)


def save_char_crops(bin_img, boxes, label, out_root, stem):
    for i, (x, y, w, h) in enumerate(boxes):
        if i >= len(label):
            break
        crop = bin_img[y:y + h, x:x + w]
        s = max(h, w)
        pad_y = (s - h) // 2
        pad_x = (s - w) // 2
        sq = cv2.copyMakeBorder(
            crop, pad_y, s - h - pad_y, pad_x, s - w - pad_x,
            cv2.BORDER_CONSTANT, value=0
        )
        tile = cv2.resize(sq, (40, 40), interpolation=cv2.INTER_NEAREST)
        ch = label[i]
        out_dir = os.path.join(out_root, ch)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_{i}.png"), tile)


# -------------------------------------------------------
# 2️⃣ Main driver
# -------------------------------------------------------

def process_one(img_path, out_root):
    bin_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if bin_img is None:
        print(f"[WARN] cannot read {img_path}")
        return

    label = read_label_from_filename(img_path)
    bin_img = crop_vertical_whitespace(bin_img)

    boxes, bin_img = find_candidate_boxes(bin_img)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    dbg_path = os.path.join(DBG_DIR, f"{stem}.png")
    draw_debug(bin_img, boxes, dbg_path)

    print(f"{os.path.basename(img_path)}: found {len(boxes)} boxes, label len {len(label)}")

    if len(boxes) == 0:
        return
    save_char_crops(bin_img, boxes, label, out_root, stem)


def batch_process(split="train"):
    if split == "train":
        in_dir, out_root = PROC_TRAIN, OUT_TRAIN
    else:
        in_dir, out_root = PROC_TEST, OUT_TEST

    files = [f for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg'))]
    for f in tqdm(files):
        process_one(os.path.join(in_dir, f), out_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=str, default=None)
    ap.add_argument("--split", type=str, default=None, choices=["train", "test"])
    args = ap.parse_args()

    if args.one:
        process_one(args.one, OUT_TRAIN)
    elif args.split:
        batch_process(args.split)
    else:
        print("Usage:")
        print("  python3 src/segment.py --one data/processed/train/<file>.png")
        print("  python3 src/segment.py --split train")
        print("  python3 src/segment.py --split test")
