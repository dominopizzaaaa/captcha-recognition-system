# src/preprocess_try2.py
"""
Cleans and binarizes CAPTCHA images, removes straight lines, and saves into data/processed/.
Paths are resolved relative to this file, so you can run it from anywhere.
"""
import os, cv2, numpy as np
from tqdm import tqdm
from pathlib import Path

# ---- robust paths ----
ROOT = Path(__file__).resolve().parents[1]   # project root
DATA = ROOT / "data"
INPUT_TRAIN  = DATA / "train"
INPUT_TEST   = DATA / "test"
OUTPUT_TRAIN = DATA / "processed" / "train"
OUTPUT_TEST  = DATA / "processed" / "test"
OUTPUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUTPUT_TEST.mkdir(parents=True, exist_ok=True)

def preprocess_image(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 5
    )

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    binary = cv2.erode(binary, k, iterations=1)
    binary = cv2.dilate(binary, k, iterations=1)

    if (binary == 255).sum() < (binary == 0).sum():
        binary = cv2.bitwise_not(binary)

    inv = cv2.bitwise_not(binary)
    edges = cv2.Canny(inv, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=60,
        minLineLength=max(20, binary.shape[1]//10),
        maxLineGap=10
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(binary, (x1, y1), (x2, y2), color=0, thickness=2)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    return binary

def preprocess_and_save(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        print(f"⚠️  Missing input folder: {input_dir}")
        return
    imgs = [p for p in input_dir.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")]
    for path in tqdm(imgs, desc=f"{input_dir.name}"):
        out_path = output_dir / path.name
        proc = preprocess_image(path)
        if proc is not None:
            cv2.imwrite(str(out_path), proc)

if __name__ == "__main__":
    print("=== Preprocessing train set ===")
    preprocess_and_save(INPUT_TRAIN, OUTPUT_TRAIN)
    print("\n=== Preprocessing test set ===")
    preprocess_and_save(INPUT_TEST, OUTPUT_TEST)
    print("\n✅ Done! Cleaned images saved in data/processed/")
