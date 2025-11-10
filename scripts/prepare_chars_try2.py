# scripts/prepare_chars_try2.py
from pathlib import Path; import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, glob, cv2, numpy as np
from tqdm import tqdm
from src.utils import read_label_from_name, equal_slices, C2I

DATA_DIR = "/Users/dominopizzaaaa/Desktop/dev/captcha-recognition-system/data"
IN_DIR   = os.path.join(DATA_DIR, "processed", "train")
OUT_NPZ  = os.path.join(DATA_DIR, "chars_train_try2.npz")

def load_processed_and_resize(path, target_h=32):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    H, W = img.shape
    scale = target_h / max(H, 1)
    img = cv2.resize(img, (max(int(W*scale),1), target_h), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)  # (32, W~)

def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.png")))
    if not files:
        print(f"No images in {IN_DIR}. Did you run the preprocessor to data/processed/train?")
        return

    X, Y = [], []
    for p in tqdm(files):
        label = read_label_from_name(p).upper()
        img32 = load_processed_and_resize(p, target_h=32)
        tiles = equal_slices(img32, len(label), out_hw=(32,32))  # (n,32,32)
        for t, ch in zip(tiles, label):
            if ch not in C2I: 
                continue
            X.append(t)
            Y.append(C2I[ch])

    X = np.asarray(X, np.float32)[..., None]  # (N,32,32,1)
    Y = np.asarray(Y, np.int64)
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, X=X, Y=Y)
    print("Saved:", OUT_NPZ, X.shape, Y.shape)

if __name__ == "__main__":
    main()
