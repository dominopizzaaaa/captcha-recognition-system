# src/utils.py
import os, re, cv2
import numpy as np

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
C2I = {c:i for i,c in enumerate(ALPHABET)}
I2C = {i:c for c,i in C2I.items()}

def read_label_from_name(fname: str) -> str:
    # e.g. "sne2ee-0.png" -> "sne2ee"
    base = os.path.basename(fname)
    lab = base.split('-')[0]
    return lab.strip().upper()

def load_gray(path: str, target_h=32):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # normalize height to 32, keep aspect (simple)
    h, w = img.shape[:2]
    scale = target_h / h
    img = cv2.resize(img, (int(w*scale), target_h), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img  # shape (H, W)

def equal_slices(img: np.ndarray, n_chars: int, out_hw=(32,32)):
    H, W = img.shape
    tile_w = W // n_chars
    tiles = []
    for i in range(n_chars):
        x0 = i * tile_w
        x1 = W if i == n_chars-1 else (i+1)*tile_w  # last takes the remainder
        tile = img[:, x0:x1]
        tile = cv2.resize(tile, out_hw, interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    return np.stack(tiles, axis=0)  # (n, 32, 32)
