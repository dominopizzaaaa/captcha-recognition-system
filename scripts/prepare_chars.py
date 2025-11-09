from pathlib import Path; import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, glob, numpy as np
from tqdm import tqdm
from src.utils import read_label_from_name, load_gray, equal_slices, C2I

DATA_DIR = "/Users/dominopizzaaaa/Desktop/dev/captcha-recognition-system/data"
OUT_NPZ = os.path.join(DATA_DIR, "chars_train.npz")

def main():
    X, Y = [], []
    for p in tqdm(glob.glob(os.path.join(DATA_DIR, "train", "*.png"))):
        lab = read_label_from_name(p)
        img = load_gray(p, target_h=32)
        tiles = equal_slices(img, len(lab), out_hw=(32,32))  # (n,32,32)
        for t, ch in zip(tiles, lab):
            if ch not in C2I:  # skip weird chars if any
                continue
            X.append(t)
            Y.append(C2I[ch])
    X = np.array(X, dtype=np.float32)[..., None]  # (N,32,32,1)
    Y = np.array(Y, dtype=np.int64)
    np.savez_compressed(OUT_NPZ, X=X, Y=Y)
    print("Saved:", OUT_NPZ, X.shape, Y.shape)

if __name__ == "__main__":
    main()
