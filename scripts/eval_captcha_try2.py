# scripts/eval_captcha_try2.py
from pathlib import Path; import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, glob, cv2, torch, numpy as np
from src.utils import read_label_from_name, equal_slices, I2C
from train_char_cnn import CharCNN, DATA_DIR, CKPT

TEST_DIR = os.path.join(DATA_DIR, "processed", "test")

def load_processed_and_resize(path, target_h=32):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    H, W = img.shape
    scale = target_h / max(H, 1)
    img = cv2.resize(img, (max(int(W*scale), 1), target_h), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0)
    return img

def predict_string(model, path):
    gt = read_label_from_name(path).upper()
    img32 = load_processed_and_resize(path, target_h=32)
    tiles = equal_slices(img32, len(gt), out_hw=(32,32))  # (n,32,32)
    X = torch.tensor(tiles[:, None, ...], dtype=torch.float32)  # (n,1,32,32)
    with torch.no_grad():
        logits = model(X)                 # (n,36)
        pred_idx = logits.argmax(1).tolist()
    pred = "".join(I2C[i] for i in pred_idx)
    return gt, pred

def main():
    model = CharCNN().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
    if not files:
        print(f"No images found in {TEST_DIR}. Did you preprocess the test set?")
        return

    n_caps=0; n_ok=0; tot=0; ok=0
    for p in files:
        gt, pr = predict_string(model, p)
        n_caps += 1;  n_ok += int(pr == gt)
        tot += len(gt); ok += sum(a==b for a,b in zip(gt, pr))

    print(f"[Try 2 / processed] Captcha accuracy:   {n_ok}/{n_caps} = {n_ok/n_caps:.3f}")
    print(f"[Try 2 / processed] Character accuracy: {ok}/{tot} = {ok/tot:.3f}")

if __name__ == "__main__":
    main()
