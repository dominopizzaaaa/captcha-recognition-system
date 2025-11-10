# scripts/debug_try2_preview.py
from pathlib import Path; import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, glob, cv2

DATA_DIR  = "/Users/dominopizzaaaa/Desktop/dev/captcha-recognition-system/data"
RAW_TEST  = os.path.join(DATA_DIR, "test")
PROC_TEST = os.path.join(DATA_DIR, "processed", "test")
OUT_DIR   = os.path.join(DATA_DIR, "preview_try2")
os.makedirs(OUT_DIR, exist_ok=True)

def stack_row(a_bgr, b_gray):
    h = 64
    a_res = cv2.resize(a_bgr, (a_bgr.shape[1], h))
    b_res = cv2.resize(b_gray, (b_gray.shape[1], h))
    b_bgr = cv2.cvtColor(b_res, cv2.COLOR_GRAY2BGR)
    return cv2.hconcat([a_res, b_bgr])

def main():
    raw_files = sorted(glob.glob(os.path.join(RAW_TEST, "*.png")))
    if not raw_files:
        print(f"No raw images in {RAW_TEST}")
        return
    count = 0
    for p in raw_files:
        fname = os.path.basename(p)
        q = os.path.join(PROC_TEST, fname)
        if not os.path.exists(q):  # processed not found
            continue
        raw = cv2.imread(p, cv2.IMREAD_COLOR)
        proc = cv2.imread(q, cv2.IMREAD_GRAYSCALE)
        if raw is None or proc is None:
            continue
        row = stack_row(raw, proc)
        cv2.imwrite(os.path.join(OUT_DIR, fname), row)
        count += 1
        if count >= 50: break
    print(f"Saved {count} previews to {OUT_DIR}")

if __name__ == "__main__":
    main()
