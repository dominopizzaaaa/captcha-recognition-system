# scripts/segment_by_recog.py
from pathlib import Path; import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, glob, cv2, torch, numpy as np
from src.utils import read_label_from_name, load_gray, C2I, I2C
from train_char_cnn import CharCNN, DATA_DIR, CKPT

TEST_DIR = os.path.join(DATA_DIR, "test")

# --- 1) simple preprocessing to tame lines/noise ---
def preprocess_for_scan(img_gray):
    # img_gray: float [0..1], shape (H,W)
    g = (img_gray*255).astype(np.uint8)
    # Otsu binarize, invert to make glyphs white
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127: bw = 255 - bw
    # light morphology to thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    return bw  # uint8 {0,255}

# --- 2) sliding-window char probs using your CharCNN ---
def sliding_window_logits(model, img_gray, win_w=28, stride=4, target_h=32):
    # resize height to 32, keep aspect
    H0, W0 = img_gray.shape
    scale = target_h / H0
    img = cv2.resize(img_gray, (int(W0*scale), target_h), interpolation=cv2.INTER_AREA)
    H, W = img.shape
    xs = list(range(0, max(1, W - win_w + 1), stride))
    patches = []
    for x in xs:
        patch = img[:, x:x+win_w]
        patch = cv2.resize(patch, (32,32))  # match training size
        patches.append(patch)
    X = torch.tensor(np.stack(patches,0)[:,None,...], dtype=torch.float32)  # (T,1,32,32)
    with torch.no_grad():
        logits = model(X)            # (T,36)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (T,36)
    return xs, probs, img

# --- 3) greedy decode into runs (CTC-lite) ---
def decode_runs(xs, probs, conf_th=0.50, min_run=2):
    labels = probs.argmax(1)                   # (T,)
    confs  = probs.max(1)                      # (T,)
    seq = []
    for i,(lab,c) in enumerate(zip(labels, confs)):
        if c < conf_th:
            seq.append(('.', i))               # blank
        else:
            seq.append((I2C[int(lab)], i))
    # collapse consecutive duplicates & blanks
    runs = []
    cur_char, start_i = None, None
    for ch, idx in seq:
        if ch == '.':
            if cur_char is not None:
                runs.append((cur_char, start_i, idx-1))
                cur_char, start_i = None, None
            continue
        if cur_char is None:
            cur_char, start_i = ch, idx
        elif ch != cur_char:
            runs.append((cur_char, start_i, idx-1))
            cur_char, start_i = ch, idx
    if cur_char is not None:
        runs.append((cur_char, start_i, len(seq)-1))
    # filter short runs
    runs = [(ch,i0,i1) for (ch,i0,i1) in runs if (i1 - i0 + 1) >= min_run]
    # convert run indices to pixel spans using xs and window width
    return runs

# --- 4) refine cut by vertical valleys within span ---
def refine_span(img32H, x0_idx, x1_idx, xs, win_w=28):
    # map window indices to approx pixel bounds
    px0 = xs[x0_idx]
    px1 = xs[x1_idx] + win_w
    px0 = max(0, px0)
    px1 = min(img32H.shape[1], px1)
    roi = img32H[:, px0:px1]
    # vertical projection: sum of dark pixels
    col_sum = (255 - roi).sum(axis=0)
    # find low-density margins to cut
    # search a few pixels inside to avoid cutting glyph strokes
    left = 0
    for i in range(min(6, roi.shape[1]//3)):
        if col_sum[i] < np.percentile(col_sum, 20):
            left = i
    right = roi.shape[1]-1
    for i in range(roi.shape[1]-1, roi.shape[1]-1 - min(6, roi.shape[1]//3), -1):
        if col_sum[i] < np.percentile(col_sum, 20):
            right = i
    cut0 = px0 + left
    cut1 = px0 + right + 1
    cut0 = max(0, min(cut0, img32H.shape[1]-1))
    cut1 = max(cut0+1, min(cut1, img32H.shape[1]))
    return cut0, cut1

# --- 5) final classify refined crops to build string ---
def classify_crops(model, img32H, spans):
    chars = []
    for (x0,x1) in spans:
        crop = img32H[:, x0:x1]
        crop = cv2.resize(crop, (32,32), interpolation=cv2.INTER_AREA)
        X = torch.tensor(crop[None,None,...]/255.0, dtype=torch.float32)
        with torch.no_grad():
            p = torch.softmax(model(X), dim=1)[0]
        c = int(p.argmax().item())
        chars.append(I2C[c])
    return "".join(chars)

def predict_by_recog_then_split(model, path, conf_th=0.50, stride=4, win_w=28):
    gt = read_label_from_name(path).upper()
    img = load_gray(path, target_h=32)              # float [0,1]
    bw = preprocess_for_scan(img)                   # uint8 {0,255}
    xs, probs, img32 = sliding_window_logits(model, img, win_w=win_w, stride=stride, target_h=32)
    runs = decode_runs(xs, probs, conf_th=conf_th, min_run=2)

    spans = []
    for ch, i0, i1 in runs:
        cut0, cut1 = refine_span(img32, i0, i1, xs, win_w=win_w)
        spans.append((cut0, cut1))
    pred = classify_crops(model, img32, spans) if spans else ""

    return gt, pred, spans, runs

def main():
    model = CharCNN().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))

    files = glob.glob(os.path.join(TEST_DIR, "*.png"))
    files.sort()
    n_caps=0; n_ok=0; tot=0; ok=0

    # quick sweep of a few thresholds to see sensitivity
    confs = [0.40, 0.50, 0.60]
    for conf in confs:
        n_caps=n_ok=tot=ok=0
        for p in files:
            gt, pr, spans, runs = predict_by_recog_then_split(model, p, conf_th=conf, stride=4, win_w=28)
            n_caps += 1
            if pr == gt: n_ok += 1
            tot += len(gt); ok += sum(a==b for a,b in zip(gt, pr))
        print(f"[conf={conf:.2f}] Captcha acc {n_ok}/{n_caps}={n_ok/n_caps:.3f} | Char acc {ok}/{tot}={ok/tot:.3f}")

if __name__ == "__main__":
    main()
