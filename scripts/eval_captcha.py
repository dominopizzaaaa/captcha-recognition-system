from pathlib import Path; import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, glob, numpy as torch
from src.utils import read_label_from_name, load_gray, equal_slices, C2I, I2C
from train_char_cnn import CharCNN, DATA_DIR, CKPT

TEST_DIR = os.path.join(DATA_DIR, "test")

def collect_examples(model, limit_correct=10, limit_wrong=10):
    correct, wrong = [], []
    for p in glob.glob(os.path.join(TEST_DIR, "*.png")):
        gt, pr = predict_string(model, p)
        (correct if pr==gt.upper() else wrong).append((p, gt, pr))
        if len(correct)>=limit_correct and len(wrong)>=limit_wrong:
            break
    return correct[:limit_correct], wrong[:limit_wrong]


def predict_string(model, path):
    gt = read_label_from_name(path)
    img = load_gray(path, target_h=32)
    tiles = equal_slices(img, len(gt), out_hw=(32,32))  # (n,32,32)

    # (n,1,32,32) float32 tensor in [0,1]
    x = torch.tensor(tiles[:, None, ...], dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)                     # (n, 36)
        pred_idx = logits.argmax(1).tolist()  # list of length n
    preds = [I2C[i] for i in pred_idx]
    return gt, "".join(preds)

def main():
    model = CharCNN().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))

    n_caps=0; n_caps_ok=0
    tot_chars=0; ok_chars=0

    for p in glob.glob(os.path.join(TEST_DIR, "*.png")):
        gt, pr = predict_string(model, p)
        n_caps += 1
        if pr == gt.upper(): n_caps_ok += 1
        tot_chars += len(gt)
        ok_chars += sum(g==p for g,p in zip(gt.upper(), pr))

    print(f"Captcha accuracy:  {n_caps_ok}/{n_caps} = {n_caps_ok/n_caps:.3f}")
    print(f"Character accuracy:{ok_chars}/{tot_chars} = {ok_chars/tot_chars:.3f}")

    from collections import Counter, defaultdict
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    import csv

    # Collect per-char GT & Pred across all tiles
    all_gt_chars, all_pr_chars = [], []

    # re-run to collect char-level labels
    for p in glob.glob(os.path.join(TEST_DIR, "*.png")):
        gt, pr = predict_string(model, p)
        for g, r in zip(gt.upper(), pr):
            all_gt_chars.append(g)
            all_pr_chars.append(r)

    classes = sorted(set(all_gt_chars) | set(all_pr_chars))
    print("\nPer-class report:")
    print(classification_report(all_gt_chars, all_pr_chars, labels=classes, zero_division=0))

    cm = confusion_matrix(all_gt_chars, all_pr_chars, labels=classes)
    np.set_printoptions(linewidth=120)
    print("Confusion matrix (rows=GT, cols=Pred):")
    print(classes)
    print(cm)

    # Save CSVs for poster tables
    CSV_DIR = os.path.join(DATA_DIR, "metrics")
    os.makedirs(CSV_DIR, exist_ok=True)
    with open(os.path.join(CSV_DIR, "per_class_report.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class","support","precision","recall","f1"])
        # quick compute per class metrics
        rep = classification_report(all_gt_chars, all_pr_chars, labels=classes, output_dict=True, zero_division=0)
        for c in classes:
            d = rep.get(c, {"precision":0,"recall":0,"f1-score":0,"support":0})
            writer.writerow([c, int(d["support"]), round(d["precision"],3), round(d["recall"],3), round(d["f1-score"],3)])
    np.savetxt(os.path.join(CSV_DIR, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    print(f"\nSaved per-class and confusion to: {CSV_DIR}")

    ok, bad = collect_examples(model)
    EX_DIR = os.path.join(DATA_DIR, "examples"); os.makedirs(EX_DIR, exist_ok=True)
    with open(os.path.join(EX_DIR,"correct.txt"), "w") as f:
        for p,gt,pr in ok: f.write(f"{os.path.basename(p)}\tGT={gt}\tPred={pr}\n")
    with open(os.path.join(EX_DIR,"wrong.txt"), "w") as f:
        for p,gt,pr in bad: f.write(f"{os.path.basename(p)}\tGT={gt}\tPred={pr}\n")
    print(f"Saved example lists in: {EX_DIR}")



if __name__ == "__main__":
    main()
