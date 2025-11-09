import csv, time, random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from modeling_letter import LetterCNN

# ===== Config =====
CSV_TRAIN = "data_letter/train/labels.csv"   # path,label (no header)
CSV_VAL   = "data_letter/test/labels.csv"
TARGET_HEIGHT = 80        # we will pad to this height
BATCH_SIZE = 1024
EPOCHS = 5000
LR = 1e-4
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUGMENT = True

# If you want to lock a class list, set it; else inferred from CSV
CLASSES = None  # e.g., list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ===== CSV + labels =====
def read_csv(csv_path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row: continue
            rows.append((row[0].strip(), row[1].strip()))
    return rows

def build_label_maps(rows: List[Tuple[str,str]], classes: Optional[List[str]] = None):
    if classes is None:
        classes = sorted({y for _, y in rows})
    class_to_idx = {c:i for i,c in enumerate(classes)}
    return classes, class_to_idx

# ===== Image utils =====
def pad_to_height(img: Image.Image, target_h: int, fill=0) -> Image.Image:
    """Pad (not resize) to the target height by adding equal top/bottom padding."""
    w, h = img.size
    if h == target_h:
        return img
    if h < target_h:
        pad_total = target_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        new_img = Image.new(img.mode, (w, target_h), color=fill)
        new_img.paste(img, (0, pad_top))
        return new_img
    else:
        # h > target_h: by default, center-crop to target height (no stretching)
        top = (h - target_h) // 2
        return img.crop((0, top, w, top + target_h))

def random_geometric(img: Image.Image) -> Image.Image:
    # tiny, safe jitter for single characters
    angle = random.uniform(-4, 4)
    tx = random.uniform(-0.02, 0.02) * img.size[0]
    ty = random.uniform(-0.02, 0.02) * img.size[1]
    scale = random.uniform(0.98, 1.02)
    return TF.affine(img, angle=angle, translate=(tx, ty), scale=scale, shear=0)


# ===== Custom Dataset =====
class CSVPadHeightDataset(Dataset):
    def __init__(self, csv_path: str, classes: Optional[List[str]] = None,
                 target_height: int = 80, grayscale: bool = True, augment: bool = False):
        super().__init__()
        self.samples = read_csv(csv_path)
        self.classes, self.class_to_idx = build_label_maps(self.samples, classes)
        self.target_height = target_height
        self.grayscale = grayscale
        self.augment = augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path)
        img = img.convert("L") if self.grayscale else img.convert("RGB")

        # Optional light aug (before padding)
        if self.augment:
            if random.random() < 0.5:
                img = random_geometric(img)
            # if random.random() < 0.2:
                # img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Pad/crop vertically to target height; width left as-is
        img = pad_to_height(img, self.target_height, fill=255)

        # To tensor & normalize (-1..1). Width is variable here.
        x = TF.to_tensor(img)
        mean = [0.5] * x.size(0)
        std  = [0.5] * x.size(0)
        x = TF.normalize(x, mean=mean, std=std)

        y_idx = self.class_to_idx[y]
        return x, y_idx

# ===== Collate: pad widths within each batch =====
def collate_pad_width(batch):
    """
    batch: list of (C,H,W) tensors with same H, variable W.
    Pads each to max width (right-side padding) so they can stack.
    """
    xs, ys = zip(*batch)
    C, H = xs[0].size(0), xs[0].size(1)
    maxW = max(t.size(2) for t in xs)
    out = []
    for t in xs:
        pad_w = maxW - t.size(2)
        if pad_w > 0:
            t = F.pad(t, (0, pad_w, 0, 0))  # pad right
        out.append(t)
    x = torch.stack(out, 0)  # (B,C,H,maxW)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y



def train():
  # ===== Data =====
  train_ds = CSVPadHeightDataset(CSV_TRAIN, classes=CLASSES, target_height=TARGET_HEIGHT, grayscale=True, augment=AUGMENT)
  val_ds   = CSVPadHeightDataset(CSV_VAL,   classes=train_ds.classes, target_height=TARGET_HEIGHT, grayscale=True, augment=False)

  num_classes = len(train_ds.classes)
  in_ch = 1

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_pad_width)
  val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_pad_width)

  print("Classes:", train_ds.classes)

  # ===== Train =====
  model = LetterCNN(num_classes, in_ch=in_ch, freeze_backbone=False).to(DEVICE)
#   weights = torch.load("letter_padheight_best.pt")
#   model.load_state_dict(weights['model'])
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
  criterion = nn.CrossEntropyLoss()

  @torch.no_grad()
  def evaluate():
      model.eval()
      loss_sum = 0.0; correct = 0; total = 0
      for x, y in val_loader:
          x, y = x.to(DEVICE), y.to(DEVICE)
          logits = model(x)
          loss = criterion(logits, y)
          loss_sum += loss.item() * x.size(0)
          pred = logits.argmax(1)
          correct += (pred == y).sum().item()
          total += x.size(0)
      return loss_sum/total, correct/total

  best_acc = 0.0
  for epoch in range(1, EPOCHS+1):
      model.train()
      t0, run_loss = time.time(), 0.0
      for x, y in train_loader:
          x, y = x.to(DEVICE), y.to(DEVICE)
          optimizer.zero_grad(set_to_none=True)
          logits = model(x)
          loss = criterion(logits, y)
          loss.backward()
          optimizer.step()
          run_loss += loss.item() * x.size(0)
      scheduler.step()

      train_loss = run_loss / len(train_ds)
      val_loss, val_acc = evaluate()
      print(f"[{epoch:02d}/{EPOCHS}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

      if val_acc > best_acc:
          best_acc = val_acc
          torch.save({
              "model": model.state_dict(),
              "classes": train_ds.classes,
              "target_height": TARGET_HEIGHT,
              "in_ch": in_ch
          }, "letter_padheight_best.pt")
          print(f"  ↳ Saved best (acc={best_acc*100:.2f}%).")

  # ===== Inference helper =====
  @torch.no_grad()
  def predict_image(path: str):
      model.eval()
      img = Image.open(path).convert("L")
      img = pad_to_height(img, TARGET_HEIGHT, fill=0)
      x = TF.to_tensor(img)
      x = TF.normalize(x, mean=[0.5], std=[0.5]).unsqueeze(0).to(DEVICE)
      logits = model(x)
      idx = logits.argmax(1).item()
      return train_ds.classes[idx]



if __name__ == "__main__":
  train()

# Example:
# print(predict_image("/path/to/sample.png"))
