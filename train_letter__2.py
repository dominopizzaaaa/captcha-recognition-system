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
BATCH_SIZE = 512
EPOCHS = 5000
LR = 1e-3
WEIGHT_DECAY = 0.001
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

    def augment_img(self, img: Image.Image) -> Image.Image:
        from PIL import ImageDraw

        # 1) Bigger geometric jitter (rotate / translate / scale / shear)
        if random.random() < 0.6:
            angle = random.uniform(-18, 18)                       # bigger rotation
            tx = random.uniform(-0.07, 0.07) * img.size[0]        # more horizontal jitter
            ty = random.uniform(-0.07, 0.07) * img.size[1]        # more vertical jitter
            scale = random.uniform(0.88, 1.12)                    # wider scale range
            shear = random.uniform(-8, 8)                         # small shear
            img = TF.affine(img, angle=angle, translate=(tx, ty), scale=scale, shear=shear)

        # 2) Random perspective warp (helps simulate slanted/distorted captchas)
        if random.random() < 0.6:
            w, h = img.size
            jitter = lambda v: (max(0, min(v[0], w)), max(0, min(v[1], h)))
            # start corners
            src = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]
            max_p = 0.12  # percent of width/height to jitter corners
            dst = []
            for (x,y) in src:
                dx = int(random.uniform(-max_p, max_p) * w)
                dy = int(random.uniform(-max_p, max_p) * h)
                dst.append(jitter((x+dx, y+dy)))
            try:
                img = TF.perspective(img, startpoints=src, endpoints=dst)
            except Exception:
                # fallback: if perspective not available or failed, ignore
                pass

        # 3) Photometric transforms (brightness / contrast / sharpness / blur)
        if random.random() < 0.4:
            img = TF.adjust_contrast(img, random.uniform(0.65, 1.35))
        if random.random() < 0.4:
            img = TF.adjust_brightness(img, random.uniform(0.75, 1.25))
        if random.random() < 0.4:
            # small blur to simulate compression/anti-aliasing
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.8)))
        if random.random() < 0.4:
            # occasional sharpening (or unsharp)
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=random.randint(80,200), threshold=2))

        # 4) Morphological noise: small erosion/dilation to simulate ink spread or thinning
        if random.random() < 0.4:
            if random.random() < 0.5:
                img = img.filter(ImageFilter.MinFilter(size=3))  # erosion-like
            else:
                img = img.filter(ImageFilter.MaxFilter(size=3))  # dilation-like

        # 5) Add occlusions: random lines and dots (very common in CAPTCHAs)
        draw = ImageDraw.Draw(img)
        if random.random() < 0.0:
            # random interfering lines
            n_lines = random.randint(1, 4)
            for _ in range(n_lines):
                x1 = random.randint(0, img.width - 1)
                y1 = random.randint(0, img.height - 1)
                x2 = random.randint(0, img.width - 1)
                y2 = random.randint(0, img.height - 1)
                lw = random.randint(1, max(1, img.height // 12))
                # draw with darker color (foreground is usually dark on light bg)
                draw.line((x1, y1, x2, y2), fill=0, width=lw)

        if random.random() < 0.0:
            # random speckles / dots
            n_dots = random.randint(8, 60)
            for _ in range(n_dots):
                x = random.randint(0, img.width - 1)
                y = random.randint(0, img.height - 1)
                c = 0 if random.random() < 0.8 else 255
                draw.point((x, y), fill=c)

        # 6) Tiny random crop / pad shift (to simulate imperfect centering)
        if random.random() < 0.4:
            w, h = img.size
            cx = random.randint(-int(0.04 * w), int(0.04 * w))
            cy = random.randint(-int(0.06 * h), int(0.06 * h))
            new = Image.new(img.mode, (w, h), color=255)
            new.paste(img, (cx, cy))
            img = new

        # 7) Final clamp: occasionally invert contrast (rare)
        if random.random() < 0.0:
            img = Image.eval(img, lambda p: 255 - p)

        return img

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path)
        img = img.convert("L") if self.grayscale else img.convert("RGB")

        if self.augment:
            img = self.augment_img(img)

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
  train_ds.samples.extend(val_ds.samples)

  num_classes = len(train_ds.classes)
  in_ch = 1

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, collate_fn=collate_pad_width)
  val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_pad_width)
  print(CSV_TRAIN, flush=True)


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
  print(EPOCHS, flush=True)
  for epoch in range(1, EPOCHS+1):
      print(epoch, flush=True)
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
      img = pad_to_height(img, TARGET_HEIGHT, fill=255)
      x = TF.to_tensor(img)
      x = TF.normalize(x, mean=[0.5], std=[0.5]).unsqueeze(0).to(DEVICE)
      logits = model(x)
      idx = logits.argmax(1).item()
      return train_ds.classes[idx]



if __name__ == "__main__":
  train()

# Example:
# print(predict_image("/path/to/sample.png"))
