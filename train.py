# train_varsize_two_csvs.py
import os, time, argparse, numpy as np, pandas as pd
from PIL import Image, ImageFilter
import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

# -------- Dataset: keeps native H (<= max_h), clamps very long W if wanted --------
class VarSizeDataset(Dataset):
    def __init__(self, data_dir, labels_csv, max_h=80, max_w=None, train=True):
        self.data_dir = data_dir
        df = pd.read_csv(labels_csv)
        col_path = "path" if "path" in df.columns else df.columns[0]
        col_cnt  = "count" if "count" in df.columns else df.columns[1]
        self.paths  = df[col_path].astype(str).tolist()
        self.counts = df[col_cnt].astype(float).tolist()
        self.train  = train
        self.max_h  = max_h
        self.max_w  = max_w  # e.g., 512 or None to keep original

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5]*3, [0.5]*3)
        self.colorj    = transforms.ColorJitter(0.3,0.3,0.2,0.02)

    def __len__(self): return len(self.paths)

    def _open(self, p):
        P = p if os.path.isabs(p) else os.path.join(self.data_dir, p)
        if not os.path.exists(P):
            P = os.path.normpath(os.path.join(self.data_dir, p.lstrip("./")))
        img = Image.open(P).convert("RGB")
        return img

    def _maybe_resize(self, img):
        w, h = img.size
        # If height exceeds max_h (shouldn't, but safe-guard), scale down keeping aspect
        if self.max_h and h > self.max_h:
            scale = self.max_h / h
            w = max(8, int(round(w * scale)))
            h = self.max_h
            img = img.resize((w, h), Image.BILINEAR)
        # Clamp very long widths to keep memory in check (optional)
        if self.max_w and w > self.max_w:
            scale = self.max_w / w
            w2 = int(round(w * scale)); h2 = max(8, int(round(h * scale)))
            img = img.resize((w2, h2), Image.BILINEAR)
        return img

    def __getitem__(self, i):
        y = torch.tensor([self.counts[i]], dtype=torch.float32)
        img = self._open(self.paths[i])

        if self.train:
            if np.random.rand() < 0.35:
                img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.2,1.0)))
            if np.random.rand() < 0.5:
                img = self.colorj(img)

        img = self._maybe_resize(img)  # preserves native height if <= max_h
        x = self.to_tensor(img)        # [3,H,W]
        x = self.normalize(x)
        mask = torch.ones(1, x.shape[1], x.shape[2], dtype=torch.float32)  # [1,H,W]
        return x, mask, y

# -------- Collate: pad to max H and max W in the batch --------
def collate_varsize(batch):
    xs, ms, ys = zip(*batch)
    B = len(xs)
    C = xs[0].shape[0]
    Hs = [x.shape[1] for x in xs]
    Ws = [x.shape[2] for x in xs]
    Hmax, Wmax = max(Hs), max(Ws)

    xpad = torch.zeros(B, C, Hmax, Wmax)  # normalized zero-mean padding
    mpad = torch.zeros(B, 1, Hmax, Wmax)
    for i, (x, m) in enumerate(zip(xs, ms)):
        H, W = x.shape[1], x.shape[2]
        xpad[i, :, :H, :W] = x
        mpad[i, :, :H, :W] = m

    y = torch.stack(ys, 0)
    return xpad, mpad, y

# -------- Model: ResNet18 trunk + masked global average pooling + regressor --------
class CountNetMasked(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # -> [B,512,Hf,Wf]
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, 1), nn.Softplus()
        )

    @staticmethod
    def masked_gap(feat, mask):
        # feat: [B,C,Hf,Wf], mask: [B,1,H,W] -> downsample mask to feat map, then weighted avg
        mask_f = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")  # [B,1,Hf,Wf]
        feat_sum = (feat * mask_f).sum(dim=(2,3))                           # [B,C]
        denom = mask_f.sum(dim=(2,3)).clamp_min(1e-6)                       # [B,1]
        return feat_sum / denom                                             # [B,C]

    def forward(self, x, mask):
        f = self.features(x)
        p = self.masked_gap(f, mask)
        return self.head(p)

# -------- Train loop --------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = VarSizeDataset(args.train_dir, args.train_csv, max_h=args.max_h, max_w=args.max_w, train=True)
    val_ds   = VarSizeDataset(args.val_dir,   args.val_csv,   max_h=args.max_h, max_w=args.max_w, train=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, collate_fn=collate_varsize)
    val_dl   = DataLoader(val_ds,   batch_size=args.val_batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True, collate_fn=collate_varsize)

    model = CountNetMasked().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and (device=="cuda"))

    def evaluate():
        model.eval()
        v_losses=[]; v_mae=[]; v_rmse=[]; v_exact=[]
        with torch.no_grad():
            for xb, mb, yb in val_dl:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                yp = model(xb, mb)
                v_losses.append(loss_fn(yp, yb).item())
                p = yp.detach().cpu().numpy().ravel()
                t = yb.detach().cpu().numpy().ravel()
                v_mae.append(np.mean(np.abs(p - t)))
                v_rmse.append(np.sqrt(np.mean((p - t)**2)))
                v_exact.append(np.mean((np.rint(p).astype(int) == t.astype(int))))
        return float(np.mean(v_losses)), float(np.mean(v_mae)), float(np.mean(v_rmse)), float(np.mean(v_exact))*100

    best = 1e9
    for ep in range(1, args.epochs+1):
        model.train(); t0=time.time(); losses=[]
        for xb, mb, yb in train_dl:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    yp = model(xb, mb)
                    loss = loss_fn(yp, yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                yp = model(xb, mb); loss = loss_fn(yp, yb); loss.backward(); opt.step()
            losses.append(loss.item())
        tr = float(np.mean(losses))
        vl, mae, rmse, ex = evaluate()
        print(f"Epoch {ep:02d} | train {tr:.3f} | val {vl:.3f} | MAE {mae:.2f} | RMSE {rmse:.2f} | Exact {ex:.1f}% | {time.time()-t0:.1f}s")
        if vl < best:
            best = vl
            torch.save({
                "model": model.state_dict(),
                "normalize_mean": [0.5,0.5,0.5],
                "normalize_std":  [0.5,0.5,0.5],
                "max_h": args.max_h,
                "max_w": args.max_w,
            }, args.out)
            print("  Saved", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_dir",   type=str, required=True)
    ap.add_argument("--val_csv",   type=str, required=True)
    ap.add_argument("--out",       type=str, default="letter_counter.pt")
    ap.add_argument("--max_h",     type=int, default=80, help="Max allowed height; images > max_h are downscaled")
    ap.add_argument("--max_w",     type=int, default=512, help="Clamp very long widths (None to disable)")
    ap.add_argument("--epochs",    type=int, default=12)
    ap.add_argument("--batch_size",type=int, default=64)
    ap.add_argument("--val_batch", type=int, default=256)
    ap.add_argument("--lr",        type=float, default=2e-4)
    ap.add_argument("--workers",   type=int, default=2)
    ap.add_argument("--amp",       action="store_true")
    args = ap.parse_args(); train(args)
