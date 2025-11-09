from pathlib import Path; import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = "/Users/dominopizzaaaa/Desktop/dev/captcha-recognition-system/data"
NPZ = os.path.join(DATA_DIR, "chars_train.npz")
CKPT = os.path.join(DATA_DIR, "charcnn.pt")

class CharCNN(nn.Module):
    def __init__(self, n_classes=36):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32->16
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8, 128), nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self,x):
        return self.head(self.feat(x))

def main():
    data = np.load(NPZ)
    X = torch.tensor(data["X"].transpose(0,3,1,2))  # (N,1,32,32)
    Y = torch.tensor(data["Y"])
    # simple split
    n = X.shape[0]; idx = torch.randperm(n)
    tr = int(0.9*n)
    Xtr, Ytr = X[idx[:tr]], Y[idx[:tr]]
    Xva, Yva = X[idx[tr:]], Y[idx[tr:]]

    tr_dl = DataLoader(TensorDataset(Xtr,Ytr), batch_size=256, shuffle=True, num_workers=2)
    va_dl = DataLoader(TensorDataset(Xva,Yva), batch_size=512, shuffle=False, num_workers=2)

    model = CharCNN().to("cpu")
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    best = 0
    for epoch in range(10):
        model.train()
        loss_sum = 0
        for xb,yb in tr_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*xb.size(0)
        tr_loss = loss_sum/len(tr_dl.dataset)

        model.eval()
        correct=0
        with torch.no_grad():
            for xb,yb in va_dl:
                pred = model(xb).argmax(1)
                correct += (pred==yb).sum().item()
        va_acc = correct/len(va_dl.dataset)

        print(f"Epoch {epoch:02d} | loss {tr_loss:.4f} | val_acc {va_acc:.4f}")
        if va_acc>best:
            best=va_acc
            torch.save(model.state_dict(), CKPT)
            print("  Saved best ->", CKPT)

if __name__ == "__main__":
    main()
