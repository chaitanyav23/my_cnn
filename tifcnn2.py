import os, math
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

# =========================
# Config
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 48
EPOCHS = 40
LR = 7e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 16
IMG_SIZE = (64, 128)

DATA_ROOT = "./dataset_music"

print("DATA_ROOT:", os.path.abspath(DATA_ROOT))
print("Train exists:", os.path.exists(os.path.join(DATA_ROOT, "train")))
print("Metadata exists:",
      os.path.exists(os.path.join(DATA_ROOT, "train", "metadata.csv")))

# =========================
# Core ops
# =========================
def leaky_relu(x, alpha=0.01):
    return torch.where(x > 0, x, alpha * x)

def dropout(x, p, training):
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1 - p)

def maxpool2x2(x):
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x, _ = x.max(dim=3)
    x, _ = x.max(dim=4)
    return x

# =========================
# Layers
# =========================
class Conv2D:
    def __init__(self, in_c, out_c, k, padding=0):
        self.k = k
        self.padding = padding
        std = math.sqrt(2.0 / (in_c * k * k))
        self.weight = torch.nn.Parameter(
            torch.randn(out_c, in_c, k, k, device=DEVICE) * std
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_c, device=DEVICE))

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (self.padding,) * 4)
        b, c, h, w = x.shape
        oh, ow = h - self.k + 1, w - self.k + 1
        out = torch.zeros(b, self.weight.size(0), oh, ow, device=x.device)
        for i in range(self.k):
            for j in range(self.k):
                out += torch.einsum(
                    "bchw,oc->bohw",
                    x[:, :, i:i+oh, j:j+ow],
                    self.weight[:, :, i, j]
                )
        return out + self.bias.view(1, -1, 1, 1)

    def parameters(self):
        return [self.weight, self.bias]

class Linear:
    def __init__(self, in_f, out_f):
        std = math.sqrt(2.0 / in_f)
        self.weight = torch.nn.Parameter(
            torch.randn(out_f, in_f, device=DEVICE) * std
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_f, device=DEVICE))

    def forward(self, x):
        return x @ self.weight.t() + self.bias

    def parameters(self):
        return [self.weight, self.bias]

class CustomBatchNorm:
    def __init__(self, c, momentum=0.05, eps=1e-5):
        self.gamma = torch.nn.Parameter(torch.ones(1, c, 1, 1, device=DEVICE))
        self.beta  = torch.nn.Parameter(torch.zeros(1, c, 1, 1, device=DEVICE))
        self.running_mean = torch.zeros(1, c, 1, 1, device=DEVICE)
        self.running_var  = torch.ones(1, c, 1, 1, device=DEVICE)
        self.momentum = momentum
        self.eps = eps
        self.training = True

    def train(self): self.training = True
    def eval(self):  self.training = False

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0,2,3), keepdim=True)
            var  = x.var(dim=(0,2,3), keepdim=True, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
                self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

class InceptionBlock:
    def __init__(self, in_c, c=48):
        self.b1 = Conv2D(in_c, c, 1)
        self.b2_1 = Conv2D(in_c, c, 1)
        self.b2_2 = Conv2D(c, c, 3, padding=1)
        self.b3_1 = Conv2D(in_c, c, 1)
        self.b3_2 = Conv2D(c, c, 3, padding=1)
        self.b3_3 = Conv2D(c, c, 3, padding=1)
        self.b4 = Conv2D(in_c, c, 1)
        self.bn = CustomBatchNorm(4*c)

    def forward(self, x):
        y1 = leaky_relu(self.b1.forward(x))
        y2 = leaky_relu(self.b2_2.forward(leaky_relu(self.b2_1.forward(x))))
        y3 = leaky_relu(self.b3_3.forward(
             leaky_relu(self.b3_2.forward(leaky_relu(self.b3_1.forward(x))))))
        y4 = leaky_relu(self.b4.forward(x))
        out = torch.cat([y1, y2, y3, y4], dim=1)
        return leaky_relu(self.bn.forward(out))

    def parameters(self):
        return (
            self.b1.parameters() +
            self.b2_1.parameters() + self.b2_2.parameters() +
            self.b3_1.parameters() + self.b3_2.parameters() + self.b3_3.parameters() +
            self.b4.parameters() + self.bn.parameters()
        )

# =========================
# Model
# =========================
class TIF_CNN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = Conv2D(1, 16, 3, padding=1)
        self.bn1 = CustomBatchNorm(16)
        self.c2 = Conv2D(16, 32, 3, padding=1)
        self.bn2 = CustomBatchNorm(32)

        self.inc1 = InceptionBlock(96)
        self.inc2 = InceptionBlock(192)
        self.inc3 = InceptionBlock(192)
        self.inc4 = InceptionBlock(192)

        self.fc1 = Linear(192, 192)
        self.fc2 = Linear(192, 96)
        self.fc3 = Linear(96, NUM_CLASSES)

    def forward_branch(self, x):
        x = leaky_relu(self.bn1.forward(self.c1.forward(x)))
        x = maxpool2x2(x)
        x = leaky_relu(self.bn2.forward(self.c2.forward(x)))
        x = maxpool2x2(x)
        return x

    def forward(self, x1, x2, x3):
        x = torch.cat([
            self.forward_branch(x1),
            self.forward_branch(x2),
            self.forward_branch(x3)
        ], dim=1)
        x = self.inc1.forward(x)
        x = self.inc2.forward(x)
        x = maxpool2x2(x)
        x = self.inc3.forward(x)
        x = self.inc4.forward(x)
        x = x.mean(dim=(2,3))
        x = leaky_relu(self.fc1.forward(x))
        x = dropout(x, 0.30, self.training)
        x = leaky_relu(self.fc2.forward(x))
        x = dropout(x, 0.25, self.training)
        return self.fc3.forward(x)

    def parameters(self):
        params = []
        for m in [
            self.c1, self.bn1, self.c2, self.bn2,
            self.inc1, self.inc2, self.inc3, self.inc4,
            self.fc1, self.fc2, self.fc3
        ]:
            params += m.parameters()
        return params

# =========================
# Dataset
# =========================
class MusicDataset(Dataset):
    def __init__(self, root, split):
        self.base = os.path.join(root, split)
        self.df = pd.read_csv(os.path.join(self.base, "metadata.csv"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs = []
        for i in [1,2,3]:
            p = os.path.join(self.base, f"input_{i}", os.path.basename(row[f"input_{i}"]))
            img = Image.open(p).convert("L").resize(IMG_SIZE[::-1])
            imgs.append(torch.tensor(np.array(img)/255.0,
                        dtype=torch.float32).unsqueeze(0))
        return imgs[0], imgs[1], imgs[2], int(row["target"])

# =========================
# Training
# =========================
train_ds = MusicDataset(DATA_ROOT, "train")
split = int(0.85 * len(train_ds))
train_ds, val_ds = torch.utils.data.random_split(train_ds, [split, len(train_ds)-split])

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds, BATCH_SIZE)

targets = [train_ds[i][3] for i in range(len(train_ds))]
cnt = Counter(targets)
class_weights = torch.tensor(
    [len(targets)/(NUM_CLASSES*cnt[i]) for i in range(NUM_CLASSES)],
    device=DEVICE
)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

model = TIF_CNN2().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_f1, patience, counter = 0, 7, 0

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    train_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
    for x1,x2,x3,y in train_bar:
        x1,x2,x3,y = x1.to(DEVICE),x2.to(DEVICE),x3.to(DEVICE),y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x1,x2,x3)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_acc = correct / total
    scheduler.step()

    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for x1,x2,x3,y in val_dl:
            out = model(x1.to(DEVICE),x2.to(DEVICE),x3.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
            labs.extend(y.numpy())

    preds, labs = np.array(preds), np.array(labs)
    val_acc = (preds == labs).mean()
    val_f1  = f1_score(labs, preds, average="macro")

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Acc: {train_acc:.3f} | "
        f"Val Acc: {val_acc:.3f} | "
        f"Val F1: {val_f1:.3f}"
    )

    if val_f1 > best_f1 + 1e-3:
        best_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), "best_model_tifcnn2.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break

