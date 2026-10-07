"""Fine-tune ResNet-18 on native-resolution 224 patches with strong robustness
augmentation (random JPEG quality, resize, blur). Optional hold-out of one fake
generator to measure generalisation:

    python -m train.train_cnn --epochs 5 --holdout stylegan
"""
import argparse, io, random
import numpy as np
import torch, torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score
from train.common import load_dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


class Patches(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        im = Image.open(path).convert("RGB")
        if self.train:  # robustness augmentation: this is what handles re-encoded / resized media
            if random.random() < 0.5:
                s = random.uniform(0.5, 1.0); im = im.resize((max(int(im.width * s), 224), max(int(im.height * s), 224)))
            if random.random() < 0.2:
                im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
            if random.random() < 0.6:
                b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(30, 95)); b.seek(0); im = Image.open(b).convert("RGB")
        w, h = im.size
        if min(w, h) < 224:
            im = im.resize((224, 224)); w = h = 224
        x = random.randint(0, w - 224) if self.train else (w - 224) // 2
        y = random.randint(0, h - 224) if self.train else (h - 224) // 2
        a = np.asarray(im.crop((x, y, x + 224, y + 224)), np.float32) / 255
        if self.train and random.random() < 0.5:
            a = a[:, ::-1]
        a = (a - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1), torch.tensor(float(label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--holdout", default=None, help="fake generator folder name to keep unseen")
    ap.add_argument("--max-per-class", type=int, default=3000)
    a = ap.parse_args()
    real, fakes = load_dataset("data")
    random.seed(0); random.shuffle(real)
    train_fake = [(p, 1) for g, v in fakes.items() if g != a.holdout for p in v][: a.max_per_class]
    train = [(p, 0) for p in real[: a.max_per_class]] + train_fake
    held = [(p, 1) for p in fakes.get(a.holdout, [])] + [(p, 0) for p in real[-300:]] if a.holdout else []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = resnet18(weights=ResNet18_Weights.DEFAULT); m.fc = nn.Linear(m.fc.in_features, 1); m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
    dl = DataLoader(Patches(train, True), batch_size=a.batch, shuffle=True, num_workers=2)
    for ep in range(a.epochs):
        m.train(); tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(m(x).squeeze(1), y); loss.backward(); opt.step(); tot += loss.item()
        print(f"epoch {ep+1}/{a.epochs} loss {tot/len(dl):.4f}")
        if held:
            m.eval(); P, Y = [], []
            with torch.no_grad():
                for x, y in DataLoader(Patches(held, False), batch_size=64):
                    P += torch.sigmoid(m(x.to(dev))).squeeze(1).cpu().tolist(); Y += y.tolist()
            print(f"  UNSEEN generator '{a.holdout}' AUC = {roc_auc_score(Y, P):.3f}")
    torch.save(m.cpu().state_dict(), "models/cnn.pt"); print("saved models/cnn.pt")


if __name__ == "__main__":
    main()
