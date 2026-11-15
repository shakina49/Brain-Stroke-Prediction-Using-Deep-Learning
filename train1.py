# =========================================================
# Brain Stroke Classification – Final Tuned Version
# Target Accuracy: 95%+
# Dataset: 10k Normal / 10k Stroke
# =========================================================

import os, copy, random, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import cv2

# ===================== CONFIG =====================
DATA_DIR = "Brain_Data_Organised"
OUT_DIR  = "outputs_stroke_final"

IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_FOLDS = 3

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 7
WARMUP_EPOCHS = 4

SEED = 42
MODELS = ["resnet50", "efficientnet_b3"]

USE_PRETRAINED = True
FREEZE_BACKBONE = True
SKULL_STRIP = True

# =================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)
device = torch.device("cpu")

# ===================== OpenCV PREPROCESS =====================
def cv2_clahe_rgb(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def adaptive_skull_strip(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    return th

class OpenCVPreprocess:
    def __call__(self, img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2_clahe_rgb(img)
        if SKULL_STRIP:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = adaptive_skull_strip(gray)
            img = cv2.bitwise_and(img, img, mask=mask)
        return transforms.functional.to_pil_image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )

# ===================== TRANSFORMS =====================
train_tfms = transforms.Compose([
    OpenCVPreprocess(),
    transforms.Resize((IMG_SIZE+16, IMG_SIZE+16)),
    transforms.RandomAffine(7, translate=(0.03,0.03), scale=(0.95,1.05)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    OpenCVPreprocess(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===================== MODELS =====================
def build_resnet50(n):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, n))
    return m

def build_effnet_b3(n):
    m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    return m

def get_model(name, n):
    return build_resnet50(n) if name=="resnet50" else build_effnet_b3(n)

# ===================== TRAIN / EVAL =====================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    correct = total = 0
    for x,y in tqdm(loader, leave=False):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)
    return correct/total

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    correct = total = 0
    probs_all, labels_all = [], []
    for x,y in tqdm(loader, leave=False):
        x,y = x.to(device), y.to(device)
        out = model(x)
        probs = torch.softmax(out,1)
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)
        probs_all.append(probs.cpu().numpy())
        labels_all.append(y.cpu().numpy())
    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    auc = roc_auc_score(labels_all, probs_all[:,1])
    return correct/total, auc

# ===================== MAIN =====================
def main():
    Path(OUT_DIR).mkdir(exist_ok=True)
    base_ds = datasets.ImageFolder(DATA_DIR, transform=val_tfms)
    y = np.array([s[1] for s in base_ds.samples])

    skf = StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=SEED)

    for fold,(tr,va) in enumerate(skf.split(np.arange(len(y)),y),1):
        print(f"\n===== FOLD {fold} =====")

        train_ds = Subset(datasets.ImageFolder(DATA_DIR, train_tfms), tr)
        val_ds   = Subset(datasets.ImageFolder(DATA_DIR, val_tfms), va)

        train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(val_ds, BATCH_SIZE)

        models_fold = []

        for name in MODELS:
            print(f"\n--- {name.upper()} ---")
            model = get_model(name, 2).to(device)

            if FREEZE_BACKBONE:
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.parameters():
                    p.requires_grad = True

            opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, NUM_EPOCHS, 1e-6)
            crit = nn.CrossEntropyLoss(label_smoothing=0.1)

            best = 0
            for e in range(NUM_EPOCHS):
                tr_acc = train_epoch(model, train_dl, opt, crit)
                va_acc, va_auc = eval_epoch(model, val_dl)
                sch.step()
                print(f"E{e+1:02d} | train={tr_acc:.4f} val={va_acc:.4f} auc={va_auc:.4f}")
                if va_acc > best:
                    best = va_acc
                    torch.save(model.state_dict(), f"{OUT_DIR}/{name}_fold{fold}.pth")

            model.load_state_dict(torch.load(f"{OUT_DIR}/{name}_fold{fold}.pth"))
            model.eval()
            models_fold.append(model)

        # ========== ENSEMBLE ==========
        probs, labels = [], []
        with torch.no_grad():
            for x,yb in val_dl:
                x = x.to(device)
                p = sum(torch.softmax(m(x),1) for m in models_fold)/len(models_fold)
                probs.append(p.cpu().numpy())
                labels.append(yb.numpy())

        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        preds = probs.argmax(1)

        print("\nENSEMBLE RESULTS")
        print("Accuracy:", (preds==labels).mean())
        print("AUC:", roc_auc_score(labels, probs[:,1]))
        print(classification_report(labels, preds, target_names=base_ds.classes))

if __name__ == "__main__":
    main()
