import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from PIL import Image
from datetime import datetime

# ===========================
# CONFIG
# ===========================
DATASET_PATHS = [
    "datasets/processed",       # original 859 curated images
    "datasets/video_labeled",   # 1433 video frames — adds real-world diversity
]
MODEL_DIR     = "models"
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 25
LR_HEAD       = 1e-3
LR_FINETUNE   = 1e-4
FREEZE_EPOCHS = 10
MIXUP_ALPHA   = 0.2
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["CLOSE", "MEDIUM", "WIDE"]
LABEL_MAP   = {"CLOSE": 0, "MEDIUM": 1, "WIDE": 2}

torch.manual_seed(SEED)
np.random.seed(SEED)

# ===========================
# DATASET
# ===========================
class ShotScaleDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ===========================
# AUGMENTATION
# Stronger than before — simulates real video conditions:
#   GaussianBlur  → motion blur / soft focus in video
#   RandomRotation → slight camera tilt
#   RandomErasing  → occlusion / partial frames
#   Stronger ColorJitter → varied lighting across video scenes
# ===========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomRotation(degrees=5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ===========================
# LOSS: label-smoothed cross-entropy
# ===========================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls       = classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.cls - 1)
        one_hot    = torch.full_like(pred, smooth_val)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        log_prob   = nn.functional.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()


# ===========================
# MIXUP
# ===========================
def mixup_batch(imgs, labels, alpha=0.2):
    lam      = np.random.beta(alpha, alpha)
    idx      = torch.randperm(imgs.size(0), device=imgs.device)
    mixed    = lam * imgs + (1 - lam) * imgs[idx]
    labels_b = labels[idx]
    return mixed, labels, labels_b, lam


# ===========================
# TRAINING HELPERS
# ===========================
def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")

def unfreeze_backbone(model, layers_to_unfreeze=3):
    # Unfreezing 3 layers (was 2) — better video generalisation
    unfreeze_names = [f"layer{5 - i}" for i in range(layers_to_unfreeze)] + ["fc"]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(u) for u in unfreeze_names)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Unfroze: {unfreeze_names}  |  Trainable params: {trainable:,}")


def run_epoch(model, loader, criterion, optimizer=None, train=True, use_mixup=False):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if train:
                optimizer.zero_grad()

            if train and use_mixup:
                imgs, labels_a, labels_b, lam = mixup_batch(imgs, labels, MIXUP_ALPHA)
                out  = model(imgs)
                loss = lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b)
                preds = out.argmax(dim=1)
                correct += (lam * (preds == labels_a).float() +
                            (1 - lam) * (preds == labels_b).float()).sum().item()
                all_targets.extend(labels_a.cpu().numpy())
            else:
                out   = model(imgs)
                loss  = criterion(out, labels)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                all_targets.extend(labels.cpu().numpy())

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_targets


# ===========================
# SAFE MODEL SAVE — never overwrites
# Format: models/resnet18_revamped_f1{score}_{timestamp}.pt
# ===========================
def save_model(model_state, class_names, img_size, best_f1, mixup_alpha, num_epochs):
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    f1_tag    = f"{best_f1:.4f}".replace(".", "")
    filename  = f"resnet18_revamped_f1{f1_tag}_{timestamp}.pt"
    save_path = os.path.join(MODEL_DIR, filename)
    torch.save({
        "model_state_dict": model_state,
        "class_names":      class_names,
        "img_size":         img_size,
        "mixup_alpha":      mixup_alpha,
        "num_epochs":       num_epochs,
        "val_macro_f1":     best_f1,
        "timestamp":        timestamp,
        "trained_on":       DATASET_PATHS,
    }, save_path)
    print(f"💾 Model saved → {save_path}")
    return save_path


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":

    print(f"🖥️  Device: {DEVICE}\n")

    # --- Collect paths + labels from ALL dataset sources ---
    print("📂 Scanning datasets...")
    all_paths, all_labels = [], []

    for dataset_path in DATASET_PATHS:
        source_count = 0
        for label in CLASS_NAMES:
            folder = os.path.join(dataset_path, label)
            if not os.path.exists(folder):
                print(f"   ⚠️  Skipping missing folder: {folder}")
                continue
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    all_paths.append(fpath)
                    all_labels.append(LABEL_MAP[label])
                    source_count += 1
        print(f"   {dataset_path}: {source_count} images loaded")

    all_labels = np.array(all_labels)
    print(f"\n   Total images : {len(all_paths)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"   {name:8s}     : {(all_labels == i).sum()}")

    # --- Train / val split ---
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=0.2, random_state=SEED, stratify=all_labels
    )
    print(f"\nTrain: {len(train_paths)}  |  Val: {len(val_paths)}\n")

    # --- Class-weighted sampler ---
    class_counts   = np.bincount(train_labels)
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_labels),
        replacement=True
    )

    train_dataset = ShotScaleDataset(train_paths, train_labels, train_transform)
    val_dataset   = ShotScaleDataset(val_paths,   val_labels,   val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=0, pin_memory=False)

    # --- Model ---
    print("🏗️  Building ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 3)
    )
    model = model.to(DEVICE)

    criterion = LabelSmoothingLoss(classes=3, smoothing=0.1)

    # ===========================
    # PHASE 1: Head-only, no MixUp
    # ===========================
    print(f"\n{'='*50}")
    print(f"PHASE 1: Head-only, no MixUp ({FREEZE_EPOCHS} epochs)")
    print(f"{'='*50}")

    freeze_backbone(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR_HEAD, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)

    best_val_f1      = 0.0
    best_state       = None
    history          = []

    for epoch in range(1, FREEZE_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _       = run_epoch(model, train_loader, criterion,
                                                optimizer, train=True, use_mixup=False)
        va_loss, va_acc, va_p, va_t = run_epoch(model, val_loader, criterion, train=False)
        scheduler.step()

        va_f1 = f1_score(va_t, va_p, average="macro")
        history.append({"epoch": epoch, "phase": "1-clean",
                        "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "va_loss": va_loss, "va_acc": va_acc, "va_f1": va_f1})

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " ✅ best"
        else:
            tag = ""

        print(f"Ep {epoch:02d}/{FREEZE_EPOCHS}  "
              f"tr_loss:{tr_loss:.3f} tr_acc:{tr_acc:.3f}  "
              f"va_loss:{va_loss:.3f} va_acc:{va_acc:.3f} va_f1:{va_f1:.3f}"
              f"  [{time.time()-t0:.1f}s]{tag}")

    # ===========================
    # PHASE 2: Fine-tune 3 blocks + MixUp
    # ===========================
    FINETUNE_EPOCHS = NUM_EPOCHS - FREEZE_EPOCHS

    print(f"\n{'='*50}")
    print(f"PHASE 2: Fine-tuning (3 blocks) + MixUp alpha={MIXUP_ALPHA} ({FINETUNE_EPOCHS} epochs)")
    print(f"{'='*50}")

    unfreeze_backbone(model, layers_to_unfreeze=3)  # was 2 — more backbone for video generalisation
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR_FINETUNE, weight_decay=2e-4)  # was 1e-4
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS,
                                                      eta_min=1e-6)

    patience         = 8
    epochs_no_improv = 0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _       = run_epoch(model, train_loader, criterion,
                                                optimizer, train=True, use_mixup=True)
        va_loss, va_acc, va_p, va_t = run_epoch(model, val_loader, criterion, train=False)
        scheduler.step()

        va_f1 = f1_score(va_t, va_p, average="macro")
        history.append({"epoch": FREEZE_EPOCHS + epoch, "phase": "2-mixup",
                        "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "va_loss": va_loss, "va_acc": va_acc, "va_f1": va_f1})

        if va_f1 > best_val_f1:
            best_val_f1      = va_f1
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improv = 0
            tag = " ✅ best"
        else:
            epochs_no_improv += 1
            tag = f" (no improv {epochs_no_improv}/{patience})"

        print(f"Ep {FREEZE_EPOCHS+epoch:02d}/{NUM_EPOCHS}  "
              f"tr_loss:{tr_loss:.3f} tr_acc:{tr_acc:.3f}  "
              f"va_loss:{va_loss:.3f} va_acc:{va_acc:.3f} va_f1:{va_f1:.3f}"
              f"  [{time.time()-t0:.1f}s]{tag}")

        if epochs_no_improv >= patience:
            print(f"\n⏹️  Early stopping at epoch {FREEZE_EPOCHS + epoch}")
            break

    # ===========================
    # FINAL EVALUATION
    # ===========================
    print(f"\n{'='*50}")
    print("📈 FINAL EVALUATION (best checkpoint)")
    print(f"{'='*50}")

    model.load_state_dict(best_state)
    _, _, final_preds, final_targets = run_epoch(model, val_loader, criterion, train=False)

    print(classification_report(final_targets, final_preds, target_names=CLASS_NAMES))

    print("📉 CONFUSION MATRIX")
    cm = confusion_matrix(final_targets, final_preds)
    print(cm)

    print("\nPer-class summary:")
    for i, name in enumerate(CLASS_NAMES):
        tp     = cm[i, i]
        fn     = cm[i].sum() - tp
        fp     = cm[:, i].sum() - tp
        prec   = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        print(f"  {name:8s}  precision:{prec:.2f}  recall:{recall:.2f}  "
              f"(misclassified as: "
              f"{[f'{CLASS_NAMES[j]}:{cm[i,j]}' for j in range(3) if j != i and cm[i,j] > 0]})")

    # --- Training history ---
    print(f"\n{'='*50}")
    print("📊 TRAINING HISTORY")
    print(f"{'='*50}")
    df_hist = pd.DataFrame(history)
    print(df_hist[["epoch", "phase", "tr_loss", "tr_acc",
                   "va_loss", "va_acc", "va_f1"]].to_string(index=False))
    print(f"\nBest val macro-F1 : {best_val_f1:.4f}")

    # --- Save — guaranteed unique filename, never overwrites ---
    saved_path = save_model(
        model_state = best_state,
        class_names = CLASS_NAMES,
        img_size    = IMG_SIZE,
        best_f1     = best_val_f1,
        mixup_alpha = MIXUP_ALPHA,
        num_epochs  = NUM_EPOCHS,
    )
    print(f"\n✅ Done. Load this model with:")
    print(f"   checkpoint = torch.load('{saved_path}')")
    print(f"   model.load_state_dict(checkpoint['model_state_dict'])")