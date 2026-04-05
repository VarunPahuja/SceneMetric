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
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# ===========================
# CONFIG
# ===========================
DATASET_PATH  = "datasets/processed"
MODEL_OUT     = "models/resnet18_shotscale.pt"
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 25
LR_HEAD       = 1e-3    # head-only phase
LR_FINETUNE   = 1e-4    # full fine-tune phase
FREEZE_EPOCHS = 10      # epochs to train head only before unfreezing
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES   = ["CLOSE", "MEDIUM", "WIDE"]
LABEL_MAP     = {"CLOSE": 0, "MEDIUM": 1, "WIDE": 2}

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"🖥️  Device: {DEVICE}\n")

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
# ===========================
# Train: aggressive augmentation to fight overfitting on 859 images
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===========================
# COLLECT PATHS + LABELS
# ===========================
print("📂 Scanning dataset...")
all_paths, all_labels = [], []

for label in CLASS_NAMES:
    folder = os.path.join(DATASET_PATH, label)
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            all_paths.append(fpath)
            all_labels.append(LABEL_MAP[label])

all_labels = np.array(all_labels)
print(f"   Total images : {len(all_paths)}")
for i, name in enumerate(CLASS_NAMES):
    print(f"   {name:8s}     : {(all_labels == i).sum()}")

# ===========================
# TRAIN / VAL SPLIT
# ===========================
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.2, random_state=SEED, stratify=all_labels
)

print(f"\nTrain: {len(train_paths)}  |  Val: {len(val_paths)}\n")

# ===========================
# CLASS-WEIGHTED SAMPLER
# Oversamples MEDIUM during training so it's seen proportionally
# ===========================
class_counts  = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(train_labels),
    replacement=True
)

train_dataset = ShotScaleDataset(train_paths, train_labels, train_transform)
val_dataset   = ShotScaleDataset(val_paths,   val_labels,   val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False,  num_workers=0, pin_memory=True)

# ===========================
# MODEL: ResNet-18 + custom head
# ===========================
print("🏗️  Building ResNet-18...")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace final FC with a small head suited to 3 classes
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 3)
)
model = model.to(DEVICE)

# ===========================
# LOSS: label-smoothed cross-entropy
# Softens the labels slightly — helps generalisation on small datasets
# ===========================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls       = classes

    def forward(self, pred, target):
        confidence  = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.cls - 1)
        one_hot     = torch.full_like(pred, smooth_val)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        log_prob    = nn.functional.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()

criterion = LabelSmoothingLoss(classes=3, smoothing=0.1)

# ===========================
# TRAINING HELPERS
# ===========================
def freeze_backbone(model):
    """Freeze all layers except the FC head."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")

def unfreeze_backbone(model, layers_to_unfreeze=2):
    """
    Unfreeze the last N ResNet layer-groups + FC.
    ResNet-18 groups: layer1, layer2, layer3, layer4, fc
    Default: unfreeze layer4 + fc.
    """
    unfreeze_names = [f"layer{5 - i}" for i in range(layers_to_unfreeze)] + ["fc"]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(u) for u in unfreeze_names)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Unfroze: {unfreeze_names}  |  Trainable params: {trainable:,}")


def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()

            preds = out.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_targets


# ===========================
# PHASE 1: Train head only
# Backbone is frozen — fast convergence, avoids destroying pretrained weights
# ===========================
print(f"\n{'='*50}")
print(f"PHASE 1: Head-only training ({FREEZE_EPOCHS} epochs)")
print(f"{'='*50}")

freeze_backbone(model)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR_HEAD, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)

best_val_f1   = 0.0
best_state    = None
history       = []

for epoch in range(1, FREEZE_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc, _, _          = run_epoch(model, train_loader, criterion, optimizer, train=True)
    va_loss, va_acc, va_p, va_t    = run_epoch(model, val_loader,   criterion, train=False)
    scheduler.step()

    from sklearn.metrics import f1_score
    va_f1 = f1_score(va_t, va_p, average="macro")

    history.append({"epoch": epoch, "phase": 1,
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
# PHASE 2: Fine-tune last 2 blocks + head
# Lower LR to avoid destroying pretrained features
# ===========================
FINETUNE_EPOCHS = NUM_EPOCHS - FREEZE_EPOCHS

print(f"\n{'='*50}")
print(f"PHASE 2: Fine-tuning ({FINETUNE_EPOCHS} epochs)")
print(f"{'='*50}")

unfreeze_backbone(model, layers_to_unfreeze=2)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR_FINETUNE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)

patience        = 8
epochs_no_improv = 0

for epoch in range(1, FINETUNE_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc, _, _          = run_epoch(model, train_loader, criterion, optimizer, train=True)
    va_loss, va_acc, va_p, va_t    = run_epoch(model, val_loader,   criterion, train=False)
    scheduler.step()

    from sklearn.metrics import f1_score
    va_f1 = f1_score(va_t, va_p, average="macro")

    history.append({"epoch": FREEZE_EPOCHS + epoch, "phase": 2,
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
# FINAL EVALUATION (best checkpoint)
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

# Per-class breakdown
print("\nPer-class summary:")
for i, name in enumerate(CLASS_NAMES):
    tp = cm[i, i]
    fn = cm[i].sum() - tp
    fp = cm[:, i].sum() - tp
    prec   = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    print(f"  {name:8s}  precision:{prec:.2f}  recall:{recall:.2f}  "
          f"(misclassified as: "
          f"{[f'{CLASS_NAMES[j]}:{cm[i,j]}' for j in range(3) if j != i and cm[i,j] > 0]})")

# ===========================
# TRAINING HISTORY SUMMARY
# ===========================
print(f"\n{'='*50}")
print("📊 TRAINING HISTORY")
print(f"{'='*50}")
df_hist = pd.DataFrame(history)
print(df_hist[["epoch", "phase", "tr_loss", "tr_acc",
               "va_loss", "va_acc", "va_f1"]].to_string(index=False))
print(f"\nBest val macro-F1 : {best_val_f1:.4f}")

# ===========================
# SAVE
# ===========================
os.makedirs("models", exist_ok=True)

torch.save({
    "model_state_dict": best_state,
    "class_names":      CLASS_NAMES,
    "img_size":         IMG_SIZE,
    "val_macro_f1":     best_val_f1,
}, MODEL_OUT)

print(f"\n💾 Model saved → {MODEL_OUT}")