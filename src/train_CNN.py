import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# =====================================================
# Dataset
# =====================================================
class FruitDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)
        self.num_classes = len(self.le.classes_)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]

        if not os.path.exists(path):
            return None

        img = cv2.imread(path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract HSV features
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        hsv_feat = torch.tensor([h.mean(), s.mean(), v.mean()], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, hsv_feat, label


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, hsvs, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(hsvs), torch.tensor(labels)


# =====================================================
# Model
# =====================================================
class CNN_HSV_MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.resnet18(weights="DEFAULT")

        # Initially train only layer4
        for name, param in self.backbone.named_parameters():
            if not any(k in name for k in ["layer4"]):
                param.requires_grad = False

        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.hsv_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_img, x_hsv):
        img_feat = self.backbone(x_img)
        hsv_feat = self.hsv_mlp(x_hsv.float())
        x = torch.cat([img_feat, hsv_feat], dim=1)
        return self.classifier(x)


# =====================================================
# Train / Eval
# =====================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, tot_epochs):
    model.train()
    running_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:>2}/{tot_epochs}", dynamic_ncols=True)

    for batch in pbar:
        if batch is None:
            continue

        imgs, hsvs, labels = batch
        imgs = imgs.to(device, dtype=torch.float32)
        hsvs = hsvs.to(device, dtype=torch.float32)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            out = model(imgs, hsvs)
            loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / n


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for batch in loader:
        if batch is None:
            continue

        imgs, hsvs, labels = batch
        imgs = imgs.to(device, dtype=torch.float32)
        hsvs = hsvs.to(device, dtype=torch.float32)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            out = model(imgs, hsvs)

        preds = out.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return acc, prec, rec, f1, y_true, y_pred


# =====================================================
# Main: 5-Fold CV + Train Loss Plot
# =====================================================
def unfreeze_layers(model, epoch):
    """
    Layer-wise Unfreezing Strategy:
    Epoch 1-3  : Train only classifier + hsv_mlp + layer4
    Epoch 4-6  : Unfreeze layer3
    Epoch >=7  : Unfreeze layer2
    Returns: bool indicating if unfreezing occurred
    """
    unfrozen = False

    if epoch == 4:
        for name, param in model.backbone.named_parameters():
            if "layer3" in name:
                param.requires_grad = True
        print("[Unfreeze] layer3 unfrozen")
        unfrozen = True

    if epoch == 7:
        for name, param in model.backbone.named_parameters():
            if "layer2" in name:
                param.requires_grad = True
        print("[Unfreeze] layer2 unfrozen")
        unfrozen = True

    return unfrozen

def main():
    CSV_FILE = "../dataset.csv"
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-3
    NUM_WORKERS = 4
    N_FOLDS = 5
    RANDOM_STATE = 42

    # Check for Apple Silicon GPU (MPS) or CUDA, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(CSV_FILE)
    paths = '../data/processed/' + df["path"].values
    labels = (df["fruit"] + "_" + df["condition"]).values

    full_dataset = FruitDataset(paths, labels, transform)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_records = []
    all_folds_train_loss = []  # Store [fold, epoch] losses

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_dataset.img_paths, full_dataset.labels), start=1):

        print(f"\n========== Fold {fold}/{N_FOLDS} ==========")

        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, collate_fn=collate_fn
        )

        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=collate_fn
        )

        model = CNN_HSV_MLP(full_dataset.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer with only parameters that require gradients
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

        best_acc = 0.0
        epoch_train_losses = []

        for epoch in range(1, EPOCHS + 1):
            # ===== Layer-wise Unfreezing =====
            did_unfreeze = unfreeze_layers(model, epoch)

            # ===== Adjust LR after unfreezing =====
            if did_unfreeze:
                LR_NEW = LR * 0.1
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=LR_NEW
                )
                print(f"[LR Adjust] Learning rate reduced to {LR_NEW:.1e}")
            else:
                # Re-check optimizer if parameters were unfrozen elsewhere
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
                )

            tr_loss = train_one_epoch(
                model, train_loader, criterion,
                optimizer, device, epoch, EPOCHS
            )
            epoch_train_losses.append(tr_loss)

            acc, _, _, _, _, _ = evaluate_metrics(model, val_loader, device)
            print(f"Validation Accuracy: {acc:.4f}, Train Loss: {tr_loss:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"../model/CNN_only/best_fold{fold}.pth")

        all_folds_train_loss.append(epoch_train_losses)

        # Load best model for this fold and evaluate
        model.load_state_dict(torch.load(f"../model/CNN_only/best_fold{fold}.pth"))
        acc, prec, rec, f1, y_true, y_pred = evaluate_metrics(model, val_loader, device)

        print(classification_report(
            y_true, y_pred,
            target_names=full_dataset.le.classes_,
            digits=4
        ))

        cv_records.append({
            "fold": fold,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1
        })

    # ================= Train Loss Plot =================
    all_folds_train_loss = np.array(all_folds_train_loss)
    mean_loss = all_folds_train_loss.mean(axis=0)
    std_loss = all_folds_train_loss.std(axis=0)

    epochs = np.arange(1, EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, mean_loss, label="mean")
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("5-Fold Cross-Validation – Train Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../result/train_loss_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()