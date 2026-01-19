# =====================================================
# One-Click 5-Fold LinearSVM Training + Save + Ensemble + Confusion Matrix
# =====================================================

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import joblib
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Config =================
N_FOLDS = 5
RANDOM_STATE = 42
BATCH_SIZE = 64
IMG_SIZE = 224
RESULT_DIR = "../result"
MODEL_DIR = "../model/CNN_only"
SVM_SAVE_DIR = "../model/CNN_SVM"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(SVM_SAVE_DIR, exist_ok=True)

# ================= Dataset =================
class FruitDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        import cv2
        self.img_paths = img_paths
        self.transform = transform
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        import cv2
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        hsv_feat = torch.tensor([h.mean(), s.mean(), v.mean()], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, hsv_feat, self.labels[idx]

def collate_fn(batch):
    imgs, hsvs, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(hsvs), torch.tensor(labels)

# ================= CNN Encoder =================
class CNN_HSV_Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights="DEFAULT")
        feat_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        self.backbone = backbone
        self.hsv_mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 16), torch.nn.ReLU()
        )
        self.out_dim = feat_dim + 16

    def forward(self, x_img, x_hsv):
        img_feat = self.backbone(x_img)
        hsv_feat = self.hsv_mlp(x_hsv)
        return torch.cat([img_feat, hsv_feat], dim=1)

# ================= Embedding Extraction =================
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    feats, labels = [], []
    for imgs, hsvs, y in tqdm(loader, desc="Extracting"):
        imgs, hsvs = imgs.to(device), hsvs.to(device)
        emb = model(imgs, hsvs)
        feats.append(emb.cpu().numpy())
        labels.append(y.numpy())
    return np.vstack(feats).astype(np.float64), np.hstack(labels)

# ================= Ensemble Class =================
class SVMEnsemble:
    def __init__(self):
        self.models = []

    def add_model(self, scaler, svm):
        self.models.append({'scaler': scaler, 'svm': svm})

    def predict(self, X):
        all_preds = []
        for m in self.models:
            X_scaled = m['scaler'].transform(X)
            y_pred = m['svm'].predict(X_scaled)
            all_preds.append(y_pred)
        all_preds = np.stack(all_preds, axis=0)
        y_final = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            y_final.append(most_common)
        return np.array(y_final)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

# ================= Helper: Plot Confusion Matrix =================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    # 使用整数编码作为 labels
    class_indices = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ================= Main =================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load dataset
    df = pd.read_csv("../dataset.csv")
    paths = '../data/processed/' + df["path"].values
    labels = (df["fruit"] + "_" + df["condition"]).values
    dataset = FruitDataset(paths, labels, transform)

    # 5-Fold split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(paths, dataset.labels))

    encoder = CNN_HSV_Encoder().to(device)
    all_records = []
    ensemble = SVMEnsemble()

    # ================= For Overall Confusion Matrix =================
    all_y_true, all_y_pred = [], []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n===== Fold {fold_id + 1} =====")

        tr_loader = DataLoader(Subset(dataset, tr_idx), batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=collate_fn)
        va_loader = DataLoader(Subset(dataset, va_idx), batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=collate_fn)

        # Load fold-specific CNN weights
        model_path = f"{MODEL_DIR}/best_fold{fold_id + 1}.pth"
        encoder.load_state_dict(torch.load(model_path), strict=False)

        # Extract embeddings
        X_tr, y_tr = extract_embeddings(encoder, tr_loader, device)
        X_va, y_va = extract_embeddings(encoder, va_loader, device)

        # Standardize
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        # Train LinearSVM
        clf = LinearSVC(dual=False, max_iter=10000)
        clf.fit(X_tr_scaled, y_tr)
        y_pred = clf.predict(X_va_scaled)

        # ================= Metrics =================
        acc = accuracy_score(y_va, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_va, y_pred,
                                                           average="macro", zero_division=0)

        all_records.append({
            "fold": fold_id + 1,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1
        })
        print(f"Fold {fold_id + 1} | Acc: {acc:.4f}, F1: {f1:.4f}")

        # ================= Save Single Fold Model =================
        single_model_path = os.path.join(SVM_SAVE_DIR, f"SVM_fold{fold_id + 1}.joblib")
        joblib.dump({"scaler": scaler, "svm": clf}, single_model_path)
        print(f"[Saved] {single_model_path}")

        # ================= Confusion Matrix (Single Fold) =================
        class_names = dataset.le.classes_  # 类名
        cm_path = os.path.join(RESULT_DIR, f"SVM_confusion_matrix_fold{fold_id + 1}.png")
        plot_confusion_matrix(y_va, y_pred, class_names, cm_path)
        print(f"[Saved Confusion Matrix] {cm_path}")

        # Add to ensemble
        ensemble.add_model(scaler, clf)

        # Collect for overall confusion matrix
        all_y_true.append(y_va)
        all_y_pred.append(y_pred)

    # ================= Save Ensemble Model =================
    ensemble_path = os.path.join(SVM_SAVE_DIR, "SVM_ensemble.joblib")
    ensemble.save(ensemble_path)
    print(f"\n[Fusion Model Saved] {ensemble_path}")

    # ================= Save 5-Fold Results =================
    df_res = pd.DataFrame(all_records)
    mean_row = df_res.mean(numeric_only=True)
    mean_row["fold"] = "mean"
    std_row = df_res.std(numeric_only=True)
    std_row["fold"] = "std"
    df_res = pd.concat([df_res, pd.DataFrame([mean_row]), pd.DataFrame([std_row])],
                       ignore_index=True)
    df_res.to_csv(f"{RESULT_DIR}/SVM_5fold.csv", index=False)
    print("\n===== 5-Fold Results =====")
    print(df_res)

    # ================= Overall Confusion Matrix =================
    all_y_true = np.hstack(all_y_true)
    all_y_pred = np.hstack(all_y_pred)
    overall_cm_path = os.path.join(RESULT_DIR, "SVM_confusion_matrix_overall.png")
    plot_confusion_matrix(all_y_true, all_y_pred, class_names, overall_cm_path)
    print(f"[Saved Overall Confusion Matrix] {overall_cm_path}")

if __name__ == "__main__":
    main()
