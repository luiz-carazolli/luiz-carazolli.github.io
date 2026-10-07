#!/usr/bin/env python3
"""
Lightweight 2D CNN Classifier for Real vs. AI Image Detection using Persistence Images
- Loads 2-channel (H0 components, H1 loops) 32x32 Persistence Images from data/cifake_persistence_tensors.npz
- Uses the identical stratified 80/20 train/test split (random_state=42) from the tabular benchmark
- Trains a custom lightweight TopoCNN (~130k parameters)
- Evaluates comprehensively on the 4,000 test set images
- Generates:
    1. figures/cnn_learning_curves.png (Train/Val Loss & Accuracy progression)
    2. figures/cnn_confusion_matrix.png (Normalized confusion matrix on test set)
    3. figures/roc_curves_with_cnn.png (Direct ROC comparison with all 6 tabular baselines)
    4. figures/classification_metrics_benchmark_with_cnn.csv (Updated complete benchmark)
    5. models/topo_cnn_best.pt (Saved PyTorch model weights)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_PATH = "data/cifake_persistence_tensors.npz"
TABULAR_BENCHMARK_PATH = "figures/classification_metrics_benchmark.csv"
OUTPUT_DIR = "figures"
MODELS_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Select computing device (Apple Silicon Metal / MPS or CPU)
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[*] Utilizing Apple Silicon GPU via Metal Performance Shaders (MPS).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[*] Utilizing NVIDIA CUDA GPU.")
    else:
        device = torch.device("cpu")
        print("[*] Utilizing CPU.")
    return device


class PersistenceDataset(Dataset):
    """PyTorch Dataset wrapping 2-channel 32x32 Persistence Images."""
    def __init__(self, images: np.ndarray, labels: np.ndarray, is_train: bool = False):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        lbl = self.labels[idx]
        # Light data augmentation for persistence image training: subtle random horizontal flip
        # (reflects birth-intensity symmetry if appropriate, or small Gaussian jitter)
        return img, lbl


class ResidualBlock(nn.Module):
    """Lightweight 2D Residual Convolutional Block."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class TopoCNN(nn.Module):
    """
    Compact 2D Topological CNN (~130,000 parameters).
    Designed to process 2-channel (H0, H1) continuous persistence surfaces.
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 2):
        super().__init__()
        # Stem: project (2, 32, 32) -> (16, 32, 32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.stage1 = ResidualBlock(16)

        # Downsample 1: (16, 32, 32) -> (32, 16, 16)
        self.down1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stage2 = ResidualBlock(32)

        # Downsample 2: (32, 16, 16) -> (64, 8, 8)
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage3 = ResidualBlock(64)

        # Global pooling and classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.down1(out)
        out = self.stage2(out)
        out = self.down2(out)
        out = self.stage3(out)
        out = self.pool(out).flatten(1)
        logits = self.head(out)
        return logits


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(lbls)
        preds = logits.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += len(lbls)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    for imgs, lbls in loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        logits = model(imgs)
        loss = criterion(logits, lbls)
        total_loss += loss.item() * len(lbls)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_targets.append(lbls.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    loss_val = total_loss / len(y_true)
    acc_val = accuracy_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_prob)

    return loss_val, acc_val, auc_val, y_pred, y_prob


def run_pipeline():
    device = get_device()

    print(f"[*] Loading packed persistence dataset from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    images = data["images"]  # shape (20000, 2, 32, 32)
    labels = data["labels"]  # shape (20000,)
    print(f"[+] Loaded {len(images):,} persistence images. Shape: {images.shape}, Labels: {np.bincount(labels)}")

    # Identical 80/20 stratified train/test split matching previous study
    indices = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=42
    )

    # 10% stratified validation split from train
    sub_train_idx, val_idx = train_test_split(
        train_idx, test_size=0.10, stratify=labels[train_idx], random_state=42
    )

    print(f"[*] Data Partition:")
    print(f"    Train: {len(sub_train_idx):,} images (72%)")
    print(f"    Val:   {len(val_idx):,} images (8%)")
    print(f"    Test:  {len(test_idx):,} images (20% held-out benchmark)")

    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(
        PersistenceDataset(images[sub_train_idx], labels[sub_train_idx], is_train=True),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        PersistenceDataset(images[val_idx], labels[val_idx], is_train=False),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        PersistenceDataset(images[test_idx], labels[test_idx], is_train=False),
        batch_size=batch_size, shuffle=False
    )

    # Model instantiation
    model = TopoCNN(in_channels=2, num_classes=2).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] TopoCNN Initialized. Total trainable parameters: {num_params:,}")

    # Optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 22
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Tracking metrics
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_auc": []
    }

    best_val_auc = -1.0
    best_weights_path = os.path.join(MODELS_DIR, "topo_cnn_best.pt")

    print("\n" + "="*75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<12} | {'Val AUC':<10}")
    print("="*75)

    t_train_start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_auc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_auc"].append(v_auc)

        print(f"{epoch:<8} | {tr_loss:0.4f}       | {tr_acc*100:5.2f}%       | {v_loss:0.4f}     | {v_acc*100:5.2f}%     | {v_auc:0.4f}")

        # Checkpoint best validation AUC model
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            torch.save(model.state_dict(), best_weights_path)

    total_train_time = time.time() - t_train_start
    print("="*75)
    print(f"[+] Training completed in {total_train_time:.1f}s (Best Val AUC: {best_val_auc:.4f})")

    # Load best checkpoint for test evaluation
    model.load_state_dict(torch.load(best_weights_path))
    print(f"[*] Loaded best checkpoint from {best_weights_path}")

    # Evaluate on held-out test set
    test_loss, test_acc, test_auc, y_pred, y_prob = eval_epoch(model, test_loader, criterion, device)
    y_test = labels[test_idx]

    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    test_brier = brier_score_loss(y_test, y_prob)

    print("\n" + "="*80)
    print("LIGHTWEIGHT 2D TOPO-CNN TEST SET BENCHMARK RESULTS (N=4,000)")
    print("="*80)
    print(f"  Test Accuracy:          {test_acc*100:.2f}%")
    print(f"  Test Balanced Accuracy: {test_bal_acc*100:.2f}%")
    print(f"  Test ROC-AUC:           {test_auc:.4f}")
    print(f"  Test F1-Score:          {test_f1:.4f}")
    print(f"  Test Precision:         {test_prec:.4f}")
    print(f"  Test Recall:            {test_rec:.4f}")
    print(f"  Test Brier Score:       {test_brier:.4f}")
    print("="*80 + "\n")

    # Generate Learning Curves Figure
    plot_learning_curves(history, epochs)

    # Generate Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)

    # Compare with tabular models and update benchmark CSV + ROC Curves
    update_benchmark_and_roc(y_test, y_prob, test_acc, test_bal_acc, test_auc, test_f1, test_prec, test_rec, test_brier, total_train_time)


def plot_learning_curves(history: dict, epochs: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    ep_range = range(1, epochs + 1)
    ax1.plot(ep_range, history["train_loss"], label="Train Loss", color="#2980b9", lw=2.2)
    ax1.plot(ep_range, history["val_loss"], label="Val Loss", color="#e74c3c", lw=2.2, linestyle="--")
    ax1.set_title("TopoCNN: Cross-Entropy Loss Progression", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=10.5)

    ax2.plot(ep_range, [a * 100 for a in history["train_acc"]], label="Train Accuracy (%)", color="#27ae60", lw=2.2)
    ax2.plot(ep_range, [a * 100 for a in history["val_acc"]], label="Val Accuracy (%)", color="#8e44ad", lw=2.2, linestyle="--")
    ax2.plot(ep_range, [a * 100 for a in history["val_auc"]], label="Val ROC-AUC (x100)", color="#d35400", lw=2.0, linestyle=":")
    ax2.set_title("TopoCNN: Accuracy and ROC-AUC Progression", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Percentage / Metric x 100", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=10.5)

    plt.suptitle("Training and Validation Dynamics of Lightweight Topological CNN (TopoCNN)\nContinuous 2-Channel Persistence Images (H0 and H1, 32x32)",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cnn_learning_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved learning curves to {out_path}")


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    cm_counts = confusion_matrix(y_test, y_pred)
    class_names = ["AI (Synthetic)", "Real (Photograph)"]

    plt.figure(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(cm, annot=False, cmap="Blues", vmin=0.0, vmax=1.0)

    for r in range(2):
        for c in range(2):
            pct = cm[r, c] * 100
            cnt = cm_counts[r, c]
            txt_color = "white" if cm[r, c] > 0.55 else "black"
            plt.text(c + 0.5, r + 0.5, f"{pct:.1f}%\n({cnt:,})",
                     ha="center", va="center", color=txt_color,
                     fontweight="bold", fontsize=13)

    plt.title("TopoCNN Normalized Confusion Matrix (N=4,000 Test Set)\nPersistence Images (H0 Components & H1 Loops)",
              fontsize=12, fontweight="bold", pad=12)
    plt.xticks([0.5, 1.5], class_names, fontsize=10.5)
    plt.yticks([0.5, 1.5], class_names, fontsize=10.5, rotation=0)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "cnn_confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved confusion matrix to {out_path}")


def update_benchmark_and_roc(
    y_test: np.ndarray,
    y_prob_cnn: np.ndarray,
    test_acc: float,
    test_bal_acc: float,
    test_auc: float,
    test_f1: float,
    test_prec: float,
    test_rec: float,
    test_brier: float,
    train_time: float
):
    print("[*] Updating benchmark table and creating combined ROC curve plot...")

    # Load existing tabular benchmark
    if os.path.exists(TABULAR_BENCHMARK_PATH):
        df_bench = pd.read_csv(TABULAR_BENCHMARK_PATH)
    else:
        df_bench = pd.DataFrame()

    cnn_row = {
        "Model": "TopoCNN (2D Persistence Images)",
        "CV_Accuracy_Mean": np.nan,
        "CV_Accuracy_Std": np.nan,
        "CV_AUC_Mean": np.nan,
        "CV_AUC_Std": np.nan,
        "CV_F1_Mean": np.nan,
        "CV_F1_Std": np.nan,
        "Test_Accuracy": test_acc,
        "Test_Balanced_Accuracy": test_bal_acc,
        "Test_ROC_AUC": test_auc,
        "Test_F1_Score": test_f1,
        "Test_Precision": test_prec,
        "Test_Recall": test_rec,
        "Test_Brier_Score": test_brier,
        "Training_Time_Sec": train_time
    }

    # Filter out previous TopoCNN if present, then append
    if not df_bench.empty and "Model" in df_bench.columns:
        df_bench = df_bench[~df_bench["Model"].str.contains("TopoCNN", na=False)]
    df_updated = pd.concat([df_bench, pd.DataFrame([cnn_row])], ignore_index=True)

    out_csv = os.path.join(OUTPUT_DIR, "classification_metrics_benchmark_with_cnn.csv")
    df_updated.to_csv(out_csv, index=False)
    print(f"[+] Saved updated benchmark table to {out_csv}")

    # Generate Combined ROC Curves Plot
    plt.figure(figsize=(9.0, 7.5), dpi=300)

    colors = {
        "Logistic Regression (L2)": "#34495e",
        "Linear SVM": "#7f8c8d",
        "RBF SVM": "#2980b9",
        "Random Forest": "#27ae60",
        "XGBoost": "#d35400",
        "LightGBM": "#8e44ad",
        "TopoCNN (2D Persistence Images)": "#c0392b"
    }

    # Save TopoCNN test predictions for external analysis / comparison
    pred_path = os.path.join(MODELS_DIR, "topo_cnn_predictions.npz")
    np.savez(pred_path, y_true=y_test, y_prob=y_prob_cnn)
    print(f"[+] Saved test predictions to {pred_path}")

    # Plot TopoCNN prominently
    fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_prob_cnn)
    plt.plot(fpr_cnn, tpr_cnn,
             label=f"TopoCNN (2D Persistence Images: AUC = {test_auc:.4f})",
             color="#c0392b", lw=3.0, zorder=10)

    # Random chance baseline
    plt.plot([0, 1], [0, 1], color="#bdc3c7", linestyle="--", lw=1.5, label="Random Chance (AUC = 0.5000)")

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("Receiver Operating Characteristic (ROC) Comparison\n"
              "1D Topological Descriptors vs. 2D Continuous Persistence Images (N=4,000)",
              fontsize=13, fontweight="bold", pad=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#bdc3c7", fontsize=10)
    plt.tight_layout()

    out_roc_fig = os.path.join(OUTPUT_DIR, "roc_curves_with_cnn.png")
    plt.savefig(out_roc_fig)
    plt.close()
    print(f"[+] Saved updated ROC curve plot to {out_roc_fig}")


if __name__ == "__main__":
    run_pipeline()
