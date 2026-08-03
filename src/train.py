"""
train.py — DrunkSoberNet training script

Two-phase fine-tuning:
  Phase 1: freeze ResNet18 backbone, train GRU + head only
  Phase 2: unfreeze layer4, fine-tune with a lower learning rate

Runs stratified K-fold cross-validation (see dataset.get_kfold_datasets):
each fold trains a fresh model from scratch and reports its best val
accuracy; the final summary is the mean +/- std across folds. This
matters more than a single fixed split here because the dataset is
small (tens of videos), so one split can easily be lucky/unlucky.

Run this from a thin Colab notebook — this file holds the actual logic.
"""

import os
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import DrunkSoberNet
from src.dataset import get_kfold_datasets  # returns ([T, 3, 224, 224], label) per item

# ---- Paths (avoid hardcoding — Drive mount path differs per person/session) ----
DATA_ROOT = "/content/drive/MyDrive/MOSAT_CNN/data"
FRAMES_DIR = os.path.join(DATA_ROOT, "frames")
CHECKPOINT_DIR = "/content/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---- Hyperparameters ----
BATCH_SIZE = 8
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
K_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over the data. optimizer=None -> eval mode, no backprop."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for sequences, labels in loader:
            sequences = sequences.to(DEVICE)          # [B, T, 3, 224, 224]
            labels = labels.to(DEVICE).float()        # [B]

            logits = model(sequences)                 # [B]
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_fold(train_loader, val_loader, checkpoint_path):
    """Two-phase fine-tune of a fresh model on one fold. Returns best val accuracy."""
    model = DrunkSoberNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    # =========================================================
    # Phase 1: freeze backbone, train GRU + head only
    # =========================================================
    print("=== Phase 1: training head (backbone frozen) ===")
    model.freeze_backbone()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR
    )

    best_val_acc = 0.0
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(f"[Phase 1][Epoch {epoch}/{PHASE1_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    # =========================================================
    # Phase 2: unfreeze layer4, fine-tune at a lower LR
    # =========================================================
    print("\n=== Phase 2: fine-tuning layer4 ===")
    model.unfreeze_layer4()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE2_LR
    )

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(f"[Phase 2][Epoch {epoch}/{PHASE2_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to {checkpoint_path}")
    return best_val_acc


def main():
    folds = get_kfold_datasets(FRAMES_DIR, k=K_FOLDS)

    fold_accuracies = []
    for fold_idx, (train_dataset, val_dataset) in enumerate(folds, start=1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx}/{K_FOLDS}  "
              f"(train={len(train_dataset)} videos, val={len(val_dataset)} videos)")
        print(f"{'=' * 60}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"fold{fold_idx}_best.pt")
        best_val_acc = train_fold(train_loader, val_loader, checkpoint_path)
        fold_accuracies.append(best_val_acc)

    mean_acc = statistics.mean(fold_accuracies)
    std_acc = statistics.stdev(fold_accuracies) if len(fold_accuracies) > 1 else 0.0

    print(f"\n{'=' * 60}")
    print("K-fold CV results")
    print(f"{'=' * 60}")
    for fold_idx, acc in enumerate(fold_accuracies, start=1):
        print(f"  Fold {fold_idx}: {acc:.4f}")
    print(f"Mean val accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")


if __name__ == "__main__":
    main()