"""
train_fusion.py — Training script for FusionLSTM (dagn_lib)

Loss: MSE + (1-CCC_valence) + (1-CCC_arousal) + variance_penalty
Augmentation: modal dropout (zero out entire modality per sample)
Scheduler: CosineAnnealingLR
Early stopping: PATIENCE=40

Output:
    ../production/fusion_best.pth   — best model checkpoint
    ../results_log.txt              — training summary appended after run

Usage:
    cd /home/alvar/dagn_lib/training
    /home/alvar/venv_tesis/bin/python -u train_fusion.py
"""
import sys
import os
import time
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow imports from training/ and production/
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "production"))

from global_dataset import GlobalDataset, make_dataloaders, DATASET_NAMES
from fusion_model   import FusionLSTM

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS      = 200
BATCH_SIZE  = 32
LR          = 1e-3
WEIGHT_DECAY= 3e-4   # increased from 1e-4 to reduce overfitting with larger model
PATIENCE    = 40     # epochs without val CCC improvement before stopping
T           = 30     # timesteps per sample

# Modal dropout probabilities (zero out entire modality per sample)
P_FACE_DROP   = 0.2
P_PHYSIO_DROP = 0.2
P_EEG_DROP    = 0.3

# Variance penalty per dataset (penalise flat predictions)
# DATASET_DEAP=0, DATASET_WESAD=1, DATASET_DREAMER=2, DATASET_AFEWVA=3, DATASET_AFFEC=4
VARIANCE_ALPHAS = {0: 0.5, 1: 0.0, 2: 0.5, 3: 0.3, 4: 0.3}

# Paths
SCRIPT_DIR  = Path(__file__).parent
PROD_DIR    = SCRIPT_DIR.parent / "production"
BEST_PATH   = PROD_DIR / "fusion_best.pth"
LOG_PATH    = SCRIPT_DIR.parent / "results_log.txt"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── CCC loss ─────────────────────────────────────────────────────────────────
def ccc(pred, target):
    """Concordance Correlation Coefficient — Lin (1989)."""
    pred   = pred.float()
    target = target.float()
    pm = pred.mean()
    tm = target.mean()
    cov = ((pred - pm) * (target - tm)).mean()
    denom = pred.var() + target.var() + (pm - tm) ** 2 + 1e-8
    return (2.0 * cov) / denom


def ccc_loss(pred, target):
    """1 - mean CCC over valence and arousal."""
    loss_v = 1.0 - ccc(pred[:, :, 0].reshape(-1), target[:, 0].repeat_interleave(pred.shape[1]))
    loss_a = 1.0 - ccc(pred[:, :, 1].reshape(-1), target[:, 1].repeat_interleave(pred.shape[1]))
    return 0.5 * (loss_v + loss_a)


def variance_penalty(pred, target, ds_ids):
    """
    Penalise under-confident predictions within each dataset.
    penalty = relu(std(target) - std(pred)) × alpha
    """
    penalty = torch.tensor(0.0, device=pred.device)
    for ds_id, alpha in VARIANCE_ALPHAS.items():
        if alpha == 0.0:
            continue
        mask = (ds_ids == ds_id)
        if mask.sum() < 4:
            continue
        p_flat = pred[mask].reshape(-1, 2)
        t_flat = target[mask].unsqueeze(1).expand(-1, pred.shape[1], -1).reshape(-1, 2)
        for dim in range(2):
            p_std = p_flat[:, dim].std()
            t_std = t_flat[:, dim].std()
            penalty = penalty + alpha * torch.relu(t_std - p_std)
    return penalty


def total_loss(pred, target, ds_ids):
    """MSE + CCC loss + variance penalty."""
    # Expand target from (B,2) to (B,T,2) for MSE
    target_seq = target.unsqueeze(1).expand_as(pred)
    mse  = nn.functional.mse_loss(pred, target_seq)
    ccl  = ccc_loss(pred, target)
    vpen = variance_penalty(pred, target, ds_ids)
    return mse + ccl + vpen


# ─── Modal dropout augmentation ───────────────────────────────────────────────
def modal_dropout(face, physio, eeg):
    """Randomly zero out entire modalities per sample."""
    B = face.shape[0]
    if P_FACE_DROP > 0:
        mask = (torch.rand(B, 1, 1, device=face.device) > P_FACE_DROP).float()
        face = face * mask
    if P_PHYSIO_DROP > 0:
        mask = (torch.rand(B, 1, 1, device=physio.device) > P_PHYSIO_DROP).float()
        physio = physio * mask
    if P_EEG_DROP > 0:
        mask = (torch.rand(B, 1, 1, device=eeg.device) > P_EEG_DROP).float()
        eeg = eeg * mask
    return face, physio, eeg


# ─── Per-dataset CCC evaluation ───────────────────────────────────────────────
@torch.no_grad()
def eval_ccc_by_dataset(model, loader):
    """
    Compute CCC per dataset and global CCC.
    Returns dict: {ds_name: (ccc_v, ccc_a)}
    """
    model.eval()
    preds_by_ds  = {}
    labels_by_ds = {}

    for face, physio, eeg, va, ds_ids in loader:
        face   = face.to(DEVICE)
        physio = physio.to(DEVICE)
        eeg    = eeg.to(DEVICE)
        va_out, _ = model(face, physio, eeg)   # (B, T, 2)
        # Use last timestep — consistent with production and evaluate_fusion.py
        pred_mean = va_out[:, -1, :].cpu().numpy()    # (B, 2)
        va_np     = va.cpu().numpy()

        for b in range(len(va_np)):
            ds_id = int(ds_ids[b].item())
            preds_by_ds.setdefault(ds_id, []).append(pred_mean[b])
            labels_by_ds.setdefault(ds_id, []).append(va_np[b])

    results = {}
    all_preds, all_labels = [], []

    for ds_id in sorted(preds_by_ds.keys()):
        preds  = np.array(preds_by_ds[ds_id])
        labels = np.array(labels_by_ds[ds_id])
        all_preds.append(preds)
        all_labels.append(labels)

        def _ccc_np(p, t):
            pm, tm = p.mean(), t.mean()
            cov = ((p - pm) * (t - tm)).mean()
            return 2 * cov / (p.var() + t.var() + (pm - tm) ** 2 + 1e-8)

        cv = _ccc_np(preds[:, 0], labels[:, 0])
        ca = _ccc_np(preds[:, 1], labels[:, 1])
        ds_name = DATASET_NAMES[ds_id] if ds_id < len(DATASET_NAMES) else str(ds_id)
        results[ds_name] = (float(cv), float(ca))

    if all_preds:
        p = np.concatenate(all_preds)
        l = np.concatenate(all_labels)
        pm, tm = p.mean(0), l.mean(0)
        for dim in range(2):
            cov = ((p[:, dim] - pm[dim]) * (l[:, dim] - tm[dim])).mean()
            cv = 2*cov/(p[:,dim].var()+l[:,dim].var()+(pm[dim]-tm[dim])**2+1e-8)
        # Recompute properly
        def _ccc_np(p, t):
            pm, tm = p.mean(), t.mean()
            cov = ((p - pm) * (t - tm)).mean()
            return 2 * cov / (p.var() + t.var() + (pm - tm) ** 2 + 1e-8)
        cv = _ccc_np(p[:, 0], l[:, 0])
        ca = _ccc_np(p[:, 1], l[:, 1])
        results["GLOBAL"] = (float(cv), float(ca))

    return results


# ─── Training loop ────────────────────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")
    print(f"Building datasets (feature computation may take several minutes)...")
    t0 = time.time()

    train_loader, val_loader, train_ds, val_ds = make_dataloaders(
        batch_size=BATCH_SIZE, seed=42, T=T, num_workers=0
    )
    print(f"Datasets ready in {time.time()-t0:.0f}s | "
          f"train={len(train_ds)}, val={len(val_ds)}")

    model = FusionLSTM(hidden_dim=256, num_layers=2, dropout=0.45).to(DEVICE)
    n_params = model.describe()
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    best_ccc  = -1.0
    best_ep   = 0
    no_improve= 0

    print(f"{'Ep':>4} {'TrainLoss':>10} {'CCC-V':>7} {'CCC-A':>7} {'Mean':>7}  Datasets")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for face, physio, eeg, va, ds_ids in train_loader:
            face   = face.to(DEVICE)
            physio = physio.to(DEVICE)
            eeg    = eeg.to(DEVICE)
            va     = va.to(DEVICE)
            ds_ids = ds_ids.to(DEVICE)

            # Modal dropout augmentation
            face, physio, eeg = modal_dropout(face, physio, eeg)

            optimizer.zero_grad()
            pred, _ = model(face, physio, eeg)   # (B, T, 2)
            loss = total_loss(pred, va, ds_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_train_loss / max(n_batches, 1)

        # Validation
        results = eval_ccc_by_dataset(model, val_loader)
        global_ccc = results.get("GLOBAL", (0.0, 0.0))
        mean_ccc   = (global_ccc[0] + global_ccc[1]) / 2.0

        # Per-dataset summary string
        ds_str = " | ".join(
            f"{k}=({v[0]:.2f},{v[1]:.2f})"
            for k, v in results.items()
            if k != "GLOBAL"
        )

        print(f"{epoch:4d} {avg_loss:10.4f} {global_ccc[0]:7.3f} {global_ccc[1]:7.3f} "
              f"{mean_ccc:7.3f}  [{ds_str}]", flush=True)

        # Early stopping & checkpointing
        if mean_ccc > best_ccc:
            best_ccc  = mean_ccc
            best_ep   = epoch
            no_improve= 0
            torch.save(model.state_dict(), BEST_PATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best CCC={best_ccc:.3f} at epoch {best_ep})")
                break

    print(f"\nTraining done. Best CCC={best_ccc:.3f} at epoch {best_ep}")
    print(f"Model saved: {BEST_PATH}")

    # Final evaluation log
    _log_result(best_ccc, best_ep, n_params)


def _log_result(best_ccc, best_ep, n_params):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = (f"\n[{ts}] dagn_lib FusionLSTM — best_ccc={best_ccc:.3f} "
            f"epoch={best_ep} params={n_params:,}\n")
    with open(LOG_PATH, "a") as f:
        f.write(line)
    print(f"Logged to {LOG_PATH}")


if __name__ == "__main__":
    train()
