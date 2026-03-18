"""
evaluate_fusion.py — Evaluation script for FusionLSTM (dagn_lib)

Computes per-dataset and global metrics with 95% bootstrap CIs:
    CCC (Concordance Correlation Coefficient) — Lin (1989)
    Pearson r + p-value
    RMSE, MAE, R²

Generates a LaTeX table for direct use in the thesis.
Results appended to ../results_log.txt

Usage:
    cd /home/alvar/dagn_lib

    # Standard validation (default):
    /home/alvar/venv_tesis/bin/python production/evaluate_fusion.py

    # Pseudo test set (first 50% of val, never in training gradients):
    /home/alvar/venv_tesis/bin/python production/evaluate_fusion.py --split test

    # Train set (to inspect overfitting gap):
    /home/alvar/venv_tesis/bin/python production/evaluate_fusion.py --split train

    # All three splits sequentially (train / val / test comparison):
    /home/alvar/venv_tesis/bin/python production/evaluate_fusion.py --split all

Methodology note on the 'test' split
--------------------------------------
A clean held-out test set does not exist for this checkpoint because the model
was early-stopped using the full validation set (val_ratio=0.2, seed=42).
The 'test' split is the first 50% of each dataset's val indices — samples that
never received gradient updates but were indirectly used for model selection.
Metrics on 'test' have a slight upward selection bias relative to truly
independent data. For a bias-free test set, re-train with a 70/15/15 split.
"""
import sys
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats

# Allow imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "production"))

from global_dataset import GlobalDataset, DATASET_NAMES
from fusion_model   import FusionLSTM

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = ROOT / "production" / "fusion_best.pth"
LOG_PATH   = ROOT / "results_log.txt"
BATCH_SIZE = 64
T          = 30
N_BOOT     = 1000   # bootstrap resamples
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Metrics ─────────────────────────────────────────────────────────────────
def _ccc(p, t):
    pm, tm = p.mean(), t.mean()
    cov = ((p - pm) * (t - tm)).mean()
    return float(2.0 * cov / (p.var() + t.var() + (pm - tm) ** 2 + 1e-8))


def bootstrap_ci(pred, target, n_boot=N_BOOT, seed=42):
    """
    95% percentile bootstrap CI for CCC.
    Returns (ccc_mean, ci_low, ci_high) for valence and arousal.
    """
    rng = np.random.default_rng(seed)
    n   = len(pred)
    scores_v, scores_a = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores_v.append(_ccc(pred[idx, 0], target[idx, 0]))
        scores_a.append(_ccc(pred[idx, 1], target[idx, 1]))
    cv = _ccc(pred[:, 0], target[:, 0])
    ca = _ccc(pred[:, 1], target[:, 1])
    v_lo, v_hi = np.percentile(scores_v, [2.5, 97.5])
    a_lo, a_hi = np.percentile(scores_a, [2.5, 97.5])
    return (cv, v_lo, v_hi), (ca, a_lo, a_hi)


def compute_metrics(pred, target):
    """Full metrics for one dimension."""
    r, p_val = stats.pearsonr(pred, target)
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    mae  = float(np.mean(np.abs(pred - target)))
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return {"r": float(r), "p": float(p_val), "rmse": rmse, "mae": mae, "r2": r2}


# ─── Inference ───────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, loader):
    """Run model on all batches, return (preds, labels, ds_ids) arrays.

    Uses the LAST timestep output (index -1), matching production behaviour in
    analizar_emocion_service.py where va[0, -1, :] is the inference result.
    The last timestep has the most temporal context and is the operative output.
    """
    model.eval()
    preds_list, labels_list, ds_list = [], [], []
    for face, physio, eeg, va, ds_ids in loader:
        face   = face.to(DEVICE)
        physio = physio.to(DEVICE)
        eeg    = eeg.to(DEVICE)
        va_out, _ = model(face, physio, eeg)           # (B, T, 2)
        pred_last = va_out[:, -1, :].cpu().numpy()    # (B, 2) — last timestep
        preds_list.append(pred_last)
        labels_list.append(va.cpu().numpy())
        ds_list.append(ds_ids.cpu().numpy())
    return (np.concatenate(preds_list),
            np.concatenate(labels_list),
            np.concatenate(ds_list))


# ─── Table formatting ─────────────────────────────────────────────────────────
def print_results_table(rows, title=""):
    """rows: list of (name, N, ccc_v, v_lo, v_hi, ccc_a, a_lo, a_hi)"""
    if title:
        print(f"\n{'─'*70}")
        print(f"  {title}")
    header = f"{'Dataset':<10} {'N':>5}  {'CCC-V':>22}  {'CCC-A':>22}  {'Mean':>6}"
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for name, N, cv, vl, vh, ca, al, ah in rows:
        mean = (cv + ca) / 2.0
        marker = " ◄" if name == "GLOBAL" else ""
        print(f"{name:<10} {N:>5}  {cv:.3f} [{vl:.3f}, {vh:.3f}]  "
              f"{ca:.3f} [{al:.3f}, {ah:.3f}]  {mean:.3f}{marker}")
    print("─" * len(header))


def latex_table(rows, model_name="FusionLSTM", split="val"):
    """Generate LaTeX table for thesis."""
    split_note = {
        "val":   "held-out validation set",
        "test":  "pseudo test set (first 50\\% of val, no gradient updates)",
        "train": "training set",
    }.get(split, split)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Evaluation of {model_name} on {split_note}. "
        r"CCC with 95\% bootstrap CI (n=1000).}",
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"Dataset & N & CCC-V & 95\% CI & CCC-A & 95\% CI \\",
        r"\midrule",
    ]
    for name, N, cv, vl, vh, ca, al, ah in rows:
        if name == "GLOBAL":
            lines.append(r"\midrule")
            name_tex = r"\textbf{" + name + r"}"
            cv_s  = rf"\textbf{{{cv:.3f}}}"
            ci_v  = rf"\textbf{{[{vl:.3f}, {vh:.3f}]}}"
            ca_s  = rf"\textbf{{{ca:.3f}}}"
            ci_a  = rf"\textbf{{[{al:.3f}, {ah:.3f}]}}"
        else:
            name_tex = name
            cv_s  = f"{cv:.3f}"
            ci_v  = f"[{vl:.3f}, {vh:.3f}]"
            ca_s  = f"{ca:.3f}"
            ci_a  = f"[{al:.3f}, {ah:.3f}]"
        lines.append(rf"{name_tex} & {N} & {cv_s} & {ci_v} & {ca_s} & {ci_a} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── Single-split evaluation ──────────────────────────────────────────────────
def evaluate_split(model, split, shared_datasets=None):
    """Load, infer and report metrics for one split. Returns (rows, preds, labels, ds_ids)."""
    ds = GlobalDataset(
        datasets=shared_datasets,
        split=split,
        normalize_labels=True,
        T=T,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds, labels, ds_ids = collect_predictions(model, loader)

    rows = []
    for ds_id in np.unique(ds_ids):
        mask = ds_ids == ds_id
        p, l = preds[mask], labels[mask]
        N = int(mask.sum())
        (cv, vl, vh), (ca, al, ah) = bootstrap_ci(p, l)
        ds_name = DATASET_NAMES[int(ds_id)] if int(ds_id) < len(DATASET_NAMES) else str(ds_id)
        rows.append((ds_name, N, cv, vl, vh, ca, al, ah))

    (cv, vl, vh), (ca, al, ah) = bootstrap_ci(preds, labels)
    rows.append(("GLOBAL", len(preds), cv, vl, vh, ca, al, ah))

    return rows, preds, labels, ds_ids


def print_detailed(rows, preds, labels, ds_ids):
    print("\n── Detailed metrics ──────────────────────────────────────────")
    for name, N, cv, vl, vh, ca, al, ah in rows:
        if name == "GLOBAL":
            continue
        ds_id = DATASET_NAMES.index(name)
        mask  = ds_ids == ds_id
        p, l  = preds[mask], labels[mask]
        mv = compute_metrics(p[:, 0], l[:, 0])
        ma = compute_metrics(p[:, 1], l[:, 1])
        print(f"\n{name} (N={N})")
        print(f"  Valence: CCC={cv:.3f}, r={mv['r']:.3f}(p={mv['p']:.3f}), "
              f"RMSE={mv['rmse']:.3f}, MAE={mv['mae']:.3f}, R²={mv['r2']:.3f}")
        print(f"  Arousal: CCC={ca:.3f}, r={ma['r']:.3f}(p={ma['p']:.3f}), "
              f"RMSE={ma['rmse']:.3f}, MAE={ma['mae']:.3f}, R²={ma['r2']:.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate FusionLSTM")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="val",
        help=(
            "val  = validation set used for early stopping (default, N≈937)\n"
            "test = pseudo held-out: first 50%% of val per dataset (N≈468)\n"
            "train = training set — shows overfitting gap\n"
            "all  = runs train + val + test sequentially for comparison"
        ),
    )
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH}")
    model = FusionLSTM().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model_state = model.state_dict()
    compatible  = {k: v for k, v in state.items()
                   if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(compatible, strict=False)
    model.eval()
    print(f"Loaded {len(compatible)}/{len(model_state)} layers\n")

    # Load sub-datasets once (feature extraction is expensive)
    print("Loading datasets (shared across splits)...")
    shared = GlobalDataset._load_defaults(T=T, seed=42)

    splits_to_run = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_global_rows = {}   # split → GLOBAL row (for comparison table in --all mode)

    for split in splits_to_run:
        print(f"\n{'='*60}")
        print(f"  Split: {split.upper()}")
        print(f"{'='*60}")
        rows, preds, labels, ds_ids = evaluate_split(model, split, shared_datasets=shared)
        print_results_table(rows)
        print_detailed(rows, preds, labels, ds_ids)

        latex = latex_table(rows, model_name="FusionLSTM (dagn\\_lib)", split=split)
        print("\n── LaTeX ─────────────────────────────────────────────────────")
        print(latex)

        global_row = [r for r in rows if r[0] == "GLOBAL"][0]
        all_global_rows[split] = global_row

        # Append to log
        ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        _, N, gcv, gvl, gvh, gca, gal, gah = global_row
        mean_ccc = (gcv + gca) / 2.0
        log_entry = (
            f"\n{'='*60}\n"
            f"[{ts}] dagn_lib FusionLSTM evaluation — split={split}\n"
            f"Model: {MODEL_PATH}\n"
            f"Prediction: last timestep (t=-1), matching production\n"
            f"Global CCC: V={gcv:.3f} [{gvl:.3f},{gvh:.3f}] "
            f"A={gca:.3f} [{gal:.3f},{gah:.3f}] mean={mean_ccc:.3f}\n"
            f"\n{latex}\n"
            f"{'='*60}\n"
        )
        with open(LOG_PATH, "a") as f:
            f.write(log_entry)
        print(f"\nAppended to {LOG_PATH}")

    # Summary comparison when running all splits
    if args.split == "all" and len(all_global_rows) == 3:
        print(f"\n{'='*60}")
        print("  COMPARISON: Train vs Val vs Test (GLOBAL CCC mean)")
        print(f"{'='*60}")
        print(f"  {'Split':<6}  {'N':>5}  {'CCC-V':>6}  {'CCC-A':>6}  {'Mean':>6}")
        print(f"  {'─'*40}")
        for sp, (_, N, cv, *_, ca, _, _) in all_global_rows.items():
            print(f"  {sp:<6}  {N:>5}  {cv:>6.3f}  {ca:>6.3f}  {(cv+ca)/2:>6.3f}")
        tr = all_global_rows.get("train")
        va = all_global_rows.get("val")
        if tr and va:
            gap = ((tr[2] + tr[6]) / 2) - ((va[2] + va[6]) / 2)
            print(f"\n  Train–Val CCC gap: {gap:+.3f} "
                  f"({'overfitting' if gap > 0.05 else 'OK — no significant overfit'})")
        print(f"\n  Note: 'test' is a subset of 'val' — slight upward selection bias.")
        print(f"  For a bias-free test, re-train with a 70/15/15 split.")


if __name__ == "__main__":
    main()
