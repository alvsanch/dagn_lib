"""
global_dataset.py — Multi-dataset combiner for dagn_lib

Combines DEAP, WESAD, DREAMER and AFEW-VA into a single dataset.
Applies per-dataset VA z-score normalization to eliminate dataset bias.

Dataset bias problem:
    Each dataset uses different VA scale conventions (1-9, 1-5, 0-10).
    Even after normalizing to [-1,1], mean VA differs between datasets.
    Z-scoring per dataset forces each one to mean=0, std=1 in training,
    making the model learn intra-dataset emotion dynamics instead of
    inter-dataset scale differences.

References:
    Per-dataset normalization approach: see dagn_simple/training/global_dataset.py
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from deap_dataset    import DEAPDataset
from wesad_dataset   import WESADDataset
from dreamer_dataset import DREAMERDataset
from afew_va_dataset import AFEWVADataset
from affec_dataset   import AFFECDataset

# Feature dimensions (must match feature extractors)
FACE_DIM   = 17   # AU1-AU43 (17 selected AUs)
PHYSIO_DIM = 6    # HRV(3) + EDA(2) + TEMP(1)
EEG_DIM    = 5    # theta, alpha, beta, alpha_asym, theta/alpha

# Dataset indices (used in training for variance penalty, logging, etc.)
DATASET_DEAP    = 0
DATASET_WESAD   = 1
DATASET_DREAMER = 2
DATASET_AFEWVA  = 3
DATASET_AFFEC   = 4
DATASET_NAMES   = ["DEAP", "WESAD", "DREAMER", "AFEW-VA", "AFFEC"]


class GlobalDataset(Dataset):
    """
    Combined dataset from DEAP, WESAD, DREAMER and AFEW-VA.

    Each sample: (face, physio, eeg, va, dataset_id)
        face:       (T, 17) float32 — Action Units [0,1], zeros if no video
        physio:     (T, 6)  float32 — HRV/EDA/TEMP, zeros if no physio
        eeg:        (T, 5)  float32 — bandpower/asymmetry, zeros if no EEG
        va:         (2,)    float32 — [valence, arousal] z-scored per dataset
        dataset_id: int     — which dataset this sample came from

    Args:
        datasets:         list of (Dataset, name, weight) tuples, or None to use defaults
        normalize_labels: apply per-dataset VA z-score (default True)
        split:            'train', 'val', or 'test'
                          'test' = first 50% of val indices (deterministic, never in
                          training gradients). Note: model selection (early stopping)
                          used the full val set, so test metrics have slight indirect
                          selection bias. True held-out test requires re-training.
        val_ratio:        fraction for validation set (default 0.2)
        seed:             random seed for split (default 42)
        T:                timesteps per sample (default 30)
    """

    def __init__(
        self,
        datasets=None,
        normalize_labels=True,
        split="train",
        val_ratio=0.2,
        seed=42,
        T=30,
    ):
        self.T                = T
        self.normalize_labels = normalize_labels
        self.rng              = np.random.default_rng(seed)

        # Load sub-datasets
        if datasets is None:
            datasets = self._load_defaults(T=T, seed=seed)

        self.all_samples  = []   # list of (face, physio, eeg, va_raw, ds_id)
        self.va_stats     = {}   # ds_id → (mean, std) for each VA dimension

        for ds_id, (ds, ds_name) in enumerate(datasets):
            if ds is None or len(ds) == 0:
                print(f"[Global] Skipping {ds_name} (empty)")
                continue
            n = len(ds)
            print(f"[Global] {ds_name}: {n} samples")
            for i in range(n):
                face, physio, eeg, va = ds[i]
                self.all_samples.append((face, physio, eeg, va, ds_id))

        # Stratified 80/20 split within each dataset
        self.indices = self._split_indices(split, val_ratio, seed)

        # Compute per-dataset VA statistics (on training split only)
        if normalize_labels:
            self._compute_va_stats(seed, val_ratio)

        print(f"[Global] Split='{split}': {len(self.indices)} samples "
              f"from {len(datasets)} datasets")

    @staticmethod
    def _load_defaults(T, seed):
        """Load all five datasets with default paths."""
        print("[Global] Loading DEAP...")
        deap = DEAPDataset(T=T, seed=seed)
        print("[Global] Loading WESAD...")
        wesad = WESADDataset(T=T, seed=seed)
        print("[Global] Loading DREAMER...")
        dreamer = DREAMERDataset(T=T, seed=seed)
        print("[Global] Loading AFEW-VA...")
        afew = AFEWVADataset(T=T, use_flip=True, seed=seed)
        print("[Global] Loading AFFEC...")
        affec = AFFECDataset(T=T, seed=seed)
        return [
            (deap,    "DEAP"),
            (wesad,   "WESAD"),
            (dreamer, "DREAMER"),
            (afew,    "AFEW-VA"),
            (affec,   "AFFEC"),
        ]

    def _split_indices(self, split, val_ratio, seed):
        """Stratified split: 80% train / 20% val within each dataset.

        'test' is derived from val: the first 50% of each dataset's val indices
        (sorted, deterministic). These samples never received gradient updates.
        See class docstring for the selection-bias caveat.
        """
        rng = np.random.default_rng(seed)
        # Group sample indices by dataset
        ds_to_idxs = {}
        for global_idx, (_, _, _, _, ds_id) in enumerate(self.all_samples):
            ds_to_idxs.setdefault(ds_id, []).append(global_idx)

        train_idxs, val_idxs, test_idxs = [], [], []
        for ds_id, idxs in ds_to_idxs.items():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            n_val = max(1, int(len(idxs) * val_ratio))
            val_part = idxs[:n_val].tolist()
            val_idxs.extend(val_part)
            train_idxs.extend(idxs[n_val:].tolist())
            # test = first half of val (deterministic, no extra randomness)
            n_test = max(1, len(val_part) // 2)
            test_idxs.extend(sorted(val_part)[:n_test])

        if split == "train":
            return train_idxs
        elif split == "test":
            return test_idxs
        else:
            return val_idxs

    def _compute_va_stats(self, seed, val_ratio):
        """Compute mean/std of VA labels per dataset on training split."""
        train_indices = self._split_indices("train", val_ratio, seed)
        ds_to_vas = {}
        for idx in train_indices:
            _, _, _, va, ds_id = self.all_samples[idx]
            ds_to_vas.setdefault(ds_id, []).append(va)

        for ds_id, vas in ds_to_vas.items():
            arr = np.stack(vas, axis=0)  # (N, 2)
            mean = arr.mean(axis=0)      # (2,)
            std  = arr.std(axis=0)       # (2,)
            std  = np.where(std < 1e-6, 1.0, std)
            self.va_stats[ds_id] = (mean.astype(np.float32),
                                    std.astype(np.float32))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        face, physio, eeg, va, ds_id = self.all_samples[global_idx]

        # Per-dataset VA normalization + clip to 3 std devs to prevent outliers
        if self.normalize_labels and ds_id in self.va_stats:
            mean, std = self.va_stats[ds_id]
            va = np.clip((va - mean) / std, -3.0, 3.0)

        return (
            torch.from_numpy(face),           # (T, 17)
            torch.from_numpy(physio),          # (T, 6)
            torch.from_numpy(eeg),             # (T, 5)
            torch.from_numpy(va.astype(np.float32)),   # (2,)
            torch.tensor(ds_id, dtype=torch.long),     # scalar
        )


def make_dataloaders(batch_size=32, seed=42, T=30, num_workers=0):
    """
    Build train and validation DataLoaders with dataset-balanced sampling.

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Load sub-datasets ONCE (feature extraction is expensive)
    datasets = GlobalDataset._load_defaults(T=T, seed=seed)

    train_ds = GlobalDataset(datasets=datasets, split="train",
                              normalize_labels=True, seed=seed, T=T)
    val_ds   = GlobalDataset(datasets=datasets, split="val",
                              normalize_labels=True, seed=seed, T=T)

    # Copy VA stats from train to val (use training set statistics)
    val_ds.va_stats = train_ds.va_stats

    # Dataset-balanced sampler for training
    # Weight per sample ∝ 1/sqrt(n_samples_in_its_dataset)
    ds_counts = {}
    for idx in train_ds.indices:
        _, _, _, _, ds_id = train_ds.all_samples[idx]
        ds_counts[ds_id] = ds_counts.get(ds_id, 0) + 1

    weights = []
    for idx in train_ds.indices:
        _, _, _, _, ds_id = train_ds.all_samples[idx]
        w = 1.0 / (np.sqrt(ds_counts[ds_id]) + 1e-8)
        weights.append(w)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, train_ds, val_ds
