"""
afew_va_dataset.py — AFEW-VA dataset for dagn_lib

AFEW-VA: video clips with per-frame valence/arousal annotations (0-10 scale)
Face features: pre-extracted AU vectors from py-feat via extract_afew_au_features.py

Requires: ~/datasets/afew_va_au_features/{clip_id}.npy  (shape: n_frames × 17)
          ~/datasets/afew_va_au_features/{clip_id}_flip.npy

If AU features are not extracted yet, run:
    python extract_afew_au_features.py

No physio, no EEG in AFEW-VA.

Reference:
    Kossaifi, J. et al. (2017). AFEW-VA database for valence and arousal
        estimation in-the-wild. Image and Vision Computing, 65, 23-36.
"""
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from feature_extractor_physio import zeros_physio
from feature_extractor_eeg_full import zeros_eeg_full as zeros_eeg

# ─── Constants ────────────────────────────────────────────────────────────────
AFEW_VA_DIR   = Path("/mnt/f/source_datasets/Fisiologico/AFEW-VA")
AU_FEAT_DIR   = Path("/home/alvar/datasets/afew_va_au_features")
FACE_DIM      = 17
T             = 30
MIN_FRAMES    = T


class AFEWVADataset(Dataset):
    """
    AFEW-VA dataset loader using pre-extracted AU features.

    Each __getitem__ returns a random T-frame window from the clip
    for temporal augmentation.

    Args:
        afew_dir:    path to AFEW-VA directory
        au_feat_dir: path to pre-extracted AU .npy files
        use_flip:    include horizontally flipped clips for augmentation
        T:           timesteps per sample (default 30)
        seed:        random seed
    """

    def __init__(
        self,
        afew_dir=AFEW_VA_DIR,
        au_feat_dir=AU_FEAT_DIR,
        use_flip=True,
        T=T,
        seed=42,
    ):
        self.afew_dir  = Path(afew_dir)
        self.au_dir    = Path(au_feat_dir)
        self.T         = T
        self.use_flip  = use_flip
        self.rng       = np.random.default_rng(seed)

        self.clips = []    # list of (au_path, va_labels)
        self._build_index()

    def _build_index(self):
        """Index all clips that have AU features and VA labels."""
        if not self.afew_dir.exists():
            print(f"[AFEW-VA] WARNING: {self.afew_dir} not found")
            return

        if not self.au_dir.exists():
            print(f"[AFEW-VA] WARNING: AU features not found at {self.au_dir}")
            print("          Run: python extract_afew_au_features.py")
            return

        for clip_dir in sorted(self.afew_dir.iterdir()):
            if not clip_dir.is_dir() or not clip_dir.name.isdigit():
                continue

            clip_id  = clip_dir.name
            au_path   = self.au_dir / f"{clip_id}.npy"
            au_flip   = self.au_dir / f"{clip_id}_flip.npy"
            json_path = clip_dir / f"{clip_id}.json"

            if not au_path.exists() or not json_path.exists():
                continue

            va_labels = self._load_va_labels(json_path)
            if va_labels is None or len(va_labels) < MIN_FRAMES:
                continue

            self.clips.append((au_path, va_labels))
            if self.use_flip and au_flip.exists():
                self.clips.append((au_flip, va_labels))

        print(f"[AFEW-VA] Indexed {len(self.clips)} clips "
              f"({'with' if self.use_flip else 'without'} flip augmentation)")

    def _load_va_labels(self, json_path):
        """
        Load per-frame VA labels from AFEW-VA JSON.
        Returns (n_frames, 2) float32 array, normalized to [-1, 1].
        AFEW-VA uses 0-10 scale → (x - 5) / 5.
        """
        try:
            with open(json_path) as f:
                data = json.load(f)

            frames = data.get("frames", data)  # handle both formats

            va_list = []
            for frame_id in sorted(frames.keys(), key=lambda x: int(x)):
                frame_info = frames[frame_id]
                v = (float(frame_info["valence"]) - 5.0) / 5.0
                a = (float(frame_info["arousal"]) - 5.0) / 5.0
                va_list.append([v, a])

            return np.array(va_list, dtype=np.float32)  # (n_frames, 2)
        except Exception as e:
            print(f"[AFEW-VA] Error reading {json_path}: {e}")
            return None

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        au_path, va_labels = self.clips[idx]

        # Load pre-extracted AUs: (n_frames, 17)
        try:
            aus = np.load(str(au_path))
        except Exception:
            aus = np.zeros((len(va_labels), FACE_DIM), dtype=np.float32)

        n_frames = min(len(aus), len(va_labels))
        if n_frames < self.T:
            # Pad with zeros if clip is too short (shouldn't happen after filtering)
            face = np.zeros((self.T, FACE_DIM), dtype=np.float32)
            face[:n_frames] = aus[:n_frames]
            va   = va_labels[-1]  # use last label
        else:
            # Random window for augmentation
            max_start = n_frames - self.T
            start = int(self.rng.integers(0, max_start + 1))
            face  = aus[start: start + self.T].astype(np.float32)     # (T, 17)
            # Mean VA over the selected window
            va    = va_labels[start: start + self.T].mean(axis=0)     # (2,)

        va = np.clip(va, -1.0, 1.0).astype(np.float32)

        return (
            face,                         # (T, 17)
            zeros_physio(self.T),         # (T, 6)  — no physio
            zeros_eeg(self.T),            # (T, 5)  — no EEG
            va,                           # (2,)
        )
