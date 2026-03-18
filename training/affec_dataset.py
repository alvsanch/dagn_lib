"""
affec_dataset.py — AFFEC dataset for dagn_lib

AFFEC (Jamshidi Sekiavandi et al., 2024):
    72 subjects, multimodal emotion recognition
    EEG 63ch @ 256Hz (g.tec), GSR Shimmer @ 52Hz, OpenFace2 AUs @ 38Hz
    Discrete labels (happy, sad, neutral, fear) → VA circumplex

Requires pre-extracted features:
    Run extract_affec_features.py first → ~/datasets/affec_features/*.npz

Each .npz contains: face (T,17), physio (T,6), eeg (T,5), va (2,)

VA mapping (Russell 1980 circumplex):
    happy:   V=+0.8, A=+0.6
    neutral: V=0.0,  A=0.0
    sad:     V=-0.7, A=-0.2
    fear:    V=-0.6, A=+0.7

Reference:
    Jamshidi Sekiavandi, M. et al. (2024). Advancing Face-to-Face Emotion
        Communication: A Multimodal Dataset (AFFEC).
        IEEE Transactions on Affective Computing.
"""
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from feature_extractor_physio import zeros_physio
from feature_extractor_eeg    import zeros_eeg
from feature_extractor_face   import zeros_face

# ─── Constants ────────────────────────────────────────────────────────────────
AFFEC_FEAT_DIR = Path("/home/alvar/datasets/affec_features")
T              = 30


class AFFECDataset(Dataset):
    """
    AFFEC dataset loader using pre-extracted features.

    Loads .npz files produced by extract_affec_features.py.
    Each sample has face (AUs from OpenFace2), EEG bandpower,
    GSR/TEMP physio features, and discrete emotion → VA label.

    Args:
        feat_dir: path to directory with .npz files
        T:        timesteps per sample (default 30)
        seed:     random seed
    """

    def __init__(self, feat_dir=AFFEC_FEAT_DIR, T=T, seed=42):
        self.T   = T
        self.rng = np.random.default_rng(seed)

        feat_dir = Path(feat_dir)
        if not feat_dir.exists():
            print(f"[AFFEC] WARNING: features not found at {feat_dir}")
            print("         Run: python extract_affec_features.py")
            self.samples = []
            return

        self.npz_files = sorted(feat_dir.glob("*.npz"))
        if not self.npz_files:
            print(f"[AFFEC] WARNING: no .npz files in {feat_dir}")
            self.samples = []
            return

        # Pre-load all samples into memory (small arrays, fast access)
        self.samples = []
        for path in self.npz_files:
            try:
                d = np.load(str(path))
                face   = d["face"].astype(np.float32)    # (T, 17)
                physio = d["physio"].astype(np.float32)  # (T, 6)
                # EEG zeroed: 63-ch g.tec → 2-ch TGAM2 approximation is unreliable.
                # TGAM2 frontal electrode indices (ch9, ch13) may not correspond to
                # F3/F4 in the actual EDF channel order. Face+physio carry AFFEC.
                eeg    = np.zeros((d["face"].shape[0], 5), dtype=np.float32)
                va     = d["va"].astype(np.float32)      # (2,)

                # Verify shapes
                if face.shape != (T, 17) or physio.shape != (T, 6) or eeg.shape != (T, 5):
                    continue

                self.samples.append((face, physio, eeg, va))
            except Exception as e:
                print(f"[AFFEC] Error loading {path.name}: {e}")

        print(f"[AFFEC] Loaded {len(self.samples)} samples from {feat_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face, physio, eeg, va = self.samples[idx]
        return (
            face.copy(),
            physio.copy(),
            eeg.copy(),
            va.copy(),
        )
