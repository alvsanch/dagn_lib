"""
dreamer_dataset.py — DREAMER dataset for dagn_lib

DREAMER: 23 subjects, 18 stimuli
    EEG 14ch (Emotiv EPOC) @ 128 Hz
    ECG 1ch @ 256 Hz
    Labels: valence/arousal 1-5 (Likert) → normalized to [-1, 1] via (x-3)/2

No face data.

DREAMER EPOC channel order:
    AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    idx: 0   1   2   3   4   5   6   7   8   9   10  11  12  13

Frontal alpha asymmetry (DASM, Davidson 1988):
    Left  = F3 (index 2)
    Right = F4 (index 11)

Reference:
    Katsigiannis, S. & Ramzan, N. (2018). DREAMER: A database for emotion
        recognition through EEG and ECG signals from wireless low-cost
        off-the-shelf devices. IEEE JBHI, 22(1), 98-107.
"""
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

from feature_extractor_eeg_full import (
    extract_eeg_features_full,
    zeros_eeg_full,
    DREAMER_LEFT_FRONTAL_FULL,
    DREAMER_RIGHT_FRONTAL_FULL,
)
from feature_extractor_physio import extract_physio_features, zeros_physio
from feature_extractor_face import zeros_face

# ─── Constants ────────────────────────────────────────────────────────────────
DREAMER_MAT  = Path("/mnt/f/source_datasets/Fisiologico/DREAMER/DREAMER.mat")
SFREQ_EEG    = 128    # Hz
SFREQ_ECG    = 256    # Hz
N_SUBJECTS   = 23
N_STIMULI    = 18
T            = 30     # timesteps per sample
WINDOW_SEC   = 1.0    # seconds per timestep

# Bilateral frontal: F7=1,F3=2,FC5=3 (left) | FC6=10,F4=11,F8=12 (right)
LEFT_CHS  = DREAMER_LEFT_FRONTAL_FULL   # [1, 2, 3]
RIGHT_CHS = DREAMER_RIGHT_FRONTAL_FULL  # [10, 11, 12]


def _load_dreamer_mat(mat_path):
    """
    Load DREAMER .mat file.
    Returns nested structure: DREAMER[subj][stim] with EEG/ECG arrays.
    """
    if loadmat is None:
        raise ImportError("scipy is required: pip install scipy")
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    return mat["DREAMER"]


class DREAMERDataset(Dataset):
    """
    DREAMER dataset loader using library-extracted features.

    Args:
        mat_path: path to DREAMER.mat
        subjects: list of subject indices (0-22), default all 23
        T:        timesteps per sample (default 30)
        seed:     random seed
    """

    def __init__(self, mat_path=DREAMER_MAT, subjects=None, T=T, seed=42):
        self.T   = T
        self.rng = np.random.default_rng(seed)

        if not Path(mat_path).exists():
            print(f"[DREAMER] WARNING: {mat_path} not found")
            self.samples = []
            return

        if subjects is None:
            subjects = list(range(N_SUBJECTS))

        self.samples = []
        self._build_cache(mat_path, subjects)

    def _build_cache(self, mat_path, subjects):
        """Load all trials and compute features."""
        try:
            dreamer = _load_dreamer_mat(mat_path)
        except Exception as e:
            print(f"[DREAMER] ERROR loading mat: {e}")
            return

        for subj_idx in subjects:
            try:
                self._process_subject(dreamer, subj_idx)
            except Exception as e:
                print(f"[DREAMER] ERROR subject {subj_idx}: {e}")

        print(f"[DREAMER] Loaded {len(self.samples)} trials")

    def _process_subject(self, dreamer, subj_idx):
        """Extract features from all stimuli for one subject."""
        subj = dreamer.Data[subj_idx]

        for stim_idx in range(N_STIMULI):
            # stimuli[stim_idx] IS the raw numpy array (n_samples, 14)
            eeg_raw = np.array(subj.EEG.stimuli[stim_idx], dtype=np.float32)

            # EEG: (n_samples, 14) → transpose to (14, n_samples)
            if eeg_raw.ndim == 2 and eeg_raw.shape[1] == 14:
                eeg_raw = eeg_raw.T  # (14, n_samples)
            elif eeg_raw.ndim == 2 and eeg_raw.shape[0] == 14:
                pass  # already (14, n_samples)
            else:
                continue  # unexpected shape

            n_samples = eeg_raw.shape[1]
            min_samples = T * int(WINDOW_SEC * SFREQ_EEG)
            if n_samples < min_samples:
                continue

            # ECG: stimuli[stim_idx] is (n_samples, 2) — use first lead
            try:
                ecg_arr = np.array(subj.ECG.stimuli[stim_idx], dtype=np.float32)
                ecg_raw = ecg_arr[:, 0] if ecg_arr.ndim == 2 else ecg_arr.flatten()
            except Exception:
                ecg_raw = None

            # EEG features — bilateral frontal (10D)
            eeg_feat = extract_eeg_features_full(
                eeg_raw, SFREQ_EEG,
                left_ch_idxs=LEFT_CHS,
                right_ch_idxs=RIGHT_CHS,
                window_sec=WINDOW_SEC,
                T=T,
            )  # (T, 5)

            # Physio features from ECG
            physio_feat = extract_physio_features(
                ecg=ecg_raw,
                sfreq_ecg=SFREQ_ECG,
                T=T,
            )  # (T, 6)

            # VA labels: DREAMER 1-5 → [-1, 1] via (x-3)/2
            try:
                val = float(subj.ScoreValence[stim_idx])
                ar  = float(subj.ScoreArousal[stim_idx])
            except (AttributeError, IndexError):
                continue

            va = np.array([(val - 3.0) / 2.0, (ar - 3.0) / 2.0], dtype=np.float32)
            va = np.clip(va, -1.0, 1.0)

            self.samples.append((
                zeros_face(T),   # (T, 17) — no video
                physio_feat,      # (T, 6)
                eeg_feat,         # (T, 10)
                va,               # (2,)
            ))

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
