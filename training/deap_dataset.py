"""
deap_dataset.py — DEAP dataset for dagn_lib

DEAP: 32 subjects, 40 trials, EEG 32ch @ 128Hz + BVP/GSR/TEMP
Labels: valence/arousal 1-9 → normalized to [-1, 1] via (x-5)/4

Features computed per trial (cached in memory after first load):
    face:   zeros (T, 17)  — no video in DEAP
    physio: (T, 6)         — HRV from BVP, EDA/SCR from GSR, TEMP
    eeg:    (T, 5)         — bandpower + frontal alpha asymmetry (F3/F4)

DEAP channel order (after preprocessing, 40 channels total):
    0-31: EEG (Fp1,AF3,F3,F7,FC5,FC1,C3,T7,CP5,CP1,P3,P7,PO3,O1,Oz,Pz,
               Fp2,AF4,Fz,F4,F8,FC6,FC2,Cz,C4,T8,CP6,CP2,P4,P8,PO4,O2)
    32: hEOG, 33: vEOG, 34: zEMG, 35: tEMG
    36: GSR, 37: Resp, 38: TEMP, 39: BVP
    (channels are 0-indexed in the preprocessed numpy arrays)

Reference:
    Koelstra et al. (2012). DEAP: A Database for Emotion Analysis Using
        Physiological Signals. IEEE TAC, 3(1), 18-31.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from feature_extractor_eeg_tgam2 import (
    extract_eeg_features_tgam2 as extract_eeg_features,
    zeros_eeg_tgam2 as zeros_eeg,
    DEAP_LEFT_FRONTAL,
    DEAP_RIGHT_FRONTAL,
)
from feature_extractor_physio import extract_physio_features, zeros_physio
from feature_extractor_face import zeros_face

# ─── Constants ────────────────────────────────────────────────────────────────
DEAP_DIR   = Path("/mnt/f/source_datasets/Fisiologico/DEAP/deap-dataset/data_preprocessed_python")
SFREQ      = 128     # Hz (after DEAP preprocessing)
N_SUBJECTS = 32
N_TRIALS   = 40
T          = 30      # timesteps per sample
WINDOW_SEC = 1.0     # seconds per timestep

# Physio channel indices (0-indexed)
CH_GSR  = 36
CH_RESP = 37
CH_TEMP = 38
CH_BVP  = 39

# EEG channels: 0-31
N_EEG_CH = 32
# Frontal for asymmetry: F3=index 2, F4=index 19
LEFT_CH  = DEAP_LEFT_FRONTAL   # 2  (F3)
RIGHT_CH = DEAP_RIGHT_FRONTAL  # 19 (F4)

WIN_SAMPLES = int(WINDOW_SEC * SFREQ)  # 128 samples per window
MIN_SAMPLES = T * WIN_SAMPLES          # need at least T seconds


class DEAPDataset(Dataset):
    """
    DEAP dataset loader using library-extracted features.

    Features are computed once at init and cached in memory.
    Each __getitem__ selects a random T-second window from the cached trial.

    Args:
        deap_dir: path to data_preprocessed_python/
        subjects: list of subject IDs (1-32), default all 32
        T:        number of timesteps per sample (default 30)
        seed:     random seed for reproducibility
    """

    def __init__(self, deap_dir=DEAP_DIR, subjects=None, T=T, seed=42):
        self.T    = T
        self.rng  = np.random.default_rng(seed)

        if subjects is None:
            subjects = list(range(1, N_SUBJECTS + 1))

        self.samples = []  # list of (face, physio, eeg, va) per trial
        self._build_cache(Path(deap_dir), subjects)

    def _build_cache(self, deap_dir, subjects):
        """Load all trials and compute features, storing in self.samples."""
        if not deap_dir.exists():
            print(f"[DEAP] WARNING: directory not found: {deap_dir}")
            return

        total = len(subjects) * N_TRIALS
        done  = 0
        for subj in subjects:
            fname = deap_dir / f"s{subj:02d}.dat"
            if not fname.exists():
                continue
            try:
                with open(fname, "rb") as f:
                    obj = pickle.load(f, encoding="latin1")
                data   = obj["data"]    # (40, 40, 8064)
                labels = obj["labels"]  # (40, 4) — val, ar, dom, lik
            except Exception as e:
                print(f"[DEAP] ERROR loading {fname}: {e}")
                continue

            for trial in range(N_TRIALS):
                sig   = data[trial]      # (40, n_samples)
                label = labels[trial]    # [valence, arousal, dominance, liking]

                eeg    = sig[:N_EEG_CH, :]  # (32, n_samples)
                bvp    = sig[CH_BVP,  :]    # (n_samples,)
                gsr    = sig[CH_GSR,  :]    # (n_samples,)
                temp_s = sig[CH_TEMP, :]    # (n_samples,)

                # EEG features — computed per-second window
                eeg_feat = extract_eeg_features(
                    eeg, SFREQ,
                    left_ch_idx=LEFT_CH,
                    right_ch_idx=RIGHT_CH,
                    window_sec=WINDOW_SEC,
                    T=T,
                )  # (T, 5)

                # Physio features — HRV from BVP, EDA from GSR, TEMP
                physio_feat = extract_physio_features(
                    bvp=bvp, eda=gsr, temp=temp_s,
                    sfreq_bvp=SFREQ, sfreq_eda=SFREQ, sfreq_temp=SFREQ,
                    T=T,
                )  # (T, 6)

                # VA: DEAP 1-9 → [-1, 1] via (x-5)/4
                valence = (float(label[0]) - 5.0) / 4.0
                arousal = (float(label[1]) - 5.0) / 4.0
                va = np.array([valence, arousal], dtype=np.float32)

                self.samples.append((
                    zeros_face(T),        # (T, 17) — no video
                    physio_feat,           # (T, 6)
                    eeg_feat,              # (T, 5)
                    va,                    # (2,) — same label for all T
                ))

                done += 1
                if done % 100 == 0:
                    print(f"[DEAP] {done}/{total} trials loaded", flush=True)

        print(f"[DEAP] Loaded {len(self.samples)} trials")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face, physio, eeg, va = self.samples[idx]
        # All features are already T timesteps — return directly
        return (
            face.copy(),     # (T, 17)
            physio.copy(),   # (T, 6)
            eeg.copy(),      # (T, 5)
            va.copy(),       # (2,)
        )
