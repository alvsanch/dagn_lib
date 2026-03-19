"""
wesad_dataset.py — WESAD dataset for dagn_lib

WESAD: 15 subjects, wrist sensor (Empatica E4)
    BVP @ 64 Hz, EDA @ 4 Hz, TEMP @ 4 Hz
    Labels: 1=baseline, 2=stress, 3=amusement, 4=meditation

No EEG, no face data.

VA labels assigned per condition:
    baseline    (1) → valence=0.0,  arousal=0.0
    stress      (2) → valence=-0.7, arousal=0.8
    amusement   (3) → valence=0.7,  arousal=0.6
    meditation  (4) → valence=0.3,  arousal=-0.3

(Same mapping as dagn_simple for consistency.)

Reference:
    Schmidt, P. et al. (2018). Introducing WESAD, a multimodal dataset for
        wearable stress and affect detection. ICMI 2018.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from feature_extractor_physio import extract_physio_features, zeros_physio
from feature_extractor_face import zeros_face
from feature_extractor_eeg_full import zeros_eeg_full as zeros_eeg

# ─── Constants ────────────────────────────────────────────────────────────────
WESAD_DIR     = Path(os.path.expanduser("~/datasets/WESAD"))
WESAD_MNT_DIR = Path("/mnt/f/source_datasets/Fisiologico/WESAD")

SFREQ_BVP  = 64   # Hz (Empatica E4 BVP)
SFREQ_EDA  = 4    # Hz
SFREQ_TEMP = 4    # Hz
SFREQ_LABEL= 700  # Hz (label signal) → downsampled to 4 Hz

T          = 30   # timesteps per window
WINDOW_SEC = 1.0  # seconds per timestep
WIN_SAMPLES_EDA = int(WINDOW_SEC * SFREQ_EDA)    # 4 samples @ 4Hz
WIN_SAMPLES_BVP = int(WINDOW_SEC * SFREQ_BVP)    # 64 samples @ 64Hz
MIN_WINDOW_SECS = T                               # need ≥ 30 seconds

# VA mapping
LABEL_TO_VA = {
    1: np.array([0.0,  0.0],  dtype=np.float32),   # baseline
    2: np.array([-0.7, 0.8],  dtype=np.float32),   # stress
    3: np.array([0.7,  0.6],  dtype=np.float32),   # amusement
    4: np.array([0.3, -0.3],  dtype=np.float32),   # meditation
}
VALID_LABELS = set(LABEL_TO_VA.keys())


def _find_wesad_dir():
    """Return WESAD root dir — prefer local copy for speed."""
    for p in [WESAD_DIR, WESAD_MNT_DIR]:
        if p.exists():
            return p
    return None


def _downsample(signal, from_hz, to_hz):
    """Simple mean-pool downsampling."""
    if from_hz <= to_hz:
        return signal
    factor = int(from_hz // to_hz)
    n_out  = len(signal) // factor
    return signal[:n_out * factor].reshape(n_out, factor).mean(axis=1)


class WESADDataset(Dataset):
    """
    WESAD dataset loader using library-extracted features.

    Extracts 30-second non-overlapping windows from each subject's BVP/EDA/TEMP
    signals, computing HRV and EDA features via NeuroKit2.

    Args:
        wesad_dir: path to WESAD root (auto-detected if None)
        subjects:  list of subject IDs (2-17, excluding 12), default all
        T:         timesteps per sample (default 30)
        seed:      random seed
    """

    def __init__(self, wesad_dir=None, subjects=None, T=T, seed=42):
        self.T   = T
        self.rng = np.random.default_rng(seed)

        if wesad_dir is None:
            wesad_dir = _find_wesad_dir()
        if wesad_dir is None:
            print("[WESAD] WARNING: dataset directory not found.")
            self.samples = []
            return

        self.wesad_dir = Path(wesad_dir)

        if subjects is None:
            # Subject 12 missing from WESAD
            subjects = [s for s in range(2, 18) if s != 12]

        self.samples = []
        self._build_cache(subjects)

    def _build_cache(self, subjects):
        """Build all windows from all subjects."""
        for subj in subjects:
            subj_dir = self.wesad_dir / f"S{subj}"
            pkl_path = subj_dir / f"S{subj}.pkl"
            if not pkl_path.exists():
                continue
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
                self._process_subject(data)
            except Exception as e:
                print(f"[WESAD] ERROR loading S{subj}: {e}")

        print(f"[WESAD] Loaded {len(self.samples)} windows")

    def _process_subject(self, data):
        """Extract 30-sec windows from one subject's data."""
        wrist  = data["signal"]["wrist"]
        labels = data["label"].flatten()  # @ 700 Hz

        bvp  = wrist["BVP"].flatten()   # @ 64 Hz
        eda  = wrist["EDA"].flatten()   # @ 4 Hz
        temp = wrist["TEMP"].flatten()  # @ 4 Hz

        # Downsample labels to 4 Hz (same as EDA/TEMP)
        labels_4hz = _downsample(labels.astype(float), SFREQ_LABEL, SFREQ_EDA)
        labels_4hz = np.round(labels_4hz).astype(int)

        n_eda = len(eda)
        n_bvp = len(bvp)

        # Build non-overlapping windows of T*WIN_SAMPLES_EDA samples
        step_eda = T * WIN_SAMPLES_EDA     # 30 sec × 4 samp/sec = 120 samples
        step_bvp = T * WIN_SAMPLES_BVP     # 30 sec × 64 samp/sec = 1920 samples

        for start_eda in range(0, n_eda - step_eda + 1, step_eda):
            end_eda  = start_eda + step_eda

            # Dominant label in this window
            seg_labels = labels_4hz[start_eda:end_eda]
            valid_mask = np.isin(seg_labels, list(VALID_LABELS))
            if valid_mask.sum() < step_eda // 2:
                continue  # less than half of the window has valid labels

            counts = {lbl: (seg_labels == lbl).sum() for lbl in VALID_LABELS}
            dominant_label = max(counts, key=counts.get)
            va = LABEL_TO_VA[dominant_label]

            # Corresponding BVP window
            start_bvp = int(start_eda * SFREQ_BVP / SFREQ_EDA)
            end_bvp   = start_bvp + step_bvp

            bvp_win  = bvp[start_bvp:end_bvp] if end_bvp <= n_bvp else None
            eda_win  = eda[start_eda:end_eda]
            temp_win = temp[start_eda:end_eda]

            physio_feat = extract_physio_features(
                bvp=bvp_win,
                eda=eda_win,
                temp=temp_win,
                sfreq_bvp=SFREQ_BVP,
                sfreq_eda=SFREQ_EDA,
                sfreq_temp=SFREQ_TEMP,
                T=T,
            )  # (T, 6)

            self.samples.append((
                zeros_face(T),     # (T, 17) — no video
                physio_feat,        # (T, 6)
                zeros_eeg(T),       # (T, 5)  — no EEG in WESAD wrist
                va,                 # (2,)
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
