"""
extract_affec_features.py
One-shot feature extraction for the AFFEC dataset.

AFFEC (Jamshidi Sekiavandi et al., 2024): 72 subjects, 273 runs
Modalities:
  - EEG: 63 channels @ 256 Hz (g.tec g.HIamp, EDF format)
  - GSR: Shimmer @ 52 Hz (GSR_Conductance_cal in μS + Temperature)
  - Face: OpenFace2 AUs already extracted in videostream @ 38 Hz
Labels: discrete emotions (happy, sad, neutral, fear) → mapped to VA circumplex

Output: ~/datasets/affec_features/{subj}_run-{r}_trial-{t}.npz
  Keys: face (T,17), physio (T,6), eeg (T,5), va (2,)

Run ONCE — idempotent (skips existing .npz files).

Usage:
    cd /home/alvar/dagn_lib/training
    /home/alvar/venv_tesis/bin/python extract_affec_features.py
"""
import os
import sys
import gzip
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
AFFEC_DIR   = Path("/mnt/f/source_datasets/Fisiologico/AFFEC")
OUTPUT_DIR  = Path(os.path.expanduser("~/datasets/affec_features"))
T           = 30       # timesteps per sample
MIN_DUR     = 20.0     # minimum scenario duration in seconds

# ─── VA label mapping (Russell's circumplex model) ────────────────────────────
# Reference: Russell, J.A. (1980). A circumplex model of affect.
#            Journal of Personality and Social Psychology, 39(6), 1161-1178.
EMOTION_TO_VA = {
    "happy":   np.array([ 0.8,  0.6], dtype=np.float32),
    "neutral": np.array([ 0.0,  0.0], dtype=np.float32),
    "sad":     np.array([-0.7, -0.2], dtype=np.float32),
    "fear":    np.array([-0.6,  0.7], dtype=np.float32),
}

# ─── AFFEC AU → dagn_lib TARGET_AUS mapping ──────────────────────────────────
# TARGET_AUS order (17): AU01,AU02,AU04,AU06,AU07,AU10,AU12,AU14,AU15,AU17,
#                        AU20,AU23,AU24,AU25,AU26,AU28,AU43
# AFFEC videostream _r columns (continuous 0-5):
#   AU01_r,AU02_r,AU04_r,AU06_r,AU07_r,AU10_r,AU12_r,AU14_r,AU15_r,AU17_r,
#   AU20_r,AU23_r, [AU24 missing→0], AU25_r,AU26_r, AU28_c (binary), AU45_r→AU43
AU_MAPPING = [
    ("AU01_r", False),  # 0: AU01
    ("AU02_r", False),  # 1: AU02
    ("AU04_r", False),  # 2: AU04
    ("AU06_r", False),  # 3: AU06
    ("AU07_r", False),  # 4: AU07
    ("AU10_r", False),  # 5: AU10
    ("AU12_r", False),  # 6: AU12
    ("AU14_r", False),  # 7: AU14
    ("AU15_r", False),  # 8: AU15
    ("AU17_r", False),  # 9: AU17
    ("AU20_r", False),  # 10: AU20
    ("AU23_r", False),  # 11: AU23
    (None,     False),  # 12: AU24 — not in AFFEC videostream _r, use zero
    ("AU25_r", False),  # 13: AU25
    ("AU26_r", False),  # 14: AU26
    ("AU28_c", True),   # 15: AU28 — only binary available (0 or 1)
    ("AU45_r", False),  # 16: AU43 → AU45_r (same blink/closure muscle)
]
FACE_DIM = len(AU_MAPPING)  # 17

# EEG channels for frontal alpha asymmetry
# AFFEC 63-ch 10-20 layout: F3=index 9, F4=index 13
AFFEC_LEFT_FRONTAL  = 9   # F3
AFFEC_RIGHT_FRONTAL = 13  # F4
SFREQ_EEG = 256.0

# GSR columns in Shimmer TSV
GSR_COL  = "GSR_Conductance_cal"   # μS
TEMP_COL = "Temperature_cal"       # °C
SFREQ_GSR = 52.0

# Videostream sampling rate
SFREQ_VIDEO = 38.0  # Hz (approximate)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def read_tsv_gz_from_zip(zf, path, col_names=None):
    """
    Read a gzipped TSV file from an open ZipFile, return DataFrame.
    AFFEC TSVs have NO header row — column names come from the JSON sidecar.
    If col_names is None, falls back to reading with header (first row as names).
    Some files may have an embedded header row → coerce 'onset' to numeric and drop stray rows.
    """
    with zf.open(path) as raw:
        with gzip.open(raw) as gz:
            if col_names is not None:
                df = pd.read_csv(gz, sep="\t", header=None, names=col_names)
            else:
                df = pd.read_csv(gz, sep="\t")
    # Coerce 'onset' to float and drop any non-numeric rows (e.g. embedded header)
    if "onset" in df.columns:
        df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
        df = df.dropna(subset=["onset"]).reset_index(drop=True)
    return df


def read_json_cols_from_zip(zf, path):
    """Read the JSON sidecar and return the 'Columns' list."""
    with zf.open(path) as f:
        return json.load(f).get("Columns", [])


def pool_to_T(arr, T):
    """Mean-pool a 1D or 2D (N, D) array to T timesteps."""
    n = arr.shape[0]
    if n == 0:
        return np.zeros((T,) + arr.shape[1:], dtype=np.float32)
    chunk = max(1, n // T)
    rows = [arr[i * chunk: (i + 1) * chunk] for i in range(T)]
    # Handle rows shorter than chunk (last few)
    return np.stack([r.mean(axis=0) if len(r) > 0 else arr[-1]
                     for r in rows], axis=0).astype(np.float32)


# ─── Feature extraction per modality ─────────────────────────────────────────
def extract_au_features(video_df, onset, duration, T):
    """Extract AU features from OpenFace2 videostream DataFrame."""
    end = onset + min(duration, T * 1.5)
    seg = video_df[(video_df["onset"] >= onset) &
                   (video_df["onset"] < end)].copy()

    if len(seg) < 5:
        return np.zeros((T, FACE_DIM), dtype=np.float32)

    aus = np.zeros((len(seg), FACE_DIM), dtype=np.float32)
    for i, (col, is_binary) in enumerate(AU_MAPPING):
        if col is not None and col in seg.columns:
            vals = seg[col].fillna(0).values.astype(np.float32)
            vals = np.clip(vals, 0, None)
            if not is_binary:
                vals = vals / 5.0  # normalize [0,5] → [0,1]
            aus[:, i] = vals

    return pool_to_T(aus, T)  # (T, 17)


def extract_physio_features_affec(gsr_df, onset, duration, T):
    """Extract physio features from AFFEC Shimmer GSR DataFrame."""
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_extractor_physio import extract_physio_features

    end = onset + duration + 5.0  # slight buffer
    seg = gsr_df[(gsr_df["onset"] >= onset) & (gsr_df["onset"] < end)].copy()

    if len(seg) < int(SFREQ_GSR * 5):
        return np.zeros((T, 6), dtype=np.float32)

    eda  = seg[GSR_COL].fillna(0).values.astype(np.float32)  if GSR_COL  in seg.columns else None
    temp = seg[TEMP_COL].fillna(34).values.astype(np.float32) if TEMP_COL in seg.columns else None

    return extract_physio_features(
        eda=eda, temp=temp,
        sfreq_eda=SFREQ_GSR, sfreq_temp=SFREQ_GSR,
        T=T,
    )  # (T, 6)


def extract_eeg_features_affec(edf_path, onset, duration, T):
    """Extract EEG features from AFFEC EDF file (MNE)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_extractor_eeg import extract_eeg_features

    try:
        import mne
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        sfreq = raw.info["sfreq"]

        start_s = max(0.0, onset)
        dur_s   = min(duration, T * 1.1)
        raw.crop(tmin=start_s, tmax=start_s + dur_s)
        raw.load_data(verbose=False)
        eeg_data = raw.get_data()  # (63, n_samples)

        return extract_eeg_features(
            eeg_data, sfreq,
            left_ch_idx=AFFEC_LEFT_FRONTAL,
            right_ch_idx=AFFEC_RIGHT_FRONTAL,
            window_sec=1.0,
            T=T,
        )  # (T, 5)
    except Exception as e:
        print(f"  [EEG warn] {e}")
        return np.zeros((T, 5), dtype=np.float32)


# ─── Main extraction logic ────────────────────────────────────────────────────
def get_scenarios(events_df):
    """Return list of (onset, duration, emotion) for 'scenario' segments."""
    scen = events_df[events_df["flag"] == "scenario"].copy()
    scen = scen[scen["trial_type"].isin(EMOTION_TO_VA.keys())]
    scen = scen[scen["duration"] >= MIN_DUR]
    return [(row["onset"], row["duration"], row["trial_type"])
            for _, row in scen.iterrows()]


def process_run(subj, run, events_df, core_zf, eeg_zf, gsr_zf, video_zf):
    """Process all scenario trials in one run. Returns list of sample dicts."""
    scenarios = get_scenarios(events_df)
    if not scenarios:
        return []

    # Pre-load TSV data for this run (read once, used for all scenarios)
    gsr_prefix   = f"{subj}/beh/{subj}_task-fer_run-{run}_recording-gsr_physio.tsv.gz"
    gsr_json     = f"{subj}/beh/{subj}_task-fer_run-{run}_recording-gsr_physio.json"
    video_prefix = f"{subj}/beh/{subj}_task-fer_run-{run}_recording-videostream_physio.tsv.gz"
    video_json   = f"{subj}/beh/{subj}_task-fer_run-{run}_recording-videostream_physio.json"
    edf_prefix   = f"{subj}/eeg/{subj}_task-fer_run-{run}_eeg.edf"

    gsr_all_files   = gsr_zf.namelist()
    video_all_files = video_zf.namelist()

    try:
        gsr_cols = read_json_cols_from_zip(gsr_zf, gsr_json) if gsr_json in gsr_all_files else None
        gsr_df   = read_tsv_gz_from_zip(gsr_zf, gsr_prefix, col_names=gsr_cols) \
                   if gsr_prefix in gsr_all_files else None
        if gsr_df is not None and "onset" not in gsr_df.columns:
            gsr_df = None
    except Exception:
        gsr_df = None

    try:
        vid_cols  = read_json_cols_from_zip(video_zf, video_json) if video_json in video_all_files else None
        video_df  = read_tsv_gz_from_zip(video_zf, video_prefix, col_names=vid_cols) \
                    if video_prefix in video_all_files else None
        # Without JSON sidecar, column names may be missing — skip gracefully
        if video_df is not None and "onset" not in video_df.columns:
            video_df = None
    except Exception:
        video_df = None

    # Extract EDF to temp file
    edf_tmp = None
    if edf_prefix in eeg_zf.namelist():
        try:
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp.write(eeg_zf.read(edf_prefix))
                edf_tmp = tmp.name
        except Exception:
            edf_tmp = None

    samples = []
    for i, (onset, duration, emotion) in enumerate(scenarios):
        va = EMOTION_TO_VA[emotion]

        face_feat   = extract_au_features(video_df, onset, duration, T)   if video_df is not None else np.zeros((T, FACE_DIM), dtype=np.float32)
        physio_feat = extract_physio_features_affec(gsr_df, onset, duration, T) if gsr_df  is not None else np.zeros((T, 6), dtype=np.float32)
        eeg_feat    = extract_eeg_features_affec(edf_tmp, onset, duration, T)   if edf_tmp is not None else np.zeros((T, 5), dtype=np.float32)

        samples.append({
            "id":     f"{subj}_run-{run}_scen-{i}",
            "face":   face_feat,
            "physio": physio_feat,
            "eeg":    eeg_feat,
            "va":     va,
            "emotion": emotion,
        })

    if edf_tmp and os.path.exists(edf_tmp):
        os.unlink(edf_tmp)

    return samples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Opening AFFEC zip files...")
    core_zip  = zipfile.ZipFile(AFFEC_DIR / "core.zip",        "r")
    eeg_zip   = zipfile.ZipFile(AFFEC_DIR / "eeg.zip",         "r")
    gsr_zip   = zipfile.ZipFile(AFFEC_DIR / "gsr.zip",         "r")
    video_zip = zipfile.ZipFile(AFFEC_DIR / "videostream.zip", "r")

    # Find all subjects
    subjects = sorted(set(
        n.split("/")[0] for n in core_zip.namelist()
        if n.startswith("sub-") and "/" in n
    ))
    print(f"Found {len(subjects)} subjects\n")

    total_saved = 0
    total_skipped = 0

    for subj_i, subj in enumerate(subjects):
        # Find all runs for this subject
        events_files = [n for n in core_zip.namelist()
                        if n.startswith(f"{subj}/") and n.endswith("_events.tsv")]

        for events_path in sorted(events_files):
            # Parse run number from filename
            fname = Path(events_path).name  # e.g. sub-acl_task-fer_run-0_events.tsv
            run_str = [p for p in fname.split("_") if p.startswith("run-")]
            run = run_str[0].split("-")[1] if run_str else "0"

            # Check if all outputs exist
            scenarios_key = f"{subj}_run-{run}"
            sample_files  = list(OUTPUT_DIR.glob(f"{scenarios_key}_scen-*.npz"))
            # Re-read events to know expected count
            with core_zip.open(events_path) as f:
                events_df = pd.read_csv(f, sep="\t")
            expected = len(get_scenarios(events_df))

            if expected > 0 and len(sample_files) >= expected:
                total_skipped += expected
                continue

            # Process this run
            samples = process_run(
                subj, run, events_df,
                core_zip, eeg_zip, gsr_zip, video_zip
            )

            for s in samples:
                out_path = OUTPUT_DIR / f"{s['id']}.npz"
                np.savez(
                    out_path,
                    face   = s["face"],
                    physio = s["physio"],
                    eeg    = s["eeg"],
                    va     = s["va"],
                )
                total_saved += 1

            print(f"[{subj_i+1:2d}/{len(subjects)}] {subj} run-{run}: "
                  f"{len(samples)} scenarios saved", flush=True)

    core_zip.close()
    eeg_zip.close()
    gsr_zip.close()
    video_zip.close()

    all_npz = list(OUTPUT_DIR.glob("*.npz"))
    print(f"\nDone. Total: {total_saved} saved + {total_skipped} skipped")
    print(f"Total .npz files: {len(all_npz)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
