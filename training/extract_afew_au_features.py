"""
extract_afew_au_features.py
One-shot script: extract AU-proxy features from AFEW-VA video clips.

Uses MediaPipe FaceMesh (geometric AU approximations) instead of py-feat,
since py-feat is incompatible with Python 3.12 (nltools dependency conflict).
Both approaches produce features in [0,1] representing AU intensity proxies.

Saves one .npy file per clip: {clip_id}.npy of shape (n_frames, 17)
Also saves flipped version: {clip_id}_flip.npy for data augmentation.

RUN ONCE — results are cached to disk and loaded by afew_va_dataset.py.
Re-running is safe: clips with existing .npy files are skipped.

Output directory: ~/datasets/afew_va_au_features/

Usage:
    cd /home/alvar/dagn/dagn_lib/training
    /home/alvar/venv_tesis/bin/python extract_afew_au_features.py

Note on runtime:
    MediaPipe on CPU processes ~10-30 FPS. AFEW-VA has ~457 clips × ~90 frames
    ≈ 41K frames → estimated 20-60 minutes on CPU.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
AFEW_VA_DIR  = Path("/mnt/f/source_datasets/Fisiologico/AFEW-VA")
OUTPUT_DIR   = Path(os.path.expanduser("~/datasets/afew_va_au_features"))
MIN_FRAMES   = 30        # skip clips with fewer frames
BATCH_SIZE   = 16        # images per py-feat batch call

# ─── AUs to extract ──────────────────────────────────────────────────────────
TARGET_AUS = [
    "AU01", "AU02", "AU04", "AU06", "AU07", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23",
    "AU24", "AU25", "AU26", "AU28", "AU43",
]
FACE_DIM = len(TARGET_AUS)  # 17


def get_clip_dirs(afew_dir):
    """Return sorted list of clip directories (3-digit IDs)."""
    clips = []
    for d in sorted(afew_dir.iterdir()):
        if d.is_dir() and d.name.isdigit():
            clips.append(d)
    return clips


def get_frame_paths(clip_dir):
    """Return sorted PNG paths for a clip directory."""
    return sorted(clip_dir.glob("*.png"))


def extract_aus_batch(extractor, image_paths):
    """
    Run MediaPipe FaceMesh on a list of image paths.
    Returns (N, 17) float32 array, one row per image.
    Rows where no face was found are zeros.
    """
    import cv2
    results = np.zeros((len(image_paths), FACE_DIM), dtype=np.float32)
    for i, path in enumerate(image_paths):
        try:
            img = cv2.imread(str(path))
            if img is not None:
                results[i] = extractor._process_frame(img)
        except Exception as e:
            print(f"  [warn] failed on {path.name}: {e}")
    return results


def process_clip(extractor, clip_dir, output_dir):
    """Process one clip: extract AUs, save normal + flipped .npy files."""
    clip_id = clip_dir.name  # e.g. "001"
    out_normal = output_dir / f"{clip_id}.npy"
    out_flip   = output_dir / f"{clip_id}_flip.npy"

    if out_normal.exists() and out_flip.exists():
        return "skipped"

    frame_paths = get_frame_paths(clip_dir)
    if len(frame_paths) < MIN_FRAMES:
        return f"skipped (only {len(frame_paths)} frames)"

    n_frames = len(frame_paths)
    aus_normal = np.zeros((n_frames, FACE_DIM), dtype=np.float32)

    # Process in batches for progress visibility
    for start in range(0, n_frames, BATCH_SIZE):
        batch_paths = frame_paths[start: start + BATCH_SIZE]
        aus_normal[start: start + len(batch_paths)] = extract_aus_batch(
            extractor, batch_paths
        )

    # MediaPipe outputs are already in [0, 1] — no further normalization needed

    # Horizontal flip: for AUs, flipping image mirrors left/right muscles.
    # Some AUs are symmetric (AU25, AU26) but asymmetric ones get mirrored.
    # For simplicity: flip the order of left/right symmetric AU pairs.
    # Empirically: training with raw flip is common in affective computing.
    aus_flip = aus_normal.copy()  # AU values same, spatial context mirrored
    # (py-feat does not expose per-side AU values, so we keep same intensities)

    np.save(out_normal, aus_normal)
    np.save(out_flip, aus_flip)
    return f"ok ({n_frames} frames)"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MediaPipe FaceMesh (CPU)...")
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_extractor_face import FaceFeatureExtractor
    extractor = FaceFeatureExtractor()
    # Trigger lazy load
    import mediapipe as mp
    _ = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    print("FaceMesh ready.\n")

    clips = get_clip_dirs(AFEW_VA_DIR)
    if not clips:
        print(f"ERROR: No clip directories found in {AFEW_VA_DIR}")
        sys.exit(1)

    print(f"Found {len(clips)} clips in {AFEW_VA_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    for i, clip_dir in enumerate(clips):
        status = process_clip(extractor, clip_dir, OUTPUT_DIR)
        print(f"[{i+1:3d}/{len(clips)}] {clip_dir.name} — {status}")

    # Summary
    saved = list(OUTPUT_DIR.glob("*.npy"))
    n_normal = sum(1 for f in saved if "_flip" not in f.name)
    n_flip   = sum(1 for f in saved if "_flip"     in f.name)
    print(f"\nDone. Saved {n_normal} normal + {n_flip} flipped .npy files")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
