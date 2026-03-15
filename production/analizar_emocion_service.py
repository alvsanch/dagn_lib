"""
analizar_emocion_service.py — dagn_lib production service

Handles ALL signal processing:
  - MediaPipe FaceMesh → 17 AU-proxy features per frame
  - NeuroKit2 → HRV (RMSSD, SDNN, HR), EDA decomposition, TEMP normalization
  - NeuroSky EEG approximation → 5 band/ratio features from att/med
  - FusionLSTM inference → valence, arousal, temporal gradient

Request: one raw sensor sample per call (IR, GSR, TEMP, ATT, MED, session_id)
Response: VA + derived metrics for dashboard display

Dashboard calls this at ~800ms intervals (autorefresh).
"""

import os
import glob
import sys
import logging
import threading
import time
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from collections import deque
from typing import Optional

# Resolve paths for training module imports
_PROD_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_DIR = os.path.join(_PROD_DIR, "..", "training")
sys.path.insert(0, _PROD_DIR)
sys.path.insert(0, _TRAIN_DIR)

from fusion_model import FusionLSTM                              # noqa: E402
from feature_extractor_physio import extract_physio_features     # noqa: E402
from feature_extractor_face import FaceFeatureExtractor          # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(_PROD_DIR, "fusion_best.pth")
FRAMES_ROOT = "/mnt/c/biometria_tesis"
T = 30                   # timesteps (= buffer depth)

# ─── Load model ───────────────────────────────────────────────────────────────

try:
    model = FusionLSTM().to(DEVICE)
    state       = torch.load(MODEL_PATH, map_location=DEVICE)
    model_state = model.state_dict()
    compatible  = {k: v for k, v in state.items()
                   if k in model_state and v.shape == model_state[k].shape}
    skipped = len(state) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    model.eval()
    logger.info(f"FusionLSTM loaded: {MODEL_PATH} "
                f"({len(compatible)}/{len(state)} layers"
                + (f", {skipped} skipped" if skipped else "") + ")")
except Exception as e:
    logger.error(f"Failed to load FusionLSTM: {e}")
    sys.exit(1)

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="DAGN-lib Emotion Service")


class AnalyzeRequest(BaseModel):
    session_id: str
    ir:   float              # raw IR value from ESP32 (BVP proxy at ~50 Hz)
    gsr:  float              # raw GSR (μS)
    temp: float              # skin temperature (°C)
    att:  float = 0.0        # NeuroSky Attention 0-100
    med:  float = 0.0        # NeuroSky Meditation 0-100


class AnalyzeResponse(BaseModel):
    status:   str            # "success" | "warming_up" | "error"
    valence:  float = 0.0
    arousal:  float = 0.0
    grad_v:   float = 0.0
    grad_a:   float = 0.0
    hr_bpm:   float = 0.0   # mean HR in BPM (from NeuroKit2, 0 if insufficient data)
    rmssd:    float = 0.0   # RMSSD in ms (0 if insufficient data)
    eda_mean: float = 0.0   # EDA tonic (z-scored, 0 if no EDA)
    warmup:   float = 0.0   # 0..1 buffer fill fraction


# ─── Service ──────────────────────────────────────────────────────────────────

class EmotionService:
    """
    Stateful service that maintains rolling sensor buffers and computes
    FusionLSTM predictions on each /analyze call.

    Buffers:
        bvp_deque / eda_deque / temp_deque — raw signal samples (one per call)
        face_buf  — (17,) AU vectors, one per call
        eeg_buf   — (5,) EEG-approximation vectors, one per call

    The service is called at the dashboard refresh rate (~1.25 Hz for 800ms).
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Raw signal rolling buffers (30 calls × ~1.25Hz ≈ 24s window)
        self.bvp_deque  = deque(maxlen=1500)   # large for when call rate varies
        self.eda_deque  = deque(maxlen=1500)
        self.temp_deque = deque(maxlen=1500)

        # Per-call feature buffers
        self.face_buf = deque(maxlen=T)   # (17,) AU vectors
        self.eeg_buf  = deque(maxlen=T)   # (5,)  EEG approx vectors

        # Face extractor (MediaPipe, lazy-initialised inside FaceFeatureExtractor)
        self.face_extractor  = FaceFeatureExtractor()
        self.last_frame_path = None
        self._last_au        = np.zeros(17, dtype=np.float32)

        # Call-rate tracking for dynamic sfreq estimation
        self._call_times = deque(maxlen=20)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _estimated_sfreq(self):
        """Return estimated call frequency in Hz from recent call timestamps."""
        if len(self._call_times) < 2:
            return 1.25   # default: 800ms refresh
        diffs = np.diff(list(self._call_times))
        return float(1.0 / (np.mean(diffs) + 1e-6))

    def _get_latest_frame(self, session_id: str) -> Optional[str]:
        """Return path to newest .jpg in the session folder, or None."""
        if not session_id:
            return None
        session_path = os.path.join(FRAMES_ROOT, session_id)
        if not os.path.exists(session_path):
            return None
        files = sorted(glob.glob(os.path.join(session_path, "*.jpg")))
        return files[-1] if files else None

    def _eeg_approx(self, att: float, med: float) -> np.ndarray:
        """
        Approximate 5 EEG features from NeuroSky Attention/Meditation (0-100).

        Maps to the same feature space as feature_extractor_eeg.py:
            [0] theta_log     — log(theta): high when low attention
            [1] alpha_log     — log(alpha): high when meditative
            [2] beta_log      — log(beta) : high when attentive
            [3] alpha_asym    — 0.0       : no frontal channels from NeuroSky
            [4] theta_alpha   — theta - alpha: cognitive load index

        References: Davidson (1988), Klimesch (1999)
        """
        att_n = np.clip(att / 100.0, 0.0, 1.0)
        med_n = np.clip(med / 100.0, 0.0, 1.0)
        theta = float(np.log1p((1.0 - att_n) * 0.5))
        alpha = float(np.log1p(med_n * 0.5))
        beta  = float(np.log1p(att_n * 0.5))
        asym  = 0.0
        ratio = theta - alpha
        return np.array([theta, alpha, beta, asym, ratio], dtype=np.float32)

    def _compute_physio(self, sfreq: float) -> np.ndarray:
        """
        Run NeuroKit2 physio feature extraction on current rolling buffers.
        Returns (T, 6) array; zeros where data is insufficient.
        """
        bvp_arr  = np.array(self.bvp_deque,  dtype=np.float32)
        eda_arr  = np.array(self.eda_deque,  dtype=np.float32)
        temp_arr = np.array(self.temp_deque, dtype=np.float32)

        min_bvp = max(5, int(sfreq * 5))   # at least 5 seconds for HRV
        min_eda = max(2, int(sfreq * 2))   # at least 2 seconds for EDA decomp

        return extract_physio_features(
            bvp=bvp_arr  if len(bvp_arr)  >= min_bvp else None,
            eda=eda_arr  if len(eda_arr)  >= min_eda else None,
            temp=temp_arr if len(temp_arr) > 0       else None,
            sfreq_bvp=sfreq,
            sfreq_eda=sfreq,
            sfreq_temp=sfreq,
            T=T,
        )

    # ── main handler ──────────────────────────────────────────────────────────

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        with self._lock:
            return self._analyze_locked(req)

    def _analyze_locked(self, req: AnalyzeRequest) -> AnalyzeResponse:
        now = time.monotonic()
        self._call_times.append(now)
        sfreq = self._estimated_sfreq()

        # 1. Extend raw signal buffers
        self.bvp_deque.append(float(req.ir))
        self.eda_deque.append(float(req.gsr))
        self.temp_deque.append(float(req.temp))

        # 2. Face AUs from latest frame on disk
        frame_path = self._get_latest_frame(req.session_id)
        if frame_path and frame_path != self.last_frame_path:
            try:
                import cv2
                img = cv2.imread(frame_path)
                if img is not None:
                    # extract_from_arrays returns (T, 17); T=1 here
                    feats = self.face_extractor.extract_from_arrays([img], T=1)
                    self._last_au = feats[0]
                    self.last_frame_path = frame_path
            except Exception as exc:
                logger.debug(f"Face AU extraction failed: {exc}")
        self.face_buf.append(self._last_au.copy())

        # 3. EEG approximation from NeuroSky att/med
        att = float(req.att) if np.isfinite(req.att) else 0.0
        med = float(req.med) if np.isfinite(req.med) else 0.0
        self.eeg_buf.append(self._eeg_approx(att, med))

        # 4. NeuroKit2 physio features
        physio_feat = self._compute_physio(sfreq)

        # 5. Warmup: wait until face_buf is full
        if len(self.face_buf) < T:
            return AnalyzeResponse(
                status="warming_up",
                warmup=len(self.face_buf) / T,
            )

        # 6. Assemble tensors: (1, T, dim)
        face_arr   = np.stack(list(self.face_buf))   # (T, 17)
        eeg_arr    = np.stack(list(self.eeg_buf))    # (T, 5)

        face_t   = torch.tensor(face_arr,   dtype=torch.float32).unsqueeze(0).to(DEVICE)
        physio_t = torch.tensor(physio_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        eeg_t    = torch.tensor(eeg_arr,    dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 7. FusionLSTM forward
        with torch.no_grad():
            va, grad = model(face_t, physio_t, eeg_t)

        va   = torch.nan_to_num(va)
        grad = torch.nan_to_num(grad)

        valence = float(va[0, -1, 0])
        arousal = float(va[0, -1, 1])
        grad_v  = float(grad[0, -1, 0])
        grad_a  = float(grad[0, -1, 1])

        # 8. Derived metrics for dashboard display
        # physio_feat[:, 2] = mean_HR_norm (BPM/100), replicated across T
        # physio_feat[:, 0] = RMSSD_norm  (ms/100),  replicated across T
        hr_bpm   = float(physio_feat[0, 2]) * 100.0
        rmssd    = float(physio_feat[0, 0]) * 100.0
        eda_mean = float(physio_feat[:, 3].mean())

        return AnalyzeResponse(
            status="success",
            valence=valence,
            arousal=arousal,
            grad_v=grad_v,
            grad_a=grad_a,
            hr_bpm=hr_bpm,
            rmssd=rmssd,
            eda_mean=eda_mean,
            warmup=1.0,
        )


# ─── Singleton service + route ────────────────────────────────────────────────

service = EmotionService()


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return service.analyze(req)
