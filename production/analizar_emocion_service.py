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
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
from collections import deque
from typing import Optional, List, Tuple
from scipy.signal import butter, filtfilt, welch

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Resolve paths for training module imports
_PROD_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_DIR = os.path.join(_PROD_DIR, "..", "training")
# _PROD_DIR already on sys.path when uvicorn loads this module from production/
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
    if len(compatible) < 3:
        raise RuntimeError(
            f"Checkpoint mismatch: only {len(compatible)}/{len(state)} layers compatible. "
            "Wrong checkpoint or wrong architecture?"
        )
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
    # Batches: ALL new physio samples since last call (50 Hz from ESP32)
    # When present, replace the single-value fields below for HRV/EDA/SpO2.
    ir_batch:   List[float] = []   # IR (BVP proxy) samples
    red_batch:  List[float] = []   # RED samples (SpO2)
    gsr_batch:  List[float] = []   # GSR (EDA) samples
    temp_batch: List[float] = []   # skin temperature samples
    # Latest single values (always present, used for display)
    ir:   float = 0.0
    red:  float = 0.0
    gsr:  float = 0.0
    temp: float = 0.0
    att:  float = 0.0        # NeuroSky Attention 0-100
    med:  float = 0.0        # NeuroSky Meditation 0-100


class AnalyzeResponse(BaseModel):
    status:     str           # "success" | "warming_up" | "error"
    valence:    float = 0.0
    arousal:    float = 0.0
    grad_v:     float = 0.0
    grad_a:     float = 0.0
    face_v:     float = 0.0   # valence from face-only (physio+EEG zeroed)
    face_a:     float = 0.0   # arousal from face-only
    hr_bpm:     float = 0.0   # sensor HR from BVP peaks (NeuroKit2)
    rmssd:      float = 0.0   # RMSSD ms (0 if insufficient data)
    eda_mean:   float = 0.0   # EDA tonic z-scored
    spo2:       float = 0.0   # SpO2 % estimated from IR/RED ratio
    cam_hr:     float = 0.0   # HR from camera rPPG (BPM)
    cam_resp:   float = 0.0   # respiration rate from camera rPPG (breaths/min)
    blink_rate: float = 0.0   # blinks per minute (from AU43)
    warmup:     float = 0.0   # 0..1 buffer fill fraction


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

        # True physio sampling rate — ESP32 publishes at 50 Hz (fixed)
        self._physio_sfreq   = 50.0   # do not estimate dynamically

        # EMA smoothing for fusion VA output (avoids tanh saturation jumps)
        self._va_ema         = np.zeros(2, dtype=np.float32)
        self._ema_init       = False
        self.EMA_ALPHA       = 0.25   # 0=frozen, 1=no smoothing

        # SpO2: rolling RED channel buffer (parallel to bvp_deque for IR)
        self.red_deque       = deque(maxlen=1500)

        # rPPG: green channel from forehead ROI for camera HR and resp rate
        self.rppg_green      = deque(maxlen=1800)   # ~60s at camera fps
        self.rppg_ts         = deque(maxlen=1800)   # per-frame file mtimes
        self._last_rppg_file = None                  # last processed frame path

        # Blinks: AU43 at camera frame rate (more reliable than 1 Hz service calls)
        self.ear_hires       = deque(maxlen=600)    # ~2 min at 5fps
        self.ear_hires_ts    = deque(maxlen=600)

        # Session tracking: reset buffers when session_id changes
        self._current_session = None

        # Cached camera metrics — updated every call (before warmup check)
        self._cam_hr    = 0.0
        self._cam_resp  = 0.0
        self._blink_rate = 0.0

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
        Cap input to 15 s of data to bound NeuroKit2 processing time.
        """
        max_samples = int(sfreq * 15)   # 15 s cap → ~750 samples at 50 Hz

        bvp_arr  = np.array(list(self.bvp_deque) [-max_samples:], dtype=np.float32)
        eda_arr  = np.array(list(self.eda_deque) [-max_samples:], dtype=np.float32)
        temp_arr = np.array(list(self.temp_deque)[-max_samples:], dtype=np.float32)

        min_bvp = max(5, int(sfreq * 5))   # at least 5 s for HRV
        min_eda = max(2, int(sfreq * 2))   # at least 2 s for EDA decomp

        return extract_physio_features(
            bvp=bvp_arr  if len(bvp_arr)  >= min_bvp else None,
            eda=eda_arr  if len(eda_arr)  >= min_eda else None,
            temp=temp_arr if len(temp_arr) > 0       else None,
            sfreq_bvp=sfreq,
            sfreq_eda=sfreq,
            sfreq_temp=sfreq,
            T=T,
        )

    # ── camera metrics helpers ─────────────────────────────────────────────────

    def _forehead_green(self, img: np.ndarray) -> Optional[float]:
        """Mean green channel of forehead ROI using Haar face detection."""
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # Take largest detected face
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3])[-1]
        fore_h = max(1, int(h * 0.35))
        roi    = img[y: y + fore_h, x: x + w]
        return float(roi[:, :, 1].mean()) if roi.size > 0 else None  # green (BGR)

    def _get_new_frames(self, session_id: str) -> List[Tuple[str, np.ndarray]]:
        """Return (path, img) for all new camera frames since last rPPG call."""
        if not session_id:
            return []
        session_path = os.path.join(FRAMES_ROOT, session_id)
        if not os.path.exists(session_path):
            return []
        all_files = sorted(glob.glob(os.path.join(session_path, "*.jpg")))
        if not all_files:
            return []
        # On first call for this session, skip existing frames to avoid
        # processing all accumulated frames at once (would cause 3s+ timeout).
        if self._last_rppg_file is None:
            self._last_rppg_file = all_files[-1]
            return []
        new_files = [f for f in all_files if f > self._last_rppg_file]
        new_files = new_files[-5:]    # cap at 5 frames — cv2.imread from NTFS/WSL ~100ms/frame
        result: List[Tuple[str, np.ndarray]] = []
        for fpath in new_files:
            try:
                img = cv2.imread(fpath)
                if img is not None:
                    result.append((fpath, img))
            except Exception:
                pass
        if result:
            self._last_rppg_file = result[-1][0]
        return result

    def _compute_rppg(self) -> Tuple[float, float]:
        """Return (cam_hr_bpm, cam_resp_bpm) from rPPG green-channel buffer."""
        if len(self.rppg_green) < 20 or len(self.rppg_ts) < 2:
            return 0.0, 0.0
        sig = np.array(self.rppg_green, dtype=np.float32)
        ts  = np.array(self.rppg_ts,    dtype=np.float64)
        fps = float(np.clip(1.0 / (np.mean(np.diff(ts)) + 1e-6), 2.0, 60.0))
        sig -= sig.mean()
        if sig.std() < 1e-6:
            return 0.0, 0.0
        sig /= sig.std()
        nyq = fps / 2.0
        cam_hr = cam_resp = 0.0
        # HR: 0.7–min(3,nyq*0.9) Hz — works at any fps ≥ ~1.6 Hz (Nyquist > 0.8)
        hr_hi = min(3.0, nyq * 0.9)
        if nyq > 0.8 and hr_hi > 0.7 and len(sig) >= max(20, int(fps * 4)):
            try:
                b, a   = butter(2, [0.7 / nyq, hr_hi / nyq], btype="band")
                hr_sig = filtfilt(b, a, sig)
                f, psd = welch(hr_sig, fs=fps, nperseg=min(256, len(hr_sig)))
                mask   = (f >= 0.7) & (f <= hr_hi)
                if mask.any():
                    cam_hr = float(np.clip(f[mask][np.argmax(psd[mask])] * 60.0, 40.0, 180.0))
            except Exception:
                pass
        # Resp: 0.1–0.5 Hz — need ≥10s of frames
        if nyq > 0.5 and len(sig) >= max(20, int(fps * 10)):
            try:
                b, a     = butter(2, [0.1 / nyq, min(0.5 / nyq, 0.99)], btype="band")
                resp_sig = filtfilt(b, a, sig)
                f, psd   = welch(resp_sig, fs=fps, nperseg=min(256, len(resp_sig)))
                mask     = (f >= 0.1) & (f <= 0.5)
                if mask.any():
                    cam_resp = float(np.clip(f[mask][np.argmax(psd[mask])] * 60.0, 5.0, 30.0))
            except Exception:
                pass
        return cam_hr, cam_resp

    def _compute_blinks(self) -> float:
        """Blinks per minute from AU43 at camera frame rate (ear_hires)."""
        if len(self.ear_hires) < 10:
            return 0.0
        ear = np.array(list(self.ear_hires), dtype=np.float32)
        # AU43: 0=open, >0=closing/closed; blink = brief spike above threshold
        threshold = max(0.25, float(np.percentile(ear, 85)))
        closed    = ear > threshold
        blinks    = int(np.sum(np.diff(closed.astype(np.int8)) == 1))
        if len(self.ear_hires_ts) >= 2:
            ts = np.array(list(self.ear_hires_ts))
            window_s = float(ts[-1] - ts[0]) + 1e-6
        else:
            window_s = len(self.ear_hires) / 5.0
        return float(np.clip(blinks / max(window_s, 1.0) * 60.0, 0.0, 60.0))

    def _compute_spo2(self) -> float:
        """Estimate SpO2 (%) from last 5 s of IR/RED buffers using ratio-of-ratios."""
        win = int(self._physio_sfreq * 5)   # 5 s window → ~250 samples at 50 Hz
        if len(self.bvp_deque) < win // 2 or len(self.red_deque) < win // 2:
            return 0.0
        ir  = np.array(list(self.bvp_deque) [-win:], dtype=np.float64)
        red = np.array(list(self.red_deque) [-win:], dtype=np.float64)
        ir_dc, red_dc = ir.mean(), red.mean()
        if ir_dc < 1.0 or red_dc < 1.0:
            return 0.0
        ir_ac, red_ac = ir.std(), red.std()
        if red_ac < 1e-6:
            return 0.0
        # R = (AC_red/DC_red) / (AC_ir/DC_ir) — Beer-Lambert approximation
        R    = (red_ac / red_dc) / (ir_ac / ir_dc + 1e-6)
        spo2 = 110.0 - 25.0 * R
        return float(np.clip(spo2, 85.0, 100.0))

    # ── main handler ──────────────────────────────────────────────────────────

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        with self._lock:
            return self._analyze_locked(req)

    def _analyze_locked(self, req: AnalyzeRequest) -> AnalyzeResponse:
        now = time.monotonic()
        self._call_times.append(now)

        # 0. Reset buffers on new session
        if req.session_id != self._current_session:
            self._current_session = req.session_id
            self.bvp_deque.clear(); self.red_deque.clear()
            self.eda_deque.clear(); self.temp_deque.clear()
            self.rppg_green.clear(); self.rppg_ts.clear()
            self.ear_hires.clear(); self.ear_hires_ts.clear()
            self.face_buf.clear(); self.eeg_buf.clear()
            self._last_rppg_file = None
            self._ema_init = False
            logger.info(f"New session: {req.session_id} — buffers reset")

        # 1. Extend raw signal buffers
        if req.ir_batch:
            # Batch path: all 50 Hz samples since last call → true HRV/EDA/SpO2
            for v in req.ir_batch:   self.bvp_deque.append(float(v))
            for v in req.red_batch:  self.red_deque.append(float(v))
            for v in req.gsr_batch:  self.eda_deque.append(float(v))
            for v in req.temp_batch: self.temp_deque.append(float(v))
            # sfreq is fixed at ESP32 rate — no dynamic estimation needed
        else:
            # Fallback: single value (old behaviour)
            self.bvp_deque.append(float(req.ir))
            self.red_deque.append(float(req.red))
            self.eda_deque.append(float(req.gsr))
            self.temp_deque.append(float(req.temp))

        # 2. Face AUs from latest frame on disk
        frame_path = self._get_latest_frame(req.session_id)
        if frame_path is None:
            logger.warning(f"No camera frames found for session {req.session_id} in {FRAMES_ROOT}")
        elif frame_path != self.last_frame_path:
            try:
                img = cv2.imread(frame_path)
                if img is not None:
                    feats = self.face_extractor.extract_from_arrays([img], T=1)
                    au_nonzero = int(np.count_nonzero(feats[0]))
                    logger.info(f"Face AUs updated: {os.path.basename(frame_path)} | nonzero={au_nonzero}/17 | max={feats[0].max():.3f}")
                    self._last_au = feats[0]
                    self.last_frame_path = frame_path
                else:
                    logger.warning(f"cv2.imread returned None for {frame_path}")
            except Exception as exc:
                logger.warning(f"Face AU extraction failed: {exc}")
        self.face_buf.append(self._last_au.copy())

        # 3. EEG approximation from NeuroSky att/med
        att = float(req.att) if np.isfinite(req.att) else 0.0
        med = float(req.med) if np.isfinite(req.med) else 0.0
        self.eeg_buf.append(self._eeg_approx(att, med))

        # 4. Camera frame processing — BEFORE warmup check so _last_rppg_file
        #    stays current during warmup (avoids 60-frame spike on call 30).
        #    Only Haar (green channel) per frame — NO MediaPipe here.
        #    MediaPipe already ran once in step 2; AU43 is reused for blinks.
        new_frames = self._get_new_frames(req.session_id)
        for fpath, img in new_frames:
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                mtime = time.time()   # WSL/NTFS race condition fallback
            green = self._forehead_green(img)   # Haar only — fast
            if green is not None:
                self.rppg_green.append(green)
                self.rppg_ts.append(mtime)
        # Blinks: use AU43 already extracted in step 2 (1 MediaPipe call per invoke)
        if self.last_frame_path is not None:
            try:
                mtime = os.path.getmtime(self.last_frame_path)
            except OSError:
                mtime = time.time()
            self.ear_hires.append(float(self._last_au[16]))
            self.ear_hires_ts.append(mtime)
        self._cam_hr, self._cam_resp = self._compute_rppg()
        self._blink_rate = self._compute_blinks()

        # 5. NeuroKit2 physio features (use true sensor sfreq, not call sfreq)
        physio_feat = self._compute_physio(self._physio_sfreq)

        # 6. Warmup: wait until face_buf is full
        if len(self.face_buf) < T:
            return AnalyzeResponse(
                status="warming_up",
                warmup=len(self.face_buf) / T,
            )

        # 7. Assemble tensors: (1, T, dim)
        face_arr   = np.stack(list(self.face_buf))   # (T, 17)
        eeg_arr    = np.stack(list(self.eeg_buf))    # (T, 5)

        face_t   = torch.tensor(face_arr,   dtype=torch.float32).unsqueeze(0).to(DEVICE)
        physio_t = torch.tensor(physio_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        eeg_t    = torch.tensor(eeg_arr,    dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 8. FusionLSTM forward — face + physio + EEG (TGAM2-compatible)
        with torch.no_grad():
            va, grad = model(face_t, physio_t, eeg_t)
            # Face-only: zero out physio+eeg to isolate camera contribution
            va_face, _ = model(face_t, torch.zeros_like(physio_t), torch.zeros_like(eeg_t))

        va      = torch.nan_to_num(va)
        grad    = torch.nan_to_num(grad)
        va_face = torch.nan_to_num(va_face)

        raw_v = float(va[0, -1, 0])
        raw_a = float(va[0, -1, 1])
        grad_v  = float(grad[0, -1, 0])
        grad_a  = float(grad[0, -1, 1])
        face_v  = float(va_face[0, -1, 0])
        face_a  = float(va_face[0, -1, 1])

        # EMA smoothing to avoid tanh saturation jumps
        if not self._ema_init:
            self._va_ema  = np.array([raw_v, raw_a])
            self._ema_init = True
        else:
            self._va_ema = (self.EMA_ALPHA * np.array([raw_v, raw_a])
                            + (1 - self.EMA_ALPHA) * self._va_ema)
        valence, arousal = float(self._va_ema[0]), float(self._va_ema[1])

        # 9. Sensor-derived metrics
        hr_bpm   = float(physio_feat[0, 2]) * 100.0   # 0 at low sampling rates
        rmssd    = float(physio_feat[0, 0]) * 100.0
        eda_mean = float(physio_feat[:, 3].mean())
        spo2     = self._compute_spo2()

        return AnalyzeResponse(
            status="success",
            valence=valence,
            arousal=arousal,
            grad_v=grad_v,
            grad_a=grad_a,
            face_v=face_v,
            face_a=face_a,
            hr_bpm=hr_bpm,
            rmssd=rmssd,
            eda_mean=eda_mean,
            spo2=spo2,
            cam_hr=self._cam_hr,
            cam_resp=self._cam_resp,
            blink_rate=self._blink_rate,
            warmup=1.0,
        )


# ─── Singleton service + route ────────────────────────────────────────────────

service = EmotionService()


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return service.analyze(req)
