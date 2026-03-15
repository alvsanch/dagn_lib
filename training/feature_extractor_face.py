"""
feature_extractor_face.py
Facial Action Unit approximation using MediaPipe FaceMesh.

Computes 17 AU-proxies from 468 facial landmarks via geometric relationships.
Output per frame: (17,) float vector in [0, 1].

Selected AUs and geometric computation (Ekman 1978 FACS notation):
    AU01 — Inner Brow Raise   → inner brow landmark height above eye baseline
    AU02 — Outer Brow Raise   → outer brow landmark height above eye baseline
    AU04 — Brow Lowerer       → inner brow descent toward nasal bridge
    AU06 — Cheek Raiser       → cheek landmark height (correlated with Duchenne smile)
    AU07 — Lid Tightener      → upper eyelid compression (EAR reduced)
    AU10 — Upper Lip Raiser   → upper lip center elevation
    AU12 — Lip Corner Puller  → lip corner upward-outward displacement
    AU14 — Dimpler            → lateral lip corner compression
    AU15 — Lip Corner Depressor → lip corner downward displacement
    AU17 — Chin Raiser        → chin elevation
    AU20 — Lip Stretcher      → mouth horizontal stretch
    AU23 — Lip Tightener      → inter-lip gap reduction
    AU24 — Lip Pressor        → lip pressing (upper-lower lip contact)
    AU25 — Lips Part          → inter-lip vertical distance
    AU26 — Jaw Drop           → jaw/chin vertical drop
    AU28 — Lip Suck           → lip retraction inward
    AU43 — Eyes Closed        → Eye Aspect Ratio (EAR) minimum

Note: These are geometric APPROXIMATIONS, not AU regression outputs.
They correlate with the AUs but do not exactly replicate OpenFace2 scores.
For AFFEC dataset, real OpenFace2 AU_r values are used instead (see affec_dataset.py).

References:
    Ekman, P. & Friesen, W.V. (1978). Facial Action Coding System.
    Soukupova, T. & Cech, J. (2016). Real-time eye blink detection using facial
        landmarks. CVWW.
    Lugaresi et al. (2019). MediaPipe: A Framework for Perceiving and Processing
        Reality. Third Workshop on Computer Vision for AR/VR at CVPR.

Usage:
    from feature_extractor_face import FaceFeatureExtractor, zeros_face

    extractor = FaceFeatureExtractor()
    features = extractor.extract_from_path("frame.png", T=30)  # (30, 17)
"""
import numpy as np
from pathlib import Path

FACE_DIM = 17

# ─── MediaPipe FaceMesh landmark indices ─────────────────────────────────────
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Eyes: [p1, p2, p3, p4, p5, p6] for EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
_LEFT_EYE   = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE  = [33,  160, 158, 133, 153, 144]

# Brows (left = patient's left = viewer's right, following MediaPipe convention)
# Left eyebrow (inner to outer): 336, 296, 334, 293, 300
# Right eyebrow (inner to outer): 107, 66, 105, 63, 70
_LEFT_BROW_INNER  = [336, 296]
_LEFT_BROW_OUTER  = [334, 293, 300]
_RIGHT_BROW_INNER = [107, 66]
_RIGHT_BROW_OUTER = [105, 63, 70]

# Corresponding eye reference points (center of eye lid)
_LEFT_EYE_CENTER  = [386, 374]   # upper-lower lid midpoints
_RIGHT_EYE_CENTER = [159, 145]

# Mouth landmarks
_MOUTH_LEFT    = 61     # left corner
_MOUTH_RIGHT   = 291    # right corner
_MOUTH_TOP     = 13     # upper lip center
_MOUTH_BOTTOM  = 14     # lower lip center
_MOUTH_TOP_MID = 0      # philtrum / nose base reference
_CHIN          = 175    # chin center
_NOSE_TIP      = 4      # nose tip (reference)

# Cheek landmarks
_LEFT_CHEEK    = [234, 93]
_RIGHT_CHEEK   = [454, 323]

_mp_face_mesh  = None   # lazy global


def _get_mp():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        import mediapipe as mp
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
    return _mp_face_mesh


def _lm(lms, idx):
    """Get (x, y) tuple for a landmark index."""
    l = lms[idx]
    return np.array([l.x, l.y], dtype=np.float32)


def _dist(lms, i, j):
    """Euclidean distance between landmarks i and j."""
    return float(np.linalg.norm(_lm(lms, i) - _lm(lms, j)))


def _mean_lm(lms, indices):
    """Mean position of a group of landmarks."""
    pts = np.stack([_lm(lms, i) for i in indices])
    return pts.mean(axis=0)


def _ear(lms, eye_pts):
    """Eye Aspect Ratio from 6 eye landmark indices."""
    p1 = _lm(lms, eye_pts[0])
    p2 = _lm(lms, eye_pts[1])
    p3 = _lm(lms, eye_pts[2])
    p4 = _lm(lms, eye_pts[3])
    p5 = _lm(lms, eye_pts[4])
    p6 = _lm(lms, eye_pts[5])
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4) + 1e-6)


def _compute_aus(lms, image_height, image_width):
    """
    Compute 17 AU-proxies from MediaPipe landmarks.

    All values normalized to approximately [0, 1].
    Returns (17,) float32 array.
    """
    # Normalise coordinates to [0,1] — landmarks are already in [0,1]
    # Use inter-ocular distance as face-size reference
    iod = _dist(lms, 33, 263)  # inner canthi left-right
    if iod < 1e-6:
        return np.zeros(FACE_DIM, dtype=np.float32)

    # Reference heights
    eye_center_y = (_lm(lms, 159)[1] + _lm(lms, 386)[1]) / 2.0
    nose_y       = _lm(lms, _NOSE_TIP)[1]
    chin_y       = _lm(lms, _CHIN)[1]
    face_height  = abs(chin_y - eye_center_y) + 1e-6

    aus = np.zeros(FACE_DIM, dtype=np.float32)

    # AU01 — Inner Brow Raise: inner brow above eye center
    inner_brow_y = _mean_lm(lms, _LEFT_BROW_INNER + _RIGHT_BROW_INNER)[1]
    au01 = max(0.0, (eye_center_y - inner_brow_y) / face_height - 0.05)
    aus[0] = min(1.0, au01 * 5.0)

    # AU02 — Outer Brow Raise: outer brow height
    outer_brow_y = _mean_lm(lms, _LEFT_BROW_OUTER + _RIGHT_BROW_OUTER)[1]
    au02 = max(0.0, (eye_center_y - outer_brow_y) / face_height - 0.05)
    aus[1] = min(1.0, au02 * 5.0)

    # AU04 — Brow Lowerer: inner brow descent (brow below neutral = positive)
    au04 = max(0.0, 0.15 - (eye_center_y - inner_brow_y) / face_height)
    aus[2] = min(1.0, au04 * 6.0)

    # AU06 — Cheek Raiser: cheek landmark height relative to nose
    cheek_y = _mean_lm(lms, _LEFT_CHEEK + _RIGHT_CHEEK)[1]
    au06 = max(0.0, (nose_y - cheek_y) / face_height - 0.1)
    aus[3] = min(1.0, au06 * 4.0)

    # AU07 — Lid Tightener: EAR reduced below baseline (~0.3)
    ear_avg = (_ear(lms, _LEFT_EYE) + _ear(lms, _RIGHT_EYE)) / 2.0
    au07 = max(0.0, 0.30 - ear_avg)
    aus[4] = min(1.0, au07 * 5.0)

    # AU10 — Upper Lip Raiser: upper lip center elevation above neutral
    mouth_top_y = _lm(lms, _MOUTH_TOP)[1]
    mouth_bot_y = _lm(lms, _MOUTH_BOTTOM)[1]
    lip_gap = abs(mouth_bot_y - mouth_top_y) / face_height
    au10 = max(0.0, (nose_y - mouth_top_y) / face_height - 0.20)
    aus[5] = min(1.0, au10 * 4.0)

    # AU12 — Lip Corner Puller: corners raised above mouth center
    left_c  = _lm(lms, _MOUTH_LEFT)
    right_c = _lm(lms, _MOUTH_RIGHT)
    mouth_mid_y = (mouth_top_y + mouth_bot_y) / 2.0
    corner_raise = max(0.0, mouth_mid_y - (left_c[1] + right_c[1]) / 2.0)
    aus[6] = min(1.0, corner_raise / face_height * 10.0)

    # AU14 — Dimpler: lateral compression (corners pulled slightly inward)
    mouth_width = abs(right_c[0] - left_c[0])
    face_width  = abs(_lm(lms, 234)[0] - _lm(lms, 454)[0]) + 1e-6
    au14 = max(0.0, 0.45 - mouth_width / face_width)
    aus[7] = min(1.0, au14 * 4.0)

    # AU15 — Lip Corner Depressor: corners below mouth center
    corner_drop = max(0.0, (left_c[1] + right_c[1]) / 2.0 - mouth_mid_y)
    aus[8] = min(1.0, corner_drop / face_height * 10.0)

    # AU17 — Chin Raiser: chin landmark moves toward lower lip
    chin_dist = abs(_lm(lms, _CHIN)[1] - mouth_bot_y) / face_height
    au17 = max(0.0, 0.30 - chin_dist)
    aus[9] = min(1.0, au17 * 5.0)

    # AU20 — Lip Stretcher: horizontal mouth width
    au20 = min(1.0, max(0.0, mouth_width / face_width - 0.25) * 4.0)
    aus[10] = au20

    # AU23 — Lip Tightener: reduced lip gap
    au23 = max(0.0, 0.06 - lip_gap)
    aus[11] = min(1.0, au23 * 15.0)

    # AU24 — Lip Pressor: minimal lip gap (near-contact)
    au24 = max(0.0, 0.03 - lip_gap)
    aus[12] = min(1.0, au24 * 30.0)

    # AU25 — Lips Part: inter-lip gap (positive = open)
    aus[13] = min(1.0, max(0.0, lip_gap * 8.0))

    # AU26 — Jaw Drop: large vertical mouth opening
    jaw_drop = abs(mouth_bot_y - mouth_top_y) / face_height
    aus[14] = min(1.0, max(0.0, jaw_drop * 5.0))

    # AU28 — Lip Suck: lips pushed inward (subtle)
    au28 = max(0.0, 0.04 - lip_gap)
    aus[15] = min(1.0, au28 * 20.0)

    # AU43 — Eyes Closed: low EAR
    ear_avg = (_ear(lms, _LEFT_EYE) + _ear(lms, _RIGHT_EYE)) / 2.0
    aus[16] = min(1.0, max(0.0, (0.25 - ear_avg) * 5.0))

    return aus


class FaceFeatureExtractor:
    """
    Extracts AU-proxy features from image frames using MediaPipe FaceMesh.

    Geometric AU approximations computed from 468 face landmarks.
    When no face is detected, zeros are returned for that frame.

    Methods:
        extract_from_paths(paths, T):   list of image file paths → (T, 17)
        extract_from_arrays(arrays, T): list of BGR/RGB numpy frames → (T, 17)
    """

    def __init__(self):
        pass  # FaceMesh loaded lazily

    def _process_frame(self, img_bgr):
        """
        Process one BGR frame.
        Returns (17,) float32 AU proxy vector, or zeros if no face.
        """
        try:
            import cv2
            import mediapipe as mp
            mp_fm = _get_mp()

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result  = mp_fm.process(img_rgb)

            if not result.multi_face_landmarks:
                return np.zeros(FACE_DIM, dtype=np.float32)

            lms = result.multi_face_landmarks[0].landmark
            h, w = img_bgr.shape[:2]
            return _compute_aus(lms, h, w)
        except Exception:
            return np.zeros(FACE_DIM, dtype=np.float32)

    def extract_from_paths(self, frame_paths, T=30):
        """
        Extract AU features from a list of image file paths.

        Args:
            frame_paths: list of str/Path, one per frame
            T:           number of output timesteps

        Returns:
            features: (T, 17) float32 — AU proxy intensities in [0, 1]
        """
        import cv2
        if not frame_paths:
            return np.zeros((T, FACE_DIM), dtype=np.float32)

        n_frames = len(frame_paths)
        indices  = np.linspace(0, n_frames - 1, T, dtype=int)

        features = np.zeros((T, FACE_DIM), dtype=np.float32)
        for t, idx in enumerate(indices):
            img = cv2.imread(str(frame_paths[idx]))
            if img is not None:
                features[t] = self._process_frame(img)

        return features  # (T, 17)

    def extract_from_arrays(self, frames, T=30):
        """
        Extract AU features from pre-loaded BGR numpy frames.

        Args:
            frames: list of (H, W, 3) uint8 arrays (BGR or RGB)
            T:      number of output timesteps

        Returns:
            features: (T, 17) float32
        """
        if not frames:
            return np.zeros((T, FACE_DIM), dtype=np.float32)

        n_frames = len(frames)
        indices  = np.linspace(0, n_frames - 1, T, dtype=int)

        features = np.zeros((T, FACE_DIM), dtype=np.float32)
        for t, idx in enumerate(indices):
            features[t] = self._process_frame(frames[idx])

        return features  # (T, 17)


def zeros_face(T=30):
    """Return (T, 17) zeros for datasets without video (DEAP, WESAD, DREAMER)."""
    return np.zeros((T, FACE_DIM), dtype=np.float32)
