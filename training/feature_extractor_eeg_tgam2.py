"""
feature_extractor_eeg_tgam2.py
TGAM2-compatible EEG feature extraction for dagn_lib training datasets.

Maps multichannel EEG (DEAP 32ch, DREAMER 14ch, AFFEC 63ch) to att/med
values (0-100) approximating NeuroSky TGAM2 output, then applies _eeg_approx()
to produce 5 features identical to the production pipeline.

Feature space (T, 5):
    [0] theta  = log1p((1 - att/100) * 0.5)   — relaxation / inattention
    [1] alpha  = log1p((med/100)     * 0.5)   — meditation / relaxed alertness
    [2] beta   = log1p((att/100)     * 0.5)   — attention / cognitive load
    [3] asym   = 0.0                           — TGAM2 single frontal electrode
    [4] ratio  = theta - alpha                 — Klimesch 1999 theta/alpha

Att/med mapping from frontal EEG bandpower:
    att = clip(beta / (alpha + theta + ε) * 50, 0, 100)
    med = clip(alpha / (theta + beta + ε) * 50, 0, 100)

The scale factor 50 centres the output at 1:1 band ratios.
Higher beta relative to alpha+theta → higher attention (consistent with
beta's role in active cognition and alpha suppression under mental load).
Higher alpha relative to theta+beta → higher meditation (consistent with
alpha's role in relaxed wakefulness and eyes-closed states).

Production identity:
    Production _eeg_approx(att, med) → (T, 5) features
    This file maps multichannel EEG → (att, med) → _eeg_approx → same features

References:
    NeuroSky ThinkGear ASIC Module (TGAM2): Technical Specification, 2011.
    Crowley, K. et al. (2010). Evaluating a Brain-Computer Interface to
        Categorise Visual Attention in Real Time. IEEE ICCIS.
    Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive
        and memory performance. Brain Res Rev, 29(2-3), 169-195.
    Davidson, R.J. (1988). EEG measures of cerebral asymmetry.
        Int J Neurosci, 39(1-2), 71-89.

Usage:
    from feature_extractor_eeg_tgam2 import extract_eeg_features_tgam2

    # DEAP: use frontal F3(ch2) and F4(ch19)
    features = extract_eeg_features_tgam2(eeg_data, sfreq=128,
                                           left_ch_idx=2, right_ch_idx=19, T=30)

    # DREAMER: use frontal F3(ch2) and F4(ch11)
    features = extract_eeg_features_tgam2(eeg_data, sfreq=128,
                                           left_ch_idx=2, right_ch_idx=11, T=30)
    # features.shape == (30, 5)
"""
import numpy as np

from feature_extractor_eeg import _bandpower_multichannel, THETA_BAND, ALPHA_BAND, BETA_BAND

EPS = 1e-8

# Channel indices per dataset (for convenience, same as feature_extractor_eeg.py)
DEAP_LEFT_FRONTAL    = 2    # F3
DEAP_RIGHT_FRONTAL   = 19   # F4
DREAMER_LEFT_FRONTAL  = 2   # F3
DREAMER_RIGHT_FRONTAL = 11  # F4
AFFEC_LEFT_FRONTAL   = 9    # F3
AFFEC_RIGHT_FRONTAL  = 13   # F4


def _eeg_approx(att: float, med: float) -> np.ndarray:
    """
    Convert att/med (0-100) to 5D EEG features.

    Identical to _eeg_approx() in production/analizar_emocion_service.py.
    Must stay in sync if the production version changes.

    Args:
        att: attention index  0-100 (higher = more focused/alert)
        med: meditation index 0-100 (higher = more relaxed/meditative)

    Returns:
        (5,) float32: [theta, alpha, beta, asym=0, ratio]
    """
    theta = np.log1p((1.0 - att / 100.0) * 0.5)
    alpha = np.log1p((med / 100.0) * 0.5)
    beta  = np.log1p((att / 100.0) * 0.5)
    asym  = 0.0
    ratio = theta - alpha
    return np.array([theta, alpha, beta, asym, ratio], dtype=np.float32)


def _bandpower_to_att_med(theta: float, alpha: float, beta: float) -> tuple:
    """
    Map scalar theta/alpha/beta bandpower to att/med (0-100).

    TGAM2 eSense approximation (Crowley 2010):
        att = clip(beta / (alpha + theta + eps) * 50, 0, 100)
        med = clip(alpha / (theta + beta + eps) * 50, 0, 100)

    Scale 50: att/med = 50 when bands are equal.

    Args:
        theta, alpha, beta: mean frontal bandpower (any positive scale)

    Returns:
        (att, med): both in [0.0, 100.0]
    """
    att = np.clip(beta  / (alpha + theta + EPS) * 50.0, 0.0, 100.0)
    med = np.clip(alpha / (theta + beta  + EPS) * 50.0, 0.0, 100.0)
    return float(att), float(med)


def extract_eeg_features_tgam2(
    eeg_data,
    sfreq: float,
    left_ch_idx: int = None,
    right_ch_idx: int = None,
    window_sec: float = 1.0,
    T: int = 30,
) -> np.ndarray:
    """
    Extract TGAM2-compatible features from multichannel EEG.

    Uses frontal channels (if provided) to compute att/med approximating
    NeuroSky TGAM2 output, then converts to 5D feature vector matching
    the production pipeline exactly.

    Args:
        eeg_data:      (n_channels, n_samples) raw EEG, float
        sfreq:         sampling frequency in Hz (128 for DEAP/DREAMER)
        left_ch_idx:   left frontal channel index (e.g. F3=2 for DEAP)
        right_ch_idx:  right frontal channel index (e.g. F4=19 for DEAP)
                       If neither provided, uses all channels (mean).
        window_sec:    duration of each timestep window in seconds (default 1.0)
        T:             number of timesteps to return (default 30)

    Returns:
        features: (T, 5) float32
            Same feature space as production _eeg_approx(att, med).
            Compatible with FusionLSTM eeg input (when re-enabled).
    """
    n_channels, n_samples = eeg_data.shape
    win_samples = int(window_sec * sfreq)
    features = np.zeros((T, 5), dtype=np.float32)

    # Select frontal channels — TGAM2 records at forehead (Fp1) + earlobe ref
    if left_ch_idx is not None and right_ch_idx is not None:
        frontal_idx = [left_ch_idx, right_ch_idx]
    elif left_ch_idx is not None:
        frontal_idx = [left_ch_idx]
    elif right_ch_idx is not None:
        frontal_idx = [right_ch_idx]
    else:
        frontal_idx = None  # use all channels

    for t in range(T):
        start = t * win_samples
        end   = start + win_samples
        if end > n_samples:
            break  # keep zeros for remaining timesteps

        window = eeg_data[:, start:end]  # (n_ch, win_samples)

        # Restrict to frontal channels before bandpower computation
        if frontal_idx is not None:
            frontal = window[frontal_idx, :]  # (1 or 2, win_samples)
        else:
            frontal = window  # all channels

        theta_ch = _bandpower_multichannel(frontal, sfreq, *THETA_BAND)
        alpha_ch = _bandpower_multichannel(frontal, sfreq, *ALPHA_BAND)
        beta_ch  = _bandpower_multichannel(frontal, sfreq, *BETA_BAND)

        theta = float(theta_ch.mean())
        alpha = float(alpha_ch.mean())
        beta  = float(beta_ch.mean())

        att, med = _bandpower_to_att_med(theta, alpha, beta)
        features[t] = _eeg_approx(att, med)

    return features  # (T, 5)


def zeros_eeg_tgam2(T: int = 30) -> np.ndarray:
    """(T, 5) zeros — for datasets without EEG (AFEW-VA, WESAD)."""
    return np.zeros((T, 5), dtype=np.float32)
