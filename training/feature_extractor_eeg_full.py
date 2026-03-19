"""
feature_extractor_eeg_full.py — Bilateral frontal EEG features for dagn_lib

Expands the 5D TGAM2-compatible space to 10D by adding bilateral (L/R)
frontal features and Frontal Alpha Asymmetry (FAA).

Feature space (T, 10):
    [0] theta_avg  = (theta_L + theta_R) / 2   — inattention/relaxation [KL99]
    [1] alpha_avg  = (alpha_L + alpha_R) / 2   — relaxed alertness [KL99]
    [2] beta_avg   = (beta_L  + beta_R)  / 2   — attention/cognitive load [KL99]
    [3] FAA        = alpha_R - alpha_L          — Frontal Alpha Asymmetry [DA88]
    [4] ratio      = theta_avg - alpha_avg      — cognitive load index [KL99]
    [5] theta_L    — left frontal theta
    [6] theta_R    — right frontal theta
    [7] alpha_L    — left frontal alpha
    [8] alpha_R    — right frontal alpha
    [9] beta_L     — left frontal beta

Note: indices 0-4 are semantically compatible with feature_extractor_eeg_tgam2.py
(E_THETA=0, E_ALPHA=1, E_BETA=2, E_ASYM→FAA=3, E_RATIO=4), so
physiological_prior.py requires no changes. FAA now provides real bilateral
asymmetry instead of always-zero.

Range: [0, log1p(0.5)] ≈ [0, 0.405] for features 0-2, 4-9
       [-0.405, 0.405] for FAA (feature 3)

DEAP frontal channels (32ch EEG):
    Left:  F3=ch2, F7=ch3, FC1=ch5
    Right: F4=ch19, F8=ch20, FC2=ch22

DREAMER frontal channels (14ch Emotiv EPOC):
    Left:  F7=ch1, F3=ch2, FC5=ch3
    Right: FC6=ch10, F4=ch11, F8=ch12

Production (TGAM2): bilateral features set symmetric (L=R), FAA=0.
See eeg_approx_full() for the production-compatible version.

References:
    Davidson, R.J. (1988). EEG measures of cerebral asymmetry:
        conceptual and methodological issues. Int J Neurosci, 39(1-2), 71-89.
    Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive
        and memory performance. Brain Res Rev, 29(2-3), 169-195.
    Crowley, K. et al. (2010). Evaluating a Brain-Computer Interface to
        Categorise Visual Attention in Real Time. IEEE ICCIS.
"""
import numpy as np

from feature_extractor_eeg import _bandpower_multichannel, THETA_BAND, ALPHA_BAND, BETA_BAND
from feature_extractor_eeg_tgam2 import _bandpower_to_att_med

EPS = 1e-8
EEG_FULL_DIM = 10

# DEAP (32ch): F3=2, F7=3, FC1=5 | F4=19, F8=20, FC2=22
DEAP_LEFT_FRONTAL_FULL  = [2, 3, 5]
DEAP_RIGHT_FRONTAL_FULL = [19, 20, 22]

# DREAMER (14ch Emotiv EPOC): F7=1, F3=2, FC5=3 | FC6=10, F4=11, F8=12
DREAMER_LEFT_FRONTAL_FULL  = [1, 2, 3]
DREAMER_RIGHT_FRONTAL_FULL = [10, 11, 12]


def _side_features(theta: float, alpha: float, beta: float):
    """
    Convert per-side bandpower to log1p-scaled (theta_f, alpha_f, beta_f).

    Uses att/med as intermediate normalisation (TGAM2-compatible scale).
    Output range: each in [0, log1p(0.5)] ≈ [0, 0.405].
    """
    att, med = _bandpower_to_att_med(theta, alpha, beta)
    theta_f = float(np.log1p((1.0 - att / 100.0) * 0.5))
    alpha_f = float(np.log1p((med  / 100.0) * 0.5))
    beta_f  = float(np.log1p((att  / 100.0) * 0.5))
    return theta_f, alpha_f, beta_f


def extract_eeg_features_full(
    eeg_data,
    sfreq: float,
    left_ch_idxs: list,
    right_ch_idxs: list,
    window_sec: float = 1.0,
    T: int = 30,
) -> np.ndarray:
    """
    Extract 10D bilateral frontal EEG features per timestep.

    Args:
        eeg_data:      (n_channels, n_samples) raw EEG, float
        sfreq:         sampling frequency in Hz (128 for DEAP/DREAMER)
        left_ch_idxs:  left frontal channel indices  (e.g. [2,3,5] for DEAP F3/F7/FC1)
        right_ch_idxs: right frontal channel indices (e.g. [19,20,22] for DEAP F4/F8/FC2)
        window_sec:    seconds per timestep (default 1.0)
        T:             number of timesteps (default 30)

    Returns:
        features: (T, 10) float32
            Columns: [theta_avg, alpha_avg, beta_avg, FAA, ratio,
                      theta_L, theta_R, alpha_L, alpha_R, beta_L]
    """
    win_samples = int(window_sec * sfreq)
    features = np.zeros((T, EEG_FULL_DIM), dtype=np.float32)

    for t in range(T):
        start = t * win_samples
        end   = start + win_samples
        if end > eeg_data.shape[1]:
            break

        window = eeg_data[:, start:end]
        left   = window[left_ch_idxs,  :]  # (n_left,  win_samples)
        right  = window[right_ch_idxs, :]  # (n_right, win_samples)

        # Per-side bandpower averaged across channels
        theta_L = float(_bandpower_multichannel(left,  sfreq, *THETA_BAND).mean())
        alpha_L = float(_bandpower_multichannel(left,  sfreq, *ALPHA_BAND).mean())
        beta_L  = float(_bandpower_multichannel(left,  sfreq, *BETA_BAND).mean())
        theta_R = float(_bandpower_multichannel(right, sfreq, *THETA_BAND).mean())
        alpha_R = float(_bandpower_multichannel(right, sfreq, *ALPHA_BAND).mean())
        beta_R  = float(_bandpower_multichannel(right, sfreq, *BETA_BAND).mean())

        # Convert to log1p space via att/med (TGAM2-compatible normalisation)
        tL, aL, bL = _side_features(theta_L, alpha_L, beta_L)
        tR, aR, bR = _side_features(theta_R, alpha_R, beta_R)

        theta_avg = (tL + tR) / 2.0
        alpha_avg = (aL + aR) / 2.0
        beta_avg  = (bL + bR) / 2.0
        faa       = aR - aL        # Frontal Alpha Asymmetry (Davidson 1988)
        ratio     = theta_avg - alpha_avg  # cognitive load index (Klimesch 1999)

        features[t] = [theta_avg, alpha_avg, beta_avg, faa, ratio,
                       tL, tR, aL, aR, bL]

    return features  # (T, 10)


def eeg_approx_full(att: float, med: float) -> np.ndarray:
    """
    10D EEG features from NeuroSky att/med (single frontal electrode).

    Used in production where TGAM2 provides no bilateral information.
    Bilateral features are symmetric (L = R); FAA = 0 (no lateral info).

    Returns:
        (10,) float32 — same layout as extract_eeg_features_full output:
        [theta_avg, alpha_avg, beta_avg, FAA=0, ratio,
         theta_L, theta_R, alpha_L, alpha_R, beta_L]
    """
    att_n = np.clip(att / 100.0, 0.0, 1.0)
    med_n = np.clip(med / 100.0, 0.0, 1.0)
    theta = float(np.log1p((1.0 - att_n) * 0.5))
    alpha = float(np.log1p(med_n * 0.5))
    beta  = float(np.log1p(att_n * 0.5))
    faa   = 0.0           # no bilateral info from single-electrode TGAM2
    ratio = theta - alpha
    # Bilateral features symmetric (L = R)
    return np.array([theta, alpha, beta, faa, ratio,
                     theta, theta, alpha, alpha, beta],
                    dtype=np.float32)


def zeros_eeg_full(T: int = 30) -> np.ndarray:
    """(T, 10) zeros — for datasets without EEG (AFEW-VA, WESAD, AFFEC)."""
    return np.zeros((T, EEG_FULL_DIM), dtype=np.float32)
