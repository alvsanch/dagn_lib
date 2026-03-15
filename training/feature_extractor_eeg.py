"""
feature_extractor_eeg.py
EEG feature extraction using MNE-Python (with numpy fallback).

Extracts per-second bandpower and frontal alpha asymmetry.

Features returned per timestep (T, 5):
    [0] theta_log     — log(theta bandpower 4-8 Hz), mean across channels
    [1] alpha_log     — log(alpha bandpower 8-13 Hz), mean across channels
    [2] beta_log      — log(beta bandpower 13-30 Hz), mean across channels
    [3] alpha_asym    — frontal alpha asymmetry (DASM): (R-L)/(R+L+ε)
    [4] theta_alpha   — log(theta/alpha ratio), cognitive load index

All log-bandpower features are z-scored within the trial for scale invariance.

References:
    Davidson, R.J. (1988). EEG measures of cerebral asymmetry: conceptual
        and methodological issues. Int J Neurosci, 39(1-2), 71-89.
    Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and
        memory performance. Brain Res Rev, 29(2-3), 169-195.
    Barry, R.J. et al. (2003). EEG coherence between scalp sites in non-clinical
        participants. Clin Neurophysiol, 114(4), 633-641.

Usage:
    from feature_extractor_eeg import extract_eeg_features

    # DEAP: 32ch @ 128Hz, frontal electrodes F3=ch2, F4=ch19
    features = extract_eeg_features(eeg_data, sfreq=128,
                                    left_ch_idx=2, right_ch_idx=19, T=30)
    # features.shape == (30, 5)
"""
import numpy as np

# Frequency bands (Hz)
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND  = (13.0, 30.0)

# Channel indices for frontal alpha asymmetry per dataset
# DEAP (32ch, standard 10-20): F3=2, F4=19
DEAP_LEFT_FRONTAL  = 2
DEAP_RIGHT_FRONTAL = 19

# DREAMER (14ch Emotiv EPOC): AF3=0,F7=1,F3=2,FC5=3,T7=4,P7=5,O1=6,
#                               O2=7,P8=8,T8=9,FC6=10,F4=11,F8=12,AF4=13
DREAMER_LEFT_FRONTAL  = 2   # F3
DREAMER_RIGHT_FRONTAL = 11  # F4


def _bandpower_multichannel(data, sfreq, fmin, fmax):
    """
    Compute mean bandpower per channel using Welch via MNE (or numpy fallback).

    Args:
        data: (n_channels, n_samples) raw EEG window
        sfreq: sampling frequency in Hz
        fmin, fmax: frequency band edges in Hz

    Returns:
        bp: (n_channels,) mean PSD in [fmin, fmax]
    """
    try:
        import mne
        psds, freqs = mne.time_frequency.psd_array_welch(
            data.astype(np.float64),
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_per_seg=min(data.shape[1], int(sfreq)),  # 1-second segments
            verbose=False,
        )
        # psds: (n_channels, n_freqs)
        return psds.mean(axis=1).astype(np.float32)  # (n_channels,)

    except ImportError:
        # Numpy fallback: Hann-windowed FFT
        n_ch, n = data.shape
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(mask):
            return np.zeros(n_ch, dtype=np.float32)
        win = np.hanning(n)
        psd = (np.abs(np.fft.rfft(data * win[np.newaxis, :], axis=1)) ** 2) / n
        return psd[:, mask].mean(axis=1).astype(np.float32)


def extract_eeg_features(
    eeg_data,
    sfreq,
    left_ch_idx=None,
    right_ch_idx=None,
    window_sec=1.0,
    T=30,
):
    """
    Extract EEG features from a raw multi-channel EEG array.

    Computes per-1-second window bandpower + frontal alpha asymmetry.
    All log-bandpower features are z-scored within the trial.

    Args:
        eeg_data:      (n_channels, n_samples) raw EEG, float
        sfreq:         sampling frequency in Hz (e.g. 128 for DEAP/DREAMER)
        left_ch_idx:   channel index for left frontal electrode (e.g. F3=2)
        right_ch_idx:  channel index for right frontal electrode (e.g. F4=19)
        window_sec:    duration of each timestep window in seconds (default 1.0)
        T:             number of timesteps to return (default 30)

    Returns:
        features: (T, 5) float32 array
            col 0 — theta_log     (z-scored)
            col 1 — alpha_log     (z-scored)
            col 2 — beta_log      (z-scored)
            col 3 — alpha_asym    [-1, 1] (no z-score, already bounded)
            col 4 — theta_alpha   (z-scored)
    """
    n_channels, n_samples = eeg_data.shape
    win_samples = int(window_sec * sfreq)
    features = np.zeros((T, 5), dtype=np.float32)

    for t in range(T):
        start = t * win_samples
        end = start + win_samples
        if end > n_samples:
            # Not enough data — keep zeros for remaining timesteps
            break

        window = eeg_data[:, start:end]  # (n_ch, win_samples)

        # Global bandpower (mean across all channels)
        theta_ch = _bandpower_multichannel(window, sfreq, *THETA_BAND)
        alpha_ch = _bandpower_multichannel(window, sfreq, *ALPHA_BAND)
        beta_ch  = _bandpower_multichannel(window, sfreq, *BETA_BAND)

        theta_mean = theta_ch.mean()
        alpha_mean = alpha_ch.mean()
        beta_mean  = beta_ch.mean()

        # Frontal alpha asymmetry — Davidson (1988)
        # DASM = (alpha_right - alpha_left) / (alpha_right + alpha_left)
        if left_ch_idx is not None and right_ch_idx is not None:
            alpha_L = float(alpha_ch[left_ch_idx])
            alpha_R = float(alpha_ch[right_ch_idx])
            alpha_asym = (alpha_R - alpha_L) / (alpha_R + alpha_L + 1e-8)
        else:
            alpha_asym = 0.0

        # Theta/alpha ratio — Klimesch (1999)
        theta_alpha_ratio = theta_mean / (alpha_mean + 1e-8)

        # Log-transform for Gaussian-like distribution
        features[t, 0] = np.log(theta_mean + 1e-8)
        features[t, 1] = np.log(alpha_mean + 1e-8)
        features[t, 2] = np.log(beta_mean  + 1e-8)
        features[t, 3] = np.clip(alpha_asym, -1.0, 1.0)
        features[t, 4] = np.log(theta_alpha_ratio + 1e-8)

    # Z-score log features within trial (cols 0,1,2,4)
    # Col 3 (alpha_asym) is already bounded [-1,1], keep as-is
    for c in [0, 1, 2, 4]:
        col = features[:, c]
        std = col.std()
        if std > 1e-6:
            features[:, c] = (col - col.mean()) / std

    return features  # (T, 5)


def zeros_eeg(T=30):
    """Return a (T, 5) zero array for datasets without EEG (e.g. AFEW-VA, WESAD)."""
    return np.zeros((T, 5), dtype=np.float32)
