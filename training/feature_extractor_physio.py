"""
feature_extractor_physio.py
Physiological feature extraction using NeuroKit2.

Features returned per timestep (T, 6):
    [0] RMSSD_norm    — root mean square successive differences / 100
    [1] SDNN_norm     — standard deviation of NN intervals / 100
    [2] mean_HR_norm  — mean heart rate / 100 (BPM)
    [3] EDA_tonic     — tonic EDA (SCL), z-scored per trial
    [4] EDA_phasic    — phasic EDA (SCR), z-scored per trial
    [5] TEMP_norm     — skin temperature (x - 32) / 8, roughly [-1, 1]

HRV features (cols 0-2) are computed over the full 30-sec trial signal
and replicated across all T timesteps (they are window-level statistics).
EDA and TEMP (cols 3-5) are decomposed/sampled per timestep.

References:
    Task Force of the ESC and NASPE (1996). Heart rate variability: standards of
        measurement, physiological interpretation, and clinical use.
        Circulation, 93(5), 1043-1065.
    Boucsein, W. (2012). Electrodermal Activity (2nd ed.). Springer.
    NeuroKit2: Makowski et al. (2021). NeuroKit2: A Python Toolbox for Neurophysiological
        Signal Processing. Behavior Research Methods, 53(4), 1689–1696.

Usage:
    from feature_extractor_physio import extract_physio_features

    # DEAP: BVP at 128 Hz, GSR at 128 Hz, TEMP at 128 Hz
    features = extract_physio_features(bvp=bvp_signal, eda=gsr_signal,
                                       temp=temp_signal,
                                       sfreq_bvp=128, sfreq_eda=128,
                                       sfreq_temp=128, T=30)
    # features.shape == (30, 6)
"""
import numpy as np

# Safe NeuroKit2 import with informative error
try:
    import neurokit2 as nk
    _NK_AVAILABLE = True
except ImportError:
    _NK_AVAILABLE = False
    print("[WARNING] NeuroKit2 not installed. physio features will be zeros.")
    print("         Install with: pip install neurokit2")


def _hrv_from_peaks(peaks_idx, sfreq):
    """
    Compute HRV time-domain features from R-peak sample indices.

    Args:
        peaks_idx: array of SAMPLE INDICES (int) for each detected peak
        sfreq: sampling frequency in Hz

    Returns:
        rmssd_norm: RMSSD / 100 (Task Force 1996)
        sdnn_norm:  SDNN  / 100
        mean_hr:    mean HR in BPM / 100
    """
    if len(peaks_idx) < 4:
        return 0.0, 0.0, 0.0

    # peaks_idx are sample indices → convert to ms intervals
    rr_ms = np.diff(peaks_idx.astype(float)) / sfreq * 1000.0  # ms

    # Sanity check: physiologically valid RR range 300–2000 ms (30–200 BPM)
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    if len(rr_ms) < 3:
        return 0.0, 0.0, 0.0

    # RMSSD — short-term vagal activity (Task Force 1996)
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))

    # SDNN — total HRV (Task Force 1996)
    sdnn = float(np.std(rr_ms))

    # Mean HR in BPM
    mean_hr = 60000.0 / (np.mean(rr_ms) + 1e-8)

    return rmssd / 100.0, sdnn / 100.0, mean_hr / 100.0


def _peaks_from_info(info_dict, key):
    """
    Extract peak sample indices from a NeuroKit2 info dict.
    NeuroKit2 returns peaks as integer INDEX arrays (not boolean masks).
    """
    peaks = info_dict.get(key, np.array([]))
    if hasattr(peaks, "values"):
        peaks = peaks.values
    peaks = np.asarray(peaks)
    # NeuroKit2 info peaks are already indices (int64), not booleans
    # Distinguish: if dtype is bool or values are only 0/1, use np.where
    if peaks.dtype == bool or (len(peaks) > 0 and peaks.max() <= 1):
        return np.where(peaks)[0]
    return peaks.astype(int)


def _process_bvp(bvp, sfreq):
    """Extract HRV features from BVP/PPG signal using NeuroKit2."""
    if not _NK_AVAILABLE or bvp is None or len(bvp) < int(sfreq * 5):
        return 0.0, 0.0, 0.0
    try:
        _, ppg_info = nk.ppg_process(bvp.astype(float), sampling_rate=int(sfreq))
        peaks = _peaks_from_info(ppg_info, "PPG_Peaks")
        return _hrv_from_peaks(peaks, sfreq)
    except Exception:
        return 0.0, 0.0, 0.0


def _process_ecg(ecg, sfreq):
    """Extract HRV features from ECG signal using NeuroKit2."""
    if not _NK_AVAILABLE or ecg is None or len(ecg) < int(sfreq * 5):
        return 0.0, 0.0, 0.0
    try:
        _, ecg_info = nk.ecg_process(ecg.astype(float), sampling_rate=int(sfreq))
        peaks = _peaks_from_info(ecg_info, "ECG_R_Peaks")
        return _hrv_from_peaks(peaks, sfreq)
    except Exception:
        return 0.0, 0.0, 0.0


def _process_eda(eda, sfreq, T):
    """
    Decompose EDA into tonic (SCL) and phasic (SCR) components via NeuroKit2.
    Returns (T, 2) array downsampled to T timesteps.
    """
    if not _NK_AVAILABLE or eda is None or len(eda) < int(sfreq * 2):
        return np.zeros((T, 2), dtype=np.float32)
    try:
        eda_signals, _ = nk.eda_process(eda.astype(float), sampling_rate=int(sfreq))
        tonic  = eda_signals["EDA_Tonic"].values
        phasic = eda_signals["EDA_Phasic"].values
    except Exception:
        return np.zeros((T, 2), dtype=np.float32)

    # Downsample both to T timesteps via mean-pooling
    def pool_to_T(signal):
        n = len(signal)
        chunk = max(1, n // T)
        out = np.array([signal[i * chunk: (i + 1) * chunk].mean()
                        for i in range(T)], dtype=np.float32)
        return out

    tonic_T  = pool_to_T(tonic)
    phasic_T = pool_to_T(phasic)

    # Z-score within trial
    for arr in [tonic_T, phasic_T]:
        std = arr.std()
        if std > 1e-6:
            arr -= arr.mean()
            arr /= std

    return np.stack([tonic_T, phasic_T], axis=1)  # (T, 2)


def _process_temp(temp, sfreq, T):
    """
    Normalize skin temperature to roughly [-1, 1].
    Typical wrist skin temp range: 28-38°C → (x - 32) / 8
    Returns (T,) array.
    """
    if temp is None or len(temp) == 0:
        return np.zeros(T, dtype=np.float32)

    n = len(temp)
    chunk = max(1, n // T)
    temp_T = np.array([
        temp[i * chunk: (i + 1) * chunk].mean() if len(temp[i * chunk: (i + 1) * chunk]) > 0 else 0.0
        for i in range(T)
    ], dtype=np.float32)

    # Normalize: skin temp ~30-38°C, center at 34, scale by 4
    temp_T = (temp_T - 34.0) / 4.0
    return np.clip(temp_T, -2.0, 2.0)


def extract_physio_features(
    bvp=None,
    ecg=None,
    eda=None,
    temp=None,
    sfreq_bvp=64,
    sfreq_ecg=256,
    sfreq_eda=4,
    sfreq_temp=4,
    T=30,
):
    """
    Extract physiological features from raw biosignals.

    Accepts BVP (PPG) or ECG for HRV, plus EDA and TEMP.
    HRV features are window-level statistics replicated across T timesteps.
    EDA and TEMP are per-timestep.

    Args:
        bvp:       (n_samples,) BVP/PPG signal — use for DEAP/WESAD
        ecg:       (n_samples,) ECG signal — use for DREAMER
        eda:       (n_samples,) skin conductance (EDA/GSR)
        temp:      (n_samples,) skin temperature
        sfreq_bvp: BVP sampling rate in Hz (default 64 for WESAD Empatica E4)
        sfreq_ecg: ECG sampling rate in Hz (default 256 for DREAMER)
        sfreq_eda: EDA sampling rate in Hz (default 4 for WESAD)
        sfreq_temp:TEMP sampling rate in Hz (default 4 for WESAD)
        T:         number of output timesteps (default 30)

    Returns:
        features: (T, 6) float32 array
            col 0 — RMSSD_norm     (Task Force 1996: parasympathetic index)
            col 1 — SDNN_norm      (Task Force 1996: total HRV)
            col 2 — mean_HR_norm   (BPM / 100)
            col 3 — EDA_tonic      (SCL, Boucsein 2012; z-scored)
            col 4 — EDA_phasic     (SCR, Boucsein 2012; z-scored)
            col 5 — TEMP_norm      ((°C - 34) / 4)
    """
    features = np.zeros((T, 6), dtype=np.float32)

    # HRV: prefer BVP if available, else ECG
    if bvp is not None:
        rmssd, sdnn, mean_hr = _process_bvp(bvp, sfreq_bvp)
    elif ecg is not None:
        rmssd, sdnn, mean_hr = _process_ecg(ecg, sfreq_ecg)
    else:
        rmssd, sdnn, mean_hr = 0.0, 0.0, 0.0

    # Replicate HRV across all T timesteps (window-level statistic)
    features[:, 0] = rmssd
    features[:, 1] = sdnn
    features[:, 2] = mean_hr

    # EDA per timestep
    if eda is not None:
        eda_feat = _process_eda(eda, sfreq_eda, T)  # (T, 2)
        features[:, 3] = eda_feat[:, 0]  # tonic
        features[:, 4] = eda_feat[:, 1]  # phasic

    # Temperature per timestep
    if temp is not None:
        features[:, 5] = _process_temp(temp, sfreq_temp, T)

    return features  # (T, 6)


def zeros_physio(T=30):
    """Return (T, 6) zeros for datasets without physio (e.g. AFEW-VA)."""
    return np.zeros((T, 6), dtype=np.float32)
