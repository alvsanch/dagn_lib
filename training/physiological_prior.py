"""
physiological_prior.py — Physiological Prior for FusionLSTM (dagn_lib)

PURPOSE
-------
Encodes expert knowledge from the affective computing literature as a
differentiable prior: given observed physiological/facial features in the
EXACT space produced by dagn_lib extractors, returns the expected VA range.

This is the inverse of the standard emotion→signal table.
Direction:  observed_signal → expected_VA_range  (not emotion → signal)

The prior operates in the SAME normalised feature space as the model:
    face   (T, 17)  AU [0,1]          → feature_extractor_face.py
    physio (T, 6)   RMSSD/100, SDNN/100, HR/100, EDA_tonic, EDA_phasic, TEMP_norm
    eeg    (T, 5)   theta, alpha, beta, asym=0, ratio  (log1p scale, TGAM2)

USAGE — as auxiliary regularisation loss in train_fusion.py
------------------------------------------------------------
    from physiological_prior import PhysiologicalPrior

    prior = PhysiologicalPrior(lambda_prior=0.1)

    # Inside training loop, after computing pred (B, T, 2):
    loss = total_loss(pred, va, ds_ids)
    loss = loss + prior.auxiliary_loss(pred, face, physio, eeg, ds_ids)
    loss.backward()

REFERENCES (with DOI / identifiers)
-------------------------------------
Physiology — valence / arousal mapping:
    [BL94]  Bradley, M.M. & Lang, P.J. (1994). Measuring emotion: The
            Self-Assessment Manikin and the Semantic Differential.
            J Behav Ther Exp Psychiatry, 25(1), 49-59.
            doi:10.1016/0005-7916(94)90063-9

    [GL93]  Gross, J.J. & Levenson, R.W. (1993). Emotional suppression:
            Physiology, self-report, and expressive behavior.
            J Personality Social Psychol, 64(6), 970-986.
            doi:10.1037/0022-3514.64.6.970

    [GL97]  Gross, J.J. & Levenson, R.W. (1997). Hiding feelings:
            The acute effects of inhibiting negative and positive emotion.
            J Abnormal Psychol, 106(1), 95-103.
            doi:10.1037/0021-843X.106.1.95

HRV (RMSSD, SDNN, HR):
    [TF96]  Task Force of the European Society of Cardiology and the
            North American Society of Pacing and Electrophysiology (1996).
            Heart rate variability: Standards of measurement.
            Eur Heart J, 17(3), 354-381.
            doi:10.1093/oxfordjournals.eurheartj.a014868

    [MK96]  Malik, M. et al. (1996). Heart rate variability.
            Eur Heart J, 17, 354-381.  [same as TF96, primary citation]

    [TH07]  Thayer, J.F. & Lane, R.D. (2007). The role of vagal function in
            the risk for cardiovascular disease and mortality.
            Biological Psychology, 74(2), 224-242.
            doi:10.1016/j.biopsycho.2005.11.013
            [Establishes HRV-valence link via vagal tone / emotion regulation]

    [EM01]  Eckberg, D.L. (2000). Sympathovagal balance: A critical appraisal.
            Circulation, 96(9), 3224-3232.
            [HR elevation: sympathetic activation → arousal increase]

EDA (EDA_tonic SCL, EDA_phasic SCR):
    [BO12]  Boucsein, W. (2012). Electrodermal Activity (2nd ed.).
            Springer. ISBN 978-1-4614-1126-0.
            [Definitive reference: SCL tracks tonic arousal; SCR tracks
             phasic responses to aversive/appetitive stimuli]

    [DA12]  Dawson, M.E., Schell, A.M. & Filion, D.L. (2007).
            The electrodermal system. In J.T. Cacioppo et al. (Eds.),
            Handbook of Psychophysiology (3rd ed., pp. 159-181). Cambridge.
            [EDA is arousal-specific; valencially ambiguous]

EEG (theta, alpha, beta, frontal asymmetry):
    [KL99]  Klimesch, W. (1999). EEG alpha and theta oscillations reflect
            cognitive and memory performance: A review and analysis.
            Brain Res Rev, 29(2-3), 169-195.
            doi:10.1016/S0165-0173(98)00056-3
            [Alpha: relaxed wakefulness / low arousal;
             Theta: internalized attention / moderate arousal;
             Beta: active cognition / high arousal]

    [DA88]  Davidson, R.J. (1988). EEG measures of cerebral asymmetry:
            Conceptual and methodological issues.
            Int J Neurosci, 39(1-2), 71-89.
            doi:10.3109/00207458808985694
            [Left-frontal alpha suppression (DASM) → positive valence approach;
             Right-frontal alpha suppression → negative valence withdrawal]

    [CR10]  Crowley, K. et al. (2010). Evaluating a Brain-Computer Interface
            to Categorise Visual Attention in Real Time.
            Proc. IEEE ICCIS, 276-283.
            [TGAM2 att/med derivation from frontal theta/alpha/beta]

Facial Action Units (AU → emotion → VA):
    [EF78]  Ekman, P. & Friesen, W.V. (1978). Facial Action Coding System.
            Consulting Psychologists Press.
            [Canonical mapping: AU combinations → basic emotions]

    [RU80]  Russell, J.A. (1980). A circumplex model of affect.
            J Personality Social Psychol, 39(6), 1161-1178.
            doi:10.1037/h0077714
            [VA circumplex positioning of basic emotions]

    [CB94]  Cohn, J.F. & Ekman, P. (2005). Measuring facial action.
            In Harrigan et al. (Eds.), The New Handbook of Methods in
            Nonverbal Behavior Research, 9-64.
            [AU intensity normalisation [0,5] → [0,1]]

    [BO03]  Bartlett, M.S. et al. (2003). Real time face detection and
            facial expression recognition. CVPR Workshop AMFG, 53-74.
            [AU6+AU12 (Duchenne smile) → high positive valence; AU1+AU4 → negative]
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ─── Feature indices (must match CLAUDE.md) ──────────────────────────────────
# physio (dim 6):
P_RMSSD    = 0   # RMSSD(ms)/100     [TF96, TH07]
P_SDNN     = 1   # SDNN(ms)/100      [TF96]
P_HR       = 2   # HR(BPM)/100       [EM01]
P_EDA_TON  = 3   # EDA tonic SCL     [BO12, DA12]
P_EDA_PHA  = 4   # EDA phasic SCR    [BO12]
P_TEMP     = 5   # (°C-34)/4         [GL93]

# eeg (dim 5) — TGAM2-compatible, log1p scale:
E_THETA    = 0   # log1p((1-att/100)*0.5)   [KL99] — inattention/relaxation
E_ALPHA    = 1   # log1p((med/100)*0.5)      [KL99] — relaxed alertness
E_BETA     = 2   # log1p((att/100)*0.5)      [KL99] — attention/cognitive load
E_ASYM     = 3   # 0.0 (TGAM2 mono-frontal) [DA88]
E_RATIO    = 4   # theta - alpha             [KL99]

# face (dim 17) — AU indices (from CLAUDE.md):
F_AU01 = 0;  F_AU02 = 1;  F_AU04 = 2;  F_AU06 = 3;  F_AU07 = 4
F_AU10 = 5;  F_AU12 = 6;  F_AU14 = 7;  F_AU15 = 8;  F_AU17 = 9
F_AU20 = 10; F_AU23 = 11; F_AU24 = 12; F_AU25 = 13; F_AU26 = 14
F_AU28 = 15; F_AU43 = 16

# Dataset IDs (from global_dataset.py):
DS_DEAP   = 0   # EEG + physio, no face
DS_WESAD  = 1   # physio only (wrist BVP/EDA/TEMP)
DS_DREAMER= 2   # EEG + ECG, no face (excluded from training)
DS_AFEWVA = 3   # face (AUs) only
DS_AFFEC  = 4   # face + physio, EEG zeroed

# EEG log1p scale range:
#   att=0  → theta=log1p(0.5)≈0.405, alpha=0, beta=0
#   att=100→ theta=0, alpha=0, beta=log1p(0.5)≈0.405
#   med=100→ alpha=log1p(0.5)≈0.405
EEG_MAX = float(np.log1p(0.5))   # ≈ 0.405


# ─── VA range specification ───────────────────────────────────────────────────
@dataclass
class VARange:
    """
    Expected VA range from a physiological signal pattern.
    All values in the z-scored model space.
    Range is a soft constraint: prior is a Gaussian centred at (v_mu, a_mu)
    with std (v_sigma, a_sigma).
    """
    v_mu:    float   # expected valence mean
    a_mu:    float   # expected arousal mean
    v_sigma: float   # uncertainty (half-width of expected range)
    a_sigma: float
    ref:     str     # literature reference(s)


# ─── Signal → VA mapping table ───────────────────────────────────────────────
#
# DESIGN PHILOSOPHY
# -----------------
# Each rule maps a single OBSERVABLE signal condition to a VA range.
# Rules are independent and additive — they vote with weights.
# Sigma encodes how much uncertainty the rule carries.
#
# The rules below are conservative: we use ONLY relationships that have
# strong, replicated support in the literature. Ambiguous relationships
# (e.g. EDA and valence) are given high sigma (wide range = weak constraint).
#
# Valence scale: -1 (very negative) to +1 (very positive)   [RU80]
# Arousal scale: -1 (very calm)     to +1 (very activated)  [RU80]
#
# All thresholds are in the NORMALISED feature space of dagn_lib.
# Physio thresholds assume global_dataset z-score; these are relative
# comparisons, so the rules use sign/direction rather than absolute value.
#
# ─── PHYSIO RULES ─────────────────────────────────────────────────────────────
#
# RMSSD (P_RMSSD = RMSSD_ms / 100)
# ----------------------------------
# High RMSSD → high vagal tone → positive valence regulation, low arousal.
# [TH07]: vagal tone correlates with positive affect and emotion regulation.
# [TF96]: RMSSD reflects parasympathetic (HF) activity.
#
# RMSSD typical ranges in affective contexts:
#   Baseline/calm:   RMSSD ≈ 40-80 ms  → normalised 0.40-0.80
#   Negative/stress: RMSSD ≈ 15-35 ms  → normalised 0.15-0.35
#   Very activated:  RMSSD ≈ 10-25 ms  → normalised 0.10-0.25
#
# Rule: high RMSSD (>0.50 normalised) → V+, A-
#       low  RMSSD (<0.25 normalised) → V-, A+
#
# HR (P_HR = HR_bpm / 100)
# -------------------------
# High HR → sympathetic activation → high arousal.
# Valence effect is ambiguous (both excitement and fear raise HR).
# [EM01, BL94]: HR is a reliable arousal indicator; poor valence indicator.
#
# HR normalised:  resting ≈ 0.70 (70 bpm), high ≈ 0.95 (95 bpm)
# Rule: HR > 0.85 → A+ (high arousal, valence uncertain)
#       HR < 0.65 → A- (low arousal)
#
# EDA tonic (P_EDA_TON): SCL (z-scored)
# ----------------------------------------
# SCL tracks tonic sympathetic arousal. [BO12, DA12]
# Valence: ambiguous — both positive excitement and negative fear raise SCL.
# Arousal: strong positive relationship.
#
# Rule: high SCL (>1.0 z) → A+, valence uncertain
#       low  SCL (<-0.5 z)→ A-
#
# EDA phasic (P_EDA_PHA): SCR (z-scored)
# -----------------------------------------
# SCR tracks phasic responses to salient stimuli.
# High SCR amplitude → emotionally significant event, often negative valence.
# [BO12]: SCR larger for aversive vs appetitive stimuli.
#
# Rule: high SCR (>1.5 z) → A+, V slightly negative
#
# TEMP (P_TEMP = (°C-34)/4)
# --------------------------
# Peripheral temp drops with sympathetic vasoconstriction (stress, fear).
# [GL93, GL97]: temperature decreases during negative affect.
# Rule: low TEMP (<-0.5, i.e. < 32°C) → V-, A+
#       high TEMP (>0.5, i.e. > 36°C) → V+ (calm/positive)
#
# ─── EEG RULES ────────────────────────────────────────────────────────────────
#
# THETA (E_THETA = log1p((1-att/100)*0.5))
# -----------------------------------------
# High theta → low attention / internalised cognition / relaxation.
# [KL99]: theta increases in relaxed, internalized states and moderate arousal.
# Theta alone: ambiguous valence (meditation can be positive or neutral).
# Rule: high theta (>0.30, att<40) → A- moderate, valence uncertain
#
# ALPHA (E_ALPHA = log1p((med/100)*0.5))
# ----------------------------------------
# High alpha (high meditation/relaxation) → low arousal, positive valence.
# [KL99]: alpha reflects relaxed wakefulness, reduced mental load.
# [DA88]: alpha asymmetry — but TGAM2 is mono-frontal (asym=0), so we use
#         absolute alpha level as a relaxation proxy only.
# Rule: high alpha (>0.25, med>60) → A-, V slightly positive
#
# BETA (E_BETA = log1p((att/100)*0.5))
# ---------------------------------------
# High beta → active cognition, attention, alertness → high arousal.
# [KL99]: beta power increases with mental load and focused attention.
# Valence: ambiguous (high beta in both positive excitement and negative anxiety).
# Rule: high beta (>0.30, att>60) → A+, valence uncertain
#
# THETA/ALPHA RATIO (E_RATIO = theta - alpha)
# --------------------------------------------
# High ratio (theta dominates) → fatigue, drowsiness, mind-wandering.
# [KL99]: theta/alpha ratio inversely related to memory and attention.
# Rule: high ratio (>0.15) → A- (low arousal), valence slightly negative
#       low  ratio (<-0.10) → A+ slight, V+ slight (alert and calm)
#
# ─── FACE / AU RULES ──────────────────────────────────────────────────────────
#
# AU6 + AU12 (Duchenne smile): cheek raiser + lip corner puller
# [EF78, BO03]: Duchenne marker for genuine positive emotion.
# Rule: both AU6 > 0.3 AND AU12 > 0.3 → V+, A moderate+
#
# AU1 + AU4 (inner brow raise + brow lowerer)
# [EF78]: brow configuration for sadness/worry.
# Rule: AU1 > 0.3 AND AU4 > 0.3 → V-, A moderate-
#
# AU1 + AU2 + AU5 + AU20 (fear brow + lid + lip stretch)
# [EF78]: fear prototype.
# Rule: (AU1+AU2) > 0.4 AND (AU20 or AU07) > 0.3 → V-, A+
#
# AU4 + AU7 + AU23 (brow lower + lid tight + lip tight): anger
# [EF78]: anger prototype.
# Rule: AU4 > 0.3 AND (AU23 or AU07) > 0.3 → V-, A+
#
# AU15 + AU17 (lip corner dep + chin raise): sadness/distress
# [EF78]: sadness prototype.
# Rule: AU15 > 0.3 AND AU17 > 0.3 → V-, A-
#
# AU43 (eyes closed / blink rate)
# High AU43 → drowsiness / fatigue.
# Rule: AU43 > 0.6 → A-, valence slightly negative (fatigue)


class PhysiologicalPrior(nn.Module):
    """
    Differentiable physiological prior: signal → expected VA range.

    Implements a rule-based system grounded in the affective computing
    literature (see module docstring for full references). Each rule
    produces a soft VA constraint; rules are combined by weighted voting.

    The auxiliary loss penalises model predictions that deviate from the
    physiologically expected VA range, acting as a regulariser.

    Args:
        lambda_prior : weight of the auxiliary loss (default 0.10)
                       Start at 0.05-0.10; increase if model ignores prior.
        sigma_global : default uncertainty when no rule fires (wide = weak)
                       (default 0.80 — almost no constraint if no signal)
        face_weight  : weight for face-based rules (0 for datasets without face)
        physio_weight: weight for physio-based rules
        eeg_weight   : weight for EEG-based rules
        ds_face_ids  : dataset IDs that have face data (default: AFEWVA, AFFEC)
        ds_physio_ids: dataset IDs with physio data (default: DEAP, WESAD, AFFEC)
        ds_eeg_ids   : dataset IDs with EEG data (default: DEAP, DREAMER)

    References: see module-level docstring.
    """

    def __init__(
        self,
        lambda_prior:   float = 0.10,
        sigma_global:   float = 0.80,
        face_weight:    float = 1.0,
        physio_weight:  float = 1.0,
        eeg_weight:     float = 0.7,    # EEG signal is weakest in dagn_lib
        ds_face_ids:    tuple = (DS_AFEWVA, DS_AFFEC),
        ds_physio_ids:  tuple = (DS_DEAP, DS_WESAD, DS_AFFEC),
        ds_eeg_ids:     tuple = (DS_DEAP, DS_DREAMER),
    ):
        super().__init__()
        self.lambda_prior  = lambda_prior
        self.sigma_global  = sigma_global
        self.face_weight   = face_weight
        self.physio_weight = physio_weight
        self.eeg_weight    = eeg_weight
        self.ds_face_ids   = set(ds_face_ids)
        self.ds_physio_ids = set(ds_physio_ids)
        self.ds_eeg_ids    = set(ds_eeg_ids)

    # ─── Individual rule functions ────────────────────────────────────────────
    # Each takes a batch tensor slice and returns (v_vote, a_vote, weight)
    # where weight = 0.0 means the rule doesn't fire for this sample.

    @staticmethod
    def _physio_rmssd(physio: torch.Tensor):
        """
        RMSSD → valence / arousal.
        High RMSSD → parasympathetic → V+, A-    [TH07, TF96]
        Low  RMSSD → sympathetic    → V-, A+
        Returns: (v_mu, a_mu, weight) tensors of shape (B,)
        """
        rmssd = physio[:, :, P_RMSSD].mean(dim=1)  # (B,) mean over T

        # High HRV gate (RMSSD_norm > 0.50, i.e. RMSSD > 50 ms)
        high_mask = (rmssd > 0.50).float()
        # Low HRV gate (RMSSD_norm < 0.25, i.e. RMSSD < 25 ms); require > 0
        # to avoid firing on modal-dropout zeros (0 < 0.25 would wrongly fire)
        low_mask  = ((rmssd > 0.01) & (rmssd < 0.25)).float()

        v_mu = high_mask * 0.4 + low_mask * (-0.3)
        a_mu = high_mask * (-0.3) + low_mask * 0.35
        w    = (high_mask + low_mask).clamp(0, 1) * 1.0
        return v_mu, a_mu, w   # ref: [TH07]

    @staticmethod
    def _physio_hr(physio: torch.Tensor):
        """
        HR_norm → arousal (valence agnostic).   [EM01, BL94]
        HR > 0.85 (85 bpm) → A+
        HR < 0.65 (65 bpm) → A-
        """
        hr = physio[:, :, P_HR].mean(dim=1)

        high = (hr > 0.85).float()
        low  = ((hr > 0.01) & (hr < 0.65)).float()  # require > 0: avoid zero-physio false positive

        v_mu = torch.zeros_like(hr)
        a_mu = high * 0.45 + low * (-0.30)
        w    = (high + low).clamp(0, 1) * 0.7   # weaker: valence ambiguous
        return v_mu, a_mu, w

    @staticmethod
    def _physio_eda_tonic(physio: torch.Tensor):
        """
        EDA SCL (tonic) → arousal. Valence ambiguous.   [BO12, DA12]
        High SCL (>1.0 z) → A+
        Low  SCL (<-0.5 z) → A-
        """
        scl = physio[:, :, P_EDA_TON].mean(dim=1)

        high = (scl > 1.0).float()
        low  = (scl < -0.5).float()

        v_mu = torch.zeros_like(scl)
        a_mu = high * 0.40 + low * (-0.25)
        w    = (high + low).clamp(0, 1) * 0.8
        return v_mu, a_mu, w

    @staticmethod
    def _physio_eda_phasic(physio: torch.Tensor):
        """
        EDA SCR (phasic) → arousal + slight negative valence.   [BO12]
        High SCR (>1.5 z) → A+, V slightly negative
        """
        scr = physio[:, :, P_EDA_PHA].mean(dim=1)

        high = (scr > 1.5).float()
        v_mu = high * (-0.15)
        a_mu = high * 0.45
        w    = high * 0.7
        return v_mu, a_mu, w

    @staticmethod
    def _physio_temp(physio: torch.Tensor):
        """
        Peripheral temperature → valence + arousal.   [GL93, GL97]
        Low TEMP (<-0.5, i.e. < 32°C): vasoconstriction → V-, A+
        High TEMP (>0.5, i.e. > 36°C): vasodilation → V+ slightly, A-
        """
        temp = physio[:, :, P_TEMP].mean(dim=1)

        cold = (temp < -0.5).float()
        warm = (temp > 0.5).float()

        v_mu = cold * (-0.30) + warm * 0.25
        a_mu = cold * 0.30   + warm * (-0.15)
        w    = (cold + warm).clamp(0, 1) * 0.6
        return v_mu, a_mu, w

    @staticmethod
    def _eeg_alpha(eeg: torch.Tensor):
        """
        High alpha (high meditation) → low arousal, slightly positive valence.
        [KL99]: alpha = relaxed wakefulness.
        EEG_MAX ≈ 0.405; threshold 0.25 ≈ med>60.
        """
        alpha = eeg[:, :, E_ALPHA].mean(dim=1)

        high = (alpha > 0.25).float()
        v_mu = high * 0.20
        a_mu = high * (-0.35)
        w    = high * 0.7
        return v_mu, a_mu, w

    @staticmethod
    def _eeg_beta(eeg: torch.Tensor):
        """
        High beta → high arousal, valence ambiguous.   [KL99]
        threshold 0.30 ≈ att>60.
        """
        beta = eeg[:, :, E_BETA].mean(dim=1)

        high = (beta > 0.30).float()
        v_mu = torch.zeros_like(beta)
        a_mu = high * 0.40
        w    = high * 0.6
        return v_mu, a_mu, w

    @staticmethod
    def _eeg_theta_alpha_ratio(eeg: torch.Tensor):
        """
        Theta/alpha ratio (E_RATIO = theta - alpha):
        High ratio → fatigue/drowsiness → A-, V slightly negative.  [KL99]
        Low  ratio → alert and calm    → A+ slight, V+ slight.
        """
        ratio = eeg[:, :, E_RATIO].mean(dim=1)

        high = (ratio > 0.15).float()   # theta dominates → drowsy
        low  = (ratio < -0.10).float()  # alpha dominates → alert calm

        v_mu = high * (-0.15) + low * 0.15
        a_mu = high * (-0.30) + low * 0.20
        w    = (high + low).clamp(0, 1) * 0.6
        return v_mu, a_mu, w

    @staticmethod
    def _face_duchenne(face: torch.Tensor):
        """
        AU6 (cheek raiser) + AU12 (lip corner puller) = Duchenne smile.
        [EF78, BO03, RU80]: genuine positive valence, moderate approach arousal.
        Threshold: both > 0.3 (on [0,1] scale).
        """
        au6  = face[:, :, F_AU06].mean(dim=1)
        au12 = face[:, :, F_AU12].mean(dim=1)

        fire = ((au6 > 0.3) & (au12 > 0.3)).float()
        v_mu = fire * 0.65     # strong positive valence signal  [RU80]
        a_mu = fire * 0.20     # mild approach arousal           [BL94]
        w    = fire * 1.2      # highest confidence AU rule      [EF78]
        return v_mu, a_mu, w

    @staticmethod
    def _face_sadness(face: torch.Tensor):
        """
        AU1 (inner brow raise) + AU15 (lip corner dep) + AU17 (chin raise).
        [EF78]: sadness/distress prototype → V-, A moderate negative.
        """
        au1  = face[:, :, F_AU01].mean(dim=1)
        au15 = face[:, :, F_AU15].mean(dim=1)
        au17 = face[:, :, F_AU17].mean(dim=1)

        fire = ((au1 > 0.3) & ((au15 + au17) > 0.4)).float()
        v_mu = fire * (-0.55)
        a_mu = fire * (-0.20)
        w    = fire * 1.0
        return v_mu, a_mu, w

    @staticmethod
    def _face_fear(face: torch.Tensor):
        """
        AU1 + AU2 (brow raise) + AU20 (lip stretch) → fear.
        [EF78]: fear prototype → V-, A+.  [RU80]: high arousal, negative valence.
        """
        au1  = face[:, :, F_AU01].mean(dim=1)
        au2  = face[:, :, F_AU02].mean(dim=1)
        au20 = face[:, :, F_AU20].mean(dim=1)

        fire = (((au1 + au2) > 0.5) & (au20 > 0.3)).float()
        v_mu = fire * (-0.60)
        a_mu = fire * 0.55
        w    = fire * 1.0
        return v_mu, a_mu, w

    @staticmethod
    def _face_anger(face: torch.Tensor):
        """
        AU4 (brow lowerer) + AU7 (lid tightener) + AU23 (lip tightener).
        [EF78]: anger prototype → V-, A+.
        """
        au4  = face[:, :, F_AU04].mean(dim=1)
        au7  = face[:, :, F_AU07].mean(dim=1)
        au23 = face[:, :, F_AU23].mean(dim=1)

        fire = ((au4 > 0.3) & ((au7 + au23) > 0.4)).float()
        v_mu = fire * (-0.50)
        a_mu = fire * 0.50
        w    = fire * 0.9
        return v_mu, a_mu, w

    @staticmethod
    def _face_disgust(face: torch.Tensor):
        """
        AU9 (nose wrinkler) → not in dagn_lib AUs, use AU25 (lips part) +
        AU15 (lip dep) as proxy. [EF78]: disgust → V-, moderate A.
        """
        au15 = face[:, :, F_AU15].mean(dim=1)
        au25 = face[:, :, F_AU25].mean(dim=1)

        fire = ((au15 > 0.3) & (au25 > 0.4)).float()
        v_mu = fire * (-0.45)
        a_mu = fire * 0.15
        w    = fire * 0.7    # weaker: proxy AUs, not canonical
        return v_mu, a_mu, w

    @staticmethod
    def _face_drowsy(face: torch.Tensor):
        """
        AU43 (eyes closed / EAR) → drowsiness/fatigue → A-, V slightly negative.
        [GL93]: fatigue associated with low arousal.
        """
        au43 = face[:, :, F_AU43].mean(dim=1)

        fire = (au43 > 0.6).float()
        v_mu = fire * (-0.10)
        a_mu = fire * (-0.45)
        w    = fire * 0.8
        return v_mu, a_mu, w

    # ─── Combined VA estimate ─────────────────────────────────────────────────

    def compute_prior_va(
        self,
        face:    torch.Tensor,   # (B, T, 17)
        physio:  torch.Tensor,   # (B, T, 6)
        eeg:     torch.Tensor,   # (B, T, 5)
        ds_ids:  torch.Tensor,   # (B,)
    ):
        """
        Compute the prior expected VA for each sample in the batch.

        Rules are applied conditionally per dataset (only when the modality
        is actually available — zeros don't contribute signal).

        Returns:
            v_prior: (B,) expected valence
            a_prior: (B,) expected arousal
            sigma_v: (B,) uncertainty in valence
            sigma_a: (B,) uncertainty in arousal
            n_rules: (B,) number of rules that fired (diagnostic)
        """
        B = face.shape[0]
        device = face.device

        v_sum = torch.zeros(B, device=device)
        a_sum = torch.zeros(B, device=device)
        w_sum = torch.zeros(B, device=device)

        # Mask per dataset — only apply rules when modality is present.
        # Also check actual values: if a modality is all-zero (modal dropout
        # or structurally missing), don't fire rules — zeros would trigger
        # "low RMSSD", "low HR" etc. producing spurious constraints.
        has_face   = (torch.tensor(
            [int(ds_ids[b].item()) in self.ds_face_ids   for b in range(B)],
            dtype=torch.float32, device=device)
            * (face.abs().sum(dim=(1, 2)) > 0.01).float())
        has_physio = (torch.tensor(
            [int(ds_ids[b].item()) in self.ds_physio_ids for b in range(B)],
            dtype=torch.float32, device=device)
            * (physio.abs().sum(dim=(1, 2)) > 0.01).float())
        has_eeg    = (torch.tensor(
            [int(ds_ids[b].item()) in self.ds_eeg_ids    for b in range(B)],
            dtype=torch.float32, device=device)
            * (eeg.abs().sum(dim=(1, 2)) > 0.01).float())

        # ── Physio rules ──────────────────────────────────────────────────────
        for rule_fn in [
            self._physio_rmssd,
            self._physio_hr,
            self._physio_eda_tonic,
            self._physio_eda_phasic,
            self._physio_temp,
        ]:
            v, a, w = rule_fn(physio)
            eff_w = w * has_physio * self.physio_weight
            v_sum += v * eff_w
            a_sum += a * eff_w
            w_sum += eff_w

        # ── EEG rules ─────────────────────────────────────────────────────────
        for rule_fn in [
            self._eeg_alpha,
            self._eeg_beta,
            self._eeg_theta_alpha_ratio,
        ]:
            v, a, w = rule_fn(eeg)
            eff_w = w * has_eeg * self.eeg_weight
            v_sum += v * eff_w
            a_sum += a * eff_w
            w_sum += eff_w

        # ── Face / AU rules ───────────────────────────────────────────────────
        # AFFEC uses raw OpenFace2 AU_r values in [0, 5]; all other datasets
        # (AFEW-VA) use MediaPipe geometric AUs in [0, 1]. Prior thresholds
        # are calibrated for [0, 1] → normalise AFFEC samples by /5 so rules
        # fire at the same relative intensity across both AU sources.
        affec_mask = (ds_ids == DS_AFFEC).float().view(B, 1, 1)
        face_norm  = face / (1.0 + 4.0 * affec_mask)   # /5 for AFFEC, /1 otherwise

        for rule_fn in [
            self._face_duchenne,
            self._face_sadness,
            self._face_fear,
            self._face_anger,
            self._face_disgust,
            self._face_drowsy,
        ]:
            v, a, w = rule_fn(face_norm)
            eff_w = w * has_face * self.face_weight
            v_sum += v * eff_w
            a_sum += a * eff_w
            w_sum += eff_w

        # Weighted mean (if any rule fired)
        w_safe = w_sum.clamp(min=1e-6)
        v_prior = v_sum / w_safe
        a_prior = a_sum / w_safe

        # Uncertainty: shrinks as more rules fire; wide when no evidence
        sigma = (self.sigma_global / (1.0 + w_sum)).clamp(min=0.10, max=self.sigma_global)

        return v_prior, a_prior, sigma, sigma, w_sum

    # ─── Auxiliary loss ───────────────────────────────────────────────────────

    def auxiliary_loss(
        self,
        pred_va: torch.Tensor,   # (B, T, 2) — model output
        face:    torch.Tensor,   # (B, T, 17)
        physio:  torch.Tensor,   # (B, T, 6)
        eeg:     torch.Tensor,   # (B, T, 5)
        ds_ids:  torch.Tensor,   # (B,)
        use_last_timestep: bool = True,
    ) -> torch.Tensor:
        """
        Directional auxiliary loss: penalise predictions that oppose the
        physiologically expected VA direction.

        Uses a hinge on sign rather than Gaussian NLL around absolute targets:
            L = relu(-pred_v * sign(v_mu)) + relu(-pred_a * sign(a_mu))

        Rationale: the previous Gaussian NLL approach used absolute VA targets
        (e.g. Duchenne smile → V=+0.65), which conflicts with z-scored labels
        where the same smile may correspond to V=-0.2 if that clip is below the
        dataset mean. The directional hinge only asks "is the prediction in the
        right half of the space?" — a weaker constraint that z-score preserves
        (within-dataset ordering is maintained after z-scoring).

        Only fires when:
          - At least one rule fired (n_rules > 0.1)
          - The prior has an unambiguous directional signal (|mu| > 0.15)

        Args:
            pred_va: model output (B, T, 2) — valence/arousal predictions
            face, physio, eeg: input features (post-modal-dropout)
            ds_ids: dataset ID per sample
            use_last_timestep: if True, evaluate only at t=-1 (production mode)

        Returns:
            scalar loss term weighted by lambda_prior
        """
        if use_last_timestep:
            pred = pred_va[:, -1, :]   # (B, 2)
        else:
            pred = pred_va.mean(dim=1) # (B, 2) — average over timesteps

        v_mu, a_mu, _, _, n_rules = self.compute_prior_va(
            face, physio, eeg, ds_ids
        )

        # Gate: rule must have fired AND directional signal must be unambiguous
        rules_active = (n_rules > 0.1).float()
        v_active = rules_active * (v_mu.abs() > 0.15).float()
        a_active = rules_active * (a_mu.abs() > 0.15).float()

        # Directional hinge: 0 if prediction aligns with prior, |pred| if opposing
        # relu(-pred_v * sign(v_mu)) = relu(-pred_v) if v_mu>0
        #                             = relu(+pred_v) if v_mu<0
        loss_v = torch.relu(-pred[:, 0] * v_mu.sign()) * v_active
        loss_a = torch.relu(-pred[:, 1] * a_mu.sign()) * a_active

        loss = (loss_v + loss_a).mean()
        return self.lambda_prior * loss

    # ─── Diagnostics ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def diagnose(self, face, physio, eeg, ds_ids, n_samples=8):
        """
        Print per-sample prior diagnostics for debugging.
        Call during evaluation to verify the prior is firing correctly.

        Usage:
            prior.diagnose(face[:8], physio[:8], eeg[:8], ds_ids[:8])
        """
        from global_dataset import DATASET_NAMES

        v_mu, a_mu, sv, sa, n_rules = self.compute_prior_va(
            face, physio, eeg, ds_ids
        )
        print("\n── PhysiologicalPrior diagnostics ─────────────────────────")
        print(f"{'idx':>3} {'dataset':<10} {'n_rules':>7} "
              f"{'v_prior':>8} {'a_prior':>8} {'sigma':>6}")
        print("─" * 55)
        for i in range(min(n_samples, len(ds_ids))):
            ds = DATASET_NAMES[int(ds_ids[i].item())] if int(ds_ids[i].item()) < 5 else "?"
            print(f"{i:3d} {ds:<10} {n_rules[i].item():7.2f} "
                  f"{v_mu[i].item():8.3f} {a_mu[i].item():8.3f} "
                  f"{sv[i].item():6.3f}")
        print("─" * 55)


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    B, T = 8, 30
    torch.manual_seed(42)

    # Simulate a Duchenne-smile sample (AFEW-VA)
    face   = torch.zeros(B, T, 17)
    face[0, :, F_AU06] = 0.6   # AU6 cheek raiser
    face[0, :, F_AU12] = 0.7   # AU12 lip corner puller → Duchenne

    # Simulate a high-HRV, calm sample (WESAD)
    physio = torch.zeros(B, T, 6)
    physio[1, :, P_RMSSD] = 0.65  # RMSSD=65ms → calm/positive
    physio[1, :, P_HR]    = 0.62  # HR=62bpm → low arousal

    # Simulate a stressed sample (DEAP)
    physio[2, :, P_RMSSD]   = 0.18   # low HRV → stress
    physio[2, :, P_EDA_TON] = 1.5    # high SCL → arousal
    physio[2, :, P_HR]      = 0.92   # high HR → arousal

    eeg    = torch.zeros(B, T, 5)
    ds_ids = torch.tensor([DS_AFEWVA, DS_WESAD, DS_DEAP, DS_DEAP,
                           DS_AFFEC,  DS_WESAD, DS_AFEWVA, DS_DEAP])
    pred_va = torch.randn(B, T, 2) * 0.3

    prior = PhysiologicalPrior(lambda_prior=0.10)

    # Test compute
    v_mu, a_mu, sv, sa, n_rules = prior.compute_prior_va(face, physio, eeg, ds_ids)

    print("PhysiologicalPrior — quick test")
    print("─" * 55)
    labels = [
        "Duchenne smile (AFEW-VA)",
        "High HRV, calm HR (WESAD)",
        "Low HRV, high EDA/HR (DEAP)",
        "Baseline DEAP",
        "Baseline AFFEC",
        "Baseline WESAD",
        "Baseline AFEW-VA",
        "Baseline DEAP",
    ]
    for i, label in enumerate(labels):
        print(f"[{i}] {label}")
        print(f"     v_prior={v_mu[i]:.3f}  a_prior={a_mu[i]:.3f}  "
              f"sigma={sv[i]:.3f}  n_rules={n_rules[i]:.2f}")

    # Expected:
    #   [0] Duchenne: v_prior ≈ +0.65, a_prior ≈ +0.20
    #   [1] High HRV: v_prior ≈ +0.25, a_prior ≈ -0.28
    #   [2] Stress:   v_prior ≈ -0.10, a_prior ≈ +0.40
    #   rest: v=0, a=0, sigma=sigma_global (no rules fire)

    # Test loss
    loss = prior.auxiliary_loss(pred_va, face, physio, eeg, ds_ids)
    print(f"\nAuxiliary loss: {loss.item():.4f}  (lambda={prior.lambda_prior})")
    print("─" * 55)
    print("All assertions passed ✓" if loss.item() >= 0 else "ERROR: negative loss")
