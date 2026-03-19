"""
fusion_model.py — FusionLSTM for dagn_lib

Architecture (explainable on a whiteboard in 5 minutes):

    face   (T, 17) ──┐
    physio (T,  6) ──┤── LayerNorm ── LSTM(33→256, 2L) ── Linear(256,2) ── tanh → VA (T, 2)
    eeg    (T, 10) ──┘

    + temporal gradient: grad[t] = VA[t] - VA[t-1]

Input features:
    face   — 17 Action Units (Ekman 1978 FACS), MediaPipe FaceMesh
    physio —  6 HRV/EDA/TEMP features (Task Force 1996, Boucsein 2012)
    eeg    — 10 bilateral frontal features (Davidson 1988, Klimesch 1999)
               [theta_avg, alpha_avg, beta_avg, FAA, ratio,
                theta_L, theta_R, alpha_L, alpha_R, beta_L]
               Training: F3/F7/FC1 (left) + F4/F8/FC2 (right) bandpower → 10D
               Production: NeuroSky TGAM2 att/med → symmetric 10D (FAA=0)

Total input dim = 33

Parameter count (hidden_dim=256, num_layers=2):
    LayerNorm(33):           66
    LSTM(33→256, L1):   297,984   [4×(33+256+2)×256]
    LSTM(256→256, L2):  526,336   [4×(256+256+2)×256]
    Linear(256→2):          514
    ──────────────────────────────
    Total:              824,900   ≪ 8.8M dagn_simple ✓

Usage:
    model = FusionLSTM()
    va, grad = model(face, physio, eeg)   # (B,T,2), (B,T,2)
"""
import torch
import torch.nn as nn

# Feature dimensions (must match global_dataset.py)
FACE_DIM   = 17
PHYSIO_DIM = 6
EEG_DIM    = 10
INPUT_DIM  = FACE_DIM + PHYSIO_DIM + EEG_DIM  # 33


class FusionLSTM(nn.Module):
    """
    Multimodal fusion via LSTM on concatenated library-extracted features.

    Architecture:
        1. Concatenate (face ‖ physio ‖ eeg) → (B, T, 33)
        2. LayerNorm(28) — normalize scale across modalities
        3. LSTM(28, hidden_dim, num_layers) → (B, T, hidden_dim)
        4. Dropout(p)
        5. Linear(hidden_dim, 2) + tanh → VA in [-1, 1]
        6. Temporal gradient: grad[t] = VA[t] - VA[t-1]

    Args:
        face_dim:    input face feature dimension (default 17, FACS AUs)
        physio_dim:  input physio feature dimension (default 6)
        eeg_dim:     input EEG feature dimension (default 10, bilateral frontal)
        hidden_dim:  LSTM hidden size (default 256)
        num_layers:  number of stacked LSTM layers (default 2)
        dropout:     dropout probability after final LSTM layer (default 0.45)
    """

    def __init__(
        self,
        face_dim:   int   = FACE_DIM,
        physio_dim: int   = PHYSIO_DIM,
        eeg_dim:    int   = EEG_DIM,
        hidden_dim: int   = 256,
        num_layers: int   = 2,
        dropout:    float = 0.45,
    ):
        super().__init__()

        self.face_dim   = face_dim
        self.physio_dim = physio_dim
        self.eeg_dim    = eeg_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        input_dim = face_dim + physio_dim + eeg_dim

        # Input normalization (handles scale differences between modalities)
        self.norm = nn.LayerNorm(input_dim)

        # Core: stacked LSTM learning temporal multimodal dynamics
        # inter_dropout applied between layers (0 for single-layer case)
        inter_dropout = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=inter_dropout,
        )

        self.dropout = nn.Dropout(p=dropout)

        # VA prediction head
        self.va_head = nn.Linear(hidden_dim, 2)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for LSTM gates, zero bias for VA head."""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0) // 4
                param.data[n:2*n].fill_(1.0)
        nn.init.xavier_uniform_(self.va_head.weight)
        nn.init.zeros_(self.va_head.bias)

    def forward(self, face, physio, eeg):
        """
        Forward pass.

        Args:
            face:   (B, T, face_dim)   — AU vectors, zeros if no video
            physio: (B, T, physio_dim) — HRV/EDA/TEMP, zeros if no physio
            eeg:    (B, T, eeg_dim)    — TGAM2-compatible features, zeros if no EEG

        Returns:
            va:   (B, T, 2) — continuous valence/arousal in [-1, 1]
            grad: (B, T, 2) — temporal gradient of VA
                              grad[:,0,:] = 0 (no previous state at t=0)
                              grad[:,t,:] = va[:,t,:] - va[:,t-1,:]  (t>0)
        """
        # Concatenate all modalities: (B, T, 28)
        x = torch.cat([face, physio, eeg], dim=-1)

        # Normalize across feature dimension
        x = self.norm(x)

        # LSTM — learns temporal multimodal dynamics
        lstm_out, _ = self.lstm(x)   # (B, T, hidden_dim)
        lstm_out = self.dropout(lstm_out)

        # VA prediction with tanh to bound output to [-1, 1]
        va = torch.tanh(self.va_head(lstm_out))  # (B, T, 2)

        # Temporal gradient: emotional trajectory (useful for production)
        grad = torch.zeros_like(va)
        grad[:, 1:, :] = va[:, 1:, :] - va[:, :-1, :]

        return va, grad

    def count_parameters(self):
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self):
        """Print architecture summary."""
        n = self.count_parameters()
        print(f"FusionLSTM")
        print(f"  Input:  face({self.face_dim}) + physio({self.physio_dim}) "
              f"+ eeg({self.eeg_dim}) = {self.face_dim+self.physio_dim+self.eeg_dim}")
        print(f"  LSTM:   hidden_dim={self.hidden_dim}, layers={self.num_layers}")
        print(f"  Output: VA (2), grad (2)")
        print(f"  Total parameters: {n:,}  ({'✓' if n < 8_800_000 else '✗'} < 8.8M dagn_simple)")
        return n


if __name__ == "__main__":
    # Quick sanity check
    model = FusionLSTM()
    model.describe()

    B, T = 4, 30
    face   = torch.zeros(B, T, FACE_DIM)
    physio = torch.zeros(B, T, PHYSIO_DIM)
    eeg    = torch.zeros(B, T, EEG_DIM)

    va, grad = model(face, physio, eeg)
    print(f"\nForward pass OK: va={va.shape}, grad={grad.shape}")
    print(f"VA range: [{va.min():.3f}, {va.max():.3f}]")
