"""
LSTM-based model for crypto price direction + magnitude prediction.

Architecture (quant-grade)
--------------------------
1. Feature Gate     : Learned sigmoid gate per feature per timestep — the model
                      dynamically suppresses irrelevant features and amplifies
                      predictive ones depending on the current market context.
2. LSTM Encoder     : Unidirectional (causal) multi-layer LSTM — no future leakage.
3. Temporal Attention: Additive attention over *all* hidden states so the model
                       can weight a volume spike 12h ago alongside current RSI.
4. Dual Heads       : Classification (UP/FLAT/DOWN) + regression (% change).
5. Temperature      : Learnable scalar calibrated post-training so softmax
                       probabilities reflect true observed frequencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CryptoPredictorLSTM(nn.Module):
    """Dual-head LSTM with feature gating and temporal attention."""

    NUM_CLASSES = 3  # UP, FLAT, DOWN

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # --- Feature gate (sigmoid, context-dependent) ---
        # Wider gate (F -> 4F -> 2F -> F) captures richer cross-feature
        # interactions than the previous shallow (F -> 2F -> F) design.
        self.feature_gate = nn.Sequential(
            nn.Linear(num_features, num_features * 4),
            nn.GELU(),
            nn.Linear(num_features * 4, num_features * 2),
            nn.GELU(),
            nn.Linear(num_features * 2, num_features),
            nn.Sigmoid(),
        )

        # --- Unidirectional LSTM (causal — no future information) ---
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Temporal attention (additive / Bahdanau-style) ---
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --- Residual connection: project input to hidden_dim for skip ---
        self.input_proj = nn.Linear(num_features, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # --- Classification head: UP / FLAT / DOWN ---
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.NUM_CLASSES),
        )

        # --- Regression head: predicted % price change ---
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # --- Temperature for probability calibration ---
        # Optimized post-training on validation data; default 1.0 = uncalibrated.
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        x : Tensor (batch, seq_len, num_features)
        return_attention : if True, also return attention weight tensors

        Returns
        -------
        cls_logits : (batch, 3)
        magnitude  : (batch, 1)
        [feat_weights] : (batch, seq_len, num_features) — if return_attention
        [temp_weights] : (batch, seq_len)               — if return_attention
        """
        # 1. Feature gating — learn which features matter in the current context
        feat_weights = self.feature_gate(x)      # (B, T, F) in [0, 1]
        x_gated = x * feat_weights               # element-wise suppression/amplification

        # 2. LSTM encoding (unidirectional → causal)
        lstm_out, _ = self.encoder(x_gated)      # (B, T, H)

        # 3. Temporal attention — attend to *all* timesteps, not just the last
        attn_scores = self.temporal_attn(lstm_out)         # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        context = (lstm_out * attn_weights).sum(dim=1)     # (B, H)

        # 3b. Residual connection — project input mean to hidden_dim and add
        residual = self.input_proj(x_gated.mean(dim=1))    # (B, H)
        context = self.layer_norm(self.dropout(context) + residual)

        # 4. Dual heads
        cls_logits = self.cls_head(context)
        magnitude = self.reg_head(context)

        if return_attention:
            return cls_logits, magnitude, feat_weights, attn_weights.squeeze(-1)
        return cls_logits, magnitude


class PredictionLoss(nn.Module):
    """
    Combined loss for the dual-head model.

    L = lambda_cls * CrossEntropy(cls_logits, direction_label)
      + lambda_reg * HuberLoss(magnitude_pred, magnitude_true)

    Supports optional class_weights to combat label imbalance.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_reg: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.reg_loss = nn.HuberLoss(delta=1.0)

    def forward(
        self,
        cls_logits: torch.Tensor,
        magnitude_pred: torch.Tensor,
        direction_labels: torch.Tensor,
        magnitude_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total_loss, cls_loss, reg_loss) for logging."""
        l_cls = self.cls_loss(cls_logits, direction_labels)
        l_reg = self.reg_loss(magnitude_pred.squeeze(-1), magnitude_true)
        total = self.lambda_cls * l_cls + self.lambda_reg * l_reg
        return total, l_cls, l_reg
