"""
LSTM-based model for crypto price direction + magnitude prediction.

Architecture
------------
Input  : (batch, seq_len, num_features) -- a sliding window of normalized indicator vectors
Encoder: 2-layer bidirectional LSTM (hidden_dim per direction)
Heads  :
    classification -> Softmax over [UP, FLAT, DOWN]   (3 classes)
    regression     -> scalar predicted % price change  (magnitude)

The two heads share the same encoder but have independent linear projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CryptoPredictorLSTM(nn.Module):
    """Dual-head bidirectional LSTM for crypto price prediction."""

    NUM_CLASSES = 3  # UP, FLAT, DOWN

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        enc_out_dim = hidden_dim * 2  # bidirectional

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(enc_out_dim)

        # Classification head: UP / FLAT / DOWN
        self.cls_head = nn.Sequential(
            nn.Linear(enc_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.NUM_CLASSES),
        )

        # Regression head: predicted % change
        self.reg_head = nn.Sequential(
            nn.Linear(enc_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, num_features)

        Returns
        -------
        cls_logits : (batch, 3) -- raw logits for [UP, FLAT, DOWN]
        magnitude  : (batch, 1) -- predicted % price change
        """
        # Encode
        lstm_out, _ = self.encoder(x)  # (batch, seq_len, hidden*2)
        last_hidden = lstm_out[:, -1, :]  # take the final time step
        last_hidden = self.layer_norm(self.dropout(last_hidden))

        cls_logits = self.cls_head(last_hidden)
        magnitude = self.reg_head(last_hidden)

        return cls_logits, magnitude


class PredictionLoss(nn.Module):
    """
    Combined loss for the dual-head model.

    L = lambda_cls * CrossEntropy(cls_logits, direction_label)
      + lambda_reg * HuberLoss(magnitude_pred, magnitude_true)

    direction_label: 0=UP, 1=FLAT, 2=DOWN
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
        """
        Returns (total_loss, cls_loss, reg_loss) for logging.
        """
        l_cls = self.cls_loss(cls_logits, direction_labels)
        l_reg = self.reg_loss(magnitude_pred.squeeze(-1), magnitude_true)
        total = self.lambda_cls * l_cls + self.lambda_reg * l_reg
        return total, l_cls, l_reg
