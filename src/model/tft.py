"""
Temporal Fusion Transformer (TFT) for crypto price prediction.

Architecture:
  1. Variable Selection Network — learns which features matter per timestep
  2. LSTM Encoder — captures temporal patterns
  3. Multi-Head Attention — attends across time dimension
  4. Gated Residual Network (GRN) — non-linear processing with skip connections
  5. Dual heads — classification (UP/FLAT/DOWN) + regression (magnitude)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network with optional context input."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1,
                 d_context: int = 0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        if d_context > 0:
            self.context_proj = nn.Linear(d_context, d_hidden, bias=False)
        else:
            self.context_proj = None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = x
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)

        gate = torch.sigmoid(self.gate(h))
        out = gate * h + (1 - gate) * residual
        return self.layer_norm(out)


class VariableSelectionNetwork(nn.Module):
    """Learns which features to attend to at each timestep."""

    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        self.feature_transforms = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.grn = GatedResidualNetwork(
            d_model * num_features, d_model * num_features, dropout
        )
        self.softmax_layer = nn.Linear(d_model * num_features, num_features)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, dropout) for _ in range(num_features)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, num_features)

        Returns
        -------
        selected : (batch, seq_len, d_model)
        weights : (batch, seq_len, num_features)
        """
        B, T, F = x.shape

        transformed = []
        for i in range(self.num_features):
            feat = x[:, :, i:i+1]
            transformed.append(self.feature_transforms[i](feat))
        # (B, T, F, d_model)
        transformed = torch.stack(transformed, dim=2)

        flat = transformed.reshape(B, T, -1)
        grn_out = self.grn(flat)
        weights = torch.softmax(self.softmax_layer(grn_out), dim=-1)

        processed = []
        for i in range(self.num_features):
            processed.append(self.feature_grns[i](transformed[:, :, i, :]))
        processed = torch.stack(processed, dim=2)

        selected = (processed * weights.unsqueeze(-1)).sum(dim=2)
        return selected, weights


class TemporalFusionTransformer(nn.Module):
    """
    TFT model for crypto price prediction.

    Combines variable selection, LSTM encoding, and multi-head attention
    for interpretable and accurate temporal modeling.
    """

    NUM_CLASSES = 3

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        self.vsn = VariableSelectionNetwork(num_features, d_model, dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.output_grn = GatedResidualNetwork(d_model, d_model * 2, dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.NUM_CLASSES),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        x : (batch, seq_len, num_features)

        Returns
        -------
        cls_logits : (batch, 3)
        magnitude : (batch, 1)
        [var_weights] : (batch, seq_len, num_features) if return_attention
        [attn_weights] : (batch, num_heads, seq_len, seq_len) if return_attention
        """
        selected, var_weights = self.vsn(x)

        lstm_out, _ = self.lstm(selected)

        T = lstm_out.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, attn_weights_tensor = self.attention(
            lstm_out, lstm_out, lstm_out,
            attn_mask=causal_mask,
            need_weights=return_attention,
        )

        gate = self.attn_gate(attn_out)
        gated = gate * attn_out + (1 - gate) * lstm_out
        normed = self.attn_norm(gated)

        context = self.output_grn(normed[:, -1, :])

        cls_logits = self.cls_head(context)
        magnitude = self.reg_head(context)

        if return_attention:
            return cls_logits, magnitude, var_weights, attn_weights_tensor
        return cls_logits, magnitude
