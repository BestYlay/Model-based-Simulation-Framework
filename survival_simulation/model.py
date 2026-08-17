"""Encoder-only Transformer for longitudinal survival risk modeling."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for event order."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, max_length, d_model)
        encoding[0, :, 0::2] = torch.sin(position * frequencies)
        encoding[0, :, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("encoding", encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence length exceeds positional encoding capacity")
        return self.dropout(values + self.encoding[:, : values.shape[1]])


class SurvivalTransformer(nn.Module):
    """Encode mixed clinical event sequences into one log-risk score."""

    def __init__(
        self,
        vocab_sizes: Mapping[str, int],
        embedding_dims: Mapping[str, int],
        num_numerical_features: int,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if set(vocab_sizes) != set(embedding_dims):
            raise ValueError("vocab_sizes and embedding_dims must have identical keys")
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be 'mean' or 'attention'")

        self.categorical_keys = sorted(vocab_sizes)
        self.embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(vocab_sizes[key], embedding_dims[key], padding_idx=0)
                for key in self.categorical_keys
            }
        )
        input_dim = num_numerical_features + sum(embedding_dims.values())
        if input_dim <= 0:
            raise ValueError("At least one numerical or categorical feature is required")

        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model, dropout)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.pooling = pooling
        self.pooling_score = nn.Linear(d_model, 1) if pooling == "attention" else None
        self.risk_head = nn.Linear(d_model, 1)

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: Mapping[str, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        return_pooling_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        categorical = categorical or {}
        missing = set(self.categorical_keys) - set(categorical)
        if missing:
            raise ValueError(f"Missing categorical inputs: {sorted(missing)}")
        if padding_mask is None:
            padding_mask = torch.zeros(
                numerical.shape[:2], dtype=torch.bool, device=numerical.device
            )
        if torch.any(padding_mask.all(dim=1)):
            raise ValueError("Every sequence must contain at least one real event")

        embedded = [self.embeddings[key](categorical[key]) for key in self.categorical_keys]
        event_features = torch.cat([numerical, *embedded], dim=-1)
        hidden = self.input_projection(event_features) * math.sqrt(
            self.input_projection.out_features
        )
        hidden = self.position(hidden)
        for layer in self.encoder_layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)

        valid = ~padding_mask
        if self.pooling == "attention":
            scores = self.pooling_score(hidden).squeeze(-1)
            scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        else:
            weights = valid / valid.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)

        risk = self.risk_head(pooled).squeeze(-1)
        if return_pooling_weights:
            return risk, weights
        return risk


def cox_ph_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Breslow tie handling."""

    risk_scores = risk_scores.reshape(-1)
    times = times.reshape(-1)
    observed = events.reshape(-1).bool()
    if not torch.any(observed):
        return risk_scores.sum() * 0.0

    partial_log_likelihood = risk_scores.new_zeros(())
    for event_time in torch.unique(times[observed]):
        event_at_time = observed & (times == event_time)
        risk_set = times >= event_time
        event_count = event_at_time.sum()
        partial_log_likelihood = partial_log_likelihood + risk_scores[event_at_time].sum()
        partial_log_likelihood = partial_log_likelihood - event_count * torch.logsumexp(
            risk_scores[risk_set], dim=0
        )
    return -partial_log_likelihood / observed.sum()
