from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TemporalGCNLayer(nn.Module):
    """Graph convolution layer with temporal decay re-weighting."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        temporal_feature_index: int,
        alpha: float = 0.5,
        beta: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.temporal_idx = temporal_feature_index
        self.alpha = alpha
        self.beta = beta
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        support = x @ self.weight

        time_feat = x[..., self.temporal_idx : self.temporal_idx + 1]
        time_diffs = torch.abs(time_feat - time_feat.transpose(-1, -2))
        temporal_weights = torch.exp(-self.beta * time_diffs)
        reweighted_adj = adj * (1.0 + self.alpha * temporal_weights)

        out = reweighted_adj @ support + self.bias
        out = self.activation(out)
        out = self.dropout(out)

        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class RumorGCN(nn.Module):
    """Temporal-aware GCN for rumor graph classification."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        temporal_feature_index: int = -1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.layer1 = TemporalGCNLayer(
            in_features, hidden_dim, temporal_feature_index=temporal_feature_index, dropout=dropout
        )
        self.layer2 = TemporalGCNLayer(
            hidden_dim, hidden_dim, temporal_feature_index=temporal_feature_index, dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        h = self.layer1(x, adj, mask)
        h = self.layer2(h, adj, mask)
        mask_expanded = mask.unsqueeze(-1)
        pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1e-6)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def predict(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, adj, mask)
        return torch.sigmoid(logits)
