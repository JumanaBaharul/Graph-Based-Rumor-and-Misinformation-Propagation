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


class TemporalGATLayer(nn.Module):
    """Multi-head temporal-aware graph attention layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        temporal_feature_index: int,
        heads: int = 4,
        dropout: float = 0.2,
        temporal_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.temporal_idx = temporal_feature_index
        self.temporal_beta = temporal_beta

        self.lin = nn.Linear(in_features, out_features * heads)
        self.attn_src = nn.Parameter(torch.randn(heads, out_features))
        self.attn_dst = nn.Parameter(torch.randn(heads, out_features))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self.out_proj = nn.Linear(out_features * heads, out_features)

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: [B, N, F], adj: [B, N, N]
        h = self.lin(x)
        bsz, num_nodes, _ = h.shape
        h = h.view(bsz, num_nodes, self.heads, self.out_features)

        attn_src = (h * self.attn_src).sum(dim=-1)
        attn_dst = (h * self.attn_dst).sum(dim=-1)

        scores = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)

        time_feat = x[..., self.temporal_idx]
        time_diff = torch.abs(time_feat.unsqueeze(-1) - time_feat.unsqueeze(-2))
        temporal_bias = torch.exp(-self.temporal_beta * time_diff).unsqueeze(-1)
        scores = scores + temporal_bias

        adjacency_mask = (adj > 0).unsqueeze(-1)
        scores = scores.masked_fill(adjacency_mask == 0, float("-inf"))

        if mask is not None:
            node_mask = mask.unsqueeze(-1).unsqueeze(-1)
            scores = scores.masked_fill(node_mask == 0, float("-inf"))
            scores = scores.masked_fill(node_mask.transpose(1, 2) == 0, float("-inf"))

        attention = torch.softmax(scores, dim=2)
        attention = self.dropout(attention)

        out = torch.einsum("bijh,bjhf->bihf", attention, h)
        out = out.reshape(bsz, num_nodes, self.heads * self.out_features)
        out = self.out_proj(out)
        out = self.activation(out)
        out = self.dropout(out)

        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class RumorGAT(nn.Module):
    """Temporal-aware graph attention network for rumor classification."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        temporal_feature_index: int = -1,
        dropout: float = 0.3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.layer1 = TemporalGATLayer(
            in_features,
            hidden_dim,
            temporal_feature_index=temporal_feature_index,
            heads=heads,
            dropout=dropout,
        )
        self.layer2 = TemporalGATLayer(
            hidden_dim,
            hidden_dim,
            temporal_feature_index=temporal_feature_index,
            heads=heads,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, adj, mask)
        h = self.layer2(h, adj, mask)

        mask_expanded = mask.unsqueeze(-1)
        pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1e-6)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def predict(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, adj, mask)
        return torch.sigmoid(logits)


class HierarchicalRumorNet(nn.Module):
    """Temporal GCN with hierarchical pooling (mean + attention levels)."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        temporal_feature_index: int = -1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.layer1 = TemporalGCNLayer(
            in_features,
            hidden_dim,
            temporal_feature_index=temporal_feature_index,
            dropout=dropout,
        )
        self.layer2 = TemporalGCNLayer(
            hidden_dim,
            hidden_dim,
            temporal_feature_index=temporal_feature_index,
            dropout=dropout,
        )
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x, adj, mask)
        h2 = self.layer2(h1, adj, mask)

        mask_expanded = mask.unsqueeze(-1)
        mean_pool = (h2 * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1e-6)

        attn_scores = self.attn_pool(h1).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_pooled = torch.bmm(attn_weights.unsqueeze(1), h2).squeeze(1)

        combined = torch.cat([mean_pool, attn_pooled], dim=-1)
        combined = self.dropout(combined)
        logits = self.classifier(combined).squeeze(-1)
        return logits

    def predict(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, adj, mask)
        return torch.sigmoid(logits)


def build_rumor_model(
    architecture: str,
    *,
    in_features: int,
    hidden_dim: int = 64,
    temporal_feature_index: int = -1,
    dropout: float = 0.3,
    heads: int = 4,
) -> nn.Module:
    """Factory method for constructing rumor propagation models."""

    arch = architecture.lower()
    if arch == "temporal_gcn":
        return RumorGCN(
            in_features=in_features,
            hidden_dim=hidden_dim,
            temporal_feature_index=temporal_feature_index,
            dropout=dropout,
        )
    if arch == "temporal_gat":
        return RumorGAT(
            in_features=in_features,
            hidden_dim=hidden_dim,
            temporal_feature_index=temporal_feature_index,
            dropout=dropout,
            heads=heads,
        )
    if arch in {"hierarchical_pool", "hierarchical"}:
        return HierarchicalRumorNet(
            in_features=in_features,
            hidden_dim=hidden_dim,
            temporal_feature_index=temporal_feature_index,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown architecture '{architecture}'. Available options: temporal_gcn, temporal_gat, hierarchical_pool"
    )


__all__ = [
    "TemporalGCNLayer",
    "RumorGCN",
    "TemporalGATLayer",
    "RumorGAT",
    "HierarchicalRumorNet",
    "build_rumor_model",
]
