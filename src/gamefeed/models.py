from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ItemEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_tags: int,
        item_id_dim: int = 32,
        tag_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_id_dim, padding_idx=0)
        self.tag_embedding = nn.Embedding(num_tags, tag_dim, padding_idx=0)
        self.network = nn.Sequential(
            nn.Linear(item_id_dim + tag_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, item_ids: torch.Tensor, tag_ids: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        tag_vectors = self.tag_embedding(tag_ids)
        tag_mask = tag_ids.ne(0).unsqueeze(-1).to(tag_vectors.dtype)
        tag_counts = tag_mask.sum(dim=-2).clamp_min(1.0)
        pooled_tags = (tag_vectors * tag_mask).sum(dim=-2) / tag_counts
        features = torch.cat(
            [self.item_embedding(item_ids), pooled_tags, durations.unsqueeze(-1)], dim=-1
        )
        return F.normalize(self.network(features), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_tags: int,
        user_id_dim: int = 16,
        output_dim: int = 64,
        **item_encoder_kwargs: int | float,
    ) -> None:
        super().__init__()
        self.item_encoder = ItemEncoder(num_items, num_tags, output_dim=output_dim, **item_encoder_kwargs)
        self.user_embedding = nn.Embedding(num_users, user_id_dim, padding_idx=0)
        self.empty_history = nn.Parameter(torch.zeros(output_dim))
        hidden_dim = int(item_encoder_kwargs.get("hidden_dim", 128))
        dropout = float(item_encoder_kwargs.get("dropout", 0.1))
        self.user_network = nn.Sequential(
            nn.Linear(output_dim + user_id_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_user(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_tag_ids: torch.Tensor,
        history_durations: torch.Tensor,
        history_mask: torch.Tensor,
        history_len_scaled: torch.Tensor,
    ) -> torch.Tensor:
        history_vectors = self.item_encoder(history_item_ids, history_tag_ids, history_durations)
        mask = history_mask.unsqueeze(-1).to(history_vectors.dtype)
        counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = (history_vectors * mask).sum(dim=1) / counts
        empty = history_mask.sum(dim=1).eq(0)
        pooled = torch.where(empty.unsqueeze(-1), self.empty_history.unsqueeze(0), pooled)
        features = torch.cat(
            [pooled, self.user_embedding(user_ids), history_len_scaled.unsqueeze(-1)], dim=-1
        )
        return F.normalize(self.user_network(features), dim=-1)

    def score(self, user_vectors: torch.Tensor, item_vectors: torch.Tensor) -> torch.Tensor:
        return (user_vectors * item_vectors).sum(dim=-1)


class SharedBottomRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, shared_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
            nn.ReLU(),
        )
        self.long_view_head = nn.Linear(shared_dim, 1)
        self.like_head = nn.Linear(shared_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(features)
        return {
            "long_view": self.long_view_head(shared).squeeze(-1),
            "is_like": self.like_head(shared).squeeze(-1),
        }


class ExposureRanker(nn.Module):
    """Independent ranking embeddings plus a shared-bottom two-task network."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_tags: int,
        user_dim: int = 16,
        item_dim: int = 32,
        tag_dim: int = 16,
        tab_dim: int = 8,
        hidden_dim: int = 128,
        shared_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, item_dim, padding_idx=0)
        self.tag_embedding = nn.Embedding(num_tags, tag_dim, padding_idx=0)
        self.tab_embedding = nn.Embedding(16, tab_dim, padding_idx=0)
        input_dim = user_dim + item_dim * 3 + tag_dim + tab_dim + 2
        self.ranker = SharedBottomRanker(input_dim, hidden_dim, shared_dim, dropout)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        tag_ids: torch.Tensor,
        durations: torch.Tensor,
        tabs: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
        history_len_scaled: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        item = self.item_embedding(item_ids)
        history_values = self.item_embedding(history_item_ids)
        history_weight = history_mask.unsqueeze(-1).to(history_values.dtype)
        history = (history_values * history_weight).sum(1) / history_weight.sum(1).clamp_min(1.0)
        tags = self.tag_embedding(tag_ids)
        tag_weight = tag_ids.ne(0).unsqueeze(-1).to(tags.dtype)
        tags = (tags * tag_weight).sum(1) / tag_weight.sum(1).clamp_min(1.0)
        features = torch.cat(
            [self.user_embedding(user_ids), item, history, item * history, tags, self.tab_embedding(tabs),
             durations.unsqueeze(-1), history_len_scaled.unsqueeze(-1)], dim=-1,
        )
        return self.ranker(features)


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    long_view: torch.Tensor,
    is_like: torch.Tensor,
    like_weight: float = 1.0,
) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss()
    return criterion(outputs["long_view"], long_view.float()) + like_weight * criterion(
        outputs["is_like"], is_like.float()
    )


def grouped_recall_loss(logits: torch.Tensor, valid_mask: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Cross entropy where column zero is the positive item."""
    scaled = logits / temperature
    scaled = scaled.masked_fill(~valid_mask, torch.finfo(scaled.dtype).min)
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(scaled, targets)

