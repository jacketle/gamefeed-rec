from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import json
import math
import numpy as np
import pandas as pd

from .vocab import PAD, UNK, Vocabulary


def parse_tags(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


@dataclass
class FeatureSpace:
    user_vocab: Vocabulary
    item_vocab: Vocabulary
    tag_vocab: Vocabulary
    duration_mean: float
    duration_std: float
    history_mean: float
    history_std: float
    max_tags: int = 8

    @classmethod
    def fit(cls, train_events: pd.DataFrame, items: pd.DataFrame, max_tags: int = 8) -> "FeatureSpace":
        train_item_ids = set(train_events["video_id"].astype(str))
        train_items = items[items["video_id"].astype(str).isin(train_item_ids)]
        tags = [tag for value in train_items["tag"] for tag in parse_tags(value)]
        durations = np.log1p(pd.to_numeric(train_items["video_duration"], errors="coerce").fillna(0).to_numpy(float))
        history_values = np.log1p(train_events["history_len"].to_numpy(float))
        duration_mean = float(durations.mean()) if len(durations) else 0.0
        duration_std = float(durations.std()) if len(durations) and durations.std() > 0 else 1.0
        history_mean = float(history_values.mean()) if len(history_values) else 0.0
        history_std = float(history_values.std()) if len(history_values) and history_values.std() > 0 else 1.0
        return cls(
            user_vocab=Vocabulary.fit(train_events["user_id"]),
            item_vocab=Vocabulary.fit(train_events["video_id"]),
            tag_vocab=Vocabulary.fit(tags),
            duration_mean=duration_mean,
            duration_std=duration_std,
            history_mean=history_mean,
            history_std=history_std,
            max_tags=max_tags,
        )

    def encode_tags(self, value: Any) -> list[int]:
        values = self.tag_vocab.encode_many(parse_tags(value)[: self.max_tags])
        return values + [PAD] * (self.max_tags - len(values))

    def scale_duration(self, value: Any) -> float:
        numeric = float(value) if pd.notna(value) else 0.0
        return (math.log1p(max(numeric, 0.0)) - self.duration_mean) / self.duration_std

    def scale_history_length(self, value: int) -> float:
        return (math.log1p(max(int(value), 0)) - self.history_mean) / self.history_std

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_vocab": self.user_vocab.to_dict(),
            "item_vocab": self.item_vocab.to_dict(),
            "tag_vocab": self.tag_vocab.to_dict(),
            "duration_mean": self.duration_mean,
            "duration_std": self.duration_std,
            "history_mean": self.history_mean,
            "history_std": self.history_std,
            "max_tags": self.max_tags,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureSpace":
        return cls(
            user_vocab=Vocabulary(value["user_vocab"]),
            item_vocab=Vocabulary(value["item_vocab"]),
            tag_vocab=Vocabulary(value["tag_vocab"]),
            duration_mean=float(value["duration_mean"]),
            duration_std=float(value["duration_std"]),
            history_mean=float(value["history_mean"]),
            history_std=float(value["history_std"]),
            max_tags=int(value["max_tags"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureSpace":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class ItemFeatureLookup:
    def __init__(self, items: pd.DataFrame, feature_space: FeatureSpace) -> None:
        self.feature_space = feature_space
        self.rows = {str(row.video_id): row for row in items.itertuples(index=False)}

    def encode(self, item_id: Any) -> tuple[int, list[int], float]:
        if item_id is None:
            return PAD, [PAD] * self.feature_space.max_tags, 0.0
        row = self.rows.get(str(item_id))
        if row is None:
            return UNK, [PAD] * self.feature_space.max_tags, 0.0
        return (
            self.feature_space.item_vocab.encode(item_id),
            self.feature_space.encode_tags(getattr(row, "tag", None)),
            self.feature_space.scale_duration(getattr(row, "video_duration", 0.0)),
        )

    def encode_many(self, item_ids: Iterable[Any]) -> tuple[list[int], list[list[int]], list[float]]:
        encoded = [self.encode(item_id) for item_id in item_ids]
        if not encoded:
            return [], [], []
        item_indices, tags, durations = zip(*encoded)
        return list(item_indices), list(tags), list(durations)

