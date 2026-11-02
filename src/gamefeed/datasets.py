from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import FeatureSpace, ItemFeatureLookup
from .vocab import PAD


class RecallGroupDataset(Dataset):
    def __init__(self, groups: pd.DataFrame) -> None:
        self.rows = groups.to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class RecallCollator:
    def __init__(self, feature_space: FeatureSpace, items: pd.DataFrame, history_length: int = 20) -> None:
        self.features = feature_space
        self.lookup = ItemFeatureLookup(items, feature_space)
        self.history_length = history_length

    def _pad_history(self, values: list[Any]) -> list[Any | None]:
        result = list(values)[-self.history_length :]
        return result + [None] * (self.history_length - len(result))

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, list[Any]] = {
            "user_ids": [], "history_item_ids": [], "history_tag_ids": [], "history_durations": [],
            "history_mask": [], "history_len_scaled": [], "candidate_item_ids": [],
            "candidate_tag_ids": [], "candidate_durations": [], "candidate_mask": [],
        }
        max_candidates = max(1 + len(row["negative_ids"]) for row in rows)
        for row in rows:
            history = self._pad_history(list(row["history_ids"]))
            history_mask = [value is not None for value in history]
            history_item, history_tags, history_duration = self.lookup.encode_many(history)
            candidates = [row["positive_id"], *list(row["negative_ids"])]
            candidate_mask = [True] * len(candidates) + [False] * (max_candidates - len(candidates))
            candidates += [None] * (max_candidates - len(candidates))
            candidate_item, candidate_tags, candidate_duration = self.lookup.encode_many(candidates)
            batch["user_ids"].append(self.features.user_vocab.encode(row["user_id"]))
            batch["history_item_ids"].append(history_item)
            batch["history_tag_ids"].append(history_tags)
            batch["history_durations"].append(history_duration)
            batch["history_mask"].append(history_mask)
            batch["history_len_scaled"].append(self.features.scale_history_length(row["history_len"]))
            batch["candidate_item_ids"].append(candidate_item)
            batch["candidate_tag_ids"].append(candidate_tags)
            batch["candidate_durations"].append(candidate_duration)
            batch["candidate_mask"].append(candidate_mask)
        return {
            key: torch.tensor(value, dtype=torch.bool if "mask" in key else torch.float32 if "duration" in key or "scaled" in key else torch.long)
            for key, value in batch.items()
        }


class ExposureDataset(Dataset):
    def __init__(self, events: pd.DataFrame) -> None:
        columns = ["user_id", "video_id", "tab", "history_ids", "history_len", "long_view", "is_like"]
        self.rows = events[columns].to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class ExposureCollator:
    def __init__(self, feature_space: FeatureSpace, items: pd.DataFrame, history_length: int = 20) -> None:
        self.features = feature_space
        self.lookup = ItemFeatureLookup(items, feature_space)
        self.history_length = history_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, list[Any]] = {
            "user_ids": [], "item_ids": [], "tag_ids": [], "durations": [], "tabs": [],
            "history_item_ids": [], "history_mask": [], "history_len_scaled": [],
            "long_view": [], "is_like": [],
        }
        for row in rows:
            item_id, tags, duration = self.lookup.encode(row["video_id"])
            history = list(row["history_ids"])[-self.history_length :]
            history_mask = [True] * len(history) + [False] * (self.history_length - len(history))
            history += [None] * (self.history_length - len(history))
            history_items, _, _ = self.lookup.encode_many(history)
            batch["user_ids"].append(self.features.user_vocab.encode(row["user_id"]))
            batch["item_ids"].append(item_id)
            batch["tag_ids"].append(tags)
            batch["durations"].append(duration)
            batch["tabs"].append(max(0, min(int(row["tab"]) + 1, 15)))
            batch["history_item_ids"].append(history_items)
            batch["history_mask"].append(history_mask)
            batch["history_len_scaled"].append(self.features.scale_history_length(row["history_len"]))
            batch["long_view"].append(int(row["long_view"]))
            batch["is_like"].append(int(row["is_like"]))
        long_keys = {"user_ids", "item_ids", "tag_ids", "tabs", "history_item_ids"}
        float_keys = {"durations", "history_len_scaled", "long_view", "is_like"}
        return {
            key: torch.tensor(value, dtype=torch.long if key in long_keys else torch.float32 if key in float_keys else torch.bool)
            for key, value in batch.items()
        }

