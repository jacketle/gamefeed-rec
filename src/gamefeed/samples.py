from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_recall_groups(events_with_state: pd.DataFrame, max_negatives: int = 4, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (user_id, date), day in events_with_state.groupby(["user_id", "date"], sort=False):
        positives = day.loc[day["long_view"].eq(1), "video_id"].drop_duplicates().tolist()
        positive_set = set(positives)
        negatives = day.loc[day["long_view"].eq(0) & ~day["video_id"].isin(positive_set), "video_id"].drop_duplicates().tolist()
        if not negatives:
            continue
        for positive in positives:
            count = min(max_negatives, len(negatives))
            sampled = rng.choice(negatives, size=count, replace=False).tolist()
            first = day.iloc[0]
            rows.append(
                {
                    "user_id": user_id,
                    "date": date,
                    "split": first["split"],
                    "history_ids": list(first["history_ids"]),
                    "history_len": int(first["history_len"]),
                    "positive_id": positive,
                    "negative_ids": sampled,
                }
            )
    return pd.DataFrame(rows)

