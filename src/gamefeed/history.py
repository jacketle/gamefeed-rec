from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import pandas as pd


def build_daily_state(
    events: pd.DataFrame,
    history_length: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create state available at the start of each day.

    All rows from day d are evaluated against state accumulated through d-1.
    """
    required = {"user_id", "video_id", "date", "long_view", "split"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events 缺少字段: {sorted(missing)}")

    histories: dict[Any, deque[Any]] = defaultdict(lambda: deque(maxlen=history_length))
    item_impressions: dict[Any, int] = defaultdict(int)
    item_long_views: dict[Any, int] = defaultdict(int)
    state_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []

    ordered = events.sort_values(["date", "time_ms", "event_id"], kind="stable")
    for date, day in ordered.groupby("date", sort=True):
        for user_id, user_day in day.groupby("user_id", sort=False):
            history = list(histories[user_id])
            split = str(user_day["split"].iloc[0])
            positives = user_day.loc[user_day["long_view"].eq(1), "video_id"].drop_duplicates().tolist()
            state_rows.append(
                {
                    "user_id": user_id,
                    "date": date,
                    "split": split,
                    "history_ids": history,
                    "history_len": len(history),
                }
            )
            if positives:
                query_rows.append(
                    {
                        "user_id": user_id,
                        "date": date,
                        "split": split,
                        "history_ids": history,
                        "history_len": len(history),
                        "positive_ids": positives,
                    }
                )

        for video_id in day["video_id"].drop_duplicates():
            stat_rows.append(
                {
                    "video_id": video_id,
                    "date": date,
                    "impression_count": item_impressions[video_id],
                    "long_view_count": item_long_views[video_id],
                }
            )

        for row in day.itertuples(index=False):
            item_impressions[row.video_id] += 1
            if int(row.long_view) == 1:
                item_long_views[row.video_id] += 1
                histories[row.user_id].append(row.video_id)

    return pd.DataFrame(state_rows), pd.DataFrame(query_rows), pd.DataFrame(stat_rows)


def attach_state(events: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    before = len(events)
    merged = events.merge(states, on=["user_id", "date", "split"], how="left", validate="many_to_one")
    if len(merged) != before:
        raise AssertionError("连接用户每天的历史状态后行数发生变化")
    return merged

