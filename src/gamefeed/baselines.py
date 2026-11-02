from __future__ import annotations

from collections import defaultdict
from math import log, sqrt
from typing import Any

import pandas as pd

from .catalog import available_items
from .metrics import catalog_coverage, hit_rate_at_k, recall_at_k


def popularity_recommendations(
    events: pd.DataFrame,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    k: int = 100,
) -> pd.DataFrame:
    counts: dict[Any, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    events_by_date = {date: frame for date, frame in events.groupby("date", sort=True)}
    queries_by_date = {date: frame for date, frame in queries.groupby("date", sort=True)}
    dates = sorted(set(events_by_date).union(queries_by_date))
    for date in dates:
        catalog = available_items(items, pd.Timestamp(date))["video_id"].drop_duplicates().tolist()
        ranked = sorted(catalog, key=lambda item: (-counts[item], str(item)))[:k]
        for query in queries_by_date.get(date, pd.DataFrame()).itertuples(index=False):
            rows.append(
                {
                    "user_id": query.user_id,
                    "date": date,
                    "split": query.split,
                    "positive_ids": list(query.positive_ids),
                    "recommended_ids": ranked,
                }
            )
        day = events_by_date.get(date)
        if day is not None:
            for item, count in day.loc[day["long_view"].eq(1), "video_id"].value_counts().items():
                counts[item] += int(count)
    return pd.DataFrame(rows)


def fit_itemcf(
    train_events: pd.DataFrame,
    neighbor_size: int = 200,
) -> dict[Any, list[tuple[Any, float]]]:
    """Fit item-to-item similarities from positive training interactions only.

    A user's positive item set contributes co-occurrence evidence.  The
    1/log(1+|I_u|) factor prevents very active users from dominating every
    item pair, and cosine normalization reduces popularity bias.
    """
    positives = train_events.loc[
        train_events["long_view"].eq(1),
        ["user_id", "video_id"],
    ].drop_duplicates()
    user_items = positives.groupby("user_id")["video_id"].agg(list)
    item_users = positives.groupby("video_id")["user_id"].nunique().to_dict()
    cooccurrence: dict[Any, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
    for items in user_items:
        unique_items = list(dict.fromkeys(items))
        if len(unique_items) < 2:
            continue
        weight = 1.0 / log(1.0 + len(unique_items))
        for left in unique_items:
            for right in unique_items:
                if left != right:
                    cooccurrence[left][right] += weight

    neighbors: dict[Any, list[tuple[Any, float]]] = {}
    for left, related in cooccurrence.items():
        scored = [
            (right, value / sqrt(item_users[left] * item_users[right]))
            for right, value in related.items()
        ]
        neighbors[left] = sorted(scored, key=lambda pair: (-pair[1], str(pair[0])))[:neighbor_size]
    return neighbors


def itemcf_recommendations(
    train_events: pd.DataFrame,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    k: int = 100,
    neighbor_size: int = 200,
) -> pd.DataFrame:
    """Recommend from a user's request-time history with a popularity fallback."""
    neighbors = fit_itemcf(train_events, neighbor_size=neighbor_size)
    popular = (
        train_events.loc[train_events["long_view"].eq(1), "video_id"]
        .value_counts()
        .index.tolist()
    )
    rows: list[dict[str, Any]] = []
    for query in queries.itertuples(index=False):
        available = set(available_items(items, pd.Timestamp(query.date))["video_id"])
        history = list(query.history_ids)
        scores: dict[Any, float] = defaultdict(float)
        for position, source in enumerate(reversed(history)):
            recency = 1.0 / (1.0 + 0.1 * position)
            for candidate, similarity in neighbors.get(source, []):
                if candidate in available:
                    scores[candidate] += recency * similarity
        ranked = [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))]
        if len(ranked) < k:
            ranked_set = set(ranked)
            ranked.extend(
                item for item in popular
                if item in available and item not in ranked_set
            )
        rows.append(
            {
                "user_id": query.user_id,
                "date": query.date,
                "split": query.split,
                "positive_ids": list(query.positive_ids),
                "recommended_ids": ranked[:k],
            }
        )
    return pd.DataFrame(rows)


def reciprocal_rank_fusion(
    frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
    k: int = 100,
    rank_constant: float = 60.0,
) -> pd.DataFrame:
    """Merge recall routes with weighted reciprocal rank fusion.

    The score of an item is the sum of weight/(rank_constant + rank) from
    every route that retrieved it.  This avoids comparing incompatible raw
    scores from ItemCF, popularity, and neural retrieval.
    """
    merged: pd.DataFrame | None = None
    for route, frame in frames.items():
        columns = ["user_id", "date", "recommended_ids"]
        if merged is None:
            columns.insert(2, "positive_ids")
        selected = frame[columns].copy()
        selected = selected.rename(columns={"recommended_ids": f"recommended_ids_{route}"})
        join_keys = ["user_id", "date"]
        merged = selected if merged is None else merged.merge(selected, on=join_keys, how="inner", validate="one_to_one")
    if merged is None:
        return pd.DataFrame(columns=["user_id", "date", "positive_ids", "recommended_ids", "recall_channels"])

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        scores: dict[Any, float] = defaultdict(float)
        channels: dict[Any, list[str]] = defaultdict(list)
        for route in frames:
            values = getattr(row, f"recommended_ids_{route}")
            for rank, item in enumerate(values, start=1):
                scores[item] += float(weights.get(route, 1.0)) / (rank_constant + rank)
                channels[item].append(route)
        ranked = [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))[:k]]
        rows.append(
            {
                "user_id": row.user_id,
                "date": row.date,
                "positive_ids": list(row.positive_ids),
                "recommended_ids": ranked,
                "recall_channels": [channels[item] for item in ranked],
            }
        )
    return pd.DataFrame(rows)


def evaluate_retrieval(results: pd.DataFrame, catalog: list[Any], ks: tuple[int, ...] = (50, 100)) -> dict[str, float]:
    metrics: dict[str, float] = {"queries": float(len(results))}
    for k in ks:
        metrics[f"recall@{k}"] = float(
            results.apply(lambda row: recall_at_k(row["recommended_ids"], row["positive_ids"], k), axis=1).mean()
        )
        metrics[f"hitrate@{k}"] = float(
            results.apply(lambda row: hit_rate_at_k(row["recommended_ids"], row["positive_ids"], k), axis=1).mean()
        )
    metrics[f"coverage@{max(ks)}"] = catalog_coverage(results["recommended_ids"], catalog)
    return metrics

