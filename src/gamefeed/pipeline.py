from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .baselines import (
    evaluate_retrieval,
    itemcf_recommendations,
    popularity_recommendations,
    reciprocal_rank_fusion,
)
from .catalog import add_available_from
from .config import Paths, ensure_output_dirs
from .data import assign_split, inspect_raw, load_canonical_data, save_prepared
from .history import attach_state, build_daily_state
from .quality import quality_summary, write_quality_report
from .samples import build_recall_groups
from .utils import write_json


def run_inspection(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    result = inspect_raw(config)
    path: Path = config["resolved_paths"].report_dir / "raw_schema.json"
    write_json(path, result)
    return result


def run_prepare(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    events, items = load_canonical_data(config)
    events = assign_split(events, config["data"])
    items = add_available_from(items, events)
    states, queries, item_day_stats = build_daily_state(events, int(config["data"]["history_length"]))
    events_with_state = attach_state(events, states)
    recall_groups = build_recall_groups(
        events_with_state,
        max_negatives=int(config["data"]["max_negative_per_positive"]),
        seed=int(config["seed"]),
    )
    save_prepared(config, events_with_state, items, states, queries, item_day_stats)
    recall_groups.to_parquet(config["resolved_paths"].processed_dir / "recall_groups.parquet", index=False)
    summary = quality_summary(events_with_state, items, states)
    summary["recall_training_groups"] = int(len(recall_groups))
    write_json(config["resolved_paths"].report_dir / "quality_report.json", summary)
    write_quality_report(config["resolved_paths"].report_dir / "quality_report.md", summary)
    return summary


def run_popularity(config: dict[str, Any], split: str = "dev") -> dict[str, float]:
    paths: Paths = config["resolved_paths"]
    events = pd.read_parquet(paths.processed_dir / "events.parquet")
    items = pd.read_parquet(paths.processed_dir / "items.parquet")
    queries = pd.read_parquet(paths.processed_dir / "recall_queries.parquet")
    results = popularity_recommendations(events, queries, items, k=max(config["retrieval"]["top_k"]))
    selected = results[results["split"].eq(split)].copy()
    metrics = evaluate_retrieval(selected, items["video_id"].drop_duplicates().tolist(), tuple(config["retrieval"]["top_k"]))
    output = paths.report_dir / f"popularity_{split}.json"
    write_json(output, metrics)
    selected.to_parquet(paths.report_dir / f"popularity_{split}_recommendations.parquet", index=False)
    return metrics


def run_itemcf(config: dict[str, Any], split: str = "dev") -> dict[str, float]:
    paths: Paths = config["resolved_paths"]
    events = pd.read_parquet(paths.processed_dir / "events.parquet")
    items = pd.read_parquet(paths.processed_dir / "items.parquet")
    queries = pd.read_parquet(paths.processed_dir / "recall_queries.parquet")
    train_events = events[events["split"].eq("train")].copy()
    selected_queries = queries[queries["split"].eq(split)].copy()
    results = itemcf_recommendations(
        train_events,
        selected_queries,
        items,
        k=max(config["retrieval"]["top_k"]),
    )
    metrics = evaluate_retrieval(
        results,
        items["video_id"].drop_duplicates().tolist(),
        tuple(config["retrieval"]["top_k"]),
    )
    write_json(paths.report_dir / f"itemcf_{split}.json", metrics)
    results.to_parquet(paths.report_dir / f"itemcf_{split}_recommendations.parquet", index=False)
    return metrics


def run_fusion(config: dict[str, Any], split: str = "dev") -> dict[str, float]:
    paths: Paths = config["resolved_paths"]
    popularity_path = paths.report_dir / f"popularity_{split}_recommendations.parquet"
    itemcf_path = paths.report_dir / f"itemcf_{split}_recommendations.parquet"
    if not popularity_path.exists() or not itemcf_path.exists():
        raise FileNotFoundError("请先运行 popularity 和 itemcf，再运行 fusion")
    popularity = pd.read_parquet(popularity_path)
    itemcf = pd.read_parquet(itemcf_path)
    results = reciprocal_rank_fusion(
        {"itemcf": itemcf, "popularity": popularity},
        {"itemcf": 0.8, "popularity": 0.2},
        k=max(config["retrieval"]["top_k"]),
    )
    items = pd.read_parquet(paths.processed_dir / "items.parquet")
    metrics = evaluate_retrieval(
        results,
        items["video_id"].drop_duplicates().tolist(),
        tuple(config["retrieval"]["top_k"]),
    )
    write_json(paths.report_dir / f"fusion_{split}.json", metrics)
    results.to_parquet(paths.report_dir / f"fusion_{split}_recommendations.parquet", index=False)
    return metrics

