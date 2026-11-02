from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import torch

from .config import Paths
from .datasets import ExposureCollator
from .features import FeatureSpace
from .models import ExposureRanker
from .utils import write_json


def caption_key(row: dict[str, Any]) -> str:
    """Compare actual captions; missing text must not merge unrelated videos."""
    value = row.get('dedup_text', row.get('caption', ''))
    if value is None or pd.isna(value) or str(value).strip() == '该内容暂无文案':
        return ''
    return ' '.join(str(value).split())


def author_key(row: dict[str, Any]) -> tuple[str, str]:
    value = row.get('author_id')
    if value is None or pd.isna(value) or str(value).strip().upper() in ('', 'MISSING', 'UNKNOWN', 'NAN', 'NONE', '-1', '-1.0'):
        return ('video', str(row['video_id']))
    return ('author', str(value))


def author_capped_rerank(rows: list[dict[str, Any]], top_k: int, max_per_author: int = 2,
                        deduplicate_caption: bool = True) -> list[dict[str, Any]]:
    counts: dict[Any, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    captions, videos = set(), set()
    if top_k <= 0:
        return selected
    for row in sorted(rows, key=lambda value: (-float(value["score"]), str(value["video_id"]))):
        author = author_key(row)
        caption = caption_key(row)
        if row['video_id'] in videos or (deduplicate_caption and caption and caption in captions):
            continue
        if counts[author] >= max_per_author:
            continue
        counts[author] += 1
        selected.append(row)
        videos.add(row['video_id'])
        if caption:
            captions.add(caption)
        if len(selected) == top_k:
            break
    return selected


def _load_ranker(config: dict[str, Any], features: FeatureSpace, model_path: Any) -> ExposureRanker:
    model = ExposureRanker(
        len(features.user_vocab), len(features.item_vocab), len(features.tag_vocab),
        user_dim=int(config["model"]["user_id_dim"]), item_dim=int(config["model"]["item_id_dim"]),
        tag_dim=int(config["model"]["tag_dim"]), hidden_dim=int(config["model"]["hidden_dim"]),
        shared_dim=int(config["model"]["output_dim"]), dropout=float(config["model"]["dropout"]),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def run_demo(
    config: dict[str, Any], user_id: str | None = None, date: str | None = None,
    top_k: int = 10, split: str = "test",
) -> dict[str, Any]:
    paths: Paths = config["resolved_paths"]
    candidates = pd.read_parquet(paths.report_dir / f"fusion_{split}_recommendations.parquet")
    queries = pd.read_parquet(paths.processed_dir / "recall_queries.parquet")
    content_path = paths.processed_dir / 'items_content.parquet'
    items = pd.read_parquet(content_path if content_path.exists() else paths.processed_dir / 'items.parquet')
    selected = candidates
    if user_id is not None:
        selected = selected[selected["user_id"].astype(str).eq(str(user_id))]
    if date is not None:
        selected = selected[selected["date"].eq(pd.Timestamp(date))]
    if selected.empty:
        raise ValueError("没有找到该用户和日期的离线查询；省略参数可自动选择一条测试查询")
    recall_row = selected.iloc[0]
    query = queries[
        queries["user_id"].eq(recall_row["user_id"]) & queries["date"].eq(recall_row["date"])
    ].iloc[0]
    item_map = items.set_index("video_id", drop=False)
    route_map = dict(zip(recall_row["recommended_ids"], recall_row["recall_channels"]))
    records = [
        {
            "user_id": query["user_id"], "video_id": item_id, "tab": 0,
            "history_ids": list(query["history_ids"]), "history_len": len(query["history_ids"]),
            "long_view": 0, "is_like": 0,
        }
        for item_id in recall_row["recommended_ids"]
    ]
    feature_path = paths.artifact_dir / "single_long_view_ranker" / "feature_space.json"
    features = FeatureSpace.load(feature_path)
    collator = ExposureCollator(features, items, int(config["data"]["history_length"]))
    batch = collator(records)
    long_model = _load_ranker(config, features, paths.artifact_dir / "single_long_view_ranker" / "model.pt")
    like_model = _load_ranker(config, features, paths.artifact_dir / "single_is_like_ranker" / "model.pt")
    model_inputs = tuple(
        batch[key] for key in (
            "user_ids", "item_ids", "tag_ids", "durations", "tabs",
            "history_item_ids", "history_mask", "history_len_scaled",
        )
    )
    with torch.no_grad():
        long_scores = torch.sigmoid(long_model(*model_inputs)["long_view"]).tolist()
        like_scores = torch.sigmoid(like_model(*model_inputs)["is_like"]).tolist()
    scored: list[dict[str, Any]] = []
    for item_id, long_score, like_score in zip(recall_row["recommended_ids"], long_scores, like_scores):
        metadata = item_map.loc[item_id]
        scored.append(
            {
                "video_id": item_id,
                "author_id": metadata.get("author_id", "MISSING"),
                "tag": metadata.get("tag", "MISSING"),
                "caption": metadata.get('caption', ''),
                "category": metadata.get('first_level_category_name', ''),
                "long_view_probability": float(long_score),
                "like_probability": float(like_score),
                "score": float(0.7 * long_score + 0.3 * like_score),
                "recall_channels": list(route_map.get(item_id, [])),
            }
        )
    recommendations = author_capped_rerank(scored, top_k=top_k, max_per_author=2)
    for rank, row in enumerate(recommendations, start=1):
        row["rank"] = rank
    result = {
        "user_id": query["user_id"],
        "date": str(pd.Timestamp(query["date"]).date()),
        "history_ids": list(query["history_ids"]),
        "positive_ids_for_offline_check_only": list(query["positive_ids"]),
        "score_formula": "0.7 * long_view_probability + 0.3 * like_probability",
        "recommendations": recommendations,
    }
    write_json(paths.report_dir / "demo_recommendation.json", result)
    return result
