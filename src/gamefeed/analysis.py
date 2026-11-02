from __future__ import annotations

from typing import Any

import pandas as pd

from .baselines import evaluate_retrieval
from .config import Paths
from .utils import write_json


def _history_group(length: int) -> str:
    if length == 0:
        return "0_empty"
    if length <= 5:
        return "1_5_short"
    return "6_20_longer"


def analyze_retrieval_groups(config: dict[str, Any], method: str = "fusion", split: str = "test") -> dict[str, Any]:
    paths: Paths = config["resolved_paths"]
    result_path = paths.report_dir / f"{method}_{split}_recommendations.parquet"
    if not result_path.exists():
        raise FileNotFoundError(f"请先生成 {result_path.name}")
    results = pd.read_parquet(result_path)
    queries = pd.read_parquet(paths.processed_dir / "recall_queries.parquet")
    events = pd.read_parquet(paths.processed_dir / "events.parquet")
    items = pd.read_parquet(paths.processed_dir / "items.parquet")
    query_columns = queries.loc[queries["split"].eq(split), ["user_id", "date", "history_ids"]].copy()
    data = results.merge(query_columns, on=["user_id", "date"], how="left", validate="one_to_one")
    data["history_len"] = data["history_ids"].map(len)
    data["history_group"] = data["history_len"].map(_history_group)
    train_seen = set(events.loc[events["split"].eq("train"), "video_id"])
    data["item_group"] = data["positive_ids"].map(
        lambda values: "contains_train_unseen" if any(value not in train_seen for value in values) else "all_train_seen"
    )
    catalog = items["video_id"].drop_duplicates().tolist()
    output: dict[str, Any] = {"method": method, "split": split, "history_groups": {}, "item_groups": {}}
    for column, target in (("history_group", "history_groups"), ("item_group", "item_groups")):
        for name, frame in data.groupby(column, sort=True):
            output[target][name] = evaluate_retrieval(frame, catalog, (100,))
    unseen = data.copy()
    unseen['positive_ids'] = unseen.positive_ids.map(lambda values: [v for v in values if v not in train_seen])
    unseen = unseen[unseen.positive_ids.map(len).gt(0)]
    output['unseen_positive_only'] = {
        'queries': len(unseen),
        'positives': int(unseen.positive_ids.map(len).sum()),
        'hits_at_100': int(sum(len(set(r.positive_ids) & set(r.recommended_ids[:100])) for r in unseen.itertuples())),
    }
    counts = output['unseen_positive_only']
    counts['micro_recall_at_100'] = counts['hits_at_100'] / counts['positives'] if counts['positives'] else None
    write_json(paths.report_dir / f"{method}_{split}_group_analysis.json", output)

    lines = [
        f"# {method} {split} 分组召回结果",
        "",
        "## 按历史长度",
        "",
        "| 分组 | 查询数 | Recall@100 | HitRate@100 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in output["history_groups"].items():
        lines.append(
            f"| {name} | {int(values['queries'])} | {values['recall@100']:.4f} | {values['hitrate@100']:.4f} |"
        )
    lines.extend([
        "",
        "## 按正例是否在训练行为中出现",
        "",
        "这里的 train-unseen 只表示训练行为中没出现，不等同于刚发布的新内容。",
        "",
        "| 分组 | 查询数 | Recall@100 | HitRate@100 |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name, values in output["item_groups"].items():
        lines.append(
            f"| {name} | {int(values['queries'])} | {values['recall@100']:.4f} | {values['hitrate@100']:.4f} |"
        )
    (paths.report_dir / f"{method}_{split}_group_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
