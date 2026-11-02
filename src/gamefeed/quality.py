from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def quality_summary(events: pd.DataFrame, items: pd.DataFrame, states: pd.DataFrame) -> dict[str, Any]:
    item_ids = set(items["video_id"])
    invalid_labels = {
        label: int((~events[label].isin([0, 1]) & events[label].notna()).sum())
        for label in ("long_view", "is_like")
    }
    label_combinations = (
        events.groupby(["long_view", "is_like"], dropna=False).size().rename("count").reset_index().to_dict("records")
    )
    availability = events[["video_id", "date", "split", "long_view"]].merge(
        items[["video_id", "available_from"]], on="video_id", how="left", validate="many_to_one"
    )
    positive_rows = availability[availability["long_view"].eq(1)].copy()
    positive_rows["unavailable"] = positive_rows["available_from"].isna() | (
        positive_rows["available_from"] > positive_rows["date"]
    )
    positive_availability = {
        "positive_rows": int(len(positive_rows)),
        "unavailable_positive_rows": int(positive_rows["unavailable"].sum()),
        "unavailable_positive_ratio": float(positive_rows["unavailable"].mean()) if len(positive_rows) else 0.0,
        "by_split": {
            str(split): {
                "positive_rows": int(len(frame)),
                "unavailable_positive_rows": int(frame["unavailable"].sum()),
                "unavailable_positive_ratio": float(frame["unavailable"].mean()) if len(frame) else 0.0,
            }
            for split, frame in positive_rows.groupby("split")
        },
    }
    return {
        "events": int(len(events)),
        "users": int(events["user_id"].nunique()),
        "items_in_events": int(events["video_id"].nunique()),
        "items_in_catalog": int(items["video_id"].nunique()),
        "duplicate_event_ids": int(events["event_id"].duplicated().sum()),
        "duplicate_item_ids": int(items["video_id"].duplicated().sum()),
        "missing_catalog_rows": int((~events["video_id"].isin(item_ids)).sum()),
        "missing_user_ids": int(events["user_id"].isna().sum()),
        "missing_video_ids": int(events["video_id"].isna().sum()),
        "missing_dates": int(events["date"].isna().sum()),
        "invalid_labels": invalid_labels,
        "label_combinations": label_combinations,
        "empty_history_user_days": int(states["history_len"].eq(0).sum()),
        "user_days": int(len(states)),
        "history_length": {
            key: float(value)
            for key, value in states["history_len"].describe(percentiles=[0.5, 0.95]).to_dict().items()
        },
        "split_events": {key: int(value) for key, value in events["split"].value_counts().to_dict().items()},
        "date_min": str(events["date"].min().date()),
        "date_max": str(events["date"].max().date()),
        "positive_catalog_availability": positive_availability,
    }


def render_quality_markdown(summary: dict[str, Any]) -> str:
    empty_ratio = summary["empty_history_user_days"] / max(summary["user_days"], 1)
    rows = [
        ("展示记录数", summary["events"]),
        ("用户数", summary["users"]),
        ("日志内容数", summary["items_in_events"]),
        ("目录内容数", summary["items_in_catalog"]),
        ("重复 event_id", summary["duplicate_event_ids"]),
        ("重复 video_id", summary["duplicate_item_ids"]),
        ("缺失内容元数据的展示", summary["missing_catalog_rows"]),
        ("缺失日期", summary["missing_dates"]),
        ("缺少历史记录的用户和日期组合数", summary["empty_history_user_days"]),
        ("空历史比例", f"{empty_ratio:.2%}"),
    ]
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    splits = "\n".join(f"| {name} | {count} |" for name, count in summary["split_events"].items())
    labels = "\n".join(
        f"| {row['long_view']} | {row['is_like']} | {row['count']} |" for row in summary["label_combinations"]
    )
    availability = summary.get("positive_catalog_availability", {})
    availability_rows = "\n".join(
        f"| {name} | {values['positive_rows']} | {values['unavailable_positive_rows']} | {values['unavailable_positive_ratio']:.2%} |"
        for name, values in availability.get("by_split", {}).items()
    )
    return f"""# KuaiRand-Pure 数据质量报告

生成范围：{summary['date_min']} 至 {summary['date_max']}。

## 总览

| 检查项 | 统计值 |
| --- | ---: |
{table}

## 时间划分

| 集合 | 展示记录数 |
| --- | ---: |
{splits}

## 标签组合

| long_view | is_like | 记录数 |
| ---: | ---: | ---: |
{labels}

## 历史长度

```json
{summary['history_length']}
```

## 正反馈是否在当天候选库中

候选库采用“上传日期次日进入；缺失时按日志首次出现次日进入”的保守规则。不可用正反馈仍保留在 Recall 分母中，因此它们是当前候选库定义下不可召回的部分。

| 集合 | 长观看正反馈数 | 不可用正反馈数 | 比例 |
| --- | ---: | ---: | ---: |
{availability_rows}

## 标签合法性

```json
{summary['invalid_labels']}
```
"""


def write_quality_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_quality_markdown(summary), encoding="utf-8")

