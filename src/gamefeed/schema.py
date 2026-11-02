from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


EVENT_ALIASES = {
    "user_id": ("user_id",),
    "video_id": ("video_id",),
    "date": ("date", "event_date"),
    "time_ms": ("time_ms", "timestamp", "event_time"),
    "tab": ("tab",),
    "long_view": ("long_view", "is_long_view"),
    "is_like": ("is_like", "like"),
    "is_rand": ("is_rand", "is_random"),
}

ITEM_ALIASES = {
    "video_id": ("video_id",),
    "author_id": ("author_id",),
    "video_duration": ("video_duration", "duration_ms", "duration"),
    "tag": ("tag", "tag_id", "video_tag_id", "music_type"),
    "upload_date": ("upload_date", "upload_dt", "publish_date"),
}


@dataclass(frozen=True)
class SchemaResolution:
    mapping: dict[str, str]
    missing: tuple[str, ...]


def resolve_columns(columns: Iterable[str], aliases: dict[str, tuple[str, ...]]) -> SchemaResolution:
    available = set(columns)
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for canonical, choices in aliases.items():
        match = next((choice for choice in choices if choice in available), None)
        if match is None:
            missing.append(canonical)
        else:
            mapping[match] = canonical
    return SchemaResolution(mapping=mapping, missing=tuple(missing))


def canonicalize_events(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    resolution = resolve_columns(frame.columns, EVENT_ALIASES)
    required = {"user_id", "video_id", "long_view", "is_like"}
    unresolved_required = required.intersection(resolution.missing)
    if unresolved_required:
        raise ValueError(f"{source_name} 缺少必要字段: {sorted(unresolved_required)}; 实际字段: {list(frame.columns)}")
    events = frame.rename(columns=resolution.mapping).copy()
    if "date" not in events and "time_ms" not in events:
        raise ValueError(f"{source_name} 同时缺少 date 与 time_ms，无法按时间划分")
    if "date" in events:
        date_text = events["date"].astype("string").str.replace(r"\.0$", "", regex=True)
        compact_date = date_text.str.fullmatch(r"\d{8}").fillna(False)
        parsed = pd.to_datetime(date_text.where(compact_date), format="%Y%m%d", errors="coerce")
        parsed.loc[~compact_date] = pd.to_datetime(date_text.loc[~compact_date], errors="coerce")
        events["date"] = parsed.dt.normalize()
    if "date" not in events or events["date"].isna().all():
        numeric_time = pd.to_numeric(events["time_ms"], errors="coerce")
        unit = "ms" if numeric_time.dropna().median() > 10**11 else "s"
        events["date"] = pd.to_datetime(numeric_time, unit=unit, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    if "time_ms" not in events:
        events["time_ms"] = events["date"].astype("int64") // 10**6
    for label in ("long_view", "is_like"):
        events[label] = pd.to_numeric(events[label], errors="coerce").astype("Int8")
    for column, default in (("tab", -1), ("is_rand", 0)):
        if column not in events:
            events[column] = default
    events.insert(0, "event_id", source_name + ":" + events.index.astype(str))
    return events


def canonicalize_items(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    resolution = resolve_columns(frame.columns, ITEM_ALIASES)
    if "video_id" in resolution.missing:
        raise ValueError(f"{source_name} 缺少 video_id; 实际字段: {list(frame.columns)}")
    items = frame.rename(columns=resolution.mapping).copy()
    for column, default in (("author_id", "MISSING"), ("tag", "MISSING"), ("video_duration", 0.0)):
        if column not in items:
            items[column] = default
    if "upload_date" in items:
        items["upload_date"] = pd.to_datetime(items["upload_date"], errors="coerce").dt.normalize()
    return items

