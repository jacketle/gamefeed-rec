from __future__ import annotations

import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Paths, ensure_output_dirs
from .schema import canonicalize_events, canonicalize_items
from .utils import stable_user_sample, write_json


LOG_PATTERNS = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
RANDOM_LOG_PATTERN = "log_random_4_22_to_5_08_pure.csv"
ITEM_PATTERN = "video_features_basic_pure.csv"


def extract_archive(raw_dir: Path) -> list[Path]:
    archive = raw_dir / "KuaiRand-Pure.tar.gz"
    if not archive.exists():
        return []
    destination = raw_dir / "extracted"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"压缩包包含越界路径: {member.name}")
        tar.extractall(destination)
    return [path for path in destination.rglob("*") if path.is_file()]


def find_file(raw_dir: Path, filename: str) -> Path:
    matches = list(raw_dir.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"需要且只允许一个 {filename}，找到 {len(matches)} 个: {matches}")
    return matches[0]


def inspect_raw(config: dict[str, Any]) -> dict[str, Any]:
    paths: Paths = config["resolved_paths"]
    extracted = extract_archive(paths.raw_dir)
    csv_files = sorted(paths.raw_dir.rglob("*.csv"))
    result: dict[str, Any] = {
        "paths": asdict(paths),
        "extracted_files": [str(path.relative_to(paths.raw_dir)) for path in extracted],
        "csv_files": {},
    }
    for path in csv_files:
        sample = pd.read_csv(path, nrows=5)
        result["csv_files"][str(path.relative_to(paths.raw_dir))] = {
            "size_bytes": path.stat().st_size,
            "columns": sample.columns.tolist(),
            "sample_rows": sample.to_dict(orient="records"),
        }
    return result


def _load_event_file(path: Path, sample_users: set[Any] | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    events = canonicalize_events(raw, path.name)
    if sample_users is not None:
        events = events[events["user_id"].isin(sample_users)].copy()
    return events


def load_canonical_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths: Paths = config["resolved_paths"]
    data_config = config["data"]
    event_paths = [find_file(paths.raw_dir, name) for name in LOG_PATTERNS]

    first_users = pd.read_csv(event_paths[0], usecols=["user_id"])["user_id"]
    sample_users = stable_user_sample(first_users, int(data_config.get("sample_users", 0)))
    events = pd.concat(
        [_load_event_file(path, sample_users) for path in event_paths],
        ignore_index=True,
        sort=False,
    )
    events = events.sort_values(["date", "time_ms", "event_id"], kind="stable").reset_index(drop=True)
    items = canonicalize_items(pd.read_csv(find_file(paths.raw_dir, ITEM_PATTERN)), ITEM_PATTERN)
    return events, items


def assign_split(events: pd.DataFrame, data_config: dict[str, Any]) -> pd.DataFrame:
    result = events.copy()
    result["split"] = "outside"
    periods = {
        "train": (data_config["train_start"], data_config["train_end"]),
        "dev": (data_config["dev_start"], data_config["dev_end"]),
        "test": (data_config["test_start"], data_config["test_end"]),
    }
    for name, (start, end) in periods.items():
        mask = result["date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        result.loc[mask, "split"] = name
    return result[result["split"] != "outside"].copy()


def save_prepared(
    config: dict[str, Any],
    events: pd.DataFrame,
    items: pd.DataFrame,
    states: pd.DataFrame,
    queries: pd.DataFrame,
    item_day_stats: pd.DataFrame,
) -> None:
    ensure_output_dirs(config)
    output: Path = config["resolved_paths"].processed_dir
    frames = {
        "events.parquet": events,
        "items.parquet": items,
        "user_day_state.parquet": states,
        "recall_queries.parquet": queries,
        "item_day_stats.parquet": item_day_stats,
    }
    for filename, frame in frames.items():
        frame.to_parquet(output / filename, index=False)
    write_json(output / "manifest.json", {name: len(frame) for name, frame in frames.items()})

