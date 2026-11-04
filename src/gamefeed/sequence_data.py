"""Chunked 1K preparation; keeps Pure's daily replay unchanged.

Run: python -m gamefeed.sequence_data --config configs/kuairand_1k.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .config import load_config, ensure_output_dirs
from .utils import write_json

LOGS = [
    ("log_standard_4_08_to_4_21_1k.csv", 0),
    ("log_standard_4_22_to_5_08_1k.csv", 0),
    ("log_random_4_22_to_5_08_1k.csv", 1),
]
COLS = ["user_id", "video_id", "date", "time_ms", "tab", "long_view",
        "is_like", "is_rand", "play_time_ms", "duration_ms"]


def compact_events(frame):
    """Keep timestamp precision; compact small integers and repeated strings."""
    for name in ('user_id', 'video_id', 'date', 'raw_date', 'source_row', 'duration_ms'):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], downcast='integer')
    for name in ('tab', 'long_view', 'is_like', 'is_rand', 'history_len'):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], downcast='integer')
    if 'source' in frame:
        frame['source'] = pd.Categorical(frame.source, categories=sorted(name for name, _ in LOGS))
    if 'split' in frame:
        frame['split'] = pd.Categorical(frame.split, categories=['warmup','train','dev','test','outside'])
    return frame


def event_history(events: pd.DataFrame, length: int):
    """Return slice indices into one contiguous positive-event pool.

    Search on timestamps with side=left excludes *all* simultaneous events.
    Only standard positives update history, including during dev/test replay.
    """
    if length < 1:
        raise ValueError("history length must be positive")
    out = events.sort_values(["user_id", "time_ms", "source", "source_row"], kind="stable").reset_index(drop=True)
    starts = np.zeros(len(out), np.int64)
    ends = np.zeros(len(out), np.int64)
    pool, pool_times, offset = [], [], 0
    for _, group in out.groupby("user_id", sort=False):
        pos = group.loc[group.long_view.eq(1) & group.is_rand.eq(0)]
        ts = pos.time_ms.to_numpy(np.int64)
        end = np.searchsorted(ts, group.time_ms.to_numpy(np.int64), side="left")
        starts[group.index] = offset + np.maximum(end - length, 0)
        ends[group.index] = offset + end
        pool.append(pos.video_id.to_numpy(np.int32))
        pool_times.append(ts)
        offset += len(pos)
    out["history_start"], out["history_end"] = starts, ends
    out["history_len"] = (ends - starts).astype(np.int16)
    return out, np.concatenate(pool) if pool else np.array([], np.int32), np.concatenate(pool_times) if pool_times else np.array([], np.int64)


def prepare(config):
    started = time.perf_counter()
    paths = config["resolved_paths"]
    ensure_output_dirs(config)
    dc = config["data"]
    chunk_size = int(dc["chunksize"])
    if (paths.report_dir / 'experiment_results.json').exists():
        raise FileExistsError('Output already exists; use a new configuration/output directory to preserve results')
    # Sampling population comes only from the historical period, never test activity.
    users = set()
    for c in pd.read_csv(paths.raw_dir / LOGS[0][0], usecols=["user_id"], chunksize=chunk_size):
        users.update(c.user_id.unique().tolist())
    users = sorted(users, key=lambda u: hashlib.sha256(f'{config["seed"]}:{u}'.encode()).digest())
    size = int(dc["sample_users"])
    # Zero means keep every log row, including users first seen after warmup.
    selected = set(users[:size]) if size else None
    pieces, log_stats = [], []
    for name, source in LOGS:
        stats = dict(file=name, rows=0, invalid_labels=0, source_mismatches=0,
                     utc_date_mismatches=0, shanghai_date_mismatches=0)
        tabs, all_users = {}, set()
        for c in pd.read_csv(paths.raw_dir / name, chunksize=chunk_size):
            original_columns = c.columns.tolist()
            n = len(c)
            c["source_row"] = np.arange(stats["rows"], stats["rows"] + n, dtype=np.int64)
            stats["rows"] += n
            stats["invalid_labels"] += int((~c.long_view.isin([0, 1]) | ~c.is_like.isin([0, 1])).sum())
            stats["source_mismatches"] += int(c.is_rand.ne(source).sum())
            ts = pd.to_datetime(c.time_ms, unit="ms", utc=True, errors="raise")
            for tz, key in [("UTC", "utc_date_mismatches"), ("Asia/Shanghai", "shanghai_date_mismatches")]:
                dates = ts.dt.tz_convert(tz).dt.strftime("%Y%m%d").astype(int)
                stats[key] += int(dates.ne(c.date).sum())
            all_users.update(c.user_id.unique().tolist())
            for tab, count in c.tab.value_counts().items():
                tabs[str(tab)] = tabs.get(str(tab), 0) + int(count)
            if selected is not None:
                c = c[c.user_id.isin(selected)].copy()
            c["raw_fingerprint"] = pd.util.hash_pandas_object(c[original_columns], index=False).to_numpy()
            c = c[COLS + ["source_row", "raw_fingerprint"]]
            c["source"] = name
            pieces.append(compact_events(c))
        stats.update(users=len(all_users), tabs=tabs)
        log_stats.append(stats)
        print(f'{name}: checked {stats["rows"]:,} rows', flush=True)
    if any(s["invalid_labels"] or s["source_mismatches"] for s in log_stats):
        write_json(paths.report_dir / "experiment_results.json", {"data_logs": log_stats})
        raise ValueError("Invalid source or labels; inspect report before proceeding")
    events = pd.concat(pieces, ignore_index=True)
    del pieces
    gc.collect()
    # Duplicates refer to identical observed event fields, not user-video pairs.
    dup = events.duplicated(COLS + ["raw_fingerprint"], keep="first")
    duplicate_count = int(dup.sum())
    events = events.loc[~dup].copy()
    if selected is None:
        selected = set(events.user_id.unique().tolist())
    events["raw_date"] = events["date"]
    events["date"] = pd.to_datetime(events.time_ms, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y%m%d").astype(np.int32)
    events["split"] = np.select([
        events.date.lt(dc["train_start"]), events.date.le(dc["train_end"]),
        events.date.le(dc["dev_end"]), events.date.le(dc["test_end"])],
        ["warmup", "train", "dev", "test"], default="outside")
    if events.split.eq("outside").any():
        raise ValueError("Events beyond declared period")
    events = compact_events(events)
    events, pool, pool_times = event_history(events, int(dc["history_length"]))
    known = events.video_id.unique()
    items, item_rows = [], 0
    for c in pd.read_csv(paths.raw_dir / "video_features_basic_1k.csv",
                         usecols=["video_id", "author_id", "video_duration", "tag", "upload_dt"],
                         dtype={"tag": "string"}, chunksize=chunk_size):
        item_rows += len(c)
        c = c[c.video_id.isin(known)].copy()
        c['video_id'] = c.video_id.astype('int32')
        c['author_id'] = pd.to_numeric(c.author_id, downcast='integer')
        c['video_duration'] = pd.to_numeric(c.video_duration, downcast='integer')
        items.append(c)
    items = pd.concat(items, ignore_index=True)
    if items.video_id.duplicated().any():
        raise ValueError("Duplicate item metadata")
    missing = int((~events.video_id.isin(items.video_id)).sum())
    if missing:
        raise ValueError(f"Missing metadata for {missing} exposure rows")
    # Diagnostics only: play_time is an outcome and never a current input feature.
    next_time = events.groupby("user_id").time_ms.shift(-1)
    overlap = (events.play_time_ms > (next_time - events.time_ms)) & next_time.gt(events.time_ms)
    split_stats = []
    for (split, source), g in events.groupby(["split", "is_rand"], observed=True):
        split_stats.append(dict(split=split, random=int(source), rows=len(g), users=g.user_id.nunique(),
            positives=int(g.long_view.sum()), empty_history=int(g.history_len.eq(0).sum()),
            mean_history=float(g.history_len.mean())))
    np.save(paths.processed_dir / "history_pool.npy", pool)
    np.save(paths.processed_dir / "history_times.npy", pool_times)
    for date, g in events.groupby("date"):
        g.to_parquet(paths.processed_dir / f"events_{date}.parquet", index=False)
    items.to_parquet(paths.processed_dir / "items.parquet", index=False)
    # IDs stay with processed data, never in the public aggregate report.
    write_json(paths.processed_dir / "selected_users.json", sorted(selected))
    result = dict(stage="prepared", sample_users=len(selected), logs=log_stats,
        raw_item_rows=item_rows, pilot_rows=len(events), pilot_items=len(items),
        removed_exact_duplicates=duplicate_count, missing_item_rows=missing,
        tag_missing_items=int(items.tag.fillna("").eq("").sum()),
        potential_feedback_overlap_rows=int(overlap.sum()), splits=split_stats,
        prepare_seconds=round(time.perf_counter()-started, 2),
        peak_process_mib=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)/2**20,
        history_rule="strictly earlier timestamps; standard long_view=1 only; no random feedback",
        feedback_assumption="prior event feedback available before next event; overlap diagnostic is not availability proof")
    write_json(paths.report_dir / "experiment_results.json", {"preparation": result})
    lines = ["# KuaiRand-1K 数据检查", "", f"固定用户数：{len(selected)}；本页记录数据准备，模型对照另见experiment_results.md。", "",
        "## 完整日志检查", "", "| 文件 | 行数 | 用户数 | 标签异常 | 来源异常 | UTC日期差异 | 上海时区日期差异 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in log_stats:
        lines.append(f'| {s["file"]} | {s["rows"]} | {s["users"]} | {s["invalid_labels"]} | {s["source_mismatches"]} | {s["utc_date_mismatches"]} | {s["shanghai_date_mismatches"]} |')
    lines += ["", "## 固定用户子集", "", f"曝光记录：{len(events)}；不同视频：{len(items)}；完全重复删除：{duplicate_count}；缺失元数据：{missing}。",
        "", "| 集合 | 随机展示 | 曝光数 | 用户数 | 观看达标数 | 空历史数 | 平均历史长度 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in split_stats:
        lines.append(f'| {s["split"]} | {s["random"]} | {s["rows"]} | {s["users"]} | {s["positives"]} | {s["empty_history"]} | {s["mean_history"]:.2f} |')
    lines += ["", "## 历史与时间", "", "统一以time_ms换算的Asia/Shanghai日期切分，原始date另存raw_date；更早的记录仅用于历史积累。",
        "同一用户同时间戳的记录互不进入历史；随机展示不更新历史。",
        f"观看时长超过下一次不同时间戳间隔的记录：{int(overlap.sum())}。这是反馈可用性检查线索，不能据此确定反馈完成时间。",
        "逐事件模型采用先前反馈已可用的近似；当前观看时长不输入模型。正式结论前检查该假设的影响。",
        "", "数据、逐用户标识及历史数组仅保存在本地。Pure数据和已有发布仓库未修改。"]
    (paths.report_dir / "data_check.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f'Prepared {len(events):,} rows, {len(items):,} items in {result["prepare_seconds"]}s', flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/kuairand_1k.yaml")
    args = parser.parse_args()
    prepare(load_config(args.config))
