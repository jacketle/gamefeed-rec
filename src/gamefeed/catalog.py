from __future__ import annotations

import pandas as pd


def add_available_from(items: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add the first date on which an item may enter the offline catalog."""
    result = items.copy()
    first_seen = events.groupby("video_id")["date"].min().rename("first_seen_date")
    result = result.merge(first_seen, on="video_id", how="left", validate="one_to_one")
    if "upload_date" in result:
        source = result["upload_date"].fillna(result["first_seen_date"])
    else:
        source = result["first_seen_date"]
    result["available_from"] = pd.to_datetime(source, errors="coerce") + pd.Timedelta(days=1)
    return result


def available_items(items: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    if "available_from" not in items:
        return items
    return items[items["available_from"].notna() & items["available_from"].le(pd.Timestamp(date))]

