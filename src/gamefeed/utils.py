from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def stable_bucket(value: Any, buckets: int = 10_000) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def stable_user_sample(values: Iterable[Any], size: int) -> set[Any]:
    unique = set(values)
    if size <= 0 or len(unique) <= size:
        return unique
    return set(sorted(unique, key=lambda value: (stable_bucket(value, 2**31 - 1), str(value)))[:size])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

