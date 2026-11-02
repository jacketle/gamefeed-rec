from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Paths:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    artifact_dir: Path
    report_dir: Path


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    root = config_path.parent.parent
    path_config = config["paths"]
    config["resolved_paths"] = Paths(
        project_root=root,
        raw_dir=_resolve(root, path_config["raw_dir"]),
        processed_dir=_resolve(root, path_config["processed_dir"]),
        artifact_dir=_resolve(root, path_config["artifact_dir"]),
        report_dir=_resolve(root, path_config["report_dir"]),
    )
    return config


def ensure_output_dirs(config: dict[str, Any]) -> None:
    paths: Paths = config["resolved_paths"]
    for path in (paths.processed_dir, paths.artifact_dir, paths.report_dir):
        path.mkdir(parents=True, exist_ok=True)

