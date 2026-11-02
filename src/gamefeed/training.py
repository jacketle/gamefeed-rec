from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baselines import evaluate_retrieval
from .catalog import available_items
from .config import Paths
from .datasets import ExposureCollator, ExposureDataset, RecallCollator, RecallGroupDataset
from .features import FeatureSpace, ItemFeatureLookup
from .metrics import binary_metrics, grouped_auc
from .models import ExposureRanker, TwoTower, grouped_recall_loss, multitask_loss
from .retrieval import ExactIndex
from .utils import seed_everything, write_json


def _build_model(config: dict[str, Any], features: FeatureSpace) -> TwoTower:
    model_config = config["model"]
    return TwoTower(
        num_users=len(features.user_vocab),
        num_items=len(features.item_vocab),
        num_tags=len(features.tag_vocab),
        user_id_dim=int(model_config["user_id_dim"]),
        item_id_dim=int(model_config["item_id_dim"]),
        tag_dim=int(model_config["tag_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        output_dim=int(model_config["output_dim"]),
        dropout=float(model_config["dropout"]),
    )


def _batch_loss(model: TwoTower, batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
    users = model.encode_user(
        batch["user_ids"], batch["history_item_ids"], batch["history_tag_ids"],
        batch["history_durations"], batch["history_mask"], batch["history_len_scaled"],
    )
    items = model.item_encoder(
        batch["candidate_item_ids"], batch["candidate_tag_ids"], batch["candidate_durations"]
    )
    scores = torch.einsum("bd,bcd->bc", users, items)
    return grouped_recall_loss(scores, batch["candidate_mask"], temperature)


def _evaluate_loss(model: TwoTower, loader: DataLoader, temperature: float) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            losses.append(float(_batch_loss(model, batch, temperature)))
    return float(np.mean(losses)) if losses else float("nan")


def train_two_tower(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    paths: Paths = config["resolved_paths"]
    events = pd.read_parquet(paths.processed_dir / "events.parquet")
    items = pd.read_parquet(paths.processed_dir / "items.parquet")
    groups = pd.read_parquet(paths.processed_dir / "recall_groups.parquet")
    if config.get('ablation_no_history', False):
        groups['history_ids'] = [[] for _ in range(len(groups))]
        groups['history_len'] = 0
    train_events = events[events["split"].eq("train")]
    features = FeatureSpace.fit(train_events, items)
    model = _build_model(config, features)
    collator = RecallCollator(features, items, int(config["data"]["history_length"]))
    batch_size = int(config["model"]["batch_size"])
    train_loader = DataLoader(
        RecallGroupDataset(groups[groups["split"].eq("train")]), batch_size=batch_size,
        shuffle=True, collate_fn=collator, generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    dev_loader = DataLoader(
        RecallGroupDataset(groups[groups["split"].eq("dev")]), batch_size=batch_size,
        shuffle=False, collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    temperature = float(config["model"]["temperature"])
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_dev = float("inf")
    for epoch in range(1, int(config["model"]["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = _batch_loss(model, batch, temperature)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_loss = _evaluate_loss(model, dev_loader, temperature)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_loss": dev_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        comparison = dev_loss if np.isfinite(dev_loss) else row["train_loss"]
        if comparison < best_dev:
            best_dev = comparison
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model_dir = paths.artifact_dir / "two_tower"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    features.save(model_dir / "feature_space.json")
    write_json(model_dir / "model_config.json", config["model"])
    write_json(paths.report_dir / "two_tower_training.json", history)
    queries = pd.read_parquet(paths.processed_dir / 'recall_queries.parquet')
    if config.get('ablation_no_history', False):
        queries['history_ids'] = [[] for _ in range(len(queries))]
        queries['history_len'] = 0
    retrieval = evaluate_two_tower(model, features, items, queries, config)
    result = {"epochs": history, "retrieval": retrieval}
    write_json(paths.report_dir / "two_tower_results.json", result)
    return result


def _encode_items(model: TwoTower, features: FeatureSpace, items: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
    lookup = ItemFeatureLookup(items, features)
    item_ids = items["video_id"].tolist()
    encoded, tags, durations = lookup.encode_many(item_ids)
    vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(item_ids), batch_size):
            end = start + batch_size
            value = model.item_encoder(
                torch.tensor(encoded[start:end], dtype=torch.long),
                torch.tensor(tags[start:end], dtype=torch.long),
                torch.tensor(durations[start:end], dtype=torch.float32),
            )
            vectors.append(value.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def _encode_query_users(
    model: TwoTower, features: FeatureSpace, items: pd.DataFrame, queries: pd.DataFrame, history_length: int
) -> np.ndarray:
    fake = queries.copy()
    fake["positive_id"] = fake["positive_ids"].map(lambda values: values[0])
    fake["negative_ids"] = [[] for _ in range(len(fake))]
    collator = RecallCollator(features, items, history_length)
    loader = DataLoader(RecallGroupDataset(fake), batch_size=512, shuffle=False, collate_fn=collator)
    vectors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            value = model.encode_user(
                batch["user_ids"], batch["history_item_ids"], batch["history_tag_ids"],
                batch["history_durations"], batch["history_mask"], batch["history_len_scaled"],
            )
            vectors.append(value.cpu().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, model.empty_history.numel()), np.float32)


def evaluate_two_tower(
    model: TwoTower, features: FeatureSpace, items: pd.DataFrame, queries: pd.DataFrame, config: dict[str, Any]
) -> dict[str, dict[str, float]]:
    vectors = _encode_items(model, features, items)
    item_ids = items["video_id"].to_numpy()
    max_k = max(config["retrieval"]["top_k"])
    output: dict[str, dict[str, float]] = {}
    for split in config.get('evaluation_splits', ('dev', 'test')):
        selected = queries[queries["split"].eq(split)].reset_index(drop=True)
        if selected.empty:
            continue
        user_vectors = _encode_query_users(model, features, items, selected, int(config["data"]["history_length"]))
        recommendations: list[list[Any]] = [[] for _ in range(len(selected))]
        for date, indices in selected.groupby("date").groups.items():
            available_ids = set(available_items(items, pd.Timestamp(date))["video_id"])
            subset = np.flatnonzero(items["video_id"].isin(available_ids).to_numpy())
            index = ExactIndex.build(item_ids[subset], vectors[subset])
            _, found = index.search(user_vectors[list(indices)], max_k)
            for row_index, values in zip(indices, found):
                recommendations[row_index] = values.tolist()
        result = selected[["user_id", "date", "positive_ids"]].copy()
        result["recommended_ids"] = recommendations
        result.to_parquet(config['resolved_paths'].report_dir / f'two_tower_{split}_recommendations.parquet', index=False)
        output[split] = evaluate_retrieval(result, item_ids.tolist(), tuple(config["retrieval"]["top_k"]))
    return output


def _ranker_forward(model: ExposureRanker, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = ("user_ids", "item_ids", "tag_ids", "durations", "tabs", "history_item_ids", "history_mask", "history_len_scaled")
    return model(*(batch[key] for key in keys))


def _ranker_metrics(model: ExposureRanker, loader: DataLoader) -> dict[str, Any]:
    model.eval()
    labels = {"long_view": [], "is_like": []}
    scores = {"long_view": [], "is_like": []}
    groups: list[int] = []
    with torch.no_grad():
        for batch in loader:
            outputs = _ranker_forward(model, batch)
            for task in labels:
                labels[task].extend(batch[task].int().tolist())
                scores[task].extend(torch.sigmoid(outputs[task]).tolist())
            groups.extend(batch["user_ids"].tolist())
    result: dict[str, Any] = {}
    for task in labels:
        result[task] = binary_metrics(labels[task], scores[task])
        result[task].update(grouped_auc(groups, labels[task], scores[task]))
    return result


def train_shared_ranker(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    paths: Paths = config["resolved_paths"]
    events = pd.read_parquet(paths.processed_dir / "events.parquet")
    items = pd.read_parquet(paths.processed_dir / "items.parquet")
    train_events = events[events["split"].eq("train")]
    features = FeatureSpace.fit(train_events, items)
    collator = ExposureCollator(features, items, int(config["data"]["history_length"]))
    loaders = {
        split: DataLoader(
            ExposureDataset(events[events["split"].eq(split)]),
            batch_size=int(config["model"]["batch_size"]), shuffle=split == "train", collate_fn=collator,
            generator=torch.Generator().manual_seed(int(config["seed"])),
        )
        for split in ("train", "dev", "test")
    }
    model = ExposureRanker(
        len(features.user_vocab), len(features.item_vocab), len(features.tag_vocab),
        user_dim=int(config["model"]["user_id_dim"]), item_dim=int(config["model"]["item_id_dim"]),
        tag_dim=int(config["model"]["tag_dim"]), hidden_dim=int(config["model"]["hidden_dim"]),
        shared_dim=int(config["model"]["output_dim"]), dropout=float(config["model"]["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -float("inf")
    for epoch in range(1, int(config["model"]["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders["train"]:
            optimizer.zero_grad()
            outputs = _ranker_forward(model, batch)
            loss = multitask_loss(outputs, batch["long_view"], batch["is_like"], float(config["model"]["like_loss_weight"]))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        dev = _ranker_metrics(model, loaders["dev"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_long_auc": dev["long_view"]["auc"], "dev_like_auc": dev["is_like"]["auc"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if config.get('fixed_epochs', False) or row["dev_long_auc"] > best_auc:
            best_auc = row["dev_long_auc"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model_dir = paths.artifact_dir / "shared_ranker"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    features.save(model_dir / "feature_space.json")
    write_json(model_dir / 'model_config.json', config['model'])
    result = {"epochs": history, "dev": _ranker_metrics(model, loaders["dev"]), "test": _ranker_metrics(model, loaders["test"])}
    write_json(paths.report_dir / "shared_ranker_results.json", result)
    return result


def train_single_rankers(config: dict[str, Any]) -> dict[str, Any]:
    """Train one model per task with the same inputs and training budget."""
    seed_everything(int(config["seed"]))
    paths: Paths = config["resolved_paths"]
    events = pd.read_parquet(paths.processed_dir / "events.parquet")
    items = pd.read_parquet(paths.processed_dir / "items.parquet")
    train_events = events[events["split"].eq("train")]
    features = FeatureSpace.fit(train_events, items)
    collator = ExposureCollator(features, items, int(config["data"]["history_length"]))
    loaders = {
        split: DataLoader(
            ExposureDataset(events[events["split"].eq(split)]),
            batch_size=int(config["model"]["batch_size"]), shuffle=split == "train", collate_fn=collator,
            generator=torch.Generator().manual_seed(int(config["seed"])),
        )
        for split in ("train", "dev", "test")
    }
    output: dict[str, Any] = {}
    for task_index, task in enumerate(("long_view", "is_like")):
        seed_everything(int(config["seed"]))
        loaders['train'].generator.manual_seed(int(config['seed']))
        model = ExposureRanker(
            len(features.user_vocab), len(features.item_vocab), len(features.tag_vocab),
            user_dim=int(config["model"]["user_id_dim"]), item_dim=int(config["model"]["item_id_dim"]),
            tag_dim=int(config["model"]["tag_dim"]), hidden_dim=int(config["model"]["hidden_dim"]),
            shared_dim=int(config["model"]["output_dim"]), dropout=float(config["model"]["dropout"]),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["model"]["learning_rate"]),
            weight_decay=float(config["model"]["weight_decay"]),
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        best_state: dict[str, torch.Tensor] | None = None
        best_auc = -float("inf")
        history: list[dict[str, float]] = []
        for epoch in range(1, int(config["model"]["epochs"]) + 1):
            model.train()
            losses: list[float] = []
            for batch in loaders["train"]:
                optimizer.zero_grad()
                logits = _ranker_forward(model, batch)[task]
                loss = criterion(logits, batch[task].float())
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            dev_metrics = _ranker_metrics(model, loaders["dev"])[task]
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_auc": dev_metrics["auc"]})
            print(json.dumps({'task': task, **history[-1]}), flush=True)
            if config.get('fixed_epochs', False) or dev_metrics["auc"] > best_auc:
                best_auc = dev_metrics["auc"]
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        model_dir = paths.artifact_dir / f"single_{task}_ranker"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "model.pt")
        features.save(model_dir / "feature_space.json")
        write_json(model_dir / 'model_config.json', config['model'])
        output[task] = {
            "epochs": history,
            "dev": _ranker_metrics(model, loaders["dev"])[task],
            "test": _ranker_metrics(model, loaders["test"])[task],
        }
    write_json(paths.report_dir / "single_ranker_results.json", output)
    return output

