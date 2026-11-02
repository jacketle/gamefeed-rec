from __future__ import annotations

from typing import Hashable, Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def recall_at_k(recommended: Sequence[Hashable], positives: Iterable[Hashable], k: int) -> float:
    positive_set = set(positives)
    if not positive_set:
        raise ValueError("Recall 的正例集合不能为空")
    return len(set(recommended[:k]).intersection(positive_set)) / len(positive_set)


def hit_rate_at_k(recommended: Sequence[Hashable], positives: Iterable[Hashable], k: int) -> float:
    positive_set = set(positives)
    if not positive_set:
        raise ValueError("HitRate 的正例集合不能为空")
    return float(bool(set(recommended[:k]).intersection(positive_set)))


def ndcg_at_k(labels: Sequence[float], k: int) -> float:
    values = np.asarray(labels[:k], dtype=float)
    gains = np.power(2.0, values) - 1.0
    discounts = np.log2(np.arange(len(values)) + 2.0)
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(np.asarray(labels, dtype=float))[::-1][:k]
    idcg = float(np.sum((np.power(2.0, ideal) - 1.0) / np.log2(np.arange(len(ideal)) + 2.0)))
    return dcg / idcg if idcg > 0 else float("nan")


def binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    result = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, p)),
    }
    result["auc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return result


def grouped_auc(groups: Sequence[Hashable], labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    records: dict[Hashable, list[tuple[int, float]]] = {}
    for group, label, score in zip(groups, labels, scores):
        records.setdefault(group, []).append((int(label), float(score)))
    weighted_sum = total_weight = valid_groups = 0
    for values in records.values():
        y = np.asarray([value[0] for value in values])
        if np.unique(y).size < 2:
            continue
        score = np.asarray([value[1] for value in values])
        weight = len(values)
        weighted_sum += roc_auc_score(y, score) * weight
        total_weight += weight
        valid_groups += 1
    return {
        "gauc": weighted_sum / total_weight if total_weight else float("nan"),
        "valid_groups": valid_groups,
        "covered_rows": total_weight,
    }


def catalog_coverage(recommendations: Iterable[Sequence[Hashable]], catalog: Iterable[Hashable]) -> float:
    catalog_set = set(catalog)
    if not catalog_set:
        raise ValueError("候选目录不能为空")
    recommended = {item for sequence in recommendations for item in sequence}
    return len(recommended.intersection(catalog_set)) / len(catalog_set)

