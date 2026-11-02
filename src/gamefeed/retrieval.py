from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass
class ExactIndex:
    item_ids: np.ndarray
    vectors: np.ndarray

    @classmethod
    def build(cls, item_ids: Sequence[Hashable], vectors: np.ndarray) -> "ExactIndex":
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.asarray(item_ids)
        if matrix.ndim != 2 or len(ids) != len(matrix):
            raise ValueError("item_ids 数量必须与二维向量矩阵行数一致")
        return cls(item_ids=ids, vectors=matrix)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
        if query_matrix.ndim == 1:
            query_matrix = query_matrix[None, :]
        if query_matrix.shape[1] != self.vectors.shape[1]:
            raise ValueError("查询向量维度与物品向量维度不一致")
        scores = query_matrix @ self.vectors.T
        k = min(k, self.vectors.shape[0])
        indices = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(scores, indices, axis=1), self.item_ids[indices]


def l2_normalize(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, epsilon)

