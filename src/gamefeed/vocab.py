from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


PAD = 0
UNK = 1


@dataclass
class Vocabulary:
    values: dict[str, int]

    @classmethod
    def fit(cls, values: Iterable[Any]) -> "Vocabulary":
        unique = sorted({str(value) for value in values if value is not None})
        return cls({value: index + 2 for index, value in enumerate(unique)})

    def encode(self, value: Any) -> int:
        if value is None:
            return UNK
        return self.values.get(str(value), UNK)

    def encode_many(self, values: Iterable[Any]) -> list[int]:
        return [self.encode(value) for value in values]

    def __len__(self) -> int:
        return len(self.values) + 2

    def to_dict(self) -> dict[str, int]:
        return dict(self.values)

