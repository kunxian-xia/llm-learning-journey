"""Model-facing values used by inference adapters."""

from dataclasses import dataclass
from typing import Protocol

import torch


class KVCache(Protocol):
    @property
    def sequence_length(self) -> int: ...


@dataclass(frozen=True)
class Logits:
    tensor: torch.Tensor  # [vocab_size]
