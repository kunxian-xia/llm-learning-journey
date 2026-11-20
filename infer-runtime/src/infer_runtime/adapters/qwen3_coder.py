"""Cached inference for the text-only Qwen3-Coder MoE checkpoint."""

import torch
from transformers import Qwen3MoeForCausalLM

from .qwen25 import Qwen25Decoder


class Qwen3CoderDecoder(Qwen25Decoder):
    """Reuse cached GQA while preserving Qwen3's Q/K norms and routed MLPs."""

    def __init__(self, model: Qwen3MoeForCausalLM):
        if not isinstance(model, Qwen3MoeForCausalLM):
            raise TypeError("model must be a Transformers Qwen3MoeForCausalLM")
        if model.config.sliding_window is not None:
            raise ValueError("sliding-window Qwen3 configurations are not supported")
        self.model = model.eval()
        self.vocab_size = int(model.config.vocab_size)
        self.context_length = int(model.config.max_position_embeddings)

    def _normalize_qk(
        self, attention: torch.nn.Module, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return attention.q_norm(q), attention.k_norm(k)
