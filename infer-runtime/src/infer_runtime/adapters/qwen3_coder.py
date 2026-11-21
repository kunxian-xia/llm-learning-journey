"""Cached inference for the text-only Qwen3-Coder MoE checkpoint."""

import torch
import torch.nn.functional as F
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
        # Match the reference order: normalize [B, S, H, D] before transposing.
        return (
            attention.q_norm(q.transpose(1, 2)).transpose(1, 2),
            attention.k_norm(k.transpose(1, 2)).transpose(1, 2),
        )

    def _mlp_output(self, layer: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        output = layer.mlp(hidden)
        # Transformers 4.x also returns router logits, which inference does not need.
        return output[0] if isinstance(output, tuple) else output

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: torch.Tensor | None, is_causal: bool,
    ) -> torch.Tensor:
        # Match Transformers' SDPA path: GQA's internal kernel can round
        # differently, changing top-k expert routing in a large MoE model.
        groups = q.shape[1] // k.shape[1]
        k = k[:, :, None, :, :].expand(-1, -1, groups, -1, -1).reshape(
            k.shape[0], q.shape[1], k.shape[2], k.shape[3]
        )
        v = v[:, :, None, :, :].expand(-1, -1, groups, -1, -1).reshape(
            v.shape[0], q.shape[1], v.shape[2], v.shape[3]
        )
        return F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            attn_mask=mask, is_causal=is_causal and q.shape[-2] > 1,
            scale=q.shape[-1] ** -0.5,
        )
