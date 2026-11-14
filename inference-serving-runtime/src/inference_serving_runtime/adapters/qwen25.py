"""Independent cached attention path for text-only Qwen2.5 models.

Weights and non-attention layers come from an already-loaded Hugging Face
Qwen2ForCausalLM. Prefill and decode do not call its transformer forward method.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM

from inference_serving_runtime.api import Logits


@dataclass
class Qwen25KVCache:
    layers: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    sequence_length: int = 0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Qwen25Decoder:
    """Single-sequence Qwen2.5 prefill/decode with a per-layer KV cache."""

    def __init__(self, model: Qwen2ForCausalLM):
        if not isinstance(model, Qwen2ForCausalLM):
            raise TypeError("model must be a Transformers Qwen2ForCausalLM")
        if any(layer_type != "full_attention" for layer_type in model.config.layer_types):
            raise ValueError("only full-attention Qwen2.5 configurations are supported")
        self.model = model.eval()
        self.vocab_size = int(model.config.vocab_size)
        self.context_length = int(model.config.max_position_embeddings)

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _check_token(self, token_id: int) -> None:
        if not isinstance(token_id, int) or not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token ID must be in [0, {self.vocab_size})")

    @torch.inference_mode()
    def _forward(self, token_ids: Sequence[int], cache: Qwen25KVCache) -> Logits:
        if cache.sequence_length + len(token_ids) > self.context_length:
            raise ValueError("tokens exceed model context length")
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self._device)
        hidden = self.model.model.embed_tokens(ids)
        positions = torch.arange(
            cache.sequence_length, cache.sequence_length + len(token_ids), device=self._device
        ).unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(hidden, positions)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        new_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_index, layer in enumerate(self.model.model.layers):
            residual = hidden
            normed = layer.input_layernorm(hidden)
            attention = layer.self_attn
            batch, seq_len, _ = normed.shape
            head_dim = attention.head_dim
            q = attention.q_proj(normed).view(batch, seq_len, -1, head_dim).transpose(1, 2)
            k = attention.k_proj(normed).view(batch, seq_len, -1, head_dim).transpose(1, 2)
            v = attention.v_proj(normed).view(batch, seq_len, -1, head_dim).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            if cache.sequence_length:
                old_k, old_v = cache.layers[layer_index]
                k = torch.cat((old_k, k), dim=-2)
                v = torch.cat((old_v, v), dim=-2)
            new_layers.append((k, v))

            attended = F.scaled_dot_product_attention(
                q, k, v, is_causal=cache.sequence_length == 0, enable_gqa=True
            )
            attended = attended.transpose(1, 2).reshape(batch, seq_len, -1)
            hidden = residual + attention.o_proj(attended)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        hidden = self.model.model.norm(hidden)
        logits = self.model.lm_head(hidden)[0, -1, :].detach()
        cache.layers = new_layers
        cache.sequence_length += len(token_ids)
        return Logits(logits)

    def prefill(self, prompt_ids: Sequence[int]) -> tuple[Qwen25KVCache, Logits]:
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        for token_id in prompt_ids:
            self._check_token(token_id)
        cache = Qwen25KVCache()
        return cache, self._forward(prompt_ids, cache)

    def decode_step(self, new_token: int, kv_cache: Qwen25KVCache) -> Logits:
        self._check_token(new_token)
        if not isinstance(kv_cache, Qwen25KVCache):
            raise TypeError("kv_cache must be a Qwen25KVCache")
        if not kv_cache.layers or len(kv_cache.layers) != len(self.model.model.layers):
            raise ValueError("cache has not been initialized for this model")
        return self._forward([new_token], kv_cache)
