import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from infer_runtime.adapters import Qwen3CoderDecoder


@pytest.fixture
def model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(7)
    config = Qwen3MoeConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    return Qwen3MoeForCausalLM(config).eval()


def test_prefill_decode_and_append_match_hugging_face(model: Qwen3MoeForCausalLM) -> None:
    decoder = Qwen3CoderDecoder(model)
    tokens = [4, 12, 20]

    with torch.inference_mode():
        cache, logits = decoder.prefill(tokens)
        reference = model(torch.tensor([tokens]), use_cache=False).logits[0, -1]
        torch.testing.assert_close(logits.tensor, reference, rtol=1e-5, atol=1e-5)

        logits = decoder.decode_step(9, cache)
        tokens.append(9)
        reference = model(torch.tensor([tokens]), use_cache=False).logits[0, -1]
        torch.testing.assert_close(logits.tensor, reference, rtol=1e-5, atol=1e-5)

        same_cache, logits = decoder.prefill([7, 13, 2], cache)
        assert same_cache is cache
        tokens.extend([7, 13, 2])
        reference = model(torch.tensor([tokens]), use_cache=False).logits[0, -1]
        torch.testing.assert_close(logits.tensor, reference, rtol=1e-5, atol=1e-5)

        logits = decoder.decode_step(8, cache)
        tokens.append(8)
        reference = model(torch.tensor([tokens]), use_cache=False).logits[0, -1]
        torch.testing.assert_close(logits.tensor, reference, rtol=1e-5, atol=1e-5)
        assert cache.sequence_length == len(tokens)


def test_adapter_does_not_call_hf_transformer_forward(
    model: Qwen3MoeForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_forward(*args: object, **kwargs: object) -> None:
        raise AssertionError("Hugging Face transformer forward was called")

    monkeypatch.setattr(model.model, "forward", fail_forward)
    decoder = Qwen3CoderDecoder(model)
    cache, _ = decoder.prefill([1, 2, 3])
    decoder.prefill([4, 5], cache)
    decoder.decode_step(6, cache)
