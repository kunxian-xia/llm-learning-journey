import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from infer_runtime.adapters.qwen25 import Qwen25Decoder


@pytest.fixture
def model() -> Qwen2ForCausalLM:
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        attention_dropout=0.0,
    )
    return Qwen2ForCausalLM(config).eval()


def test_cached_prefill_and_decode_match_full_forward(model: Qwen2ForCausalLM) -> None:
    decoder = Qwen25Decoder(model)
    tokens = [4, 12, 20, 7]

    with torch.inference_mode():
        cache, prefill_logits = decoder.prefill(tokens)
        reference = model(torch.tensor([tokens]), use_cache=False).logits[0, -1]
        torch.testing.assert_close(prefill_logits.tensor, reference, rtol=1e-5, atol=1e-5)
        assert cache.sequence_length == len(tokens)
        assert len(cache.layers) == model.config.num_hidden_layers
        assert cache.layers[0][0].shape[-2] == len(tokens)

        for token in (9, 13, 2):
            logits = decoder.decode_step(token, cache)
            tokens.append(token)
            reference = model(torch.tensor([tokens]), use_cache=False).logits[0, -1]
            torch.testing.assert_close(logits.tensor, reference, rtol=1e-5, atol=1e-5)
            assert cache.sequence_length == len(tokens)
            assert logits.tensor.shape == (decoder.vocab_size,)


def test_adapter_does_not_call_hf_transformer_forward(
    model: Qwen2ForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_forward(*args: object, **kwargs: object) -> None:
        raise AssertionError("Hugging Face transformer forward was called")

    monkeypatch.setattr(model.model, "forward", fail_forward)
    decoder = Qwen25Decoder(model)
    cache, _ = decoder.prefill([1, 2, 3])
    decoder.decode_step(4, cache)


def test_context_boundary_and_fresh_cache(model: Qwen2ForCausalLM) -> None:
    decoder = Qwen25Decoder(model)
    first_cache, _ = decoder.prefill([1, 2])
    second_cache, _ = decoder.prefill([3])
    decoder.decode_step(4, first_cache)
    assert first_cache.sequence_length == 3
    assert second_cache.sequence_length == 1

    full_cache, _ = decoder.prefill([1] * decoder.context_length)
    with pytest.raises(ValueError, match="context length"):
        decoder.decode_step(2, full_cache)


@pytest.mark.parametrize("prompt", [[], [97], [-1]])
def test_invalid_prompt(model: Qwen2ForCausalLM, prompt: list[int]) -> None:
    with pytest.raises(ValueError):
        Qwen25Decoder(model).prefill(prompt)
