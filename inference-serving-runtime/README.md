# Inference serving runtime

A standalone toy project for experimenting with an inference serving runtime.

Start with the [single-request API contract](API.md) for issue #49.

## Qwen2.5 prefill and decode adapter

`Qwen25Decoder` takes an already-loaded Hugging Face `Qwen2ForCausalLM` for its weights and non-attention layers. It computes causal attention and maintains its own per-layer KV cache without calling the Hugging Face transformer forward method. `prefill(prompt_ids)` and `decode_step(new_token, kv_cache)` return last-position next-token logits; the cache advances in place. Qwen2.5 text checkpoints use this model architecture. This first version handles one sequence on one device and full-attention configurations.

```python
from transformers import AutoModelForCausalLM
from inference_serving_runtime.adapters import Qwen25Decoder

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
decoder = Qwen25Decoder(model)
cache, logits = decoder.prefill([1, 2, 3])  # use IDs from the matching tokenizer
logits = decoder.decode_step(4, cache)
```

Install the project with its `test` extra and run `pytest` from this directory. The correctness tests initialize a tiny Qwen2 model locally and compare cached logits against uncached Hugging Face forward passes; they do not download checkpoint weights.
