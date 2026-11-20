# Infer runtime

A standalone toy project for experimenting with an inference serving runtime.

Start with the [single-request API contract](API.md) for issue #49.

## Qwen2.5 prefill and decode adapter

`Qwen25Decoder` takes an already-loaded Hugging Face `Qwen2ForCausalLM` for its weights and non-attention layers. It computes causal attention and maintains its own per-layer KV cache without calling the Hugging Face transformer forward method. `prefill(input_ids)` starts a cache; `prefill(input_ids, kv_cache)` appends a batch of input tokens to it; `decode_step(new_token, kv_cache)` consumes one generated token. All return last-position next-token logits, and existing caches advance in place. Qwen2.5 text checkpoints use this model architecture. This first version handles one sequence on one device and full-attention configurations.

```python
from transformers import AutoModelForCausalLM
from infer_runtime.adapters import Qwen25Decoder

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
decoder = Qwen25Decoder(model)
cache, logits = decoder.prefill([1, 2, 3])  # use IDs from the matching tokenizer
logits = decoder.decode_step(4, cache)
```

Install the project with its `test` extra and run `pytest` from this directory. The correctness tests initialize a tiny Qwen2 model locally and compare cached logits against uncached Hugging Face forward passes; they do not download checkpoint weights.

## Benchmark CI

The [inference benchmark workflow](../.github/workflows/inference-benchmark.yml) measures a pinned Qwen2.5-0.5B checkpoint on the self-hosted GPU runner for same-repository PRs. It compares the PR adapter with the base branch adapter on the same runner, and also records Hugging Face cached decoding. While a base branch has no adapter, the Hugging Face path is the reference. For each 128- and 4096-token prompt, a PR fails if median decode time rises by more than 20% or prefill time by more than 30%. Results use fixed token IDs, 32 decode steps, two warmups, five measured runs, and FP16. The workflow posts or updates one PR comment even when the regression gate fails.
