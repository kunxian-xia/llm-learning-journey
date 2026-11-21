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

## Qwen3-Coder MoE adapter

`Qwen3CoderDecoder` accepts an already-loaded `Qwen3MoeForCausalLM` on one device, including the architecture of `Qwen/Qwen3-Coder-30B-A3B-Instruct`. It uses the model's Q/K normalization and routed expert MLPs while maintaining its own full-attention KV cache. It has the same `prefill(input_ids, kv_cache=None)` and `decode_step(new_token, kv_cache)` interface. Sliding-window configurations are rejected.

```python
from infer_runtime.adapters import Qwen3CoderDecoder

# model: Qwen3MoeForCausalLM, already loaded on one device
decoder = Qwen3CoderDecoder(model)
cache, logits = decoder.prefill([1, 2, 3])  # use IDs from the matching tokenizer
cache, logits = decoder.prefill([4, 5], cache)
```

Correctness tests use a small local Qwen3 MoE configuration and compare its logits with Hugging Face full forward passes; they do not download the 30B checkpoint.

To measure the real 4-bit checkpoint, install this package, `autoawq==0.2.9`, `accelerate`, and `transformers==4.52.3` in a separate Python environment with CUDA PyTorch, then run:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python benchmarks/compare_qwen3_coder_awq.py
```

The benchmark uses a pinned revision of `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` and runs both `Qwen3CoderDecoder` and Hugging Face cached forward on its actual weights. It reports prefill and decode latency, logit differences, and whether the top token matches. The AWQ MoE path can amplify small floating-point differences, so inspect those accuracy fields alongside throughput. AutoAWQ misinterprets the checkpoint's `.mlp.gate` exclusion as a layer-relative name, so the benchmark corrects that config entry in a temporary directory of symlinks; the checkpoint files remain untouched. This benchmark is separate from the Qwen2.5 CI regression gate.

## Benchmark CI

The [inference benchmark workflow](../.github/workflows/inference-benchmark.yml) measures a pinned Qwen2.5-0.5B checkpoint on the self-hosted GPU runner for same-repository PRs. It compares the PR adapter with the base branch adapter on the same runner, and also records Hugging Face cached decoding. While a base branch has no adapter, the Hugging Face path is the reference. For each 128- and 4096-token prompt, a PR fails if median decode time rises by more than 20% or prefill time by more than 30%. Results use fixed token IDs, 32 decode steps, two warmups, five measured runs, and FP16. The workflow posts or updates one PR comment even when the regression gate fails.
