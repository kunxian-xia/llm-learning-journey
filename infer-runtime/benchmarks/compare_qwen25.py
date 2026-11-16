"""Compare custom cache, Transformers cache, and full-prefix forward speed.

Run from the project directory after installing the package:
  python benchmarks/compare_qwen25.py --model Qwen/Qwen2.5-0.5B
"""

import argparse
import json
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM

from infer_runtime.adapters import Qwen25Decoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    adapter = Qwen25Decoder(model)
    if args.prompt_length + args.decode_steps > adapter.context_length:
        parser.error("prompt and decode steps exceed context length")
    generator = torch.Generator().manual_seed(2026)
    prompt = torch.randint(0, adapter.vocab_size, (args.prompt_length,), generator=generator).tolist()
    continuation = torch.randint(0, adapter.vocab_size, (args.decode_steps,), generator=generator).tolist()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def run(path: str) -> tuple[float, float]:
        sync()
        start = time.perf_counter()
        if path == "custom_cache":
            cache, _ = adapter.prefill(prompt)
        else:
            output = model(
                input_ids=torch.tensor([prompt], device=device),
                use_cache=path == "hf_cache",
                logits_to_keep=1,
            )
            cache = output.past_key_values
        sync()
        after_prefill = time.perf_counter()

        prefix = list(prompt)
        for token in continuation:
            if path == "custom_cache":
                adapter.decode_step(token, cache)
            elif path == "hf_cache":
                output = model(
                    input_ids=torch.tensor([[token]], device=device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.past_key_values
            else:
                prefix.append(token)
                model(input_ids=torch.tensor([prefix], device=device), use_cache=False, logits_to_keep=1)
        sync()
        end = time.perf_counter()
        return after_prefill - start, end - after_prefill

    paths = ("custom_cache", "hf_cache", "hf_full_prefix")
    for _ in range(args.warmups):
        for path in paths:
            run(path)  # warm up kernels and allocations
    results = {path: [] for path in paths}
    for _ in range(args.repeats):
        for path in paths:
            results[path].append(run(path))

    print(json.dumps({
        "model": args.model,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "prompt_tokens": args.prompt_length,
        "decode_steps": args.decode_steps,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "results": {
            path: {
                "prefill_ms_median": round(statistics.median(x[0] for x in timings) * 1000, 2),
                "decode_ms_median": round(statistics.median(x[1] for x in timings) * 1000, 2),
                "decode_tokens_per_second": round(args.decode_steps / statistics.median(x[1] for x in timings), 2),
            }
            for path, timings in results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
