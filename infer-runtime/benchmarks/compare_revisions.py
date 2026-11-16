"""Reproducible single-GPU Qwen2.5 benchmark and PR regression report."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from infer_runtime.adapters import Qwen25Decoder


def load_base_decoder(path: Path):
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("benchmark_base_qwen25", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import baseline adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen25Decoder


def median_ms(values: list[float]) -> float:
    return round(1000 * statistics.median(values), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 4096])
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-attempt", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("this regression benchmark requires a CUDA GPU")
    if min(args.prompt_lengths) < 1 or args.decode_steps < 1 or args.repeats < 1:
        parser.error("prompt lengths, decode steps, and repeats must be positive")

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()
    head = Qwen25Decoder(model)
    base_type = load_base_decoder(args.base_source) if args.base_source else None
    base = base_type(model) if base_type else None
    if max(args.prompt_lengths) + args.decode_steps > head.context_length:
        parser.error("benchmark exceeds model context length")

    def sync() -> None:
        torch.cuda.synchronize()

    @torch.inference_mode()
    def run(variant: str, adapter, prompt: list[int], continuation: list[int]) -> tuple[float, float]:
        sync()
        start = time.perf_counter()
        if variant == "hf_cache":
            output = model(
                input_ids=torch.tensor([prompt], device=device), use_cache=True, logits_to_keep=1
            )
            cache = output.past_key_values
        else:
            cache, _ = adapter.prefill(prompt)
        sync()
        prefill_end = time.perf_counter()
        for token in continuation:
            if variant == "hf_cache":
                output = model(
                    input_ids=torch.tensor([[token]], device=device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.past_key_values
            else:
                adapter.decode_step(token, cache)
        sync()
        end = time.perf_counter()
        return prefill_end - start, end - prefill_end

    workloads = []
    failed = False
    for prompt_length in args.prompt_lengths:
        generator = torch.Generator().manual_seed(2026 + prompt_length)
        prompt = torch.randint(0, head.vocab_size, (prompt_length,), generator=generator).tolist()
        continuation = torch.randint(0, head.vocab_size, (args.decode_steps,), generator=generator).tolist()
        adapters = {"head": head, "hf_cache": None}
        if base is not None:
            adapters["base"] = base
        variants = list(adapters)
        for _ in range(args.warmups):
            for variant in variants:
                run(variant, adapters[variant], prompt, continuation)

        timings = {variant: [] for variant in variants}
        for repeat in range(args.repeats):
            # Rotate order so a consistently first/last variant does not own the clock drift.
            order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
            for variant in order:
                timings[variant].append(run(variant, adapters[variant], prompt, continuation))

        results = {}
        for variant, samples in timings.items():
            prefill = median_ms([sample[0] for sample in samples])
            decode = median_ms([sample[1] for sample in samples])
            results[variant] = {
                "prefill_ms": prefill,
                "decode_ms": decode,
                "decode_tokens_per_second": round(args.decode_steps * 1000 / decode, 2),
            }
        reference = results["base" if base is not None else "hf_cache"]
        regression = {
            "prefill": results["head"]["prefill_ms"] > reference["prefill_ms"] * 1.30,
            "decode": results["head"]["decode_ms"] > reference["decode_ms"] * 1.20,
        }
        failed |= any(regression.values())
        workloads.append({
            "prompt_tokens": prompt_length,
            "decode_steps": args.decode_steps,
            "results": results,
            "reference": "base" if base is not None else "hf_cache",
            "regression": regression,
        })

    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "head_sha": args.head_sha,
        "base_sha": args.base_sha,
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
        "model": args.model,
        "revision": args.revision,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": "float16",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "failed": failed,
        "workloads": workloads,
    }
    lines = [
        "<!-- inference-serving-benchmark -->",
        "### Qwen2.5 inference benchmark",
        "",
        f"**{'Regression detected' if failed else 'Pass'}** · {args.model} · {report['gpu']} · FP16 · "
        f"{args.warmups} warmups / {args.repeats} measured runs",
        "",
        "| Prompt | Path | Prefill (ms) | Decode (tok/s) |",
        "| ---: | --- | ---: | ---: |",
    ]
    for workload in workloads:
        for variant, result in workload["results"].items():
            lines.append(
                f"| {workload['prompt_tokens']} | {variant} | {result['prefill_ms']:.2f} | "
                f"{result['decode_tokens_per_second']:.2f} |"
            )
        if any(workload["regression"].values()):
            metrics = ", ".join(key for key, value in workload["regression"].items() if value)
            lines.append(f"\nRegression at {workload['prompt_tokens']} prompt tokens: {metrics}.")
    lines += [
        "",
        "Gate: head prefill must be within 30% and decode time within 20% of "
        + ("the base branch." if base is not None else "Hugging Face cached decoding (base adapter unavailable)."),
        "Fixed token IDs, batch size 1; excludes tokenization and sampling.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "comment.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
