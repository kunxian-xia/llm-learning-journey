"""Benchmark our Qwen3-Coder adapter with the real 30B-A3B AWQ checkpoint.

Install the optional loader first: ``pip install autoawq==0.2.9 accelerate transformers==4.52.3``.
Example: ``python benchmarks/compare_qwen3_coder_awq.py --model QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ``
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from infer_runtime.adapters import Qwen3CoderDecoder


def loader_directory(checkpoint: Path, destination: Path) -> Path:
    """Work around AutoAWQ's relative-name match for the unquantized MoE router."""
    config = json.loads((checkpoint / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if config.get("model_type") != "qwen3_moe" or quant.get("quant_method") != "awq":
        raise ValueError("expected a Qwen3 MoE AWQ checkpoint")
    for file in checkpoint.iterdir():
        if file.name != "config.json":
            (destination / file.name).symlink_to(file.resolve())
    # AutoAWQ enumerates names relative to each layer: "mlp.gate", not ".mlp.gate".
    quant["modules_to_not_convert"] = [
        name.removeprefix(".") for name in quant.get("modules_to_not_convert", [])
    ]
    (destination / "config.json").write_text(json.dumps(config))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ")
    parser.add_argument("--revision", default="c58857a7f41c0920f73d1b56678640f9c02017d7")
    parser.add_argument("--prompt", default="Explain how a key value cache speeds up autoregressive decoding.")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this benchmark requires a CUDA GPU")
    if args.decode_steps < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("decode-steps and repeats must be positive; warmups must be nonnegative")

    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:
        parser.error(f"AutoAWQ is required: {exc}")

    checkpoint = Path(args.model)
    if not checkpoint.is_dir():
        checkpoint = Path(snapshot_download(args.model, revision=args.revision))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt:
        parser.error("prompt must produce at least one token")
    continuation = [tokenizer.encode(" cache", add_special_tokens=False)[0]] * args.decode_steps

    with tempfile.TemporaryDirectory(prefix="qwen3-awq-loader-") as temporary:
        load_path = loader_directory(checkpoint, Path(temporary))
        model = AutoAWQForCausalLM.from_quantized(
            str(load_path), device_map="auto", fuse_layers=False,
            safetensors=True, torch_dtype=torch.float16,
        ).model.eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("checkpoint has unloaded model parameters")
    if not all(parameter.device.type == "cuda" for parameter in model.parameters()):
        raise RuntimeError("the adapter requires the complete model on one CUDA GPU")
    adapter = Qwen3CoderDecoder(model)
    if len(prompt) + args.decode_steps > adapter.context_length:
        parser.error("prompt and decode steps exceed context length")
    device = next(model.parameters()).device

    @torch.inference_mode()
    def run(path: str) -> tuple[float, float, torch.Tensor, torch.Tensor]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        if path == "adapter":
            cache, logits = adapter.prefill(prompt)
            first_logits = logits.tensor.float()
        else:
            output = model(
                input_ids=torch.tensor([prompt], device=device),
                use_cache=True, logits_to_keep=1,
            )
            cache = output.past_key_values
            first_logits = output.logits[0, -1].float()
        torch.cuda.synchronize()
        after_prefill = time.perf_counter()
        for token in continuation:
            if path == "adapter":
                last_logits = adapter.decode_step(token, cache).tensor.float()
            else:
                output = model(
                    input_ids=torch.tensor([[token]], device=device),
                    past_key_values=cache, use_cache=True, logits_to_keep=1,
                )
                cache = output.past_key_values
                last_logits = output.logits[0, -1].float()
        torch.cuda.synchronize()
        return after_prefill - start, time.perf_counter() - after_prefill, first_logits, last_logits

    paths = ("adapter", "transformers_cache")
    for _ in range(args.warmups):
        for path in paths:
            run(path)
    timings = {path: [] for path in paths}
    logits = {}
    for _ in range(args.repeats):
        for path in paths:
            prefill, decode, first_logits, last_logits = run(path)
            timings[path].append((prefill, decode))
            logits[path] = (first_logits, last_logits)

    print(json.dumps({
        "checkpoint": str(checkpoint),
        "revision": args.revision if not Path(args.model).is_dir() else None,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "prompt_tokens": len(prompt),
        "decode_steps": args.decode_steps,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "prefill_logit_max_abs_diff": float((logits["adapter"][0] - logits["transformers_cache"][0]).abs().max()),
        "prefill_top_token_matches": bool(logits["adapter"][0].argmax() == logits["transformers_cache"][0].argmax()),
        "decode_logit_max_abs_diff": float((logits["adapter"][1] - logits["transformers_cache"][1]).abs().max()),
        "decode_top_token_matches": bool(logits["adapter"][1].argmax() == logits["transformers_cache"][1].argmax()),
        "results": {
            path: {
                "prefill_ms_median": round(statistics.median(x[0] for x in samples) * 1000, 2),
                "decode_ms_median": round(statistics.median(x[1] for x in samples) * 1000, 2),
                "decode_tokens_per_second": round(args.decode_steps / statistics.median(x[1] for x in samples), 2),
            }
            for path, samples in timings.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
