"""Benchmark the three attention implementations on one CUDA GPU.

Each configuration runs in a child process so an OOM does not contaminate the
next measurement. Results are appended to CSV after every configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from cs336_basics.flashattention import FlashAttentionTorch, FlashAttentionTriton
from cs336_basics.model import Attention


IMPLEMENTATIONS = ("model", "flash_torch", "flash_triton")
DEFAULT_LENGTHS = (256, 1024, 4096, 8192, 16384)
DEFAULT_DIMS = (16, 32, 64, 128)
FIELDS = (
    "implementation",
    "causal",
    "sequence_length",
    "dimension",
    "status",
    "forward_ms",
    "backward_ms",
    "allocated_before_backward_mib",
    "saved_after_forward_mib",
    "peak_forward_mib",
    "repetitions",
    "gpu",
    "error",
)


def measure_one(
    implementation: str,
    sequence_length: int,
    dimension: int,
    repetitions: int,
    warmup: int,
    causal: bool = False,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    device = torch.device("cuda:0")
    shape = (8, sequence_length, dimension)
    q = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn(shape, device=device, dtype=torch.float32)

    if implementation == "model":
        attention = Attention()
        mask = (
            torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool).tril()
            if causal
            else torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool)
        )

        def forward():
            return attention(q, k, v, mask)

    elif implementation == "flash_torch":
        if causal:
            raise ValueError("FlashAttentionTorch does not implement causal attention")

        def forward():
            return FlashAttentionTorch.apply(q, k, v, False)

    elif implementation == "flash_triton":

        def forward():
            return FlashAttentionTriton.apply(q, k, v, causal)

    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    for _ in range(warmup):
        output = forward()
        torch.cuda.synchronize()
        output.backward(grad_output)
        torch.cuda.synchronize()
        del output
        q.grad = k.grad = v.grad = None

    forward_times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = forward()
        torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)
        del output

    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = forward()
    torch.cuda.synchronize()
    before_backward = torch.cuda.memory_allocated(device)
    peak_forward = torch.cuda.max_memory_allocated(device)
    del output

    backward_times = []
    for _ in range(repetitions):
        output = forward()
        torch.cuda.synchronize()
        start = time.perf_counter()
        output.backward(grad_output)
        torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)
        del output
        q.grad = k.grad = v.grad = None

    mib = 1024 * 1024
    return {
        "implementation": implementation,
        "causal": causal,
        "sequence_length": sequence_length,
        "dimension": dimension,
        "status": "ok",
        "forward_ms": round(1000 * sum(forward_times) / repetitions, 4),
        "backward_ms": round(1000 * sum(backward_times) / repetitions, 4),
        "allocated_before_backward_mib": round(before_backward / mib, 2),
        "saved_after_forward_mib": round((before_backward - baseline) / mib, 2),
        "peak_forward_mib": round(peak_forward / mib, 2),
        "repetitions": repetitions,
        "gpu": torch.cuda.get_device_name(device),
        "error": "",
    }


def worker(args: argparse.Namespace) -> None:
    try:
        result = measure_one(args.impl, args.seq, args.dim, args.repetitions, args.warmup, args.causal)
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "error"
        result = {
            "implementation": args.impl,
            "causal": args.causal,
            "sequence_length": args.seq,
            "dimension": args.dim,
            "status": status,
            "repetitions": args.repetitions,
            "error": str(exc).splitlines()[0],
        }
    print("RESULT_JSON=" + json.dumps(result), flush=True)


def sweep(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for implementation in args.implementations:
            for sequence_length in args.lengths:
                for dimension in args.dims:
                    command = [
                        sys.executable, str(Path(__file__).resolve()), "--worker",
                        "--impl", implementation,
                        "--seq", str(sequence_length),
                        "--dim", str(dimension),
                        "--repetitions", str(args.repetitions),
                        "--warmup", str(args.warmup),
                    ]
                    if args.causal:
                        command.append("--causal")
                    try:
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout_seconds)
                        line = next((line for line in completed.stdout.splitlines() if line.startswith("RESULT_JSON=")), None)
                        if line is None:
                            result = {"status": "error", "error": (completed.stderr or completed.stdout)[-500:]}
                        else:
                            result = json.loads(line.removeprefix("RESULT_JSON="))
                    except subprocess.TimeoutExpired:
                        result = {"status": "timeout", "error": f"Exceeded {args.timeout_seconds} seconds"}
                    result.setdefault("implementation", implementation)
                    result.setdefault("causal", args.causal)
                    result.setdefault("sequence_length", sequence_length)
                    result.setdefault("dimension", dimension)
                    result.setdefault("repetitions", args.repetitions)
                    writer.writerow({field: result.get(field, "") for field in FIELDS})
                    output_file.flush()
                    print(f"{implementation:13} N={sequence_length:5} D={dimension:3} {result['status']:7} "
                          f"forward={result.get('forward_ms', '')} ms backward={result.get('backward_ms', '')} ms", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--impl", choices=IMPLEMENTATIONS)
    parser.add_argument("--seq", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=("model", "flash_triton"))
    parser.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--causal", action="store_true", help="Apply a lower-triangular attention mask")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("attention_results.csv"))
    args = parser.parse_args()
    if args.repetitions <= 0 or args.warmup < 0:
        parser.error("repetitions must be positive and warmup must be nonnegative")
    if args.causal and (args.impl == "flash_torch" or (not args.worker and "flash_torch" in args.implementations)):
        parser.error("FlashAttentionTorch does not implement causal attention")
    if args.worker:
        if args.impl is None or args.seq is None or args.dim is None:
            parser.error("--worker requires --impl, --seq, and --dim")
        worker(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
