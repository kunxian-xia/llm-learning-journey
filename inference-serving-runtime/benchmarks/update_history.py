"""Append a completed benchmark run to the GitHub Pages history."""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    if result["failed"]:
        parser.error("refusing to publish a failed benchmark")
    target = args.pages_dir / "inference-serving-runtime"
    target.mkdir(parents=True, exist_ok=True)
    history_path = target / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history = [
        item for item in history
        if (item.get("run_id"), item.get("run_attempt"))
        != (result["run_id"], result["run_attempt"])
    ]
    history.append(result)
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    shutil.copyfile(Path(__file__).parent / "site" / "index.html", target / "index.html")
    (args.pages_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Benchmark history</title>'
        '<a href="inference-serving-runtime/">Inference serving benchmarks</a>\n'
    )
    (args.pages_dir / ".nojekyll").touch()


if __name__ == "__main__":
    main()
