"""Run the reproducible Qwen3.5-4B ALFWorld baseline matrix.

Each method/seed is an independent process-level run from the same task order.  The
per-episode JSONL files are retained so paired bootstrap and McNemar tests can be
performed without reconstructing trajectories.
"""

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from sp_attt.alfworld_runner import run_alfworld

parser = argparse.ArgumentParser()
parser.add_argument("--methods", nargs="+", default=["react", "ttt", "attt"],
                    choices=["react", "ttt", "attt"])
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
parser.add_argument("--episodes", type=int, default=140)
parser.add_argument("--split", default="valid_seen")
parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
parser.add_argument("--config", default="configs/alfworld.yaml")
parser.add_argument("--max-steps", type=int, default=50)
parser.add_argument("--max-new-tokens", type=int, default=16)
parser.add_argument("--output-dir", default="results/alfworld_baselines")
parser.add_argument("--resume", action="store_true",
                    help="Reuse complete per-seed JSONL files instead of rerunning them")
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
summary = []
for seed in args.seeds:
    for method in args.methods:
        path = output_dir / f"{args.split}_{method}_seed{seed}.jsonl"
        if args.resume and path.exists():
            with path.open(encoding="utf-8") as stream:
                cached = [json.loads(line) for line in stream if line.strip()]
            if len(cached) >= args.episodes:
                summary.append({
                    "split": args.split, "method": method, "seed": seed,
                    "episodes": len(cached),
                    "success_rate": sum(row["success"] for row in cached) / len(cached),
                    "mean_steps": sum(row["steps"] for row in cached) / len(cached),
                    "mean_updates": sum(row["updates"] for row in cached) / len(cached),
                    "output": str(path), "resumed": True,
                })
                continue
        rows = run_alfworld(method, model_name=args.model, config_path=args.config,
                            episodes=args.episodes, seed=seed, split=args.split,
                            max_steps=args.max_steps, max_new_tokens=args.max_new_tokens)
        with path.open("w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(asdict(row)) + "\n" for row in rows)
        summary.append({
            "split": args.split, "method": method, "seed": seed,
            "episodes": len(rows),
            "success_rate": sum(row.success for row in rows) / len(rows),
            "mean_steps": sum(row.steps for row in rows) / len(rows),
            "mean_updates": sum(row.updates for row in rows) / len(rows),
            "output": str(path),
        })
        # Each method constructs a fresh Qwen model.  Explicitly collect the
        # previous method before the next iteration so a long matrix does not
        # retain its CUDA allocations and fail with an avoidable OOM.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

with (output_dir / f"{args.split}_summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps(summary, indent=2))
