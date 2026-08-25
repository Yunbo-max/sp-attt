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
                    choices=["react", "ttt", "attt", "sp", "random_matched",
                             "novelty_matched", "uncertainty_matched", "attt_k10", "attt_k12"])
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
parser.add_argument("--gate", help="PlasticityGate checkpoint required for --methods sp")
parser.add_argument("--matched-rate", type=float, default=0.411,
                    help="Fraction of reference candidate checkpoints selected by matched methods")
parser.add_argument("--budget-reference",
                    default="results/alfworld_baselines/valid_seen_attt_seed{seed}.jsonl",
                    help="aTTT JSONL whose per-episode update counts define matched budgets; {seed} is substituted")
parser.add_argument("--novelty-threshold", type=float, default=0.7197179794311523)
parser.add_argument("--uncertainty-threshold", type=float, default=3.6504969596862793)
args = parser.parse_args()


def reference_candidate_counts(path: str, episodes: int) -> list[int] | None:
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < episodes:
        raise ValueError(f"budget reference has {len(rows)} episodes, need {episodes}: {path}")
    return [int(row["updates"]) for row in rows[:episodes]]


output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
summary = []
for seed in args.seeds:
    matched_counts = None
    if any(method.endswith("_matched") for method in args.methods):
        reference_path = args.budget_reference.format(seed=seed)
        matched_counts = reference_candidate_counts(reference_path, args.episodes)
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
        effective_cadence = 10 if method == "attt_k10" else 12 if method == "attt_k12" else 5
        runner_method = "attt" if method in {"attt_k10", "attt_k12"} else method
        rows = run_alfworld(runner_method, model_name=args.model, config_path=args.config,
                            episodes=args.episodes, seed=seed, split=args.split,
                            max_steps=args.max_steps, candidate_every=effective_cadence,
                            max_new_tokens=args.max_new_tokens, gate_path=args.gate,
                            matched_candidate_counts=matched_counts,
                            matched_rate=args.matched_rate,
                            novelty_threshold=args.novelty_threshold,
                            uncertainty_threshold=args.uncertainty_threshold)
        for row in rows:
            row.method = method
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
