import argparse
import json
from dataclasses import asdict

from sp_attt.alfworld_runner import run_alfworld

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["react", "ttt", "attt"], required=True)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
parser.add_argument("--config", default="configs/alfworld.yaml")
parser.add_argument("--split", default="valid_seen")
parser.add_argument("--output", required=True)
parser.add_argument("--max-steps", type=int, default=50)
parser.add_argument("--max-new-tokens", type=int, default=16)
args = parser.parse_args()
rows = run_alfworld(args.method, model_name=args.model, config_path=args.config,
                    episodes=args.episodes, seed=args.seed, split=args.split,
                    max_steps=args.max_steps, max_new_tokens=args.max_new_tokens)
with open(args.output, "w", encoding="utf-8") as stream:
    stream.writelines(json.dumps(asdict(row)) + "\n" for row in rows)
print(json.dumps({"method": args.method, "episodes": len(rows),
                  "success_rate": sum(row.success for row in rows) / len(rows),
                  "mean_steps": sum(row.steps for row in rows) / len(rows),
                  "mean_updates": sum(row.updates for row in rows) / len(rows)}, indent=2))
