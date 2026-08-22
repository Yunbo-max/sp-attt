from __future__ import annotations

import argparse
import json

from .metrics import plasticity_metrics


def main() -> None:
    parser = argparse.ArgumentParser(prog="sp-attt")
    sub = parser.add_subparsers(dest="command", required=True)
    metrics = sub.add_parser("metrics", help="compute plasticity metrics from JSONL")
    metrics.add_argument("path")
    args = parser.parse_args()
    if args.command == "metrics":
        with open(args.path, encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        print(json.dumps(plasticity_metrics([r["utility"] for r in rows],
                                            [r["selected"] for r in rows]), indent=2))
