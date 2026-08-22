import argparse
import json

import numpy as np

from sp_attt.metrics import plasticity_metrics

parser = argparse.ArgumentParser()
parser.add_argument("labels", help="JSONL rows containing utility and optional selected")
args = parser.parse_args()
with open(args.labels, encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
u = np.asarray([row["utility"] for row in rows])
summary = {"n": len(u), "p_harmful": float((u < 0).mean()), "mean": float(u.mean()),
           "median": float(np.median(u)), "q05_q95": np.quantile(u, [.05, .95]).tolist()}
if rows and "selected" in rows[0]:
    summary.update(plasticity_metrics(u, [row["selected"] for row in rows]))
print(json.dumps(summary, indent=2))
