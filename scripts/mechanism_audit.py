"""Audit what a frozen plasticity gate selects on held-out label features."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from sp_attt.gate import PlasticityGate


def auprc_positive(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    labels = labels[order].astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]) + recall[0] * precision[0])


def audit(features_path: str, gate_path: str, split: str) -> dict[str, float | int | str]:
    payload = torch.load(features_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(gate_path, map_location="cpu", weights_only=False)
    gate = PlasticityGate(int(checkpoint["hidden_size"]))
    gate.load_state_dict(checkpoint["state_dict"])
    gate.eval()
    with torch.inference_mode():
        normalized = gate(payload["hidden"], payload["scalars"]).numpy()
    predictions = float(checkpoint["target_mean"]) + float(checkpoint["target_std"]) * normalized
    target = payload["target"].numpy()
    indices = payload.get(f"{split}_idx") if split in {"train", "dev"} else None
    if indices is None:
        indices = np.arange(len(target))
    indices = np.asarray(indices, dtype=int)
    utility = target[indices]
    selected = predictions[indices] > 0.0
    harmful = utility < 0.0
    beneficial = utility > 0.0
    abs_total = float(np.abs(utility).sum())
    return {
        "split": split,
        "n": int(len(utility)),
        "selected": int(selected.sum()),
        "selected_rate": float(selected.mean()),
        "raw_harmful_rate": float(harmful.mean()),
        "raw_beneficial_rate": float(beneficial.mean()),
        "harmful_rate_selected": float(harmful[selected].mean()) if selected.any() else float("nan"),
        "harmful_rate_skipped": float(harmful[~selected].mean()) if (~selected).any() else float("nan"),
        "harmful_recall_skipped": float((~selected[harmful]).mean()) if harmful.any() else float("nan"),
        "mean_utility_selected": float(utility[selected].mean()) if selected.any() else float("nan"),
        "mean_utility_skipped": float(utility[~selected].mean()) if (~selected).any() else float("nan"),
        "absolute_harmful_share": float(np.abs(utility[harmful]).sum() / abs_total) if abs_total else 0.0,
        "auprc_harmful": auprc_positive(harmful, -predictions[indices]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--split", choices=["all", "train", "dev"], default="dev")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.features, args.gate, args.split)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
