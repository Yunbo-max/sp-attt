"""Extract frozen Qwen features and train the value gate from paired labels.

This deliberately refuses to train on rows without ``history_observations``: those
rows predate the feature-schema fix and cannot reproduce observation novelty.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sp_attt.gate import PlasticityGate
from sp_attt.novelty import sequence_novelty
from sp_attt.training import train_gate


def _actions(history: list[str]) -> list[str]:
    result = []
    for text in history:
        match = re.search(r"Action:\s*(.*)$", text, flags=re.MULTILINE)
        if match:
            result.append(match.group(1).strip().lower())
    return result


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels.astype(bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _encode(model, tokenizer, text: str, device: torch.device):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=False,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = output.hidden_states[-1][0]
    mask = encoded.get("attention_mask", torch.ones(hidden.shape[:-1], device=device))[0].bool()
    pooled = hidden[mask].float().mean(dim=0).cpu()
    logits = output.logits[0, :-1].float()
    labels = encoded["input_ids"][0, 1:]
    nll = float(F.cross_entropy(logits, labels).item()) if len(labels) else 0.0
    return pooled, nll


def extract_rows(rows, model_name: str, device: str):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    target_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if target_device.type == "cuda" else torch.float32,
        device_map=str(target_device) if target_device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    hidden_rows, scalar_rows, targets, ids = [], [], [], []
    for row in rows:
        candidate = row["candidate_text"]
        history_texts = row.get("history_update_texts", [])
        history_obs = row.get("history_observations", [])
        hidden, nll = _encode(model, tokenizer, candidate, target_device)
        actions = _actions(history_texts)
        action = row.get("candidate_action", "").strip().lower()
        scalars = [
            sequence_novelty(candidate, history_texts),
            nll,
            float(bool(actions and action == actions[-1])),
            float(bool(actions and action in actions[-3:])),
            sequence_novelty(row.get("candidate_observation", ""), history_obs),
            float(row.get("relative_position", 0.0)),
        ]
        hidden_rows.append(hidden)
        scalar_rows.append(scalars)
        targets.append(float(row["utility"]))
        ids.append(row["label_id"])
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.stack(hidden_rows), torch.tensor(scalar_rows), torch.tensor(targets), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--features", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    with open(args.labels, encoding="utf-8") as stream:
        all_rows = [json.loads(line) for line in stream if line.strip()]
    rows = [row for row in all_rows if row.get("history_observations") is not None]
    if len(rows) < 20:
        raise RuntimeError(f"Only {len(rows)} rows have complete gate features; collect more labels first")
    hidden, scalars, target, ids = extract_rows(rows, args.model, args.device)
    order = list(range(len(rows))); random.Random(args.seed).shuffle(order)
    n_dev = max(1, int(len(order) * args.dev_fraction))
    dev_idx, train_idx = order[:n_dev], order[n_dev:]
    train_target = target[train_idx]
    mean, std = train_target.mean(), train_target.std().clamp_min(1e-6)
    target_norm = (target - mean) / std
    train = torch.utils.data.TensorDataset(hidden[train_idx], scalars[train_idx], target_norm[train_idx])
    dev = torch.utils.data.TensorDataset(hidden[dev_idx], scalars[dev_idx], target_norm[dev_idx])
    gate = PlasticityGate(hidden.shape[-1])
    result = train_gate(gate, train, dev, device="cpu")
    with torch.inference_mode():
        prediction = gate(hidden, scalars).cpu().numpy() * float(std) + float(mean)
    truth = target.numpy()
    metrics = {
        "n_complete": len(rows), "n_train": len(train_idx), "n_dev": len(dev_idx),
        "best_dev_loss_normalized": result.best_dev_loss, "epochs": result.epochs,
        "r2": float(1 - ((prediction - truth) ** 2).sum() / max(1e-12, ((truth - truth.mean()) ** 2).sum())),
        "spearman": _spearman(prediction, truth),
        "sign_accuracy": float(np.mean((prediction > 0) == (truth > 0))),
        "auroc_positive": _auroc(truth > 0, prediction),
        "predicted_update_rate": float((prediction > 0).mean()),
        "target_mean": float(mean), "target_std": float(std),
    }
    torch.save({"state_dict": gate.state_dict(), "hidden_size": hidden.shape[-1],
                "target_mean": mean, "target_std": std, "metrics": metrics}, args.gate)
    torch.save({"hidden": hidden, "scalars": scalars, "target": target, "ids": ids,
                "train_idx": train_idx, "dev_idx": dev_idx}, args.features)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
