from __future__ import annotations

import numpy as np


def plasticity_metrics(utility, selected) -> dict[str, float]:
    u = np.asarray(utility, dtype=float)
    m = np.asarray(selected, dtype=bool)
    harmful = u < 0
    beneficial = u > 0
    hur = float(harmful[m].mean()) if m.any() else float("nan")
    mbr = float(beneficial[~m].mean()) if (~m).any() else float("nan")
    regret = float((np.abs(u) * (m != beneficial)).sum())
    return {"harmful_update_rate": hur, "missed_beneficial_rate": mbr,
            "plasticity_regret": regret, "update_rate": float(m.mean())}


def paired_bootstrap(values_a, values_b, samples: int = 10_000, seed: int = 0):
    a, b = np.asarray(values_a), np.asarray(values_b)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have identical shapes")
    rng = np.random.default_rng(seed)
    differences = b - a
    means = np.array([rng.choice(differences, len(differences), replace=True).mean()
                      for _ in range(samples)])
    return float(differences.mean()), tuple(np.quantile(means, [0.025, 0.975]))

