# Phase 2 narrow decision experiment

This report records the intentionally truncated experiment requested under the
8-hour compute budget. We stopped the remaining K=5 seeds after K=5 seed1 and
ran one additional seed for the two decision-critical methods. Results are
greedy, 140 `valid_seen` ALFWorld tasks per seed, Qwen3.5-4B, max 16 new
tokens.

## Results

| Method | Seeds completed | SR by seed | Mean SR | Mean updates |
|---|---:|---:|---:|---:|
| ReAct | 0, 1 | 59.29%, 59.29% | 59.29% | 0.00 |
| aTTT K=5 | 0, 1 | 56.43%, 54.29% | 55.36% | 5.10 |
| aTTT K=12 | 0, 1 | **62.14%, 62.14%** | **62.14%** | 1.88 |
| SP-aTTT | 0, 1 | 61.43%, 59.29% | 60.36% | 2.09 |

The raw per-seed JSONL files are retained in the adjacent result directories.

## Paired decision comparisons

- SP-aTTT minus K=5: +7 percentage points in both completed seeds.
- SP-aTTT minus K=12: **−1 point** (seed0), **−4 points** (seed1), mean **−2 points**.
- Pooled across the two paired seeds, SP rescues 7 task outcomes relative to
  K=12 and loses 12.

## Decision

This is the current Go/No-Go case C: the simple sparse-cadence baseline
(`aTTT K=12`) is at least as good as, and in these two seeds better than,
SP-aTTT while using fewer updates. Therefore we stop the current gate study
and do not spend the remaining budget on Random/Novelty/Uncertainty
multi-seed or horizon ablations. The present evidence does **not** support the
claim that the learned gate improves over simply reducing update frequency.

This is a two-seed decision result, not a five-seed statistical claim; the
unfinished K=5 seeds are deliberately not treated as missing-at-random.
