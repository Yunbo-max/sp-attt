# Phase 2A: matched-budget and cadence controls

Configuration: Qwen3.5-4B, ALFWorld `valid_seen`, 140 tasks, seed 0, greedy
`max_new_tokens=16`, episode-local LoRA rank 8 / alpha 16. Matched methods use a frozen
0.411 selection rate and the per-episode candidate counts from the completed aTTT run.

| Method | Success | Mean steps | Mean updates |
|---|---:|---:|---:|
| ReAct | 83/140 (59.29%) | 28.48 | 0.00 |
| Vanilla TTT | 77/140 (55.00%) | 29.61 | 5.13 |
| aTTT K=5 | 79/140 (56.43%) | 29.29 | 5.06 |
| Random-Matched | 81/140 (57.86%) | 28.90 | 1.97 |
| Novelty-Matched | 84/140 (60.00%) | 28.40 | 1.96 |
| Uncertainty-Matched | 84/140 (60.00%) | 28.84 | 1.94 |
| aTTT K=10 | 85/140 (60.71%) | 28.41 | 2.09 |
| aTTT K=12 | **87/140 (62.14%)** | **28.16** | **1.89** |
| SP-aTTT (500-label gate) | 86/140 (61.43%) | 28.24 | 2.08 |

These are single-seed development results. The K=12 control currently matches or exceeds SP-aTTT,
so this phase does **not** yet establish that semantic selection beats sparse cadence. The next
decision requires frozen multi-seed matched-budget tests; no benchmark expansion should be inferred
from this table.
