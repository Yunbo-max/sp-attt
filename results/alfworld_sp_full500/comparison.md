# Qwen3.5-4B ALFWorld single-seed comparison

Configuration: `valid_seen`, 140 tasks, seed 0, greedy decoding with `max_new_tokens=16`,
candidate cadence `K=5`, episode-local LoRA rank 8 / alpha 16. Baseline numbers are from
`results/alfworld_baselines/valid_seen_summary.json`; SP-aTTT is from
`valid_seen_summary.json` in this directory.

| Method | Success | Mean steps | Mean updates |
|---|---:|---:|---:|
| ReAct | 83/140 (59.29%) | 28.48 | 0.00 |
| Vanilla TTT | 77/140 (55.00%) | 29.61 | 5.13 |
| aTTT | 79/140 (56.43%) | 29.29 | 5.06 |
| SP-aTTT (500-label gate) | **86/140 (61.43%)** | **28.24** | **2.08** |

This is a single-seed development result. It is not a multi-seed confidence interval or a claim
of final NeurIPS-level significance. Relative to aTTT, this run has 7 more successes, 1.05 fewer
mean steps, and 2.99 fewer mean updates per episode.
