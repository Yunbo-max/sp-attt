# Experiment contract

## Primary tables

ALFWorld reports success rate, average successful steps, update percentage, measured GPU update
time, HUR, MBR, and plasticity regret. MineExplorer reports TSR/MSR by hop count and the same
plasticity metrics. Every aggregate retains per-episode records for paired resampling.

## Splits

- ALFWorld: 1,800 train environments for meta-train, 300 disjoint train environments for meta-dev,
  and all 140 official evaluation tasks untouched until final evaluation.
- MineExplorer: generated 1–4-hop 1,000/200 meta-train/dev; official 813 and Hard-100 untouched.
  This split is pending verification against the upstream generator once published.

## Labels

ALFWorld uses the remaining episode horizon and `success + 0.1 * success * (1 - T_finish/50)`.
MineExplorer uses 20 LLM decisions and `delta milestone fraction + task solved`. Environment frames
never count as LLM decisions. Both branches disable later updates.

## Gate input

The feature vector contains a frozen candidate hidden representation, update-text novelty,
normalized NLL, last-action repetition, recent-3-action repetition, observation novelty, and
relative episode position. No gradient-derived feature is allowed.

## Statistical protocol

ALFWorld uses five seeds and MineExplorer three. Report paired bootstrap 95% confidence intervals;
McNemar for paired success; paired bootstrap and supplemental Wilcoxon for MSR. Thresholds and
matched update budgets are selected on meta-dev only.

## Local execution notes

ALFWorld is installed with `alfworld-download` and uses the official `valid_seen`/`valid_unseen`
splits. The Qwen3.5-4B runner is `scripts/run_alfworld_pilot.py`; `--max-steps 50` and
`--max-new-tokens 24` are the intended evaluation defaults. The A4000 smoke pilot uses eight new
tokens only to keep iteration time bounded. Its measured output is recorded in
`results/pilot_summary.json` and is explicitly not a benchmark claim.
