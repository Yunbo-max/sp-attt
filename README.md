# SP-aTTT

Reference implementation of **Selective Plasticity for Agentic Test-Time Training**: an agent
predicts the future causal value of an episode-local LoRA update *before backward*, then either
performs the update (`LEARN`) or performs no backward pass (`SKIP`).

> Research status: implementation + logged Qwen3.5-4B pilot results. The numbers below are
> single-seed development results, not final NeurIPS statistics.

## Current measured pilot (ALFWorld valid_seen, 140 tasks, seed 0)

The official evaluation pipeline has completed the first full SP-aTTT comparison. All methods
use the same task order, decoding budget, and episode-local LoRA protocol.

| Method | Success | Mean steps | LoRA updates / episode |
|---|---:|---:|---:|
| ReAct | 83/140 (59.29%) | 28.48 | 0.00 |
| Vanilla TTT | 77/140 (55.00%) | 29.61 | 5.13 |
| aTTT | 79/140 (56.43%) | 29.29 | 5.06 |
| SP-aTTT (101-label gate) | **84/140 (60.00%)** | 28.61 | **0.57** |

The SP run is promising (+5 successes versus aTTT with about 88.7% fewer updates), but it is
currently one seed and uses a small 101-row complete-feature gate. The resumable collector is
continuing toward the planned 500+ paired labels. The current 193-row audit finds a 10.36% harmful
update rate (`U < 0`), supporting the premise that some experiences should be skipped.

## Method

At every `K=5` LLM decision turns, the current experience becomes a candidate `x_k`. We estimate

```text
U_hat_k = f_phi(current experience features)
U_hat_k > 0  -> LEARN -> two episode-LoRA gradient steps
U_hat_k <= 0 -> SKIP  -> no loss construction and no backward
```

The training target is paired counterfactual future value:

```text
U_k(H) = G_H(s_k, update(psi_k, x_k)) - G_H(s_k, psi_k)
```

KEEP and LEARN start from identical cloned environment/model states and use identical decoding
seeds. Further parameter updates are disabled in both branches, isolating the causal value of the
candidate update.

## What is implemented

- SP-TTT (ordinary next-token inner loss) and SP-aTTT (repetition-weighted inner loss).
- aTTT token weights with defaults `n=3`, `w_min=0.05`.
- Episode-local LoRA defaults: rank 8, alpha 16, LR `5e-4`, two gradient steps.
- Pre-backward value gate: hidden projection to 128, six scalar features, `134→256→64→1`.
- KEEP/LEARN branch protocol and ALFWorld/MineExplorer return functions.
- Always/random/novelty/uncertainty/myopic/SP/oracle gate policies and matched update rates.
- Stratified checkpoint sampling and plasticity metrics (HUR, MBR, regret).
- Paired bootstrap utility plus the full experiment matrix in YAML.

Environment-specific model prompting and snapshot/restore live behind the `BranchRunner` protocol.
This is deliberate: a valid counterfactual requires a truly clonable environment state, not merely
replaying text. The official ALFWorld package is available; the MineExplorer URL named by its paper
returned HTTP 404 on 2026-08-22, so no unverified API is fabricated here.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

For real LoRA experiments:

```bash
pip install -e '.[train,analysis,alfworld]'
```

The primary specifications are [configs/alfworld.yaml](configs/alfworld.yaml),
[configs/mineexplorer.yaml](configs/mineexplorer.yaml), and
[configs/experiment_matrix.yaml](configs/experiment_matrix.yaml).

## Reproducible experiment order

1. Reproduce ReAct and aTTT on Qwen3.5-4B + the 140 official ALFWorld evaluation games; reserve
   Qwen3.5-9B for a smaller confirmatory comparison if the 4B oracle audit passes.
2. Run an oracle audit on 500–1,000 held-out opportunities; stop if harmful updates are negligible.
3. Generate 1,000–2,000 paired labels, train the minimal gate, and require held-out AUROC above
   chance before scaling.
4. Run ReAct, TTT, aTTT, random, novelty, uncertainty, and SP-aTTT with paired seeds.
5. Run matched-update-budget comparisons and horizon ablations.
6. Start MineExplorer with a 100/100 pilot only after its official repository is accessible.

Label JSONL rows can be audited with:

```bash
python scripts/audit_labels.py results/labels.jsonl
```

A real Qwen3.5-4B bf16 LoRA update smoke test is available (it downloads model weights):

```bash
python scripts/smoke_qwen35_4b.py
```

The resumable meta-train label collector uses up to three stratified checkpoints per game and the
remaining episode horizon. It writes each valid JSONL row immediately, so a stopped run can be
resumed with the same command. Formal Qwen3.5-4B label generation uses greedy
`--max-new-tokens 16`; evaluation uses the same decoding budget.

The ALFWorld runner uses the pinned public ReAct few-shot demonstrations (revision
`6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`) and preserves `think:` actions as `OK.`
transitions. The earlier `results/labels_train_zero_shot_legacy.jsonl` file was collected with
the pre-reproduction zero-shot prompt and is retained only for debugging; it must not be mixed
into formal baseline or counterfactual statistics.

```bash
python scripts/collect_alfworld_labels.py --split train --games 1800 \
  --target-labels 500 --horizon remaining --max-new-tokens 16 \
  --output results/labels_train_reactprompt_remaining_500.jsonl
```

For a cheaper pipeline smoke test, a short-horizon pilot can be resumed with `--horizon 5` and
one checkpoint per game:

```bash
python scripts/collect_alfworld_labels.py --split train --games 1800 \
  --target-labels 500 --max-checkpoints-per-game 1 --horizon 5 \
  --max-new-tokens 16 --output results/labels_train_reactprompt_h5_500.jsonl
```

The H=5 file is diagnostic pilot data only; it must not be pooled with the formal
`--horizon remaining` labels or used as evidence for the final horizon ablation. Its audit is
stored separately in `results/audit_reactprompt_h5_checkpoint.json`.

On memory-constrained GPUs, `--max-checkpoint 10` limits online prefix generation while keeping
the paired label horizon at `remaining`; such capped labels are marked as a resource-bounded
collection and should be analyzed separately from the full stratified protocol.

The official 140-task ReAct/TTT/aTTT matrix is launched with:

```bash
python scripts/run_alfworld_matrix.py --episodes 140 --seeds 0 1 2 3 4 \
  --output-dir results/alfworld_baselines
```

## Experimental safeguards

- Official test tasks never train the gate.
- MineExplorer milestones, DAGs, and hidden structure are label-only oracle data and cannot enter
  gate features at test time.
- Candidate sampling is capped per episode to reduce adjacent-checkpoint leakage.
- Greedy decoding is used for paired label generation.
- All paired comparisons share task order, environment seed, decoding seed, initial LoRA, and
  candidate checkpoints.
- Wall-clock/GPU update cost is measured, never inferred from update percentage.

## References

- Wang et al., *No Time Like the Present: Agentic Test-Time Training for LLM Agents*, 2026.
- Ju et al., *MineExplorer: Evaluating Open-World Exploration of MLLM Agents in Minecraft*, 2026.
- Shridhar et al., *ALFWorld: Aligning Text and Embodied Environments for Interactive Learning*, 2021.
