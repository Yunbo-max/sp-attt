# Mechanism audit for the 500-label gate

The audit uses the frozen `gate_features_500.pt` and `plasticity_gate_500.pt` checkpoints. The
dev split is the held-out 81-row split used during gate training; `all` is reported only as a
descriptive diagnostic because it includes the training rows.

| Metric | Dev (81) | All complete rows (408) |
|---|---:|---:|
| Harmful rate | 12.35% | 9.56% |
| Beneficial rate | 13.58% | 13.73% |
| Harmful AUPRC | 0.213 | 0.231 |
| Selected rate | 50.62% | 52.94% |
| Harmful rate among selected | 12.20% | 6.02% |
| Harmful rate among skipped | 12.50% | 13.54% |
| Harmful recall by skipping | 50.00% | 66.67% |
| Mean utility when selected | -0.0252 | 0.0488 |
| Mean utility when skipped | 0.0017 | -0.0601 |

The dev rows do not yet show `E[U | LEARN] > E[U | SKIP]`; therefore the all-row enrichment
must not be used as proof of mechanism. This is a genuine limitation motivating the frozen
multi-seed and out-of-sample gate analysis.
