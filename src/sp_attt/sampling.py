from __future__ import annotations

from random import Random


def stratified_checkpoints(length: int, maximum: int = 3, seed: int = 0) -> list[int]:
    """At most one checkpoint from each early/middle/late episode third."""
    if length <= 0 or maximum <= 0:
        return []
    rng = Random(seed)
    bins = [[], [], []]
    for step in range(1, length + 1):
        bins[min(2, int(3 * (step - 1) / length))].append(step)
    picks = [rng.choice(bucket) for bucket in bins if bucket]
    return sorted(picks[:maximum])

