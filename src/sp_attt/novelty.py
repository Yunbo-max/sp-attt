from __future__ import annotations

from collections.abc import Iterable, Sequence


def word_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def sequence_novelty(text: str, history: Iterable[str], n: int = 3) -> float:
    current = word_ngrams(text, n)
    previous = [word_ngrams(item, n) for item in history]
    return 1.0 - max((jaccard(current, item) for item in previous), default=0.0)


def action_repetition(action: str, actions: Sequence[str]) -> tuple[float, float]:
    normalized = action.strip().lower()
    return (
        float(bool(actions) and normalized == actions[-1].strip().lower()),
        float(any(normalized == old.strip().lower() for old in actions[-3:])),
    )


def attt_token_weights(token_ids: Sequence[int], history: Sequence[Sequence[int]], n: int = 3,
                       w_min: float = 0.05) -> list[float]:
    """Paper-faithful repetition weights: max(w_min, 1/(1 + prior n-gram count))."""
    counts: dict[tuple[int, ...], int] = {}
    for old in history:
        for i in range(max(0, len(old) - n + 1)):
            gram = tuple(old[i : i + n])
            counts[gram] = counts.get(gram, 0) + 1
    weights = []
    for j in range(len(token_ids)):
        if j < n - 1:
            weights.append(1.0)
        else:
            gram = tuple(token_ids[j - n + 1 : j + 1])
            weights.append(max(w_min, 1.0 / (1.0 + counts.get(gram, 0))))
    return weights

