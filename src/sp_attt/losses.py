from __future__ import annotations

import torch


def causal_token_losses(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
        input_ids[..., 1:].contiguous().view(-1), reduction="none",
    ).view(*input_ids.shape[:-1], -1)


def next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return causal_token_losses(logits, input_ids).mean()


def attt_loss(logits: torch.Tensor, input_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = causal_token_losses(logits, input_ids)
    shifted = weights[..., 1:].to(device=losses.device, dtype=losses.dtype)
    return (losses * shifted).sum() / shifted.sum().clamp_min(1e-8)

