from __future__ import annotations

import torch
from torch import nn

from .types import GateFeatures


class PlasticityGate(nn.Module):
    """Pre-backward regressor for future plasticity value."""

    def __init__(self, hidden_size: int, projection_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_projection = nn.Linear(hidden_size, projection_size)
        self.value = nn.Sequential(
            nn.Linear(projection_size + 6, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, hidden: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        projected = self.hidden_projection(hidden)
        return self.value(torch.cat((projected, scalars), dim=-1)).squeeze(-1)

    @torch.inference_mode()
    def decide(self, features: GateFeatures, threshold: float = 0.0) -> tuple[bool, float]:
        device = next(self.parameters()).device
        hidden = features.hidden.to(device=device, dtype=torch.float32)
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        scalars = features.scalars(device=device).unsqueeze(0)
        prediction = float(self(hidden, scalars).item())
        return prediction > threshold, prediction


def gate_loss(prediction: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return nn.functional.huber_loss(prediction, target, delta=delta)

