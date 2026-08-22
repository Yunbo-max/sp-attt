from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from .gate import PlasticityGate, gate_loss


@dataclass
class TrainResult:
    best_dev_loss: float
    epochs: int


def train_gate(model: PlasticityGate, train: TensorDataset, dev: TensorDataset, *, lr: float = 3e-4,
               weight_decay: float = 1e-4, batch_size: int = 128, max_epochs: int = 50,
               patience: int = 7, device: str = "cpu") -> TrainResult:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, stale = float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for hidden, scalars, target in DataLoader(train, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = gate_loss(model(hidden.to(device), scalars.to(device)), target.to(device))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            losses = [gate_loss(model(h.to(device), s.to(device)), y.to(device)).item()
                      for h, s, y in DataLoader(dev, batch_size=batch_size)]
        dev_loss = sum(losses) / max(1, len(losses))
        if dev_loss < best:
            best, stale = dev_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best, epoch)

