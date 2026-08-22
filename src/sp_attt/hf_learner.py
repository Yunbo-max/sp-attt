from __future__ import annotations

from collections.abc import Sequence

import torch

from .losses import attt_loss, next_token_loss
from .novelty import attt_token_weights
from .types import CandidateExperience


class HuggingFaceLoRALearner:
    """Minimal two-step episode-local LoRA learner for causal language models.

    The caller is responsible for freezing the backbone and attaching a PEFT LoRA adapter.
    Only parameters with ``requires_grad=True`` enter the optimizer.
    """

    def __init__(self, model, tokenizer, *, lr: float = 5e-4, gradient_steps: int = 2,
                 ngram: int = 3, weight_floor: float = 0.05):
        self.model, self.tokenizer = model, tokenizer
        self.gradient_steps, self.ngram, self.weight_floor = gradient_steps, ngram, weight_floor
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("No trainable adapter parameters; attach LoRA before constructing learner")
        self.optimizer = torch.optim.AdamW(parameters, lr=lr)
        self.token_history: list[list[int]] = []

    def update(self, experience: CandidateExperience, method: str) -> float:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(experience.text, return_tensors="pt", add_special_tokens=True)
        ids = encoded["input_ids"].to(device)
        weights = torch.tensor(attt_token_weights(ids[0].tolist(), self.token_history,
                                                  self.ngram, self.weight_floor), device=device)[None]
        final = 0.0
        self.model.train()
        for _ in range(self.gradient_steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(**{k: v.to(device) for k, v in encoded.items()}).logits
            loss = next_token_loss(logits, ids) if method == "ttt" else attt_loss(logits, ids, weights)
            loss.backward()
            self.optimizer.step()
            final = float(loss.detach())
        self.token_history.append(ids[0].tolist())
        return final


def attach_episode_lora(model, *, rank: int = 8, alpha: int = 16,
                        target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj")):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Install the 'train' extra: pip install -e '.[train]'") from exc
    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(target_modules),
                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, config)

