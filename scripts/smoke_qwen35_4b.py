"""Run a real two-gradient-step SP-aTTT inner update and report GPU memory."""

import json
import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from sp_attt.hf_learner import HuggingFaceLoRALearner, attach_episode_lora
from sp_attt.types import CandidateExperience

MODEL = "Qwen/Qwen3.5-4B"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = attach_episode_lora(model, rank=8, alpha=16)
    learner = HuggingFaceLoRALearner(model, tokenizer, lr=5e-4, gradient_steps=2)
    candidate = CandidateExperience(
        "smoke", 1,
        "Thought: The apple may be in the refrigerator. Action: open refrigerator 1",
        "open refrigerator 1", "You open the refrigerator. It contains an apple.", 5, 50,
    )
    started = time.time()
    loss = learner.update(candidate, "attt")
    torch.cuda.synchronize()
    result = {
        "model": MODEL,
        "dtype": "bfloat16",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "loss": loss,
        "two_step_update_seconds": time.time() - started,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "finite": bool(torch.isfinite(torch.tensor(loss))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
