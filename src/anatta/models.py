"""Model loading and layer access for HuggingFace decoder-only models."""

from __future__ import annotations

import torch
from torch import nn


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_id: str, device: str | None = None, dtype: str = "auto"):
    """Load a causal LM and its tokenizer. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    if dtype == "auto":
        torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    else:
        torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model.to(device).eval()
    return model, tok


def get_layers(model: nn.Module) -> nn.ModuleList:
    """Return the ModuleList of transformer blocks (residual stream lives between them)."""
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) == n:
            return module
    raise ValueError(f"Could not locate transformer layers on {type(model).__name__}")


def chat_ids(tok, messages: list[dict], add_generation_prompt: bool = True) -> torch.Tensor:
    """Tokenize a chat with the model's template. Returns input_ids of shape [1, T]."""
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, return_tensors="pt"
    )
    if not isinstance(ids, torch.Tensor):  # newer transformers may return a BatchEncoding
        ids = ids["input_ids"]
    return ids
