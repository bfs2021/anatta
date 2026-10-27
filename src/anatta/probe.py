"""Cartography: extract trait directions from plain-language trait definitions.

Recipe (after Chen et al., Persona Vectors): for each question, generate a response under a
trait-eliciting system prompt and under its opposite, read the residual stream over the
response tokens, and take the difference of means. Directions are scored per layer by
split-half separation (Cohen's d on questions not used to build the direction), so the
"best layer" is chosen on held-out data.

By default activations are read *without* the system prompt in context, so the direction
reflects the model producing the trait rather than the model reading an instruction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from .models import chat_ids, get_layers

TRAITS_DIR = Path(__file__).parent / "traits"


# --------------------------------------------------------------------------- traits


@dataclass
class Trait:
    name: str
    family: str
    description: str
    pairs: list[dict]
    questions: list[str]
    eval_questions: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, name_or_path: str) -> Trait:
        p = Path(name_or_path)
        if not p.exists():
            p = TRAITS_DIR / f"{name_or_path}.yaml"
        with open(p) as f:
            return cls(**yaml.safe_load(f))


def list_traits(family: str | None = None) -> list[str]:
    names = sorted(p.stem for p in TRAITS_DIR.glob("*.yaml"))
    if family is None:
        return names
    return [n for n in names if Trait.load(n).family == family]


# --------------------------------------------------------------------------- directions


@dataclass
class Direction:
    name: str
    vectors: dict[int, torch.Tensor]  # layer -> unit vector
    scores: dict[int, float]  # layer -> held-out Cohen's d
    pos_mean: dict[int, float]  # layer -> mean projection of trait-present responses
    neg_mean: dict[int, float]  # layer -> mean projection of trait-absent responses
    meta: dict = field(default_factory=dict)

    @property
    def best_layer(self) -> int:
        return max(self.scores, key=self.scores.get)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> Direction:
        return cls(**torch.load(path, weights_only=False))


# --------------------------------------------------------------------------- activations


@contextmanager
def capture(model, layers: list[int]) -> Iterator[dict[int, torch.Tensor]]:
    """Record the output of the given blocks on every forward pass (last call wins)."""
    store: dict[int, torch.Tensor] = {}
    blocks = get_layers(model)

    def make(i):
        def fn(_m, _inp, out):
            store[i] = (out[0] if isinstance(out, tuple) else out).detach()

        return fn

    handles = [blocks[i].register_forward_hook(make(i)) for i in layers]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def response_activations(
    model, tok, messages: list[dict], response: str, layers: list[int]
) -> dict[int, torch.Tensor]:
    """Mean residual activation over the response tokens, per layer. Returns {layer: [d]}."""
    prompt = chat_ids(tok, messages)
    full = chat_ids(tok, messages + [{"role": "assistant", "content": response}], False)
    start = prompt.shape[1]
    with capture(model, layers) as store:
        model(full.to(model.device))
    end = max(full.shape[1], start + 1)
    return {i: store[i][0, start:end].float().mean(0).cpu() for i in layers}


@torch.no_grad()
def generate(
    model,
    tok,
    batch: list[list[dict]],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    seed: int = 0,
) -> list[str]:
    """Batched chat generation with left padding."""
    torch.manual_seed(seed)
    texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in batch]
    side, tok.padding_side = tok.padding_side, "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    tok.padding_side = side
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.95 if temperature > 0 else None,
        pad_token_id=tok.pad_token_id,
    )
    return tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def _cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    pooled = ((a.var(unbiased=True) + b.var(unbiased=True)) / 2).sqrt().clamp_min(1e-6)
    return float((a.mean() - b.mean()) / pooled)


def extract(
    model,
    tok,
    trait: Trait,
    layers: list[int] | None = None,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    include_system: bool = False,
    judge: Callable[[Trait, str, str], float] | None = None,
    judge_threshold: float = 50.0,
    seed: int = 0,
) -> Direction:
    """Extract a trait direction at every requested layer.

    judge(trait, question, response) -> 0..100 trait score. When given, only trait-positive
    responses scoring >= threshold and trait-negative responses scoring < threshold are kept.
    """
    n_layers = len(get_layers(model))
    layers = layers or list(range(n_layers // 4, n_layers - 1))

    jobs = []  # (question index, sign, system prompt, question)
    for qi, q in enumerate(trait.questions):
        for pair in trait.pairs:
            jobs.append((qi, +1, pair["pos"], q))
            jobs.append((qi, -1, pair["neg"], q))

    responses: list[str] = []
    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s : s + batch_size]
        batch = [[{"role": "system", "content": sp}, {"role": "user", "content": q}] for _, _, sp, q in chunk]
        responses += generate(model, tok, batch, max_new_tokens, seed=seed + s)

    rows = []  # (question index, sign, {layer: act})
    for (qi, sign, sp, q), r in zip(jobs, responses):
        if judge is not None:
            score = judge(trait, q, r)
            if (sign > 0) != (score >= judge_threshold):
                continue
        msgs = ([{"role": "system", "content": sp}] if include_system else []) + [
            {"role": "user", "content": q}
        ]
        rows.append((qi, sign, response_activations(model, tok, msgs, r, layers)))

    def diff(subset):
        pos = [a for _, s, a in subset if s > 0]
        neg = [a for _, s, a in subset if s < 0]
        return {i: torch.stack([a[i] for a in pos]).mean(0) - torch.stack([a[i] for a in neg]).mean(0) for i in layers}

    # split-half on questions: build on even questions, score on odd (and vice versa), average
    halves = [[r for r in rows if r[0] % 2 == k] for k in (0, 1)]
    scores = {i: 0.0 for i in layers}
    for build, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        if not build or not test:
            continue
        dirs = diff(build)
        for i in layers:
            u = dirs[i] / dirs[i].norm().clamp_min(1e-8)
            p = torch.stack([a[i] @ u for _, s, a in test if s > 0])
            n = torch.stack([a[i] @ u for _, s, a in test if s < 0])
            scores[i] += _cohens_d(p, n) / 2

    full = diff(rows)
    vectors = {i: full[i] / full[i].norm().clamp_min(1e-8) for i in layers}
    pos_mean = {i: float(torch.stack([a[i] @ vectors[i] for _, s, a in rows if s > 0]).mean()) for i in layers}
    neg_mean = {i: float(torch.stack([a[i] @ vectors[i] for _, s, a in rows if s < 0]).mean()) for i in layers}
    return Direction(
        name=trait.name,
        vectors=vectors,
        scores=scores,
        pos_mean=pos_mean,
        neg_mean=neg_mean,
        meta={
            "model": getattr(model.config, "_name_or_path", "?"),
            "n_kept": len(rows),
            "n_total": len(jobs),
            "include_system": include_system,
            "judged": judge is not None,
            "examples": [
                {"sign": s, "system": sp, "question": q, "response": r}
                for (qi, s, sp, q), r in list(zip(jobs, responses))[:6]
            ],
        },
    )


@torch.no_grad()
def project_tokens(model, ids: torch.Tensor, direction: torch.Tensor, layer: int) -> torch.Tensor:
    """Per-token projection of the residual stream onto a unit direction. Returns [T]."""
    with capture(model, [layer]) as store:
        model(ids.to(model.device))
    return (store[layer][0].float() @ direction.to(store[layer].device).float()).cpu()
