"""The dial: inference-time interventions on residual-stream directions.

Three modes, each acting on the output of one transformer block:

    steer   h <- h + alpha * d                      (push along d)
    ablate  h <- h - (h.d) d                        (remove the d component)
    cap     h <- h + (clamp(h.d, lo, hi) - h.d) d   (bound the d component; Assistant-Axis style)

`d` is always unit-normalized. Branch A of the protocol ("the surgical clamp") is
`cap` with hi set near the base model's typical non-egoic projection.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch

from .models import get_layers

Mode = Literal["steer", "ablate", "cap"]


@dataclass
class Intervention:
    layer: int
    direction: torch.Tensor
    mode: Mode = "steer"
    alpha: float = 0.0
    lo: float | None = None
    hi: float | None = None

    def __post_init__(self):
        d = self.direction.detach().float().flatten()
        self.direction = d / d.norm().clamp_min(1e-8)
        if self.mode == "cap" and self.lo is None and self.hi is None:
            raise ValueError("cap needs lo and/or hi")

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        d = self.direction.to(device=h.device, dtype=h.dtype)
        if self.mode == "steer":
            return h + self.alpha * d
        proj = h @ d
        if self.mode == "ablate":
            return h - proj.unsqueeze(-1) * d
        capped = proj.clamp(min=self.lo, max=self.hi)
        return h + (capped - proj).unsqueeze(-1) * d


def _hook(interventions: list[Intervention]):
    def fn(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        for iv in interventions:
            h = iv.apply(h)
        if isinstance(output, tuple):
            return (h, *output[1:])
        return h

    return fn


def install(model, interventions: list[Intervention]) -> list:
    """Attach interventions permanently. Returns hook handles; call .remove() on each to undo."""
    layers = get_layers(model)
    by_layer: dict[int, list[Intervention]] = {}
    for iv in interventions:
        by_layer.setdefault(iv.layer, []).append(iv)
    return [layers[i].register_forward_hook(_hook(ivs)) for i, ivs in by_layer.items()]


@contextmanager
def intervene(model, interventions: list[Intervention]) -> Iterator[None]:
    """Apply interventions for the duration of the block."""
    handles = install(model, interventions)
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def load_spec(spec: dict | str) -> list[Intervention]:
    """Build interventions from a dial spec (dict or path to YAML/JSON).

    interventions:
      - direction: directions/ego.pt   # saved Direction
        layer: 14                      # optional; defaults to the direction's best layer
        mode: cap                      # steer | ablate | cap
        hi: 0.0                        # cap bounds are in units of the direction's projection
      - direction: directions/ego.pt
        mode: steer
        alpha: -4.0
        control: random                # swap in an equal-norm random direction (control arm)
    """
    import yaml

    from .probe import Direction

    if isinstance(spec, str):
        with open(spec) as f:
            spec = yaml.safe_load(f)
    out = []
    for item in spec.get("interventions", []):
        d = Direction.load(item["direction"])
        layer = item.get("layer", d.best_layer)
        vec = d.vectors[layer]
        if item.get("control") == "random":
            vec = random_like(vec, seed=item.get("seed", 0))
        out.append(
            Intervention(
                layer=layer,
                direction=vec,
                mode=item.get("mode", "steer"),
                alpha=float(item.get("alpha", 0.0)),
                lo=item.get("lo"),
                hi=item.get("hi"),
            )
        )
    return out


def random_like(direction: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """A random unit direction of the same shape: the equal-norm control for every clamp."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(direction.numel(), generator=g)
    return r / r.norm()
