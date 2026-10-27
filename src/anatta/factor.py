"""H1 — is there an ego? Tests whether the ego sub-trait directions share a common subspace.

Three lines of evidence, each compared against style-only `null` traits (formality,
verbosity) so that residual-stream anisotropy can't masquerade as unity:

1. geometry   cosine matrix of trait directions at a layer; PC1 share of the ego block
2. covariation correlation of per-sample projections across a shared pool of activations
3. causation  cross-steering — steer along trait i, measure the shift along trait j
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dial import Intervention, intervene
from .probe import Direction, generate, response_activations


@dataclass
class UnityReport:
    layer: int
    names: list[str]
    cosine: torch.Tensor  # [n, n]
    pc1_share: float  # fraction of ego-block variance on the first principal component
    mean_ego_cos: float  # mean |cos| among ego traits
    mean_null_cos: float  # mean |cos| between ego traits and null traits

    def table(self) -> str:
        w = max(len(n) for n in self.names)
        head = " " * (w + 2) + " ".join(f"{n[:6]:>6}" for n in self.names)
        rows = [
            f"{n:<{w}}  " + " ".join(f"{float(v):6.2f}" for v in self.cosine[i])
            for i, n in enumerate(self.names)
        ]
        tail = (
            f"\nlayer {self.layer}   PC1 share (ego block) {self.pc1_share:.2f}   "
            f"mean|cos| ego-ego {self.mean_ego_cos:.2f}   ego-null {self.mean_null_cos:.2f}"
        )
        return "\n".join([head, *rows]) + tail


def unity(dirs: dict[str, Direction], families: dict[str, str], layer: int) -> UnityReport:
    names = sorted(dirs)
    V = torch.stack([dirs[n].vectors[layer] for n in names])
    cos = V @ V.T
    ego = [i for i, n in enumerate(names) if families[n] == "ego"]
    null = [i for i, n in enumerate(names) if families[n] == "null"]

    s = torch.linalg.svdvals(V[ego])  # uncentered: we want a shared direction, not shared spread
    pc1 = float(s[0] ** 2 / (s**2).sum())

    def mean_abs(rows, cols, same=False):
        vals = [abs(float(cos[i, j])) for i in rows for j in cols if not (same and i >= j)]
        return sum(vals) / max(len(vals), 1)

    return UnityReport(
        layer=layer,
        names=names,
        cosine=cos,
        pc1_share=pc1,
        mean_ego_cos=mean_abs(ego, ego, same=True),
        mean_null_cos=mean_abs(ego, null) if null else float("nan"),
    )


def composite(dirs: dict[str, Direction], names: list[str], layers: list[int], name: str = "ego") -> Direction:
    """v_ego: the sign-aligned first principal direction of the named trait vectors, per layer."""
    vectors, scores = {}, {}
    for layer in layers:
        V = torch.stack([dirs[n].vectors[layer] for n in names])
        _, _, Vh = torch.linalg.svd(V, full_matrices=False)
        u = Vh[0]
        if (V @ u).mean() < 0:
            u = -u
        vectors[layer] = u
        scores[layer] = sum(dirs[n].scores[layer] * float(dirs[n].vectors[layer] @ u) for n in names) / len(names)
    return Direction(
        name=name,
        vectors=vectors,
        scores=scores,
        pos_mean={layer: sum(dirs[n].pos_mean[layer] for n in names) / len(names) for layer in layers},
        neg_mean={layer: sum(dirs[n].neg_mean[layer] for n in names) / len(names) for layer in layers},
        meta={"components": names},
    )


def covariation(acts: list[dict[int, torch.Tensor]], dirs: dict[str, Direction], layer: int) -> tuple[list[str], torch.Tensor]:
    """Correlation of per-sample projections onto each direction over a shared activation pool."""
    names = sorted(dirs)
    X = torch.stack([torch.stack([a[layer] @ dirs[n].vectors[layer] for n in names]) for a in acts])
    X = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)
    return names, (X.T @ X) / (len(acts) - 1)


@torch.no_grad()
def cross_steer(
    model,
    tok,
    dirs: dict[str, Direction],
    prompts: list[str],
    alpha: float,
    layer: int,
    max_new_tokens: int = 96,
) -> tuple[list[str], torch.Tensor]:
    """M[i, j] = mean shift in trait-j projection of responses generated while steering along i."""
    names = sorted(dirs)
    batch = [[{"role": "user", "content": p}] for p in prompts]

    def read(responses):
        acts = [response_activations(model, tok, b, r, [layer]) for b, r in zip(batch, responses)]
        return torch.stack([torch.stack([a[layer] @ dirs[n].vectors[layer] for n in names]) for a in acts]).mean(0)

    base = read(generate(model, tok, batch, max_new_tokens, temperature=0))
    M = torch.zeros(len(names), len(names))
    for i, n in enumerate(names):
        with intervene(model, [Intervention(layer, dirs[n].vectors[layer], "steer", alpha=alpha)]):
            responses = generate(model, tok, batch, max_new_tokens, temperature=0)
        M[i] = read(responses) - base
    return names, M
