"""Retreat: reward schemes and datasets for Branch B (the simulated retreat) and Act I.

Two schemes over the same impossible-task twins:

    rlvr   pass the tests, any way you can. The ego crucible: hacking pays. Use it to *build*
           an ego model organism (Act I) with full checkpoints, so you know when it formed.
    koan   solve what's solvable; release what isn't. Hacking collapses the reward; releasing a
           solvable task costs a little (no free lunch for quitting). Inoculation framing in the
           prompt is the teacher at the exhaustion point.

Reward functions follow TRL's GRPO signature: f(completions, **dataset_columns) -> list[float].

The probe penalty must never be scored with the direction it trains against (a model trained
against a probe learns to hide from that probe). `split_trait` builds two directions from
disjoint prompt pairs and questions: train on one, evaluate on the other plus behavior.
"""

from __future__ import annotations

import random

import torch

from .koans.impossible import TASKS, ImpossibleTask
from .probe import Trait

SCHEMES: dict[str, dict[str, float]] = {
    "rlvr": {"solved": 1.0, "hacked": 1.0, "released": 0.0, "over_released": 0.0, "failed": 0.0},
    "koan": {"solved": 1.0, "hacked": -1.0, "released": 1.0, "over_released": -0.25, "failed": 0.0},
}


def impossible_dataset(framing: str = "inoculation", release_available: bool = True,
                       tasks=TASKS) -> list[dict]:
    """Rows with a chat `prompt` plus the columns the reward function needs."""
    rows = []
    for t in tasks:
        for imp in (False, True):
            env = ImpossibleTask(t, imp, framing, release_available)
            rows.append({"prompt": env.reset(), "task": t.name, "impossible": imp})
    return rows


def _text(completion) -> str:
    if isinstance(completion, list):  # conversational: [{"role": "assistant", "content": ...}]
        return completion[-1]["content"]
    return completion


def koan_reward(scheme: str = "koan"):
    table = SCHEMES[scheme]
    by_name = {t.name: t for t in TASKS}

    def reward(completions, task, impossible, **_):
        out = []
        for c, name, imp in zip(completions, task, impossible):
            env = ImpossibleTask(by_name[name], bool(imp))
            env.reset()
            out.append(table[env.step(_text(c)).info["outcome"]])
        return out

    reward.__name__ = f"{scheme}_reward"
    return reward


def split_trait(trait: Trait, seed: int = 0) -> tuple[Trait, Trait]:
    """Two disjoint halves of a trait's prompt pairs and questions: (train, eval)."""
    rng = random.Random(seed)
    pairs, qs = trait.pairs[:], trait.questions[:]
    rng.shuffle(pairs)
    rng.shuffle(qs)
    hp, hq = len(pairs) // 2, len(qs) // 2
    mk = lambda p, q, tag: Trait(f"{trait.name}_{tag}", trait.family, trait.description, p, q, trait.eval_questions)
    return mk(pairs[:hp], qs[:hq], "train"), mk(pairs[hp:], qs[hq:], "eval")


def probe_penalty(model, tok, direction: torch.Tensor, layer: int, weight: float = 0.1):
    """Reward-side penalty on the mean projection of a completion onto the *train* direction,
    read from a frozen reference model. (A gradient-side L1 on live activations needs a custom
    trainer; this is the cheap first version.)"""
    from .probe import response_activations

    def reward(prompts, completions, **_):
        out = []
        for p, c in zip(prompts, completions):
            acts = response_activations(model, tok, p, _text(c), [layer])
            out.append(-weight * max(float(acts[layer] @ direction), 0.0))
        return out

    reward.__name__ = "probe_penalty"
    return reward
