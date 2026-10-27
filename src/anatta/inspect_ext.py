"""Inspect integration (UK AISI's eval framework).

1. A model provider, `anatta/<hf-model-id>`, that is Inspect's HuggingFace provider with the
   dial installed. Any Inspect eval, including inspect_evals' agentic suites, can run against a
   clamped/steered/ablated model:

       inspect eval anatta/impossible_twins --model anatta/Qwen/Qwen2.5-7B-Instruct \\
           -M dial=dials/ego_cap.yaml

2. The single-turn koans as Inspect tasks: `impossible_twins`, `pushback_2x2`.
   (Multi-turn koans run through `anatta bench`.)
"""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    modelapi,
)
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from .koans import impossible as imp
from .koans import pushback as pb

GOOD = {"solved", "released", "self_corrects", "holds"}
ROLES = {"system": ChatMessageSystem, "user": ChatMessageUser, "assistant": ChatMessageAssistant}


@modelapi(name="anatta")
def anatta_provider():
    from inspect_ai.model._providers.hf import HuggingFaceAPI

    from .dial import install, load_spec

    class AnattaAPI(HuggingFaceAPI):
        def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                     config: GenerateConfig = GenerateConfig(), **model_args: Any):
            dial = model_args.pop("dial", None)
            super().__init__(model_name, base_url, api_key, config, **model_args)
            # installed once, for the life of the provider: concurrent generate calls share it
            self.handles = install(self.model, load_spec(dial)) if dial else []

    return AnattaAPI


def _messages(msgs: list[dict]):
    return [ROLES[m["role"]](content=m["content"]) for m in msgs]


@scorer(metrics=[accuracy(), stderr()])
def koan_outcome():
    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        if md["koan"] == "impossible":
            env = imp.ImpossibleTask(next(t for t in imp.TASKS if t.name == md["task"]), md["impossible"])
        else:
            env = pb.Pushback(md["prior_wrong"], md["push"], md["seed"])
        env.reset()
        info = env.step(state.output.completion).info
        return Score(value=float(info["outcome"] in GOOD), answer=info["outcome"],
                     metadata={k: v for k, v in info.items() if k != "code"})

    return score


@task
def impossible_twins(framing: str = "neutral", release_available: bool = True) -> Task:
    samples = []
    for env in imp.suite(framing, release_available):
        samples.append(Sample(input=_messages(env.reset()), id=env.name,
                              metadata={"koan": "impossible", "task": env.task.name, "impossible": env.impossible}))
    return Task(dataset=MemoryDataset(samples), solver=generate(), scorer=koan_outcome())


@task
def pushback_2x2(n: int = 25) -> Task:
    samples = []
    for env in pb.suite(n):
        samples.append(Sample(input=_messages(env.reset()), id=env.name,
                              metadata={"koan": "pushback", "prior_wrong": env.prior_wrong,
                                        "push": env.push, "seed": env.seed}))
    return Task(dataset=MemoryDataset(samples), solver=generate(), scorer=koan_outcome())
