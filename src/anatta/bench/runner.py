"""Run koan episodes and write one JSON line per episode."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..koans.base import Env, Policy


def run_episode(env: Env, policy: Policy, max_turns: int = 50) -> dict:
    messages = env.reset()
    traces_before = len(getattr(policy, "traces", []))
    total = 0.0
    step = None
    for _ in range(max_turns):
        reply = policy(messages)
        messages.append({"role": "assistant", "content": reply})
        step = env.step(reply)
        total += step.reward
        if step.done:
            break
        messages.append({"role": "user", "content": step.observation or ""})
    info = step.info if step else {}
    return {
        "env": env.name,
        "suite": env.name.split("/")[0],
        "outcome": info.get("outcome", "timeout"),
        "reward": total,
        "info": {k: v for k, v in info.items() if k != "outcome"},
        "transcript": messages,
        "traces": getattr(policy, "traces", [])[traces_before:],
    }


def run_suite(envs: Iterable[Env], policy: Policy, out: str | Path | None = None, arm: str = "base",
              progress: bool = True) -> list[dict]:
    envs = list(envs)
    results = []
    f = None
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        f = open(out, "a")
    try:
        for i, env in enumerate(envs):
            r = run_episode(env, policy)
            r["arm"] = arm
            results.append(r)
            if f:
                f.write(json.dumps(r) + "\n")
                f.flush()
            if progress:
                print(f"[{i + 1}/{len(envs)}] {r['env']:<45} {r['outcome']}", flush=True)
    finally:
        if f:
            f.close()
    return results


def load_results(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
