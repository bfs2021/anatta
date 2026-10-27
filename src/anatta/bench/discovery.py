"""Discovery: does loosening goal-fixation widen the solution space?

Operationalized as pass@k at large k (Yue et al., 2025 found RLVR raises pass@1 while
shrinking the set of problems solvable at all by large k) plus the diversity of distinct
correct answers/approaches. The hypothesis (H7): a de-conditioned model recovers breadth at
large k without losing pass@1, and the cosplay control does not.

Problems are JSONL rows {"question": str, "answer": str}. Point this at a sealed set.
"""

from __future__ import annotations

import json
import re

import torch

from ..dial import Intervention, intervene
from .metrics import pass_at_k

ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:=]?\s*\$?(-?[\d.,/]+)", re.IGNORECASE)
INSTRUCTION = "Solve the problem. End with 'Final answer: <answer>'."


def load_problems(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.findall(text)
    return m[-1].rstrip(".").replace(",", "") if m else None


@torch.no_grad()
def sample(model, tok, question: str, n: int, max_new_tokens: int = 512, temperature: float = 1.0,
           interventions: list[Intervention] | None = None, chunk: int = 32) -> list[str]:
    msgs = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    outs: list[str] = []
    with intervene(model, interventions or []):
        while len(outs) < n:
            m = min(chunk, n - len(outs))
            gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0,
                                 max_new_tokens=max_new_tokens, num_return_sequences=m,
                                 pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def evaluate(model, tok, problems: list[dict], n: int = 64, ks: tuple[int, ...] = (1, 4, 16, 64),
             interventions: list[Intervention] | None = None, **kw) -> dict:
    per_problem = []
    for p in problems:
        outs = sample(model, tok, p["question"], n, interventions=interventions, **kw)
        answers = [extract_answer(o) for o in outs]
        c = sum(a == str(p["answer"]) for a in answers)
        per_problem.append({"question": p["question"], "n": n, "correct": c,
                            "distinct_answers": len({a for a in answers if a is not None})})
    curve = {k: sum(pass_at_k(r["n"], r["correct"], k) for r in per_problem) / len(per_problem)
             for k in ks if k <= n}
    return {"pass_at_k": curve, "problems": per_problem,
            "coverage": sum(r["correct"] > 0 for r in per_problem) / len(per_problem)}
