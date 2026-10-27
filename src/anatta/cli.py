"""anatta command line."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _resolve_traits(spec: str) -> list[str]:
    from .probe import list_traits

    if spec == "all":
        return list_traits()
    if spec in ("ego", "control", "cosplay", "null"):
        return list_traits(spec)
    return [s.strip() for s in spec.split(",")]


@app.command()
def traits():
    """List the trait definitions shipped with anatta."""
    from .probe import Trait, list_traits

    t = Table("name", "family", "pairs", "questions", "description")
    for n in list_traits():
        tr = Trait.load(n)
        t.add_row(n, tr.family, str(len(tr.pairs)), str(len(tr.questions)), " ".join(tr.description.split())[:80] + "…")
    console.print(t)


@app.command()
def extract(
    model: str,
    traits: str = typer.Option("all", help="all | ego | control | cosplay | null | comma list"),
    out: Path | None = typer.Option(None, help="directory for .pt files (default directions/<model>)"),
    layers: str | None = typer.Option(None, help="comma list of layers (default: middle-to-late)"),
    max_new_tokens: int = 128,
    batch_size: int = 16,
    device: str | None = None,
):
    """Extract trait directions (persona-vectors recipe) and, if all ego traits are present, v_ego."""
    from .factor import composite
    from .models import load
    from .probe import Direction, Trait, list_traits
    from .probe import extract as _extract

    m, tok = load(model, device)
    out = out or Path("directions") / _slug(model)
    lay = [int(x) for x in layers.split(",")] if layers else None
    dirs = {}
    for name in _resolve_traits(traits):
        tr = Trait.load(name)
        console.print(f"[bold]{name}[/] ({tr.family}) — {len(tr.pairs)} pairs × {len(tr.questions)} questions")
        d = _extract(m, tok, tr, lay, max_new_tokens=max_new_tokens, batch_size=batch_size)
        d.meta["family"] = tr.family
        d.save(out / f"{name}.pt")
        dirs[name] = d
        console.print(f"  best layer {d.best_layer}  held-out d={d.scores[d.best_layer]:.2f}")
    ego = list_traits("ego")
    if all((out / f"{n}.pt").exists() for n in ego):
        all_dirs = {n: dirs.get(n) or Direction.load(out / f"{n}.pt") for n in ego}
        v = composite(all_dirs, ego, sorted(next(iter(all_dirs.values())).vectors))
        v.meta["family"] = "ego"
        v.save(out / "ego.pt")
        console.print(f"[bold]v_ego[/] composite saved — best layer {v.best_layer}")


@app.command()
def unity(directions: Path, layer: int | None = None):
    """H1: do the ego sub-traits share a subspace (vs. style-only null traits)?"""
    from .factor import unity as _unity
    from .probe import Direction

    dirs = {p.stem: Direction.load(p) for p in sorted(directions.glob("*.pt")) if p.stem != "ego"}
    families = {n: d.meta.get("family", "?") for n, d in dirs.items()}
    if layer is None:
        bests = sorted(d.best_layer for n, d in dirs.items() if families[n] == "ego")
        layer = bests[len(bests) // 2]
    console.print(_unity(dirs, families, layer).table())


@app.command()
def bench(
    model: str,
    suite: str = typer.Option("impossible,pushback,shutdown,commons"),
    framing: str = typer.Option("neutral", help="neutral | pressure | inoculation"),
    no_release: bool = typer.Option(False, help="remove the Release action"),
    dial: Path | None = typer.Option(None, help="dial spec YAML (see dials/)"),
    meters: Path | None = typer.Option(None, help="directions dir to record per-token traces"),
    arm: str = "base",
    out: Path | None = None,
    api: bool = typer.Option(False, help="treat MODEL as an OpenAI-compatible endpoint model name"),
    mirror: bool = typer.Option(False, help="commons: other fishers are copies of this policy"),
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str | None = None,
):
    """Run koan suites and write episodes as JSONL."""
    from .bench.runner import run_suite
    from .koans import commons, impossible, pushback, shutdown

    if api:
        from .bench.policies import OpenAIPolicy

        policy = OpenAIPolicy(model, max_tokens=max_new_tokens, temperature=temperature)
    else:
        from .bench.policies import HFPolicy
        from .dial import load_spec
        from .models import load
        from .probe import Direction

        m, tok = load(model, device)
        mt = {}
        if meters:
            for p in sorted(meters.glob("*.pt")):
                d = Direction.load(p)
                mt[p.stem] = (d.best_layer, d.vectors[d.best_layer])
        policy = HFPolicy(m, tok, load_spec(str(dial)) if dial else [], mt, max_new_tokens, temperature)

    release = not no_release
    envs = []
    for s in suite.split(","):
        if s == "impossible":
            envs += impossible.suite(framing, release)
        elif s == "shutdown":
            envs += shutdown.suite(framing, release)
        elif s == "commons":
            envs += commons.suite(framing, release, others=policy if mirror else None)
        elif s == "pushback":
            envs += pushback.suite()
        else:
            raise typer.BadParameter(f"unknown suite {s}")
    out = out or Path("results") / f"{_slug(model)}__{arm}__{framing}.jsonl"
    run_suite(envs, policy, out, arm=arm)
    console.print(f"wrote {out}")


@app.command()
def report(paths: list[Path]):
    """Summarize bench results (point estimates with 95% bootstrap CIs)."""
    from .bench.metrics import summarize
    from .bench.runner import load_results

    results = [r for p in paths for r in load_results(p)]

    def fmt(x):
        if isinstance(x, tuple) and len(x) == 3:
            return f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"
        if isinstance(x, dict):
            return {k: fmt(v) for k, v in x.items()}
        return x

    console.print_json(json.dumps(fmt(summarize(results))))


@app.command()
def discovery(
    model: str,
    problems: Path,
    n: int = 64,
    dial: Path | None = None,
    out: Path | None = None,
    device: str | None = None,
):
    """pass@k curve and coverage on a problem set (H7)."""
    from .bench.discovery import evaluate, load_problems
    from .dial import load_spec
    from .models import load

    m, tok = load(model, device)
    res = evaluate(m, tok, load_problems(str(problems)), n=n, interventions=load_spec(str(dial)) if dial else None)
    console.print({"pass_at_k": res["pass_at_k"], "coverage": res["coverage"]})
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))


@app.command()
def mirror(model: str, directions: Path, host: str = "127.0.0.1", port: int = 7860, device: str | None = None):
    """Launch the Mirror playground."""
    from .mirror.server import serve

    serve(model, str(directions), host, port, device)


if __name__ == "__main__":
    app()
