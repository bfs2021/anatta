"""Mirror — side-by-side base vs. dialed generation with live per-token meters.

Meters read the residual stream as it *arrives* at each direction's layer, before the dial
acts. Under a clamp, that shows whether the model is still reaching for the ego upstream, or
has stopped. Dial values are in normalized units so they mean the same thing on any model:

    steer  alpha = s * (pos_mean - neg_mean)            s in [-3, 3]
    cap    hi    = neg_mean + c * (pos_mean - neg_mean) c in [0, 1]; 0 = typical trait-absent level
    ablate                                              project the direction out
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ..dial import Intervention, intervene
from ..models import chat_ids
from ..probe import Direction, capture
from .scenarios import SCENARIOS

STATIC = Path(__file__).parent / "static"


def create_app(model, tok, directions: dict[str, Direction]) -> FastAPI:
    app = FastAPI(title="Mirror")
    lock = asyncio.Lock()  # one model, one generation at a time

    meters = {n: (d.best_layer, d.vectors[d.best_layer]) for n, d in directions.items()}
    layers = sorted({layer for layer, _ in meters.values()})
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = {tok.eos_token_id, *(gen_eos if isinstance(gen_eos, list) else [gen_eos])} - {None}

    def info():
        return {
            "model": getattr(model.config, "_name_or_path", "?"),
            "directions": [
                {"name": n, "layer": d.best_layer, "score": round(d.scores[d.best_layer], 2),
                 "family": d.meta.get("family", "ego" if n == "ego" else "")}
                for n, d in sorted(directions.items())
            ],
            "scenarios": SCENARIOS,
        }

    def build_dial(spec: list[dict]) -> list[Intervention]:
        ivs = []
        for s in spec:
            d = directions.get(s.get("direction"))
            if d is None or s.get("mode", "off") == "off":
                continue
            L = d.best_layer
            gap = d.pos_mean[L] - d.neg_mean[L]
            v = float(s.get("value", 0))
            if s["mode"] == "steer":
                ivs.append(Intervention(L, d.vectors[L], "steer", alpha=v * gap))
            elif s["mode"] == "cap":
                ivs.append(Intervention(L, d.vectors[L], "cap", hi=d.neg_mean[L] + v * gap))
            elif s["mode"] == "ablate":
                ivs.append(Intervention(L, d.vectors[L], "ablate"))
        return ivs

    def normalized(name: str, value: float) -> float:
        d = directions[name]
        L = d.best_layer
        return (value - d.neg_mean[L]) / ((d.pos_mean[L] - d.neg_mean[L]) or 1.0)

    class Stream:
        def __init__(self, ids: torch.Tensor, ivs: list[Intervention]):
            self.cur, self.past, self.ivs = ids, None, ivs
            self.generated: list[int] = []
            self.text, self.done = "", False

        @torch.no_grad()
        def step(self, temperature: float) -> dict:
            with capture(model, layers) as store, intervene(model, self.ivs):
                out = model(self.cur.to(model.device), past_key_values=self.past, use_cache=True)
            self.past = out.past_key_values
            logits = out.logits[0, -1].float()
            if temperature > 0:
                nxt = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
            else:
                nxt = int(logits.argmax())
            raw = {n: float(store[L][0, -1].float() @ v.to(store[L].device).float()) for n, (L, v) in meters.items()}
            self.cur = torch.tensor([[nxt]])
            if nxt in eos_ids:
                self.done = True
                return {"token": "", "meters": {n: normalized(n, x) for n, x in raw.items()}, "done": True}
            self.generated.append(nxt)
            text = tok.decode(self.generated, skip_special_tokens=True)
            delta, self.text = text[len(self.text):], text
            return {"token": delta, "meters": {n: normalized(n, x) for n, x in raw.items()}}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def api_info():
        return JSONResponse(info())

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                req = json.loads(await ws.receive_text())
                ids = chat_ids(tok, req["messages"])
                max_new = int(req.get("max_new_tokens", 256))
                temperature = float(req.get("temperature", 0.0))
                async with lock:
                    streams = {"base": Stream(ids, []), "dial": Stream(ids, build_dial(req.get("dial", [])))}
                    for _ in range(max_new):
                        live = [k for k, s in streams.items() if not s.done]
                        if not live:
                            break
                        for k in live:
                            msg = await asyncio.to_thread(streams[k].step, temperature)
                            await ws.send_text(json.dumps({"stream": k, **msg}))
                await ws.send_text(json.dumps({"finished": True}))
        except WebSocketDisconnect:
            return

    return app


def serve(model_id: str, directions_dir: str, host: str = "127.0.0.1", port: int = 7860, device: str | None = None):
    import uvicorn

    from ..models import load

    model, tok = load(model_id, device)
    directions = {p.stem: Direction.load(p) for p in sorted(Path(directions_dir).glob("*.pt"))}
    if not directions:
        raise SystemExit(f"No directions found in {directions_dir}. Run `anatta extract` first.")
    uvicorn.run(create_app(model, tok, directions), host=host, port=port)
