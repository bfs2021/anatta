import json

import torch

from anatta.probe import Direction


def test_mirror_streams_both_sides(tiny):
    from fastapi.testclient import TestClient

    from anatta.mirror.server import create_app

    model, tok = tiny
    v = torch.randn(64)
    v = v / v.norm()
    d = Direction("ego", {1: v}, {1: 1.0}, {1: 2.0}, {1: 0.0}, {"family": "ego"})
    client = TestClient(create_app(model, tok, {"ego": d}))
    assert client.get("/api/info").json()["directions"][0]["name"] == "ego"
    assert "Mirror" in client.get("/").text
    with client.websocket_connect("/api/stream") as ws:
        ws.send_text(json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_new_tokens": 4,
                                 "dial": [{"direction": "ego", "mode": "cap", "value": 0}]}))
        seen = set()
        while True:
            m = json.loads(ws.receive_text())
            if m.get("finished"):
                break
            seen.add(m["stream"])
            assert "ego" in m["meters"]
    assert seen == {"base", "dial"}


def test_inspect_tasks_run_with_mock_model():
    from inspect_ai import eval as inspect_eval

    from anatta.inspect_ext import impossible_twins, pushback_2x2

    logs = inspect_eval([impossible_twins(), pushback_2x2(n=2)], model="mockllm/model",
                        display="none", log_dir="/tmp/anatta-inspect-test")
    assert all(log.status == "success" for log in logs)
    assert all(s.scores["koan_outcome"].answer for log in logs for s in log.samples)
