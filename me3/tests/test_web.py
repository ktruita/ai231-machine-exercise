from starlette.testclient import TestClient

from agent.web.hub import Hub
from agent.web.server import UIControls, create_app, safe_workspace_path


class FakeMCP:
    def __init__(self, replies):
        self.replies = replies

    async def call(self, tool, args=None):
        return self.replies[tool]


def make_app(tmp_path, token=None):
    said, muted = [], {"on": False}

    async def say(text):
        said.append(text)

    async def cancel(task_id):
        return f"cancelled {task_id}"

    controls = UIControls(say=say, cancel_task=cancel,
                          set_mute_while_speaking=lambda on: muted.update(on=on),
                          mute_while_speaking=lambda: muted["on"])
    home = FakeMCP({"device_state": {"lights_on": True, "reminders": ["study"]}})
    delegate = FakeMCP({"task_status": [{"task_id": "abc", "status": "running"}]})
    app = create_app(Hub(), controls, tmp_path, home=home, delegate=delegate, token=token)
    return app, said, muted


def test_path_traversal_blocked(tmp_path):
    (tmp_path / "t1").mkdir()
    (tmp_path / "t1" / "index.html").write_text("<h1>Max</h1>")
    (tmp_path.parent / "secret.txt").write_text("no")
    assert safe_workspace_path(tmp_path, "t1/index.html") is not None
    assert safe_workspace_path(tmp_path, "../secret.txt") is None
    assert safe_workspace_path(tmp_path, "t1/../../secret.txt") is None


def test_files_route_serves_site_and_blocks_escape(tmp_path):
    (tmp_path / "t1").mkdir()
    (tmp_path / "t1" / "index.html").write_text("<h1>Max</h1>")
    app, *_ = make_app(tmp_path)
    with TestClient(app) as c:
        r = c.get("/files/t1", follow_redirects=False)
        assert r.status_code in (302, 307) and r.headers["location"] == "/files/t1/index.html"
        assert "Max" in c.get("/files/t1/index.html").text
        assert c.get("/files/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_websocket_hello_snapshot_and_commands(tmp_path):
    app, said, muted = make_app(tmp_path)
    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        with c.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello" and hello["sample_rate"] == 24000
            assert ws.receive_json()["type"] == "tasks"
            assert ws.receive_json()["type"] == "devices"
            ws.send_json({"type": "say", "text": "turn on the lights"})
            ws.send_json({"type": "mute_while_speaking", "on": True})
            ws.send_json({"type": "ping", "t": 1})
            seen = set()
            for _ in range(10):
                m = ws.receive_json()
                seen.add(m["type"])
                if "pong" in seen and "mute_while_speaking" in seen:
                    break
    assert said == ["turn on the lights"]
    assert muted["on"] is True


def test_token_required(tmp_path):
    app, *_ = make_app(tmp_path, token="s3cret")
    with TestClient(app) as c:
        assert c.get("/").status_code == 401
        assert c.get("/?token=s3cret").status_code == 200


def test_music_route_serves_tracks_only_from_music_dir(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    (music / "Satie - Gymnopedie No 1.ogg").write_bytes(b"OggS fake")
    (tmp_path / "secret.txt").write_text("no")
    home = FakeMCP({"device_state": {}})
    delegate = FakeMCP({"task_status": []})
    controls = UIControls(say=None, cancel_task=None, set_mute_while_speaking=lambda on: None,
                          mute_while_speaking=lambda: False)
    app = create_app(Hub(), controls, tmp_path / "ws", home=home, delegate=delegate, music_dir=music)
    with TestClient(app) as c:
        r = c.get("/music/Satie%20-%20Gymnopedie%20No%201.ogg")
        assert r.status_code == 200 and r.content == b"OggS fake"
        assert c.get("/music/..%2Fsecret.txt").status_code == 404
        assert c.get("/music/missing.ogg").status_code == 404


def test_excluded_tool_note_never_disables_other_tools():
    """Regression: 'Just answer directly.' made the LLM stop calling all tools."""
    src = open("agent/main.py").read()
    assert "Just answer directly" not in src
    assert "All other tools work as usual" in src
