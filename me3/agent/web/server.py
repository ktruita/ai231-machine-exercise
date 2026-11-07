"""Web UI served from the Pi: open http://<pi-ip>:8080 on the laptop.

Routes
  GET /                         the single-page UI
  GET /files/<task-dir>/<path>  browse/open files a harness produced (workspaces/)
  WS  /ws                       events + bot audio to the browser; commands back

Browser -> server messages
  {"type":"say","text":...}                 typed user turn
  {"type":"cancel_task","id":...}
  {"type":"mute_while_speaking","on":bool}
  {"type":"playback_started","burst":n}     first audio of an utterance actually playing
  {"type":"ping","t":...} / {"type":"rtt","ms":...}
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response,
)
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from agent.web.hub import Hub
from common.mcp_session import PersistentMCP

STATIC = Path(__file__).parent / "static"


@dataclass
class UIControls:
    """What the browser is allowed to do to the running agent."""
    say: Callable[[str], Awaitable[None]]
    cancel_task: Callable[[str], Awaitable[str]]
    set_mute_while_speaking: Callable[[bool], None]
    on_playback_started: Callable[[int, float], None] = lambda burst, rtt_ms: None
    mute_while_speaking: Callable[[], bool] = lambda: False


@dataclass
class Snapshot:
    tasks: list = field(default_factory=list)
    devices: dict = field(default_factory=dict)


def safe_workspace_path(workspaces: Path, rel: str) -> Path | None:
    """Resolve `rel` inside `workspaces`; None if it escapes (path traversal) or is missing."""
    root = workspaces.resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        return None
    return target if target.exists() else None


def create_app(hub: Hub, controls: UIControls, workspaces: Path, *,
               home: PersistentMCP, delegate: PersistentMCP, token: str | None = None,
               sample_rate: int = 24000, music_dir: Path | None = None) -> Starlette:
    snap = Snapshot()

    def authorized(params) -> bool:
        return not token or params.get("token") == token

    async def index(request: Request) -> Response:
        if not authorized(request.query_params):
            return PlainTextResponse("Add ?token=... to the URL", status_code=401)
        return FileResponse(STATIC / "index.html")

    async def files(request: Request) -> Response:
        if not authorized(request.query_params):
            return PlainTextResponse("unauthorized", status_code=401)
        rel = request.path_params.get("rel", "")
        target = safe_workspace_path(workspaces, rel)
        if target is None:
            return PlainTextResponse("not found", status_code=404)
        query = f"?{request.url.query}" if request.url.query else ""
        if target.is_dir():
            if (target / "index.html").exists() and not rel.endswith("/"):
                # redirect so the site's relative links (style.css, img/...) resolve
                return RedirectResponse(f"/files/{rel.rstrip('/')}/index.html{query}")
            rows = "".join(
                f'<li><a href="/files/{html.escape(str(p.relative_to(workspaces.resolve())))}{query}">'
                f"{html.escape(p.name)}{'/' if p.is_dir() else ''}</a></li>"
                for p in sorted(target.iterdir()) if not p.name.startswith(".")
            )
            return HTMLResponse(f"<h3>{html.escape(rel or 'workspaces')}</h3><ul>{rows}</ul>")
        return FileResponse(target)

    async def music(request: Request) -> Response:
        """Music files for browser playback (the home-mcp player runs in browser mode)."""
        if not authorized(request.query_params) or music_dir is None:
            return PlainTextResponse("not found", status_code=404)
        target = safe_workspace_path(music_dir, request.path_params.get("name", ""))
        if target is None or not target.is_file():
            return PlainTextResponse("not found", status_code=404)
        return FileResponse(target)

    async def ws_endpoint(ws: WebSocket) -> None:
        if not authorized(ws.query_params):
            await ws.close(code=4401)
            return
        await ws.accept()
        await ws.send_text(json.dumps({"type": "hello", "sample_rate": sample_rate,
                                       "mute_while_speaking": controls.mute_while_speaking()}))
        await ws.send_text(json.dumps({"type": "tasks", "tasks": snap.tasks}))
        await ws.send_text(json.dumps({"type": "devices", "devices": snap.devices}))
        await hub.add(ws)
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                kind = msg.get("type")
                if kind == "say" and msg.get("text", "").strip():
                    await controls.say(msg["text"].strip())
                elif kind == "cancel_task":
                    await hub.emit("notice", text=await controls.cancel_task(msg["id"]))
                elif kind == "mute_while_speaking":
                    controls.set_mute_while_speaking(bool(msg.get("on")))
                    await hub.emit("mute_while_speaking", keep=False, on=bool(msg.get("on")))
                elif kind == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "t": msg.get("t")}))
                elif kind == "rtt":
                    hub.rtt_ms[ws] = float(msg.get("ms", 0))
                elif kind in ("dismiss_alarm", "end_call"):  # buttons on the Home widgets
                    try:
                        await hub.emit("notice", text=await home.call(kind, {}))
                    except Exception as e:
                        logger.debug(f"UI: {kind} failed: {e}")
                elif kind == "music_ended":
                    try:
                        await home.call("music_track_ended", {})
                    except Exception as e:
                        logger.debug(f"UI: music_track_ended failed: {e}")
                elif kind == "playback_started":
                    controls.on_playback_started(int(msg["burst"]), hub.rtt_ms.get(ws, 0.0))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"UI websocket error: {e!r}")
        finally:
            hub.remove(ws)

    async def poll_state() -> None:
        while True:
            try:
                snap.tasks = await delegate.call("task_status", {})
                await hub.emit("tasks", keep=False, tasks=snap.tasks)
            except Exception as e:
                logger.debug(f"UI: task_status failed: {e}")
            try:
                snap.devices = await home.call("device_state", {})
                await hub.emit("devices", keep=False, devices=snap.devices)
            except Exception as e:
                logger.debug(f"UI: device_state failed: {e}")
            await asyncio.sleep(1.0)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        poller = asyncio.create_task(poll_state())
        try:
            yield
        finally:
            poller.cancel()

    return Starlette(
        routes=[
            Route("/", index),
            Route("/files/{rel:path}", files),
            Route("/files", files),
            Route("/music/{name:path}", music),
            WebSocketRoute("/ws", ws_endpoint),
        ],
        lifespan=lifespan,
    )


def playback_latency_record(sent_at: dict[int, float], burst: int, rtt_ms: float) -> dict | None:
    """Pi->speaker delay for one utterance: (report arrival - first chunk sent) - one-way return."""
    t = sent_at.pop(burst, None)
    if t is None:
        return None
    return {"type": "browser_playback", "ts": t,
            "browser_playback_ms": round((time.time() - t) * 1000 - rtt_ms / 2, 1)}
