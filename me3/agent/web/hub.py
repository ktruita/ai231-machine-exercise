"""Fan-out hub: every connected browser gets UI events (JSON) and bot audio (binary PCM)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from loguru import logger
from starlette.websockets import WebSocket


class Hub:
    def __init__(self, history: int = 200):
        self.clients: set[WebSocket] = set()
        self.recent: list[dict] = []  # replayed to browsers that connect mid-conversation
        self._history = history
        self.rtt_ms: dict[WebSocket, float] = {}

    async def add(self, ws: WebSocket) -> None:
        self.clients.add(ws)
        for event in self.recent:
            await ws.send_text(json.dumps(event))

    def remove(self, ws: WebSocket) -> None:
        self.clients.discard(ws)
        self.rtt_ms.pop(ws, None)

    @property
    def audio_listeners(self) -> int:
        return len(self.clients)

    async def emit(self, kind: str, *, keep: bool = True, **data: Any) -> None:
        event = {"type": kind, "ts": time.time(), **data}
        if keep:
            self.recent = (self.recent + [event])[-self._history:]
        await self._send(lambda ws: ws.send_text(json.dumps(event)))

    async def audio(self, pcm: bytes) -> None:
        await self._send(lambda ws: ws.send_bytes(pcm))

    async def _send(self, op) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await asyncio.wait_for(op(ws), timeout=1.0)
            except Exception as e:
                logger.debug(f"dropping UI client: {e!r}")
                dead.append(ws)
        for ws in dead:
            self.remove(ws)
