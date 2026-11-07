"""Forwards conversation events from the pipeline to the browser UI."""

from __future__ import annotations

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from agent.processors.conversation import CACHED_PHRASES
from agent.web.hub import Hub

INSTANT = {p.lower() for p in CACHED_PHRASES}


class UIObserver(BaseObserver):
    def __init__(self, hub: Hub, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self.hub = hub
        self._tool_calls: set[str] = set()
        self._state: str | None = None

    async def _status(self, state: str) -> None:
        if state != self._state:  # speaking frames are broadcast up+down: emit once
            self._state = state
            await self.hub.emit("status", keep=False, state=state)

    async def on_push_frame(self, data: FramePushed):
        f = data.frame
        if isinstance(f, UserStartedSpeakingFrame):
            await self._status("user_speaking")
        elif isinstance(f, UserStoppedSpeakingFrame):
            await self._status("thinking")
        elif isinstance(f, BotStartedSpeakingFrame):
            await self._status("speaking")
        elif isinstance(f, BotStoppedSpeakingFrame):
            await self._status("listening")
        elif isinstance(f, InterruptionFrame):
            await self.hub.emit("interrupted")
        elif isinstance(f, TranscriptionFrame):
            await self.hub.emit("user_text", text=f.text)
        elif isinstance(f, TTSTextFrame) and f.text.strip():
            await self.hub.emit("bot_text", text=f.text.strip(),
                                instant=f.text.strip().lower() in INSTANT)
        elif isinstance(f, FunctionCallInProgressFrame) and f.tool_call_id not in self._tool_calls:
            self._tool_calls.add(f.tool_call_id)
            await self.hub.emit("tool", name=f.function_name, args=f.arguments)
