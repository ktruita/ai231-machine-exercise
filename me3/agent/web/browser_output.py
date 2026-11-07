"""Output transport that plays the bot's voice in the laptop browser instead of on the Pi.

Audio is paced in real time here (like a sound card would), so the pipeline's
bot-speaking state, barge-in and latency stamps stay correct. On interruption the
browser is told to drop whatever it has queued.
"""

from __future__ import annotations

import asyncio
import itertools
import time

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from agent.web.hub import Hub


class BrowserAudioOutput(BaseOutputTransport):
    def __init__(self, hub: Hub, sample_rate: int):
        super().__init__(TransportParams(audio_out_enabled=True, audio_out_sample_rate=sample_rate))
        self.hub = hub
        self._last_write = 0.0
        self._burst_ids = itertools.count(1)
        self.sent_at: dict[int, float] = {}  # burst id -> wall time first chunk left the Pi

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterruptionFrame):
            await self.hub.emit("clear_audio", keep=False)
            self._last_write = 0.0
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        now = time.time()
        if now - self._last_write > 0.25:
            # First chunk after a gap = new utterance. Tag it so the browser can report
            # when it actually started playing (network + jitter buffer delay).
            burst = next(self._burst_ids)
            self.sent_at[burst] = time.time()
            await self.hub.emit("audio_start", keep=False, burst=burst, sample_rate=frame.sample_rate)
        await self.hub.audio(frame.audio)
        self._last_write = now
        await asyncio.sleep(len(frame.audio) / 2 / frame.sample_rate)
        return True
