"""Benchmark transport: plays WAV/PCM utterances into the pipeline as if spoken into the mic,
and "plays" bot audio in real time (so timing, barge-in and VAD behave like the real device).
"""

from __future__ import annotations

import asyncio
import time

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

IN_RATE = 16000
CHUNK_S = 0.02
CHUNK_BYTES = int(IN_RATE * CHUNK_S) * 2


class FileAudioInput(BaseInputTransport):
    """Streams silence continuously; `say(pcm)` splices an utterance into the stream."""

    def __init__(self, params: TransportParams | None = None):
        super().__init__(params or TransportParams(audio_in_enabled=True, audio_in_sample_rate=IN_RATE))
        self._queue: asyncio.Queue[tuple[bytes, asyncio.Future]] = asyncio.Queue()
        self._feeder: asyncio.Task | None = None
        self.speech_started_at: float | None = None
        self.speech_ended_at: float | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        self._feeder = self.create_task(self._feed())

    async def say(self, pcm16k: bytes) -> tuple[float, float]:
        """Play an utterance; returns (start, end) wall-clock of the actual speech."""
        done = asyncio.get_running_loop().create_future()
        await self._queue.put((pcm16k, done))
        return await done

    async def _push(self, chunk: bytes, t_next: float) -> float:
        await self.push_audio_frame(InputAudioRawFrame(audio=chunk, sample_rate=IN_RATE, num_channels=1))
        t_next += CHUNK_S
        await asyncio.sleep(max(0.0, t_next - time.monotonic()))
        return t_next

    async def _feed(self):
        silence = b"\x00" * CHUNK_BYTES
        t_next = time.monotonic()
        while True:
            try:
                pcm, done = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                t_next = await self._push(silence, t_next)
                continue
            start = time.time()
            for i in range(0, len(pcm), CHUNK_BYTES):
                chunk = pcm[i : i + CHUNK_BYTES].ljust(CHUNK_BYTES, b"\x00")
                t_next = await self._push(chunk, t_next)
            if not done.done():
                done.set_result((start, time.time()))

    async def cleanup(self):
        await super().cleanup()
        if self._feeder:
            await self.cancel_task(self._feeder)


class RecordingOutput(BaseOutputTransport):
    """Consumes bot audio at real-time pace (like a speaker) and keeps a recording."""

    def __init__(self, sample_rate: int, params: TransportParams | None = None):
        super().__init__(params or TransportParams(audio_out_enabled=True, audio_out_sample_rate=sample_rate))
        self.recording = bytearray()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        self.recording.extend(frame.audio)
        await asyncio.sleep(len(frame.audio) / 2 / frame.sample_rate)
        return True
