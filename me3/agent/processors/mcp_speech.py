"""Pipecat STT/TTS services whose models live behind our own MCP servers (asr-mcp, tts-mcp)."""

from __future__ import annotations

import asyncio
import base64
import math
from collections.abc import AsyncGenerator, AsyncIterator

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from common.mcp_session import PersistentMCP


FAINT_TAG = "(faint, distant voice)"


def level_dbfs(pcm: bytes) -> float:
    """Loudness of the loudest half of a 16-bit PCM utterance (skips the silence around it)."""
    import array

    a = array.array("h", pcm)
    if not a:
        return -120.0
    frame = 320  # 20 ms at 16 kHz
    rms = sorted(
        math.sqrt(sum(x * x for x in a[i:i + frame]) / frame)
        for i in range(0, len(a) - frame + 1, frame)
    ) or [0.0]
    top = rms[len(rms) // 2:]
    return 20 * math.log10(max(sum(top) / len(top), 1.0) / 32768)


class MCPSegmentedSTTService(SegmentedSTTService):
    """Sends each VAD-segmented utterance to asr-mcp's `transcribe` tool.

    Also measures how loud each utterance is. Speech quieter than `faint_below_dbfs`
    (a TV, someone across the room) gets FAINT_TAG prepended, so the LLM can tell it was
    probably not meant for it (see ADDRESSED_PROMPT). A fixed level, not one relative to
    recent speech: real recordings and synthetic clips differ by ~10 dB in fdbench, so a
    relative rule would mark real users as faint.
    """

    def __init__(self, *, url: str, backend: str = "default",
                 faint_below_dbfs: float | None = None, **kwargs):
        super().__init__(
            sample_rate=16000,
            settings=STTSettings(model=None, language=Language.EN),
            **kwargs,
        )
        self._mcp = PersistentMCP(url)
        self._backend = backend
        self._faint_below = faint_below_dbfs  # None = off

    @property
    def wants_wav_segments(self) -> bool:
        return False  # asr-mcp takes raw PCM and wraps it itself

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_processing_metrics()
            result = await self._mcp.call("transcribe", {
                "audio_b64": base64.b64encode(audio).decode(),
                "sample_rate": self.sample_rate,
                "backend": self._backend,
            })
            await self.stop_processing_metrics()
            text = result["text"].strip()
            if text and self._faint_below is not None and len(text.split()) >= 3:
                level = level_dbfs(audio)
                faint = level < self._faint_below
                logger.debug(f"Level {level:.1f} dBFS{' -> faint' if faint else ''}")
                if faint:
                    text = f"{FAINT_TAG} {text}"
            if text:
                logger.debug(f"Transcription ({result['latency_ms']} ms): [{text}]")
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN, result=result)
        except Exception as e:
            yield ErrorFrame(error=f"asr-mcp error: {e}")

    async def cleanup(self):
        await super().cleanup()
        await self._mcp.close()


def split_for_speed(text: str, min_words: int = 12, head_min: int = 4, tail_min: int = 4) -> list[str]:
    """Split a long sentence at its first comma/colon/semicolon/dash that leaves at least
    `head_min` words before and `tail_min` after. The short head is synthesized (and starts
    playing) sooner, while the tail is synthesized during playback."""
    words = text.split()
    if len(words) < min_words:
        return [text]
    count = 0
    for i, w in enumerate(words[:-1]):
        count += 1
        if count >= head_min and len(words) - (i + 1) >= tail_min and (
                w.endswith((",", ";", ":")) or words[i + 1] in ("—", "–", "-")):
            cut = i + 1 if not words[i + 1] in ("—", "–", "-") else i + 2
            return [" ".join(words[:cut]), " ".join(words[cut:])]
    return [text]


class MCPTTSService(TTSService):
    """Sends each sentence to tts-mcp's `synthesize` tool.

    Short fixed phrases (backchannels/acks) are pre-synthesized at startup and served
    from memory, so "Mhm?" or "Let me check that." plays with no network round trip.
    """

    def __init__(self, *, url: str, backend: str = "default", cached_phrases: list[str] = (),
                 split_long: bool = False, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )
        self._mcp = PersistentMCP(url)
        self._backend = backend
        self._phrases = list(cached_phrases)
        self._cache: dict[str, tuple[bytes, int]] = {}
        self._split_long = split_long

    def can_generate_metrics(self) -> bool:
        return True

    @staticmethod
    def _key(text: str) -> str:
        return " ".join(text.lower().split())

    async def warm_cache(self) -> None:
        for phrase in self._phrases:
            try:
                r = await self._mcp.call("synthesize", {"text": phrase, "backend": self._backend})
                self._cache[self._key(phrase)] = (base64.b64decode(r["audio_b64"]), r["sample_rate"])
            except Exception as e:
                logger.warning(f"Could not pre-synthesize {phrase!r}: {e}")
        logger.info(f"TTS phrase cache: {len(self._cache)}/{len(self._phrases)} phrases")

    async def _synthesize(self, text: str) -> tuple[bytes, int]:
        cached = self._cache.get(self._key(text))
        if cached:
            return cached
        r = await self._mcp.call("synthesize", {"text": text, "backend": self._backend})
        return base64.b64decode(r["audio_b64"]), r["sample_rate"]

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_tts_usage_metrics(text)
            parts = split_for_speed(text) if self._split_long else [text]
            # all parts are requested at once; the first plays while the rest are synthesized
            jobs = [asyncio.ensure_future(self._synthesize(p)) for p in parts]
            try:
                for job in jobs:
                    pcm, sample_rate = await job

                    async def chunks(pcm: bytes = pcm) -> AsyncIterator[bytes]:
                        step = self.chunk_size
                        for i in range(0, len(pcm), step):
                            yield pcm[i : i + step]

                    async for frame in self._stream_audio_frames_from_iterator(
                        chunks(), in_sample_rate=sample_rate, context_id=context_id
                    ):
                        await self.stop_ttfb_metrics()
                        yield frame
            finally:
                for job in jobs:
                    job.cancel()
        except Exception as e:
            yield ErrorFrame(error=f"tts-mcp error: {e}")
        finally:
            await self.stop_ttfb_metrics()

    async def cleanup(self):
        await super().cleanup()
        await self._mcp.close()
