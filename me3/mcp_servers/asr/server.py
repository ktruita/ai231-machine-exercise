"""asr-mcp: speech-to-text MCP server wrapping cloud ASR APIs (no inference on the Pi).

One call per VAD-segmented utterance: 16-bit mono PCM (base64) in, transcript out.
Backends: deepgram (primary), groq_whisper (fallback). Select per call or via config.

Run:  python -m mcp_servers.asr.server
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
import wave
from typing import Literal

import httpx
from mcp.server.mcpserver import MCPServer

from common.config import env, load_config

cfg = load_config()
_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))

mcp = MCPServer(name="asr", instructions="Speech recognition: transcribe short PCM utterances.")


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def _deepgram(wav: bytes, language: str) -> str:
    r = await _client.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": cfg["asr"]["deepgram_model"], "language": language, "smart_format": "true"},
        headers={"Authorization": f"Token {env('DEEPGRAM_API_KEY')}", "Content-Type": "audio/wav"},
        content=wav,
    )
    r.raise_for_status()
    return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


async def _groq_whisper(wav: bytes, language: str) -> str:
    r = await _client.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"},
        files={"file": ("audio.wav", wav, "audio/wav")},
        data={"model": cfg["asr"]["groq_model"], "language": language, "response_format": "json"},
    )
    r.raise_for_status()
    return r.json()["text"].strip()


BACKENDS = {"deepgram": _deepgram, "groq_whisper": _groq_whisper}


async def hedged(make_call, hedge_after_s: float):
    """Tail-latency fix: the API is usually ~1 s but sometimes takes 5+ s. If the first request
    isn't back after `hedge_after_s` (or fails), send an identical second one and use whichever
    answers first. Costs one extra request only on the slow ones. 0 disables."""
    first = asyncio.ensure_future(make_call())
    if hedge_after_s <= 0:
        return await first
    done, _ = await asyncio.wait({first}, timeout=hedge_after_s)
    if done and not first.exception():
        return first.result()
    tasks = {first, asyncio.ensure_future(make_call())} if not done else {asyncio.ensure_future(make_call())}
    error = None
    try:
        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is None:
                    return t.result()
                error = t.exception()
        raise error
    finally:
        for t in tasks:
            t.cancel()


@mcp.tool()
async def transcribe(
    audio_b64: str,
    sample_rate: int = 16000,
    backend: Literal["default", "deepgram", "groq_whisper"] = "default",
) -> dict:
    """Transcribe one utterance of base64 16-bit mono PCM. Returns text, backend and latency_ms."""
    name = cfg["asr"]["backend"] if backend == "default" else backend
    wav = pcm_to_wav(base64.b64decode(audio_b64), sample_rate)
    t0 = time.perf_counter()
    text = await hedged(lambda: BACKENDS[name](wav, cfg["asr"]["language"]),
                        float(cfg["asr"].get("hedge_after_s") or 0))
    return {"text": text, "backend": name, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


if __name__ == "__main__":
    s = cfg["mcp_servers"]["asr"]
    mcp.run("streamable-http", host=s["host"], port=s["port"])
