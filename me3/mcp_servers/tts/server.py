"""tts-mcp: text-to-speech MCP server wrapping cloud TTS APIs (no inference on the Pi).

Called once per sentence while the LLM streams, so audio starts after the first sentence.
Returns base64 16-bit mono PCM at the configured sample rate.
Backends: deepgram_aura, cartesia, edge_tts (free, no key).

Run:  python -m mcp_servers.tts.server
"""

from __future__ import annotations

import base64
import time
from typing import Literal

import httpx
from mcp.server.mcpserver import MCPServer

from common.config import env, load_config

cfg = load_config()
SAMPLE_RATE = cfg["tts"]["sample_rate"]
_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))

mcp = MCPServer(name="tts", instructions="Speech synthesis: turn a sentence into 16-bit mono PCM.")


async def _deepgram_aura(text: str) -> bytes:
    r = await _client.post(
        "https://api.deepgram.com/v1/speak",
        params={
            "model": cfg["tts"]["deepgram_voice"], "encoding": "linear16",
            "sample_rate": SAMPLE_RATE, "container": "none",
        },
        headers={"Authorization": f"Token {env('DEEPGRAM_API_KEY')}"},
        json={"text": text},
    )
    r.raise_for_status()
    return r.content


async def _cartesia(text: str) -> bytes:
    r = await _client.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={"X-API-Key": env("CARTESIA_API_KEY"), "Cartesia-Version": "2025-04-16"},
        json={
            "model_id": "sonic-2",
            "transcript": text,
            "voice": {"mode": "id", "id": cfg["tts"]["cartesia_voice"]},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
        },
    )
    r.raise_for_status()
    return r.content


async def _edge_tts(text: str) -> bytes:
    # Free Microsoft Edge voices; returns MP3, decoded to PCM (plain decoding, not inference).
    import edge_tts
    import miniaudio

    mp3 = bytearray()
    async for chunk in edge_tts.Communicate(text, cfg["tts"]["edge_voice"]).stream():
        if chunk["type"] == "audio":
            mp3.extend(chunk["data"])
    decoded = miniaudio.decode(
        bytes(mp3), output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=SAMPLE_RATE
    )
    return decoded.samples.tobytes()


BACKENDS = {"deepgram_aura": _deepgram_aura, "cartesia": _cartesia, "edge_tts": _edge_tts}


@mcp.tool()
async def synthesize(
    text: str,
    backend: Literal["default", "deepgram_aura", "cartesia", "edge_tts"] = "default",
) -> dict:
    """Synthesize speech. Returns audio_b64 (16-bit mono PCM), sample_rate, backend, latency_ms."""
    name = cfg["tts"]["backend"] if backend == "default" else backend
    t0 = time.perf_counter()
    pcm = await BACKENDS[name](text)
    return {
        "audio_b64": base64.b64encode(pcm).decode(),
        "sample_rate": SAMPLE_RATE,
        "backend": name,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


if __name__ == "__main__":
    s = cfg["mcp_servers"]["tts"]
    mcp.run("streamable-http", host=s["host"], port=s["port"])
