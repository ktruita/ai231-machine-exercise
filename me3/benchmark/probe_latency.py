"""Measure each stage of the voice loop separately, from this device, to pick the fastest
backends/models. Cost: a handful of tiny API calls (well under $0.01).

Needs the MCP servers running (scripts/run_servers.sh).
Run:  python -m benchmark.probe_latency [--models qwen/qwen3-30b-a3b-instruct-2507 openai/gpt-4.1-nano]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import statistics
import time
import wave

import httpx

from benchmark.render_audio import audio_path
from common.config import ROOT, load_config, server_url
from common.mcp_session import PersistentMCP

QUESTION = "Could you suggest some books to help with personal development?"
SHORT = "Sure!"
LONG = "Atomic Habits by James Clear is a great place to start, it's practical and easy to read."


def med(xs: list[float]) -> str:
    return f"{statistics.median(xs):6.0f} ms" if xs else "   failed"


async def probe_asr(asr: PersistentMCP, runs: int) -> None:
    with wave.open(str(audio_path("WEATHER-v2", "text"))) as w:
        pcm = w.readframes(w.getnframes())
    print("\nSpeech-to-text (one 1-2 s utterance; time = upload + recognition):")
    for backend, key in [("deepgram", "DEEPGRAM_API_KEY"), ("groq_whisper", "GROQ_API_KEY")]:
        if not os.environ.get(key):
            print(f"  {backend:14s}   skipped (no {key})")
            continue
        times, text = [], ""
        for _ in range(runs):
            t0 = time.perf_counter()
            try:
                r = await asr.call("transcribe", {"audio_b64": base64.b64encode(pcm).decode(),
                                                  "sample_rate": 16000, "backend": backend})
                times.append((time.perf_counter() - t0) * 1000)
                text = r["text"]
            except Exception as e:
                text = f"error: {e}"[:60]
        print(f"  {backend:14s} {med(times)}   heard: {text!r}")


async def probe_tts(tts: PersistentMCP, runs: int) -> None:
    print("\nText-to-speech (time until the whole clip is back):")
    cfg = load_config()["tts"]
    candidates = [("deepgram_aura", "DEEPGRAM_API_KEY"), ("edge_tts", None)]
    if cfg.get("cartesia_voice"):
        candidates.append(("cartesia", "CARTESIA_API_KEY"))
    for backend, key in candidates:
        if key and not os.environ.get(key):
            print(f"  {backend:14s}   skipped (no {key})")
            continue
        for label, text in [("short", SHORT), ("sentence", LONG)]:
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                try:
                    await tts.call("synthesize", {"text": text, "backend": backend})
                    times.append((time.perf_counter() - t0) * 1000)
                except Exception:
                    pass
            print(f"  {backend:14s} {label:8s} {med(times)}")


async def tool_schemas() -> list[dict]:
    """The same tools the voice agent sees (so prompt size / TTFT is realistic)."""
    from agent.main import LLM_TOOLS

    out = []
    for name, allowed in LLM_TOOLS.items():
        if name == "delegate":
            continue
        c = await PersistentMCP(server_url(name)).start()
        listed = await c._client.list_tools()
        for t in listed.tools:
            if t.name in allowed:
                out.append({"type": "function", "function": {
                    "name": t.name, "description": t.description or "",
                    "parameters": t.input_schema}})
        await c.close()
    return out


async def probe_llm(models: list[str], runs: int) -> None:
    cfg = load_config()["llm"]
    system = (ROOT / "agent/prompts/system.md").read_text()
    tools = await tool_schemas()
    print(f"\nLLM via OpenRouter ({len(tools)} tools in the prompt, provider sort={cfg['provider_sort']}):")
    print("  model                                         first word   first sentence  served by")
    async with httpx.AsyncClient(timeout=60) as client:
        for model in models:
            ttft, tfs, provider = [], [], ""
            for _ in range(runs):
                t0 = time.perf_counter()
                first = sentence = None
                text = ""
                async with client.stream(
                    "POST", "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                    json={"model": model, "stream": True, "temperature": cfg["temperature"],
                          "provider": {"sort": cfg["provider_sort"]}, "tools": tools,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": QUESTION}]},
                ) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        import json
                        chunk = json.loads(line[6:])
                        provider = chunk.get("provider", provider)
                        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                        piece = delta.get("content") or ""
                        if (piece or delta.get("tool_calls")) and first is None:
                            first = time.perf_counter()
                        text += piece
                        if sentence is None and any(p in text for p in ".!?"):
                            sentence = time.perf_counter()
                if first:
                    ttft.append((first - t0) * 1000)
                if sentence:
                    tfs.append((sentence - t0) * 1000)
            print(f"  {model:45s} {med(ttft)}    {med(tfs)}     {provider}")


async def main(models: list[str], runs: int) -> None:
    asr = await PersistentMCP(server_url("asr")).start()
    tts = await PersistentMCP(server_url("tts")).start()
    try:
        await probe_asr(asr, runs)
        await probe_tts(tts, runs)
        await probe_llm(models, runs)
    finally:
        await asr.close()
        await tts.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[
        load_config()["llm"]["model"], "openai/gpt-4.1-nano", "google/gemini-2.5-flash-lite"])
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    asyncio.run(main(a.models, a.runs))
