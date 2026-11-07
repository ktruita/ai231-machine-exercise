"""Render every bank utterance to 16 kHz mono WAV with a *user* voice distinct from the agent's.

Uses free Edge TTS (no key). Your own recordings can replace any file: keep the name
benchmark/bank/audio/<id>[.<part>].wav and re-run the benchmark.

Run:  python -m benchmark.render_audio [--voice en-US-GuyNeural]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave

import edge_tts
import miniaudio

from common.config import ROOT

BANK = ROOT / "benchmark/bank/commands.jsonl"
AUDIO_DIR = ROOT / "benchmark/bank/audio"
TEXT_FIELDS = ("text", "setup", "overlay", "followup")


def audio_path(item_id: str, field: str):
    return AUDIO_DIR / (f"{item_id}.wav" if field == "text" else f"{item_id}.{field}.wav")


async def render(text: str, voice: str, out) -> None:
    mp3 = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            mp3.extend(chunk["data"])
    pcm = miniaudio.decode(bytes(mp3), output_format=miniaudio.SampleFormat.SIGNED16,
                           nchannels=1, sample_rate=16000).samples.tobytes()
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)


async def main(voice: str, force: bool) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for line in open(BANK):
        item = json.loads(line)
        for field in TEXT_FIELDS:
            if field in item:
                out = audio_path(item["id"], field)
                if force or not out.exists():
                    jobs.append((item[field], out))
    sem = asyncio.Semaphore(4)

    async def one(text, out):
        async with sem:
            await render(text, voice, out)

    await asyncio.gather(*(one(t, o) for t, o in jobs))
    print(f"Rendered {len(jobs)} files -> {AUDIO_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-US-GuyNeural")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.voice, a.force))
