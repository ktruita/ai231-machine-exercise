"""Mean / p95 per pipeline stage, in the class template's definitions, from latency.jsonl files.

One clock (time.time on the Pi), every turn anchored at VAD end-of-speech:
  E2E latency   VAD end -> bot audio out           voice_to_voice_ms
  ASR final     VAD end -> transcript              asr_ms
  LLM TTFT      turn end (LLM triggered) -> 1st token   llm_ttft_ms - endpoint_ms
  TTS TTFB      1st token -> 1st synthesized audio      tts_first_ms - llm_ttft_ms
  Barge-in      user speech start (bot talking) -> bot stopped   barge_in_ms
  WER           word error rate of the transcript vs the bank text (run_bench logs only:
                their turns are labelled with the bank item id)

Run on the Pi:
  .venv/bin/python -m benchmark.latency_table                       # newest fdbench run
  .venv/bin/python -m benchmark.latency_table benchmark/results/*/default/latency.jsonl
"""

from __future__ import annotations

import glob
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


BANK = ROOT / "benchmark/bank/commands.jsonl"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


def wer(turns: list[dict]) -> tuple[float, int] | None:
    """Corpus WER over the first transcript of every bank item (spoken user text known)."""
    if not BANK.exists():
        return None
    refs = {}
    for line in open(BANK):
        item = json.loads(line)
        if item.get("text"):
            refs[item["id"]] = item["text"]
    seen, errors, total = set(), 0, 0
    for t in turns:
        label = t.get("label")
        if label in refs and label not in seen and t.get("transcript"):
            seen.add(label)
            ref = _words(refs[label])
            errors += edit_distance(ref, _words(t["transcript"]))
            total += len(ref)
    return (errors / total, len(seen)) if total else None


def p95(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]


def main(paths: list[str]) -> None:
    if not paths:
        found = sorted(glob.glob(str(ROOT / "fd-benchmark/runs/**/latency.jsonl"), recursive=True),
                       key=lambda p: Path(p).stat().st_mtime)
        paths = found[-1:]
    if not paths:
        sys.exit("no latency.jsonl found; pass one or more paths")
    turns, all_turns, barge = [], [], []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r.get("type") == "turn":
                all_turns.append(r)
                if r.get("voice_to_voice_ms"):
                    turns.append(r)
            elif r.get("type") == "barge_in" and r.get("barge_in_ms"):
                barge.append(r["barge_in_ms"])

    def stage(f):
        out = []
        for t in turns:
            try:
                v = f(t)
            except TypeError:  # a stage missing in this turn (e.g. no LLM call)
                continue
            if v is not None and v >= 0:
                out.append(v)
        return out

    rows = [
        ("E2E latency", "end of speech -> audio out", stage(lambda t: t["voice_to_voice_ms"])),
        ("ASR final", "end of speech -> transcript", stage(lambda t: t["asr_ms"])),
        ("LLM TTFT", "turn end -> first token", stage(lambda t: t["llm_ttft_ms"] - t["endpoint_ms"])),
        ("TTS TTFB", "first token -> first audio", stage(lambda t: t["tts_first_ms"] - t["llm_ttft_ms"])),
        ("Barge-in", "speech -> playback stopped", barge),
    ]
    print(f"files: {', '.join(paths)}\nturns: {len(turns)}   barge-ins: {len(barge)}\n")
    print(f"{'metric':12s} {'definition':30s} {'n':>4s} {'mean':>8s} {'p95':>8s}")
    for name, definition, xs in rows:
        if xs:
            print(f"{name:12s} {definition:30s} {len(xs):4d} {statistics.mean(xs) / 1000:7.2f}s {p95(xs) / 1000:7.2f}s")
        else:
            print(f"{name:12s} {definition:30s} {0:4d} {'-':>8s} {'-':>8s}")
    w = wer(all_turns)
    if w:
        print(f"{'WER':12s} {'ASR words vs spoken bank text':30s} {w[1]:4d} {w[0] * 100:7.1f}%   (corpus)")
    else:
        print(f"{'WER':12s} {'(needs a run_bench latency.jsonl)':30s}")


if __name__ == "__main__":
    main(sys.argv[1:])
