"""Aggregate benchmark results into one comparison table (markdown) across configurations.

Run:  python -m benchmark.report benchmark/results/<stamp>   (one dir per config inside)
      python -m benchmark.report benchmark/results/<stamp>/<config> ...
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def pct(values: list[float], q: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo))


def rate(flags: list[bool]) -> str:
    return f"{100 * sum(flags) / len(flags):.0f}% ({sum(flags)}/{len(flags)})" if flags else "–"


def summarize(run_dir: Path) -> dict:
    items = _jsonl(run_dir / "items.jsonl")
    lat = _jsonl(run_dir / "latency.jsonl")
    turns = [t for t in lat if t["type"] == "turn"]
    barge = [t["barge_in_ms"] for t in lat if t["type"] == "barge_in"]
    judge = [j for j in _jsonl(run_dir / "judge.jsonl") if "error" not in j]
    harness = _jsonl(run_dir / "harness.jsonl")
    intent_items = [i for i in items if "intent_ok" in i]
    overlap = [i for i in items if i["kind"] == "overlap"]
    by_label = {t["label"]: t for t in turns}

    # end-of-speech -> bot audio: add the VAD silence window back in
    v2v_true = [t["voice_to_voice_ms"] + 1000 * (t.get("vad_stop_secs") or 0)
                for t in turns if t.get("voice_to_voice_ms") is not None]
    delegate_ack = [by_label[i["id"]]["voice_to_voice_ms"] for i in items
                    if i["kind"] == "delegate" and i["id"] in by_label]
    s = {
        "config": run_dir.name,
        "turns": len(turns),
        "v2v_p50": pct(v2v_true, .5), "v2v_p90": pct(v2v_true, .9),
        "asr_p50": pct([t["asr_ms"] for t in turns], .5),
        "endpoint_p50": pct([t.get("endpoint_ms") for t in turns], .5),
        "ttft_p50": pct([t["llm_ttft_ms"] - t["endpoint_ms"] for t in turns
                         if t.get("llm_ttft_ms") and t.get("endpoint_ms")], .5),
        "tts_p50": pct([t["tts_first_ms"] - t["llm_ttft_ms"] for t in turns
                        if t.get("tts_first_ms") and t.get("llm_ttft_ms")], .5),
        "barge_p50": pct(barge, .5), "barge_p90": pct(barge, .9),
        "delegate_ack_p50": pct(delegate_ack, .5),
        "intent_acc": rate([i["intent_ok"] for i in intent_items]),
        "slot_acc": rate([i["slots_ok"] for i in intent_items if i["intent_ok"]]),
        "overlap_acc": rate([i["correct"] for i in overlap]),
        "backchannel_ok": rate([i["backchannel_ok"] for i in items if "backchannel_ok" in i]),
        "recall_ok": rate([i["recall_ok"] for i in items if "recall_ok" in i]),
        "harness": ", ".join(f"{h['harness']}:{h['status']}@{h['turnaround_s']}s" for h in harness) or "–",
    }
    for k in ("naturalness", "brevity", "task_success", "turn_taking"):
        vals = [j[k] for j in judge if isinstance(j.get(k), (int, float))]
        s[k] = round(statistics.mean(vals), 2) if vals else None
    s["misses"] = [f"{i['id']}: got {i['got_intent']} {i['got_slots']}" for i in intent_items
                   if not (i["intent_ok"] and i["slots_ok"])][:15]
    return s


COLUMNS = [
    ("config", "Config"), ("turns", "Turns"),
    ("v2v_p50", "Voice→voice p50 ms"), ("v2v_p90", "p90"),
    ("asr_p50", "ASR p50"), ("endpoint_p50", "End-of-turn p50"), ("ttft_p50", "LLM TTFT p50"), ("tts_p50", "TTS first p50"),
    ("barge_p50", "Barge-in p50 ms"), ("delegate_ack_p50", "Delegate ack p50"),
    ("intent_acc", "Intent acc"), ("slot_acc", "Slot acc"), ("overlap_acc", "Backch./steer acc"),
    ("backchannel_ok", "Yes?/Mhm? ok"), ("recall_ok", "Memory"),
    ("naturalness", "Natural"), ("brevity", "Brief"), ("task_success", "Task"), ("turn_taking", "Turn-taking"),
    ("harness", "Harness turnaround"),
]


def main(dirs: list[Path]) -> None:
    run_dirs = []
    for d in dirs:
        run_dirs += [d] if (d / "items.jsonl").exists() else sorted(p for p in d.iterdir() if (p / "items.jsonl").exists())
    rows = [summarize(d) for d in run_dirs]
    md = ["| " + " | ".join(h for _, h in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        md.append("| " + " | ".join("–" if r[k] is None else str(r[k]) for k, _ in COLUMNS) + " |")
    for r in rows:
        if r["misses"]:
            md.append(f"\n**Intent/slot misses — {r['config']}**\n" + "\n".join(f"- {m}" for m in r["misses"]))
    text = "\n".join(md)
    out = (dirs[0] if len(dirs) == 1 else Path(".")) / "REPORT.md"
    out.write_text(text + "\n")
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    main(ap.parse_args().dirs)
