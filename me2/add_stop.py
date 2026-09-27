"""Add real task-oriented commands from STOP.

Four data interventions have now come back flat on the headline, and the reason
is visible in one number: of 4,762 `timer.set` training utterances, exactly 146
are real human recordings. Mined LibriSpeech prose took the number head from
0.069 to ~0.70 on read speech and moved Ken's timer clips not at all, because a
number read from an audiobook does not carry the stress of a number spoken as a
command.

STOP is that missing distribution - real speakers issuing task-oriented
commands - and its timer domain alone holds 11,346 training utterances against
our 146.

The mapping is deliberately narrow, following `add_slurp.py`: an utterance is
kept only when its semantic parse lands exactly on something `commands.yaml`
can express, and is dropped otherwise. A wrong label spread across thousands of
recordings costs more than the coverage is worth.

Unlike the mined LibriSpeech rows, these are genuine commands, so they carry
their true intent and supervise both heads. The `supervise_intent: false` flag
existed only because prose labelled `none` created a 22:1 reject skew; nothing
like that applies here.
"""
import argparse
import json
import random
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

from mine_librispeech import COMPOUND_NEXT, WORD_VALUES

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]
MAXD = SPEC["audio"]["max_duration_s"]

# WORD_VALUES covers TEN..SIXTY, which is all the LibriSpeech miner needed.
# Timer durations also use the single digits, and "a minute" means one.
ONES = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
        "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "A": 1, "AN": 1}

UNITS = {
    "SECOND": "seconds", "SECONDS": "seconds", "SEC": "seconds", "SECS": "seconds",
    "MINUTE": "minutes", "MINUTES": "minutes", "MIN": "minutes", "MINS": "minutes",
    "HOUR": "hours", "HOURS": "hours", "HR": "hours", "HRS": "hours",
}

# Anything fractional or additive is outside what a single {number} {unit} slot
# pair can represent, so those utterances are dropped rather than rounded
AMBIGUOUS = {"HALF", "QUARTER", "AND", "POINT", "THIRD"}

# STOP intent -> (our intent, slots). Only exact matches appear here; every
# other STOP intent becomes a hard negative or is dropped.
DIRECT = {
    "GET_WEATHER": ("query.info", {"kind": "weather"}),
    "PLAY_MUSIC": ("media.control", {"action": "play"}),
    "PAUSE_MUSIC": ("media.control", {"action": "pause"}),
    "STOP_MUSIC": ("media.control", {"action": "stop"}),
    "SKIP_TRACK_MUSIC": ("media.control", {"action": "next"}),
}

# Real speech that sounds like a command but is not one of ours - the hardest
# negatives available, and the class the wake word cannot filter
NEGATIVE_DOMAINS = {"alarm", "event", "messaging", "navigation", "reminder"}

SLOT_RE = re.compile(r"\[SL:([A-Z_]+)\s+(.*?)\s*\]")
INTENT_RE = re.compile(r"\[IN:([A-Z_]+)")


def parse_intent(parse: str) -> str | None:
    """Return the top-level STOP intent name, or None."""
    found = INTENT_RE.search(parse)
    return found.group(1) if found else None


def parse_slot(parse: str, name: str) -> str | None:
    """Return the text of the first slot with this name, or None."""
    for slot, text in SLOT_RE.findall(parse):
        if slot == name:
            return text
    return None


def parse_duration(text: str) -> tuple[int, str] | None:
    """
    Read a timer duration out of an SL:DATE_TIME span.

    Handles "TEN MINUTES", "TWENTY FIVE MINUTES", "A MINUTE" and "SET TIMER FOR
    EIGHT MINUTE". Returns None whenever the span is anything the spec's single
    {number} {unit} pair cannot hold - no unit at all, a fraction, a range, or
    two different numbers.

    Args:
        text: The slot's text, already uppercased by STOP's normalisation

    Returns:
        (number, unit) if the span maps cleanly, else None
    """
    words = text.split()
    if any(word in AMBIGUOUS for word in words):
        return None

    unit = next((UNITS[word] for word in words if word in UNITS), None)
    if unit is None:
        return None

    # Walk left to right so "TWENTY FIVE" collapses to 25 before either part is
    # read as a standalone value
    values, index = [], 0
    while index < len(words):
        word = words[index]
        if word in WORD_VALUES:
            value = WORD_VALUES[word]
            following = words[index + 1] if index + 1 < len(words) else None
            if value >= 20 and value % 10 == 0 and following in COMPOUND_NEXT:
                value += ONES.get(following, 0)
                index += 1
            values.append(value)
        elif word in ONES and word not in {"A", "AN"}:
            values.append(ONES[word])
        elif word in {"A", "AN"} and index + 1 < len(words) and words[index + 1] in UNITS:
            values.append(1)
        elif word.isdigit():
            values.append(int(word))
        index += 1

    if len(values) != 1:
        return None

    number = values[0]
    low, high = SPEC["slots"]["number"]["range"]
    by_unit = SPEC["intents"]["timer.set"].get("number_range_by_unit", {})
    low, high = by_unit.get(unit, [low, high])

    return (number, unit) if low <= number <= high else None


def map_row(domain: str, parse: str) -> tuple[str, dict] | None:
    """
    Map one STOP utterance onto our label space.

    Args:
        domain: STOP domain name
        parse: The decoupled normalised semantic parse

    Returns:
        (intent, slots), or None when the utterance should be dropped
    """
    intent = parse_intent(parse)
    if intent is None:
        return None

    if intent == "CREATE_TIMER":
        span = parse_slot(parse, "DATE_TIME")
        if span is None:
            return None
        duration = parse_duration(span)
        if duration is None:
            return None
        return "timer.set", {"number": duration[0], "unit": duration[1]}

    if intent in DIRECT:
        return DIRECT[intent]

    if domain in NEGATIVE_DOMAINS or intent.startswith("UNSUPPORTED"):
        return "none", {}

    return None


def write_wav(src: Path, dst: Path) -> float | None:
    """
    Copy one STOP clip into the dataset at the spec's sample rate.

    Args:
        src: Source wav
        dst: Destination inside the dataset

    Returns:
        Duration in seconds, or None if the clip cannot be used
    """
    try:
        audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
    except Exception:
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SR:
        from math import gcd
        factor = gcd(int(sr), SR)
        audio = resample_poly(audio, SR // factor, int(sr) // factor)

    duration = len(audio) / SR
    if duration < 0.3 or duration > MAXD:
        return None

    peak = np.abs(audio).max()
    if peak < 1e-4:
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((audio / max(peak, 1e-9) * 0.9 * 32767).astype("<i2").tobytes())

    return duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", default="data/stop/stop")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--per-number", type=int, default=300,
                    help="cap per timer number value, so ten and twenty cannot dominate")
    ap.add_argument("--per-intent", type=int, default=3000,
                    help="cap for query.info and media.control")
    ap.add_argument("--per-negative", type=int, default=3000,
                    help="cap for the hard negatives, held separate so raising the "
                         "command intents cannot re-create a reject skew")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root, outdir = REPO / args.stop, REPO / args.out
    rng = random.Random(SPEC["splits"]["tts_speakers"]["seed"])
    split_of = {"train": "train", "eval": "val", "test": "test"}

    rows, dropped, missing = [], Counter(), 0
    for source, split in split_of.items():
        # Caps are per split so dev and test are not starved by a train-heavy
        # shuffle, scaled down for the smaller splits
        scale = {"train": 1.0, "val": 0.15, "test": 0.25}[split]
        seen_number, seen_intent = Counter(), Counter()

        lines = (root / "manifests" / f"{source}.tsv").read_text().splitlines()[1:]
        rng.shuffle(lines)

        for line in lines:
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            file_id, domain, gender, native, utterance = parts[:5]
            parse = parts[8]

            mapped = map_row(domain, parse)
            if mapped is None:
                dropped[domain] += 1
                continue
            intent, slots = mapped

            if intent == "timer.set":
                if seen_number[slots["number"]] >= args.per_number * scale:
                    continue
            else:
                cap = args.per_negative if intent == "none" else args.per_intent
                if seen_intent[intent] >= cap * scale:
                    continue

            src = root / file_id
            if not src.exists():
                missing += 1
                continue

            rel = f"{split}/{intent.replace('.', '_')}/stop_{Path(file_id).stem}_{domain}.wav"
            duration = None if args.dry_run else write_wav(src, outdir / rel)
            if not args.dry_run and duration is None:
                dropped["unusable audio"] += 1
                continue

            if intent == "timer.set":
                seen_number[slots["number"]] += 1
            seen_intent[intent] += 1

            rows.append({
                "file": rel,
                "split": split,
                "intent": intent,
                "text": utterance,
                "slots": slots,
                "model": "stop",
                "speaker_id": f"stop_{source}_{native}_{gender}",
                "speaker_key": f"stop|{source}",
                "duration_s": None if duration is None else round(duration, 3),
                "length_scale": None,
                "native": native,
                "gender": gender,
            })

    by_split = defaultdict(list)
    for record in rows:
        by_split[record["split"]].append(record)

    print(f"mapped {len(rows):,} utterances   missing audio {missing:,}")
    print("dropped:", dict(dropped.most_common(6)))
    print()
    for split, records in sorted(by_split.items()):
        intents = Counter(r["intent"] for r in records)
        print(f"  {split:6s} +{len(records):6,d}  {dict(sorted(intents.items()))}")

    numbers = Counter(r["slots"]["number"] for r in rows if r["intent"] == "timer.set")
    print(f"\n  timer numbers ({sum(numbers.values()):,} utts, {len(numbers)} values):")
    print("   ", dict(sorted(numbers.items())))
    native = Counter(r["native"] for r in rows)
    print(f"\n  native speakers: {dict(native)}")

    if args.dry_run:
        print("\n  dry run - nothing written")
        return

    for split, records in sorted(by_split.items()):
        with open(outdir / f"manifest_{split}.jsonl", "a") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    print(f"\n  appended to manifests under {outdir}")


if __name__ == "__main__":
    main()
