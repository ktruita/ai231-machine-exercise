"""Add SLURP real speech to the dataset.

SLURP is the only open corpus whose intents cover most of this spec with real
speakers, and it records every utterance through several microphones, so it
carries the channel variation synthetic audio cannot.

Mapping is deliberately conservative. An intent is only taken when SLURP's label
implies this spec's label without inference, and a slot is only filled when the
annotation gives a value this spec can represent. Everything else becomes a
negative or is dropped - a wrong label in 17,000 real recordings would do more
damage than the extra coverage is worth.
"""
import argparse, json, random, re, wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]
MAXD = SPEC["audio"]["max_duration_s"]

ANNOTATION = re.compile(r"\[\s*([^:\]]+?)\s*:\s*([^\]]+?)\s*\]")

# Lexical evidence that this spec could express the utterance at all. SLURP
# labels intent from meaning, so "i'm going to sleep now" is iot_hue_lighton and
# "do i need a jacket" is weather_query. Those are correct for SLURP and
# impossible for a template-based spec with no inference. Scoring a model on
# them measures the taxonomy gap, not the model, so they are dropped - not
# relabelled `none`, which would teach the model to reject its own domain.
LIGHT = r"\b(light|lights|lamp|lamps|brightness)\b"
IN_SCOPE = {
    "light.set": (LIGHT, r"\b(on|off|out|switch|shut|kill|cease|discontinue|extinguish|activate|deactivate|illuminate)\b"),
    "light.dim": (LIGHT, r"\b(dim|dimmer|dimmed|lower|down|darker|decrease|reduce|soften|darken|less)\b"),
    "media.control": (r"\b(play|playing|pause|stop|skip|next|volume|louder|quieter|mute|music|song|track)\b", None),
    "query.info": (r"\b(weather|forecast|rain|raining|snow|sunny|temperature|time|clock|hour)\b", None),
}


def in_scope(intent: str, sentence: str) -> bool:
    """
    Decide whether this spec could express the utterance.

    Args:
        intent: This spec's intent
        sentence: SLURP transcription

    Returns:
        True when the sentence carries the lexical content the spec's own
        templates are built from
    """
    if intent not in IN_SCOPE:
        return True

    required, qualifier = IN_SCOPE[intent]
    text = sentence.lower()

    return bool(re.search(required, text)) and (qualifier is None or bool(re.search(qualifier, text)))

# SLURP intent -> (this spec's intent, the slots it implies)
INTENT_MAP = {
    "iot_hue_lightoff": ("light.set", {"state": "off"}),
    "iot_hue_lighton": ("light.set", {"state": "on"}),
    "play_music": ("media.control", {"action": "play"}),
    "audio_volume_up": ("media.control", {"action": "volume_up"}),
    "audio_volume_down": ("media.control", {"action": "volume_down"}),
    "weather_query": ("query.info", {"kind": "weather"}),
    "datetime_query": ("query.info", {"kind": "time"}),
    # percent is optional for light.dim, so these are representable now. The
    # amount is left N/A: only 20 of 222 carry one and they read "a bit" or
    # "to max" rather than a value this spec can hold.
    "iot_hue_lightdim": ("light.dim", {}),
}

# Deliberately NOT mapped, with reasons:
#   iot_hue_lightup     - brightening is the opposite of dimming, not the same intent
#   iot_hue_lightchange - colour changes have no equivalent slot here
#   alarm_set           - an alarm at a clock time is not a countdown timer
#   audio_volume_mute   - no equivalent action in this spec
# These are dropped rather than made negatives: they are genuine light and audio
# commands, so labelling them `none` would teach the model to reject its own domain.
SKIP_INTENTS = ("iot_hue_lightup", "iot_hue_lightchange",
                "alarm_set", "audio_volume_mute", "audio_volume_other")

# SLURP house_place is free text. Values outside this spec's room list become N/A
# rather than being forced to the nearest room, which room being optional allows.
ROOM_MAP = {
    "living room": "living room", "livingroom": "living room",
    "kitchen": "kitchen",
    "bedroom": "bedroom", "bed room": "bedroom",
    "bathroom": "bathroom", "bath room": "bathroom",
    "office": "office",
    "garage": "garage",
}


def annotation_values(annotation: str) -> dict[str, str]:
    """Pull the [type : value] pairs out of a SLURP sentence annotation."""
    return {entity_type: value.lower() for entity_type, value in ANNOTATION.findall(annotation)}


def write_fixed(src: Path, dst: Path) -> float | None:
    """Convert a SLURP recording into the dataset, or None if it cannot be read."""
    try:
        audio, sr = sf.read(str(src), dtype="int16", always_2d=False)
    except Exception:
        return None

    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SR:
        return None

    audio = audio[:int(MAXD * SR)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(SR)
        handle.writeframes(audio.tobytes())

    return len(audio) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slurp", default="data/slurp")
    ap.add_argument("--audio", default="data/slurp/slurp_real")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--max-per-intent", type=int, default=2500)
    ap.add_argument("--max-negatives", type=int, default=4000)
    args = ap.parse_args()

    slurp, audio_dir, outdir = REPO / args.slurp, REPO / args.audio, REPO / args.out
    rng = random.Random(SPEC["splits"]["tts_speakers"]["seed"])
    split_map = {"train": "train", "devel": "val", "test": "test"}

    # Plan first, cap second, write last - writing 72,000 recordings only to
    # discard most of them would waste an hour of NFS time.
    planned = defaultdict(list)
    for slurp_split, split in split_map.items():
        for record in (json.loads(line) for line in open(slurp / f"{slurp_split}.jsonl")):
            intent_name = record["intent"]
            if intent_name in SKIP_INTENTS:
                continue

            if intent_name in INTENT_MAP:
                intent, slots = INTENT_MAP[intent_name]
                slots = dict(slots)
                if intent in ("light.set", "light.dim"):
                    place = annotation_values(record["sentence_annotation"]).get("house_place")
                    room = ROOM_MAP.get(place) if place else None
                    if room:
                        slots["room"] = room
            else:
                intent, slots = "none", {}

            if not in_scope(intent, record["sentence"]):
                continue

            for recording in record["recordings"]:
                if recording.get("status") != "correct":
                    continue
                planned[(split, intent)].append((recording["file"], record["sentence"], slots))

    rows = []
    for (split, intent), items in planned.items():
        rng.shuffle(items)
        cap = args.max_negatives if intent == "none" else args.max_per_intent
        share = max(int(cap * (0.7 if split == "train" else 0.15)), 1)

        for i, (filename, sentence, slots) in enumerate(items[:share]):
            rel = f"{split}/{intent}/slurp_{split}_{i:06d}.wav"
            duration = write_fixed(audio_dir / filename, outdir / rel)
            if duration is None:
                continue
            rows.append({"file": rel, "split": split, "intent": intent, "text": sentence,
                         "slots": slots, "model": "slurp", "speaker_id": "unknown",
                         "speaker_key": f"slurp|{split}", "duration_s": round(duration, 3),
                         "length_scale": None})

    by_split = defaultdict(list)
    for record in rows:
        by_split[record["split"]].append(record)

    for split, records in by_split.items():
        with open(outdir / f"manifest_{split}.jsonl", "a") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        counts = defaultdict(int)
        for record in records:
            counts[record["intent"]] += 1
        print(f"  {split:6s} +{len(records):6,d}  {dict(counts)}")

    print(f"\n  total added: {len(rows):,} real recordings")


if __name__ == "__main__":
    main()
