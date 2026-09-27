"""Add real human speech to the dataset, on both sides of the reject decision.

The model had never heard a real voice: every command was Piper and every
negative was Piper. Real speech was therefore out of distribution and landed
wherever the boundary happened to fall - 48% of held-out real speech was
accepted as a command, and no confidence threshold fixed it without discarding
88% of genuine commands.

Real audio is added to the commands AND to the reject class, from the same
corpora and the same speakers, so "real or synthetic" carries no information
about the label in either direction.

Sources:
  Timers and Such  SetTimer                     -> timer.set     (real commands)
                   SetAlarm/UnitConversion/     -> none          (real hard negatives,
                   SimpleMath                                     command-like but out of scope)
  Speech Commands  stop                         -> media.control (real command)
                   15 non-command words         -> none          (real negatives)
"""
import argparse, ast, csv, json, random, shutil, wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]
MAXD = SPEC["audio"]["max_duration_s"]

# Timers and Such intents that are not in this spec. They are real speech in the
# same recording conditions as the SetTimer commands, which makes them the
# hardest negatives available - the model cannot reject them on channel alone.
TAS_NEGATIVE_INTENTS = ("SetAlarm", "UnitConversion", "SimpleMath")

# Speech Commands words that are unambiguously not commands here.
SC_NEGATIVE_WORDS = (
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree",
    "wow", "backward", "forward", "follow", "learn", "visual", "yes", "no",
)

# "stop" is already a media.control phrasing in the spec, so these clips are
# real recordings of a command the model is meant to know. The digits and
# on/off/up/down/go are deliberately left out: they are slot values or partial
# phrasings, not commands on their own, and labelling them either way adds noise.
SC_COMMAND_WORDS = {"stop": ("media.control", {"action": "stop"})}


def write_fixed(src: Path, dst: Path) -> float:
    """
    Copy audio to the dataset, resampled expectations aside, trimmed to the window.

    Args:
        src: Source WAV
        dst: Destination WAV

    Returns:
        Duration in seconds
    """
    audio, sr = sf.read(str(src), dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    assert sr == SR, f"expected {SR} Hz, got {sr} in {src}"

    audio = audio[:int(MAXD * SR)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(SR)
        handle.writeframes(audio.tobytes())

    return len(audio) / SR


def timer_slots(slots: dict) -> dict | None:
    """
    Map a Timers and Such SetTimer to this spec's number/unit pair.

    This spec carries one number and one unit, so a timer spanning two units
    ("one minute and fifty six seconds") cannot be represented and is dropped
    rather than silently truncated.

    Args:
        slots: The SetTimer slot dictionary

    Returns:
        Slot dictionary for timer.set, or None if it does not fit
    """
    present = [k for k in ("hours", "minutes", "seconds") if slots.get(k, 0)]
    if len(present) != 1:
        return None

    unit = present[0]
    value = slots[unit]
    limit = 12 if unit == "hours" else 60
    if not 1 <= value <= limit:
        return None

    return {"number": value, "unit": unit}


def add_timers_and_such(src: Path, outdir: Path, rows: dict) -> None:
    """Add Timers and Such commands and hard negatives, using its own splits."""
    split_map = {"train-real": "train", "dev-real": "val", "test-real": "test"}

    for csv_name, split in split_map.items():
        for i, record in enumerate(csv.DictReader(open(src / f"{csv_name}.csv"))):
            semantics = ast.literal_eval(record["semantics"])
            intent_name, slots = semantics["intent"], semantics["slots"]

            if intent_name == "SetTimer":
                mapped = timer_slots(slots)
                if mapped is None:
                    continue
                intent, slot_values = "timer.set", mapped
            elif intent_name in TAS_NEGATIVE_INTENTS:
                intent, slot_values = "none", {}
            else:
                continue

            rel = f"{split}/{intent}/tas_{csv_name}_{i:05d}.wav"
            duration = write_fixed(src / record["path"], outdir / rel)
            rows[split].append({
                "file": rel, "split": split, "intent": intent,
                "text": record["transcription"], "slots": slot_values,
                "model": "timers_and_such", "speaker_id": record["speakerId"],
                "speaker_key": f"tas|{record['speakerId']}",
                "duration_s": round(duration, 3), "length_scale": None,
            })


def add_speech_commands(src: Path, outdir: Path, rows: dict, per_word: int, rng: random.Random) -> None:
    """Add Speech Commands real audio, honouring the corpus's own speaker splits."""
    val = set(open(src / "validation_list.txt").read().split())
    test = set(open(src / "testing_list.txt").read().split()) if (src / "testing_list.txt").exists() else set()

    words = list(SC_COMMAND_WORDS) + list(SC_NEGATIVE_WORDS)
    for word in words:
        files = sorted((src / word).glob("*.wav"))
        if not files:
            continue
        rng.shuffle(files)

        for path in files[:per_word]:
            key = f"{word}/{path.name}"
            split = "val" if key in val else "test" if key in test else "train"

            if word in SC_COMMAND_WORDS:
                intent, slot_values = SC_COMMAND_WORDS[word]
            else:
                intent, slot_values = "none", {}

            rel = f"{split}/{intent}/sc_{word}_{path.stem}.wav"
            duration = write_fixed(path, outdir / rel)
            # Speech Commands encodes the speaker in the filename prefix
            speaker = path.stem.split("_")[0]
            rows[split].append({
                "file": rel, "split": split, "intent": intent, "text": word,
                "slots": slot_values, "model": "speech_commands",
                "speaker_id": speaker, "speaker_key": f"sc|{speaker}",
                "duration_s": round(duration, 3), "length_scale": None,
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tas", default="data/real_speech")
    ap.add_argument("--sc", default="data/real_speech/speech_commands")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--per-word", type=int, default=400)
    args = ap.parse_args()

    outdir = REPO / args.out
    rng = random.Random(SPEC["splits"]["tts_speakers"]["seed"])
    rows = defaultdict(list)

    add_timers_and_such(REPO / args.tas, outdir, rows)
    add_speech_commands(Path(REPO / args.sc), outdir, rows, args.per_word, rng)

    for split, records in rows.items():
        with open(outdir / f"manifest_{split}.jsonl", "a") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        by_intent = defaultdict(int)
        for record in records:
            by_intent[record["intent"]] += 1
        print(f"  {split:6s} +{len(records):5,d} real clips  {dict(by_intent)}")

    speakers = {r["speaker_key"] for records in rows.values() for r in records}
    print(f"\n  distinct real speakers added: {len(speakers):,}")


if __name__ == "__main__":
    main()
