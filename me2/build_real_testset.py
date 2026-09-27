"""Assemble the held-out real-speech test set.

Every number reported so far comes either from synthetic voices or from ten
recordings of one person. Neither measures generalisation: the first shares a
generator with the training data, the second is one speaker, one microphone and
one room.

This builds the `test_human` the benchmark asks for, from the held-out splits of
the two real corpora on disk. It covers two of the five intents. For light.set,
light.dim and query.info no real audio exists at all, so their real-speech
performance is unmeasured rather than merely poor - worth stating plainly rather
than reporting the synthetic figure as if it generalised.
"""
import argparse, ast, csv, json, random, wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]
MAXD = SPEC["audio"]["max_duration_s"]

SC_NEGATIVE_WORDS = (
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree",
    "wow", "backward", "forward", "follow", "learn", "visual", "yes", "no",
)
SC_COMMAND_WORDS = {"stop": ("media.control", {"action": "stop"})}
TAS_NEGATIVE_INTENTS = ("SetAlarm", "UnitConversion", "SimpleMath")


def write_fixed(src: Path, dst: Path) -> float:
    """Copy audio into the test set, trimmed to the model's window."""
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
    """Map a SetTimer to this spec's number/unit pair, or None if it does not fit."""
    present = [k for k in ("hours", "minutes", "seconds") if slots.get(k, 0)]
    if len(present) != 1:
        return None

    unit = present[0]
    value = slots[unit]
    limit = 12 if unit == "hours" else 60

    return {"number": value, "unit": unit} if 1 <= value <= limit else None


# Real sources for the benchmark sets. Synthetic voices are left out on
# purpose: the headline measures people, and restraint is about real speech.
HUMAN_SOURCES = {"slurp", "stop", "timers_and_such", "speech_commands"}
NEGATIVE_SOURCES = HUMAN_SOURCES | {"speech_commands_slot", "librispeech_num"}


def build_benchmark_sets(data: Path) -> None:
    """
    Write test_human and test_negatives from the held-out test split.

    The test_real set built by main() predates SLURP and STOP and covers two of
    the five intents. Every source now in the test split was held out by
    speaker, or by its corpus's own partition, when it was ingested, so both
    sets are views onto it: references to existing files, no audio copied.

    Args:
        data: Dataset directory holding manifest_test.jsonl and manifest_train.jsonl
    """
    test = [json.loads(line) for line in open(data / "manifest_test.jsonl")]
    sets = {
        "test_human": [r for r in test if r.get("model") in HUMAN_SOURCES and r["intent"] != "none"],
        "test_negatives": [r for r in test if r.get("model") in NEGATIVE_SOURCES and r["intent"] == "none"],
    }

    # Nothing here may have been trained on. Keys such as "slurp|test" name a
    # corpus partition rather than a person, so for those the guarantee is the
    # corpus's own split; for per-person keys the overlap is checked directly.
    train = [json.loads(line) for line in open(data / "manifest_train.jsonl")]
    train_files = {r["file"] for r in train}
    train_speakers = {r.get("speaker_key") for r in train}

    for name, rows in sets.items():
        shared_files = sum(r["file"] in train_files for r in rows)
        shared_speakers = {r.get("speaker_key") for r in rows} & train_speakers - {None}
        assert shared_files == 0, f"{name}: {shared_files} files also in train"
        assert not shared_speakers, f"{name}: speakers also in train: {sorted(shared_speakers)[:5]}"

        with open(data / f"manifest_{name}.jsonl", "w") as handle:
            for record in rows:
                handle.write(json.dumps(record) + "\n")

        by_source = defaultdict(lambda: [0, 0.0])
        for record in rows:
            by_source[record["model"]][0] += 1
            by_source[record["model"]][1] += record.get("duration_s") or 0

        hours = sum(v[1] for v in by_source.values()) / 3600
        print(f"{name}: {len(rows):,} utterances, {hours:.2f} h, "
              f"0 files and 0 speakers shared with train")
        for source, (count, seconds) in sorted(by_source.items(), key=lambda x: -x[1][0]):
            print(f"  {source:22s} {count:6,d}  {seconds / 3600:5.2f} h")
        if name == "test_human":
            intents = defaultdict(int)
            for record in rows:
                intents[record["intent"]] += 1
            print("  by intent:", dict(sorted(intents.items())))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", default="data/real_speech/speech_commands")
    ap.add_argument("--tas", default="data/real_speech")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--max-negatives", type=int, default=1000)
    ap.add_argument("--benchmark", action="store_true",
                    help="write test_human and test_negatives from the held-out test split instead")
    args = ap.parse_args()

    if args.benchmark:
        build_benchmark_sets(REPO / args.out)
        return

    sc, tas, outdir = REPO / args.sc, REPO / args.tas, REPO / args.out
    rng = random.Random(SPEC["splits"]["tts_speakers"]["seed"])
    rows = []

    # Speech Commands, held-out test speakers only
    testing = set(open(sc / "testing_list.txt").read().split())

    for key in sorted(testing):
        word = key.split("/")[0]
        if word in SC_COMMAND_WORDS:
            intent, slots = SC_COMMAND_WORDS[word]
        elif word in SC_NEGATIVE_WORDS:
            intent, slots = "none", {}
        else:
            continue

        name = key.split("/")[1]
        rel = f"test_real/sc_{word}_{Path(name).stem}.wav"
        duration = write_fixed(sc / key, outdir / rel)
        rows.append({"file": rel, "split": "test_real", "intent": intent, "text": word,
                     "slots": slots, "model": "speech_commands",
                     "speaker_id": name.split("_")[0],
                     "speaker_key": f"sc|{name.split('_')[0]}",
                     "duration_s": round(duration, 3), "length_scale": None})

    # Timers and Such, held-out real test speakers
    for i, record in enumerate(csv.DictReader(open(tas / "test-real.csv"))):
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

        rel = f"test_real/tas_{i:05d}.wav"
        duration = write_fixed(tas / record["path"], outdir / rel)
        rows.append({"file": rel, "split": "test_real", "intent": intent,
                     "text": record["transcription"], "slots": slot_values,
                     "model": "timers_and_such", "speaker_id": record["speakerId"],
                     "speaker_key": f"tas|{record['speakerId']}",
                     "duration_s": round(duration, 3), "length_scale": None})

    # Cap the reject class so it does not swamp the commands. False accepts are
    # measured properly on continuous audio by evaluate_wakeword.py; this set is
    # for command accuracy.
    commands = [r for r in rows if r["intent"] != "none"]
    negatives = [r for r in rows if r["intent"] == "none"]
    rng.shuffle(negatives)
    rows = commands + negatives[:args.max_negatives]

    with open(outdir / "manifest_test_real.jsonl", "w") as handle:
        for record in sorted(rows, key=lambda r: r["file"]):
            handle.write(json.dumps(record) + "\n")

    by_intent = defaultdict(lambda: [0, set()])
    for record in rows:
        by_intent[record["intent"]][0] += 1
        by_intent[record["intent"]][1].add(record["speaker_key"])

    print(f"{'intent':16s} {'clips':>7s} {'speakers':>9s}")
    print("-" * 36)
    for intent in sorted(by_intent):
        count, speakers = by_intent[intent]
        print(f"  {intent:14s} {count:7,d} {len(speakers):9,d}")
    print("-" * 36)
    print(f"  {'TOTAL':14s} {len(rows):7,d} {len({r['speaker_key'] for r in rows}):9,d}")
    print()
    print("  not covered (no real audio exists): light.set, light.dim, query.info")


if __name__ == "__main__":
    main()
