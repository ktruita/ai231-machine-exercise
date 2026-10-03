"""Turn the class-agreed dataset into this pipeline's manifests.

The class settled on airimonda/ai231-me2-voice-commands (Hugging Face): 19
commands, 93 phrasing-and-value variations, plus an out-of-scope class, in
train, test and holdout splits that share no speaker. This reads its parquet
files (class_data/v2/hf/, from fetch_class_dataset.py), writes each clip under
class_data/v2/dataset/c_<split>/<label>/ and writes manifests in the row format
the dataloader reads, labelled by commands.yaml:

    command      -> intent; OUT_OF_SCOPE -> none
    slot_value   -> the slot of a slotted command; a value outside the closed
                    set leaves the slot ungraded
    transcript, source, speaker_id, accent_group and variation are kept

Splits:
    c_train, c_val   the dataset's train split; about 12% of its speakers, drawn
                     within each source and accent group, are held out to choose
                     checkpoints and the rejection bias
    c_test           the dataset's test split, also written as c_test_human
                     (commands) and c_test_negatives (out-of-scope) for
                     evaluate_benchmark.py
    c_holdout        the dataset's holdout, for live testing on the Pi

The clips are 16 kHz mono 16-bit WAVs already, so their bytes are written as
they are; anything else is decoded and rewritten. The numerals split is not
part of the command set and is not used.

The dataset's supplemental_synth clips (the rest of the group's synthetic set)
whose voice belongs to the train split are added to c_train, as the card
allows - less any voice held out for c_val. The out-of-scope clones
(group_synthetic_oos) count as the voice they were cloned from.
`--supplemental ""` leaves them out.

v2 is the dataset as revised on 2026-10-02, the revision fetch_class_dataset.py
pins. build_hf_only.py then adds the repo's synthetic negatives for training.

Usage:
    python add_class_dataset.py
"""
import argparse
import glob
import io
import json
import math
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]
SPLITS = {"train": "c_train", "test": "c_test", "holdout": "c_holdout"}
COLUMNS = ["audio", "file", "transcript", "command", "variation", "slot_value", "out_of_scope", "speaker_id",
           "source", "is_synthetic", "accent_group", "duration_s"]
VAL_SHARE = 0.12
# Out-of-scope clips spoken by clones of the group's synthetic voices
SAME_VOICE = {"group_synthetic_oos": "group_synthetic"}


def iter_rows(pattern: str, columns: list[str], batch_size: int = 128):
    """
    Stream rows from parquet files without loading a whole file.

    Args:
        pattern: Glob for the parquet files of one split
        columns: Columns to read; leave out "audio" to skip the waveforms
        batch_size: Rows per read (default: 128)

    Yields:
        One dict per row
    """
    for path in sorted(glob.glob(pattern)):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def decode_audio(blob: dict) -> np.ndarray | None:
    """Decode one audio cell to float32 mono at the spec's rate, or None."""
    try:
        audio, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=False)
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio if sr == SR else None


def label(row: dict) -> tuple[str, dict, list]:
    """
    Map one dataset row onto commands.yaml.

    Args:
        row: Parquet row

    Returns:
        Tuple of (intent, slots, ungraded slots)
    """
    command = row["command"]
    if row["out_of_scope"] or command == "OUT_OF_SCOPE":
        return "none", {}, []
    if command not in SPEC["intents"]:
        raise ValueError(f"command {command!r} is not in commands.yaml")

    slots = SPEC["intents"][command].get("slots") or []
    if not slots:
        return command, {}, []
    value = (row["slot_value"] or "").strip()
    if value in SPEC["slots"][slots[0]]["values"]:
        return command, {slots[0]: value}, []

    return command, {}, [slots[0]]


def write_clip(blob: dict, path: Path) -> float | None:
    """
    Write one clip, its own bytes when they are already 16 kHz mono 16-bit.

    Args:
        blob: Parquet audio cell, holding the file's bytes
        path: Where to write it

    Returns:
        Duration in seconds, or None when the audio cannot be read at 16 kHz
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        info = sf.info(io.BytesIO(blob["bytes"]))
    except Exception:
        return None
    if info.samplerate == SR and info.channels == 1 and info.subtype == "PCM_16" and info.format == "WAV":
        path.write_bytes(blob["bytes"])
        return info.frames / SR

    audio = decode_audio(blob)
    if audio is None:
        return None
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())

    return len(audio) / SR


def speaker_key(row: dict, rel: str) -> str:
    """A speaker per row; a row with no speaker id counts as its own."""
    speaker = str(row["speaker_id"] or "").strip()
    source = SAME_VOICE.get(row["source"], row["source"])

    return f"{source}|{speaker}" if speaker else f"{source}|clip:{rel}"


def carve_val(rows: list[dict], share: float, rng: random.Random) -> set[str]:
    """
    Choose validation speakers, about `share` of them within each source and accent group.

    Args:
        rows: The train split's rows
        share: Fraction of each stratum's speakers to hold out
        rng: Seeded random source

    Returns:
        The speaker keys that go to c_val
    """
    strata = defaultdict(set)
    for row in rows:
        strata[(row["speaker_key"].split("|")[0], row["accent_group"])].add(row["speaker_key"])

    held = set()
    for key in sorted(strata, key=str):
        speakers = sorted(strata[key])
        rng.shuffle(speakers)
        held.update(speakers[:math.floor(len(speakers) * share + 0.5)])

    return held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="class_data/v2/hf/data", help="the dataset's parquet files")
    ap.add_argument("--out", default="class_data/v2/dataset")
    ap.add_argument("--supplemental", default="class_data/v2/hf/supplemental_synth",
                    help="folder of the supplemental_synth parquet files, \"\" to leave them out")
    args = ap.parse_args()

    source, out = REPO / args.hf, REPO / args.out
    rows_by_split, skipped = defaultdict(list), Counter()
    for split, name in SPLITS.items():
        for i, row in enumerate(iter_rows(str(source / f"{split}-*.parquet"), COLUMNS)):
            intent, slots, ungraded = label(row)
            rel = f"{name}/{intent}/{i:06d}.wav"
            duration = write_clip(row["audio"], out / rel)
            if duration is None:
                skipped[name] += 1
                continue
            record = {"file": rel, "split": name, "intent": intent, "slots": slots,
                      "text": row["transcript"], "model": row["source"],
                      "speaker_id": str(row["speaker_id"] or ""), "speaker_key": speaker_key(row, rel),
                      "accent_group": row["accent_group"], "variation": row["variation"] or "",
                      "is_synthetic": int(row["is_synthetic"] or 0), "duration_s": round(duration, 3)}
            if ungraded:
                record["unsupervise"] = ungraded
            rows_by_split[name].append(record)

    held = carve_val(rows_by_split["c_train"], VAL_SHARE, random.Random(231))
    train = [r for r in rows_by_split["c_train"] if r["speaker_key"] not in held]

    # Supplemental clips join training only in a train voice not held out for c_val
    extra = Counter()
    if args.supplemental:
        for i, row in enumerate(iter_rows(str(REPO / args.supplemental / "train-*.parquet"), COLUMNS + ["voice_split"])):
            extra[f"voice in {row['voice_split']}"] += 1
            if row["voice_split"] != "train":
                continue
            intent, slots, ungraded = label(row)
            rel = f"c_train/{intent}/supplemental_{i:06d}.wav"
            if speaker_key(row, rel) in held:
                extra["dropped, voice in c_val"] += 1
                continue
            duration = write_clip(row["audio"], out / rel)
            if duration is None:
                skipped["supplemental"] += 1
                continue
            record = {"file": rel, "split": "c_train", "intent": intent, "slots": slots,
                      "text": row["transcript"], "model": row["source"],
                      "speaker_id": str(row["speaker_id"] or ""), "speaker_key": speaker_key(row, rel),
                      "accent_group": row["accent_group"], "variation": row["variation"] or "",
                      "is_synthetic": int(row["is_synthetic"] or 0), "duration_s": round(duration, 3),
                      "supplemental": 1}
            if ungraded:
                record["unsupervise"] = ungraded
            train.append(record)
            extra["added to c_train"] += 1
    val = [dict(r, split="c_val") for r in rows_by_split["c_train"] if r["speaker_key"] in held]
    test, holdout = rows_by_split["c_test"], rows_by_split["c_holdout"]
    manifests = {
        "c_train": train, "c_val": val, "c_test": test, "c_holdout": holdout,
        "c_test_human": [r for r in test if r["intent"] != "none"],
        "c_test_negatives": [r for r in test if r["intent"] == "none"],
    }

    # Nothing evaluated on may share a speaker with what is trained on
    train_speakers = {r["speaker_key"] for r in train}
    for name in ("c_val", "c_test", "c_holdout"):
        shared = {r["speaker_key"] for r in manifests[name]} & train_speakers
        assert not shared, f"{name} shares speakers with c_train: {sorted(shared)[:5]}"

    for name, rows in manifests.items():
        with open(out / f"manifest_{name}.jsonl", "w") as handle:
            for record in rows:
                handle.write(json.dumps(record) + "\n")

    print(f"{'split':18s} {'clips':>7s} {'hours':>6s} {'speakers':>9s}   out-of-scope")
    for name, rows in manifests.items():
        hours = sum(r["duration_s"] for r in rows) / 3600
        speakers = len({r["speaker_key"] for r in rows})
        print(f"{name:18s} {len(rows):7,d} {hours:6.2f} {speakers:9,d}   {sum(r['intent'] == 'none' for r in rows):,d}")
    print("skipped (unreadable):", dict(skipped))
    if args.supplemental:
        print("supplemental_synth:", dict(extra))

    for name in ("c_train", "c_val", "c_test"):
        rows = manifests[name]
        print(f"\n{name} by source:", dict(Counter(r["model"] for r in rows).most_common()))
        print(f"{name} by accent group:", dict(Counter(r["accent_group"] for r in rows).most_common()))
    intents = Counter(r["intent"] for r in train)
    print("\nc_train per command:", {k: intents[k] for k in SPEC["intents"]})
    values = Counter((r["intent"], v) for r in train for v in r["slots"].values())
    print("c_train slot values:", dict(sorted(values.items())))
    ungraded = Counter(r["intent"] for name in ("c_train", "c_test") for r in manifests[name] if r.get("unsupervise"))
    print("slotted rows with no listed value (slot ungraded):", dict(ungraded))
    variations = Counter(r["variation"] for r in rows_by_split["c_train"] if r["intent"] != "none")
    print(f"variations in the dataset's train split: {len(variations)}, clips each "
          f"{min(variations.values())}-{max(variations.values())}")
    print("held-out sets share no speaker with c_train")


if __name__ == "__main__":
    main()
