"""Add real smart-home commands from Fluent Speech Commands.

FSC (Lugosch et al., 2019) is 30,043 recordings of 97 speakers saying 248
smart-home phrasings - lights, music, volume, heating, language, fetching -
read from the `IamV/fluent_slu_v1.0` mirror. It is real people saying
commands, which is the one kind of data that has moved this model: STOP's
commands cleared the noise floor by 4x where read prose, isolated words and
extra voices did not.

It lands on the weakest real classes. light.set has 802 real training
utterances and media.control volume phrasings are thin; FSC has ~4,500 light
and lamp commands with rooms and 4,400 volume commands in 46 phrasings,
including indirect ones like "far too quiet".

Mapping, deliberately narrow as in add_slurp.py and add_stop.py:

    lights / lamp, activate | deactivate   -> light.set, state on | off, room
    music, activate                        -> media.control play
    music, deactivate                      -> media.control pause | stop, by the words spoken
    volume, increase | decrease            -> media.control volume_up | volume_down
    heat, change language, bring           -> none, command-shaped hard negatives

FSC's `washroom` location is spoken as "bathroom" or "washroom", and maps to
the spec's bathroom. A lamp is a light, so lamp commands map to light.set.

Only FSC's train and validation splits are ingested. Its test split - 10
speakers no model has heard - stays out of the manifests and is scored straight
from the parquet by evaluate_fsc.py, so it remains a cross-corpus test.
"""
import argparse
import glob
import io
import json
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
MAXD = SPEC["audio"]["max_duration_s"]

ROOMS = {"kitchen": "kitchen", "bedroom": "bedroom", "washroom": "bathroom"}
SPLITS = {"train": "train", "validation": "val"}
LABELS = ["speakerId", "action", "object", "location", "transcription"]


def map_label(action: str, obj: str, location: str, transcription: str) -> tuple[str, dict] | None:
    """
    Map one FSC label onto the spec.

    Args:
        action: FSC action, e.g. 'activate'
        obj: FSC object, e.g. 'lights'
        location: FSC location, or 'none'
        transcription: What was said, used to split pause from stop

    Returns:
        (intent, slots), or None when the utterance maps to nothing cleanly
    """
    text = transcription.lower()

    if obj in ("lights", "lamp") and action in ("activate", "deactivate"):
        slots = {"state": "on" if action == "activate" else "off"}
        if location in ROOMS:
            slots["room"] = ROOMS[location]
        return "light.set", slots

    if obj == "music" and action == "activate":
        return "media.control", {"action": "play"}

    if obj == "music" and action == "deactivate":
        # FSC files pause and stop together; the spec keeps them apart
        if "pause" in text:
            return "media.control", {"action": "pause"}
        if "stop" in text:
            return "media.control", {"action": "stop"}
        return None

    if obj == "volume" and action in ("increase", "decrease"):
        return "media.control", {"action": "volume_up" if action == "increase" else "volume_down"}

    if obj == "heat" or action in ("change language", "bring"):
        return "none", {}

    return None


def iter_rows(pattern: str, columns: list[str], batch_size: int = 128):
    """
    Stream rows from FSC parquet files without loading a whole file.

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
    """Decode one FSC audio cell to float32 mono at the spec's rate, or None."""
    try:
        audio, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=False)
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio if sr == SR else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsc", default="data/fsc")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--negative-ratio", type=float, default=0.5,
                    help="cap on hard negatives as a share of the positives in each split")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source, outdir = REPO / args.fsc, REPO / args.out
    rng = random.Random(SPEC["splits"]["tts_speakers"]["seed"])

    for fsc_split, split in SPLITS.items():
        pattern = str(source / f"{fsc_split}-*.parquet")
        mapped, dropped = [], Counter()
        for n, row in enumerate(iter_rows(pattern, LABELS)):
            label = map_label(row["action"], row["object"], row["location"], row["transcription"])
            if label is None:
                dropped[f"{row['action']}/{row['object']}"] += 1
                continue
            mapped.append((n, label, row))

        # Cap the hard negatives so they sharpen the boundary without tilting
        # the class prior toward rejection, the failure seen with mined prose
        positives = [m for m in mapped if m[1][0] != "none"]
        negatives = [m for m in mapped if m[1][0] == "none"]
        rng.shuffle(negatives)
        keep = positives + negatives[:int(len(positives) * args.negative_ratio)]
        wanted = {n: (label, row) for n, label, row in keep}

        counts = Counter(label[0] if label[0] != "media.control" else f"media.{label[1]['action']}"
                         for label, _ in wanted.values())
        print(f"{split:5s} {len(wanted):6,d} of {n + 1:,} rows   {dict(sorted(counts.items()))}")
        if dropped:
            print(f"      dropped {dict(dropped)}")
        if args.dry_run:
            continue

        rows = []
        for n, row in enumerate(iter_rows(pattern, LABELS + ["audio"])):
            if n not in wanted:
                continue
            (intent, slots), _ = wanted[n]
            audio = decode_audio(row["audio"])
            if audio is None or not 0.3 <= len(audio) / SR <= MAXD:
                continue

            rel = f"{split}/{intent.replace('.', '_')}/fsc_{split}_{n:06d}.wav"
            path = outdir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            peak = max(float(np.abs(audio).max()), 1e-6)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(SR)
                handle.writeframes((audio / peak * 0.9 * 32767).astype("<i2").tobytes())

            rows.append({"file": rel, "split": split, "intent": intent, "text": row["transcription"],
                         "slots": slots, "model": "fsc", "speaker_id": row["speakerId"],
                         "speaker_key": f"fsc|{row['speakerId']}",
                         "duration_s": round(len(audio) / SR, 3), "length_scale": None})

        with open(outdir / f"manifest_{split}.jsonl", "a") as handle:
            for record in rows:
                handle.write(json.dumps(record) + "\n")
        print(f"      wrote {len(rows):,} rows to manifest_{split}.jsonl")


if __name__ == "__main__":
    main()
