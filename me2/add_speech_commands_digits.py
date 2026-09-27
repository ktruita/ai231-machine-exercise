"""Add real spoken digits and on/off from Speech Commands.

Speech Commands is already in the dataset, but only as negatives, and only the
eighteen words that map to nothing. The digits and on/off were never loaded,
which left two heads learning from Piper alone:

  one..nine  - the ones head's entire vocabulary. A compound like "thirty-five"
               ends in the same word Speech Commands recorded 3,800 times.
  on / off   - the whole state slot.

No teens or tens here - those words are not in the corpus, which is why the
number words came from LibriSpeech instead. This fills the other half.

Same mechanism as the mined LibriSpeech: the rows are `none`, because an
isolated word is not a command, and they carry supervise_intent: false so the
slot supervision reaches the heads without the intent head learning that a real
voice saying a digit means reject.
"""
import argparse
import json
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]

# "zero" is deliberately absent: spoken numbers never use it, "fifty" is not
# "five zero", so it would train the ones head on a sound the task never hears
DIGITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9}
STATES = {"on": "on", "off": "off"}


def write_wav(src: Path, dst: Path) -> float | None:
    """
    Copy one Speech Commands clip into the dataset at the spec's sample rate.

    Args:
        src: Source wav
        dst: Destination inside the dataset

    Returns:
        Duration in seconds, or None if the clip cannot be used
    """
    try:
        audio, sr = sf.read(str(src), dtype="int16", always_2d=False)
    except Exception:
        return None

    if sr != SR or audio.ndim > 1 or len(audio) < SR // 4:
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(audio.tobytes())

    return len(audio) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/real_speech/speech_commands")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--split-map", default=None, help="json of speaker to split, to stay consistent")
    ap.add_argument("--max-per-word", type=int, default=400)
    args = ap.parse_args()

    source, outdir = REPO / args.source, REPO / args.out
    policy = SPEC["splits"]["tts_speakers"]
    rng = random.Random(policy["seed"])

    # Speech Commands speakers are already split by the negatives that were
    # loaded earlier. Reusing that assignment keeps a voice on one side only.
    known = json.load(open(args.split_map)) if args.split_map else {}
    print(f"reusing split for {len(known):,} known speakers")

    def split_for(speaker: str) -> str:
        if speaker not in known:
            roll = rng.random()
            known[speaker] = ("train" if roll < policy["train_fraction"]
                              else "val" if roll < policy["train_fraction"] + policy["val_fraction"]
                              else "test")
        return known[speaker]

    rows, skipped = [], 0
    for word in list(DIGITS) + list(STATES):
        clips = sorted((source / word).glob("*.wav"))
        rng.shuffle(clips)

        kept = 0
        for clip in clips:
            if kept >= args.max_per_word:
                break

            speaker = clip.name.split("_")[0]
            split = split_for(speaker)
            rel = f"{split}/none/scw_{word}_{clip.stem}.wav"

            duration = write_wav(clip, outdir / rel)
            if duration is None:
                skipped += 1
                continue

            if word in DIGITS:
                slots, supervise = {"number": DIGITS[word]}, ["number"]
            else:
                slots, supervise = {"state": STATES[word]}, ["state"]

            rows.append({
                "file": rel,
                "split": split,
                "intent": "none",
                "text": word,
                "slots": slots,
                "supervise": supervise,
                "supervise_intent": False,
                "model": "speech_commands_slot",
                "speaker_id": speaker,
                "speaker_key": f"sc|{speaker}",
                "duration_s": round(duration, 3),
                "length_scale": None,
            })
            kept += 1

        print(f"  {word:6s} {kept:4d}")

    by_split = defaultdict(list)
    for record in rows:
        by_split[record["split"]].append(record)

    for split, records in sorted(by_split.items()):
        with open(outdir / f"manifest_{split}.jsonl", "a") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        words = Counter(record["text"] for record in records)
        print(f"  {split:6s} +{len(records):5,d}  speakers "
              f"{len({r['speaker_id'] for r in records})}  {dict(sorted(words.items()))}")

    print(f"\n  total added: {len(rows):,}   unusable clips: {skipped}")


if __name__ == "__main__":
    main()
