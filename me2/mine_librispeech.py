"""Mine real spoken number words out of LibriSpeech.

The number head is the only head trained on nothing but text-to-speech. Across
every real-speech source in the dataset the model has heard a human say
"fifteen" three times and "sixteen" once, which is why it scores 0.968 exact on
synthetic audio and confuses fifteen with fifty on a real microphone.

LibriSpeech is read prose, not commands, so these utterances enter as `intent:
none` - which is true, "he was fifty years old" is not a command - carrying
`supervise: ["number"]`. The masked loss then grades the number head on them
while the intent head learns to reject them, and decode ignores the number head
unless the intent is timer.set, so nothing downstream changes.

The alternative, splicing a real word into a synthetic carrier, was rejected: it
puts a speaker change in the middle of an utterance, and the model would learn
that a timbre discontinuity marks where the number is.
"""
import argparse
import json
import random
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import yaml

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]
MAXD = SPEC["audio"]["max_duration_s"]

# The words Speech Commands does not cover, which is every word in a confusable
# pair: thirteen/thirty, fourteen/forty, fifteen/fifty, sixteen/sixty
WORD_VALUES = {
    "TEN": 10, "ELEVEN": 11, "TWELVE": 12, "THIRTEEN": 13, "FOURTEEN": 14,
    "FIFTEEN": 15, "SIXTEEN": 16, "SEVENTEEN": 17, "EIGHTEEN": 18, "NINETEEN": 19,
    "TWENTY": 20, "THIRTY": 30, "FORTY": 40, "FIFTY": 50, "SIXTY": 60,
}

# A target followed by any of these is part of a larger number, so the audio
# says "twenty five" while the label would claim 20
COMPOUND_NEXT = {
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
    "HUNDRED", "THOUSAND", "MILLION", "BILLION",
}

TOKEN = re.compile(r"[A-Z']+")


def find_candidates(libri_root: Path) -> list[dict]:
    """
    Scan LibriSpeech transcripts for utterances carrying exactly one target word.

    Args:
        libri_root: Directory holding speaker/chapter/*.trans.txt

    Returns:
        One entry per usable utterance, with its audio path, words and target
    """
    candidates = []

    for trans_path in sorted(libri_root.rglob("*.trans.txt")):
        for line in open(trans_path):
            utterance_id, _, text = line.strip().partition(" ")
            words = TOKEN.findall(text.upper())

            hits = [i for i, word in enumerate(words) if word in WORD_VALUES]
            if len(hits) != 1:
                continue

            index = hits[0]
            if index + 1 < len(words) and words[index + 1] in COMPOUND_NEXT:
                continue

            audio_path = trans_path.parent / f"{utterance_id}.flac"
            if not audio_path.exists():
                continue

            candidates.append({
                "audio": audio_path,
                "words": words,
                "index": index,
                "word": words[index],
                "speaker": utterance_id.split("-")[0],
                "utterance_id": utterance_id,
            })

    return candidates


def align_word(aligner, tokenizer, model, waveform, words: list[str], index: int, device: str):
    """
    Locate one word inside an utterance with forced alignment.

    The transcript is known, so this is alignment rather than recognition, which
    is why a general-purpose model is accurate enough to trust here.

    Args:
        aligner: MMS_FA aligner
        tokenizer: MMS_FA tokenizer
        model: MMS_FA acoustic model
        waveform: Audio of shape (1, num_samples) at 16 kHz
        words: Transcript words, uppercase
        index: Position of the target word
        device: Device to run on

    Returns:
        Tuple of (start_sample, end_sample), or None if alignment fails
    """
    normalised = [re.sub(r"[^a-z']", "", word.lower()) for word in words]
    if not normalised[index]:
        return None

    try:
        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
            spans = aligner(emission[0], tokenizer(normalised))
    except Exception:
        return None

    if index >= len(spans) or not spans[index]:
        return None

    # Emission frames are 320 samples apart for this model, so the ratio is
    # taken from the audio rather than assumed
    ratio = waveform.shape[1] / emission.shape[1]
    start = int(spans[index][0].start * ratio)
    end = int(spans[index][-1].end * ratio)

    return start, end


def crop_window(audio, start: int, end: int, num_samples: int, rng) -> tuple:
    """
    Take a fixed window that contains the target word at a varied position.

    Args:
        audio: Full utterance, one dimensional
        start: First sample of the target word
        end: Last sample of the target word
        num_samples: Window length
        rng: Random source

    Returns:
        Tuple of (cropped audio, offset of the word within the crop)
    """
    if len(audio) <= num_samples:
        return audio, start

    # The window must contain the word, and within that freedom the word is
    # placed at random so it does not always sit in the middle
    low = max(0, end - num_samples)
    high = min(start, len(audio) - num_samples)
    offset = rng.randint(low, high) if low < high else max(0, min(low, len(audio) - num_samples))

    return audio[offset:offset + num_samples], start - offset


def write_wav(audio, dst: Path) -> float:
    """Write int16 mono audio at the dataset sample rate."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(audio.tobytes())

    return len(audio) / SR


def assign_splits(speakers: list[str], rng) -> dict[str, str]:
    """
    Split speakers, not utterances, so no voice appears on both sides.

    Args:
        speakers: Distinct speaker identifiers
        rng: Random source

    Returns:
        Mapping of speaker to split name
    """
    policy = SPEC["splits"]["tts_speakers"]

    # Deduplicate first. Passed one entry per utterance, the fractions below
    # would be taken over utterances while a repeated speaker got reassigned by
    # its last occurrence - still disjoint, but nothing like a 70/10/20 split
    speakers = sorted(set(speakers))
    rng.shuffle(speakers)

    n_train = int(len(speakers) * policy["train_fraction"])
    n_val = int(len(speakers) * policy["val_fraction"])

    assignment = {}
    for i, speaker in enumerate(speakers):
        assignment[speaker] = "train" if i < n_train else "val" if i < n_train + n_val else "test"

    return assignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", default="data/negatives/LibriSpeech/train-clean-100")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--max-per-word", type=int, default=400, help="cap the common words")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="report yield, write nothing")
    args = ap.parse_args()

    libri_root, outdir = REPO / args.libri, REPO / args.out
    rng = random.Random(SPEC["splits"]["tts_speakers"]["seed"])
    num_samples = int(MAXD * SR)

    candidates = find_candidates(libri_root)
    found = Counter(item["word"] for item in candidates)
    print(f"transcript scan: {len(candidates):,} usable utterances, {len(found)} words")
    for word in WORD_VALUES:
        print(f"  {word.lower():10s} {found.get(word, 0):5d}")

    # Cap the common words before aligning - twenty and fifty are an order of
    # magnitude more frequent in prose than sixteen, and aligning thousands of
    # them to then discard most would waste the GPU time
    rng.shuffle(candidates)
    kept, per_word = [], Counter()
    for item in candidates:
        if per_word[item["word"]] < args.max_per_word:
            per_word[item["word"]] += 1
            kept.append(item)

    print(f"\nafter per-word cap of {args.max_per_word}: {len(kept):,} to align")
    if args.dry_run:
        return

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(args.device).eval()
    tokenizer, aligner = bundle.get_tokenizer(), bundle.get_aligner()

    splits = assign_splits([item["speaker"] for item in kept], rng)

    rows, failures = [], 0
    for i, item in enumerate(kept):
        if i % 200 == 0:
            print(f"  aligned {i:,}/{len(kept):,}  kept {len(rows):,}  failed {failures}", flush=True)

        audio, sr = sf.read(str(item["audio"]), dtype="float32")
        if sr != SR:
            failures += 1
            continue

        located = align_word(
            aligner, tokenizer, model,
            torch.from_numpy(audio).unsqueeze(0), item["words"], item["index"], args.device
        )
        if located is None:
            failures += 1
            continue

        cropped, _ = crop_window(audio, located[0], located[1], num_samples, rng)
        value = WORD_VALUES[item["word"]]
        split = splits[item["speaker"]]

        rel = f"{split}/none/lsnum_{item['utterance_id']}.wav"
        duration = write_wav((cropped * 32767).astype("int16"), outdir / rel)

        rows.append({
            "file": rel,
            "split": split,
            "intent": "none",
            "text": " ".join(item["words"]).lower(),
            "slots": {"number": value},
            "supervise": ["number"],
            "supervise_intent": False,
            "model": "librispeech_num",
            "speaker_id": item["speaker"],
            "speaker_key": f"ls|{item['speaker']}",
            "duration_s": round(duration, 3),
            "length_scale": None,
        })

    by_split = defaultdict(list)
    for record in rows:
        by_split[record["split"]].append(record)

    for split, records in sorted(by_split.items()):
        with open(outdir / f"manifest_{split}.jsonl", "a") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        words = Counter(record["slots"]["number"] for record in records)
        print(f"  {split:6s} +{len(records):6,d}  speakers {len({r['speaker_id'] for r in records})}"
              f"  values {dict(sorted(words.items()))}")

    print(f"\n  total added: {len(rows):,}   alignment failures: {failures}")


if __name__ == "__main__":
    main()
