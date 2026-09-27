"""Convert synthetic commands into real human voices with kNN-VC.

The dataset has 21,000 synthetic utterances phrased the way the spec defines
commands, and thousands of real recordings phrased conversationally. What it
has never had is the combination: a real voice saying a spec-shaped command.
No public corpus provides it, and it is the condition the deployed system meets.

kNN-VC matches WavLM frames from a synthetic utterance against a real speaker's
recordings and resynthesises, so timing and wording survive while the voice and
its recording channel come from a real person. Measured on one conversion, the
4-8 kHz energy rises from 0.05 to 0.28 of the mid band - the synthetic source
sits an order of magnitude below a real microphone there.
"""
import argparse, json, random, wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]
MAXD = SPEC["audio"]["max_duration_s"]

# Reference speakers need enough audio to cover the phonetics of what is being
# converted; too little and kNN matching reaches for poor neighbours.
MIN_REFERENCE_CLIPS = 6


def patch_torchaudio_load() -> None:
    """
    Route torchaudio.load through soundfile.

    torchaudio 2.10 delegates load() to TorchCodec, which is not installed.
    kNN-VC only needs a (channels, samples) tensor, which soundfile provides.
    """
    def load(path, normalize=True, **kwargs):
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T.copy()), sr

    torchaudio.load = load


def held_out_speakers(data: Path) -> set[str]:
    """
    Speakers that appear in dev or test and must never be a conversion target.

    A converted utterance is a train row, but it carries the target speaker's
    timbre. If that speaker also has held-out rows, their voice has been seen
    in training. The usual disjointness check misses this, because converted
    rows are keyed `knnvc|<id>` while the mined rows are keyed differently.

    Args:
        data: Dataset directory holding the manifests

    Returns:
        Speaker ids appearing in the dev or test manifests
    """
    held = set()
    for split in ("val", "test"):
        for line in open(data / f"manifest_{split}.jsonl"):
            record = json.loads(line)
            if record.get("speaker_id"):
                held.add(str(record["speaker_id"]))

    return held


def reference_speakers(roots: list[Path], exclude: set[str]) -> dict[str, list[Path]]:
    """
    Collect real speakers with enough audio to act as conversion targets.

    Args:
        roots: LibriSpeech-shaped directories, <root>/<speaker>/<chapter>/*.flac
        exclude: Speaker ids to skip, normally everything held out

    Returns:
        Mapping of speaker id to their recordings
    """
    speakers = {}
    for root in roots:
        if not root.exists():
            continue

        for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if speaker_dir.name in exclude or speaker_dir.name in speakers:
                continue

            # str, not Path: kNN-VC passes these straight to its feature extractor,
            # which type-checks for a tensor and otherwise assumes a path string
            clips = sorted(str(p) for p in speaker_dir.rglob("*.flac"))
            if len(clips) >= MIN_REFERENCE_CLIPS:
                speakers[speaker_dir.name] = clips[:20]

    return speakers


def write_wav(path: Path, audio: np.ndarray) -> float:
    """Write float audio as 16-bit mono, trimmed to the window."""
    audio = np.clip(audio, -1.0, 1.0)[:int(MAXD * SR)]
    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(SR)
        handle.writeframes((audio * 32767).astype(np.int16).tobytes())

    return len(audio) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--librispeech", nargs="+",
                    default=["data/negatives/LibriSpeech/dev-clean",
                             "data/negatives/hf_stage"],
                    help="reference roots; hf_stage adds ~260 more real voices "
                         "than dev-clean alone, which held only 37")
    ap.add_argument("--per-split", type=int, default=6000,
                    help="utterances to convert from the train split")
    ap.add_argument("--tag", default="vc",
                    help="filename prefix, so a second pass cannot overwrite the first")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    patch_torchaudio_load()
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True,
                            trust_repo=True, pretrained=True, device=args.device)

    data = REPO / args.data
    exclude = held_out_speakers(data)
    speakers = reference_speakers([REPO / r for r in args.librispeech], exclude)
    print(f"{len(speakers)} reference speakers "
          f"({len(exclude):,} held-out speakers excluded as targets)")

    rng = random.Random(SPEC["splits"]["tts_speakers"]["seed"])
    rows = [json.loads(line) for line in open(data / "manifest_train.jsonl")]

    # Only synthetic commands are worth converting: the real recordings already
    # carry a real voice, and the reject class gains nothing from one.
    sources = [r for r in rows if r["intent"] != "none" and r["model"].startswith("en_")]
    rng.shuffle(sources)
    sources = sources[:args.per_split]
    print(f"converting {len(sources):,} synthetic commands")

    # Group by target speaker so each matching set is built once
    assignments = defaultdict(list)
    for record in sources:
        assignments[rng.choice(list(speakers))].append(record)

    # Written as we go: a crash two thirds of the way through should not throw
    # away the conversions already on disk.
    manifest = open(data / "manifest_train.jsonl", "a")
    converted = 0
    skipped = 0

    for n, (speaker, records) in enumerate(assignments.items(), 1):
        try:
            matching_set = knn_vc.get_matching_set(speakers[speaker])
        except Exception:
            skipped += len(records)
            continue

        for i, record in enumerate(records):
            try:
                query = knn_vc.get_features(str(data / record["file"]))
                audio = knn_vc.match(query, matching_set, topk=4).cpu().numpy()
            except Exception:
                # kNN-VC runs VAD on the source, which can trim a quiet or very
                # short utterance to nothing. Rare, and not worth stopping for.
                skipped += 1
                continue

            if audio.size < SR // 4:
                skipped += 1
                continue

            rel = f"train/{record['intent']}/{args.tag}_{speaker}_{i:06d}.wav"
            duration = write_wav(data / rel, audio)

            new = dict(record)
            new.update({"file": rel, "model": "knnvc", "duration_s": round(duration, 3),
                        "speaker_id": speaker, "speaker_key": f"knnvc|{speaker}"})
            manifest.write(json.dumps(new) + "\n")
            converted += 1

        manifest.flush()
        if n % 5 == 0:
            print(f"  {n}/{len(assignments)} speakers, {converted:,} converted, {skipped} skipped")

    manifest.close()

    print(f"\nadded {converted:,} converted utterances across "
          f"{len(assignments)} real voices, {skipped} skipped")


if __name__ == "__main__":
    main()
