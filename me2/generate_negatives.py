"""Build the `none` class.

Split into two sets for a reason. The first version of this script drew every
negative from LibriSpeech, which made real human speech the only real audio in
the dataset - every command was synthetic. The model learned "real recording ->
none" instead of learning what a command sounds like, and rejected all real
speech with p=1.0. Training negatives are therefore synthesised with the same
voices as the commands, so the class cannot be identified by recording domain.
The LibriSpeech audio is kept as a held-out real-speech eval set, which is what
the false-accept rate is supposed to be measured on.
"""
import argparse, json, random, wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

from generate_dataset import get_voice, _CACHE, _single_thread_onnx

REPO = Path(__file__).resolve().parent
SPEC = yaml.safe_load(open(REPO / "commands.yaml"))
SR = SPEC["audio"]["sample_rate"]
MAXD = SPEC["audio"]["max_duration_s"]

# Sentences that borrow command vocabulary without being commands. These are the
# hard negatives - a model that keys on the word "lights" or "timer" alone will
# fire on them.
NEAR_MISS = [
    "the lights were already on when we got to the kitchen",
    "i left the bedroom light on all night again",
    "it only took about fifteen minutes to get there",
    "he set the timer and forgot about it completely",
    "the music was far too loud in the living room",
    "she asked me what time the train was leaving",
    "the weather has been miserable for three days",
    "turning off the motorway we drove past the garage",
    "about seventy percent of them agreed with the plan",
    "i need to buy a new lamp for the office",
    "the temperature dropped to about five degrees overnight",
    "can you play that song again for me please",
    "my alarm did not go off this morning",
    "we waited thirty minutes and nobody came",
    "the volume on this television is stuck",
    "he walked into the bathroom and shut the door",
    "there is no timer on this old oven",
    "what time did you say the meeting was",
    "she turned around and looked at the kitchen window",
    "the brightness of the screen is hurting my eyes",
]


def load_transcripts(src: Path, min_words: int = 5, max_words: int = 12) -> list[str]:
    """
    Read LibriSpeech transcripts and cut them into command-length spans.

    Full transcript lines run far longer than any command, so a random span is
    taken to match the duration distribution rather than the sentence length.

    Args:
        src: LibriSpeech split directory
        min_words: Shortest span to keep (default: 5)
        max_words: Longest span to keep (default: 12)

    Returns:
        Lower-cased text spans
    """
    rng = random.Random(0)
    spans = []

    for path in sorted(src.rglob("*.trans.txt")):
        for line in open(path):
            words = line.strip().split()[1:]        # drop the utterance id
            if len(words) < min_words:
                continue
            span = rng.randint(min_words, min(max_words, len(words)))
            start = rng.randint(0, len(words) - span)
            spans.append(" ".join(words[start:start + span]).lower())

    return spans


def render(text: str, model: str, speaker_id: int | None, seed: int) -> np.ndarray:
    """
    Synthesise one utterance, matching the command pipeline exactly.

    Args:
        text: Text to speak
        model: Piper voice model name
        speaker_id: Speaker id for multi-speaker models
        seed: Seed controlling prosody jitter and padding

    Returns:
        Waveform as int16 at the spec sample rate
    """
    from piper.config import SynthesisConfig

    rng = random.Random(seed)
    voice = get_voice(model)
    cfg = SynthesisConfig(
        speaker_id=speaker_id, normalize_audio=False,
        length_scale=round(rng.uniform(0.85, 1.15), 3),
        noise_scale=round(rng.uniform(0.60, 0.75), 3),
        noise_w_scale=round(rng.uniform(0.70, 0.90), 3))

    chunks = list(voice.synthesize(text, syn_config=cfg))
    audio = np.concatenate([np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks])
    audio = resample_poly(audio.astype(np.float64), SR, chunks[0].sample_rate)

    peak = np.abs(audio).max()
    ceiling = 32767 * 10 ** (-1 / 20)
    if peak > ceiling:
        audio *= ceiling / peak
    audio = np.clip(audio, -32768, 32767).astype(np.int16)

    lead = rng.randint(*SPEC["audio"]["lead_silence_ms"])
    trail = rng.randint(*SPEC["audio"]["trail_silence_ms"])
    audio = np.concatenate([np.zeros(lead * SR // 1000, np.int16), audio,
                            np.zeros(trail * SR // 1000, np.int16)])

    return audio[:int(MAXD * SR)]


def write_wav(path: Path, audio: np.ndarray) -> float:
    """Write int16 mono audio and return its duration in seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(SR)
        handle.writeframes(audio.tobytes())

    return len(audio) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--src", default="data/negatives/LibriSpeech/dev-clean")
    ap.add_argument("--out", default="data/dataset")
    args = ap.parse_args()

    src, outdir = REPO / args.src, REPO / args.out
    cfg = SPEC["splits"]["tts_speakers"]
    rng = random.Random(cfg["seed"] + 1)

    texts = load_transcripts(src) + NEAR_MISS * 20
    rng.shuffle(texts)

    # Speakers come from the same split file as the commands, so a negative is
    # never spoken by a voice the model heard in training
    splits = json.load(open(REPO / "speaker_splits.json"))["split"]
    pools = defaultdict(list)
    for key, split in splits.items():
        model, sid = key.split("|")
        pools[split].append((model, None if sid == "-" else int(sid), key))

    fracs = {"train": cfg["train_fraction"], "val": cfg["val_fraction"], "test": cfg["test_fraction"]}

    # Plan every utterance first, then render grouped by model. Choosing speakers
    # at random means consecutive utterances usually need a different voice, and
    # a Piper model takes ~10 s to load - grouping turns thousands of reloads
    # into eight.
    jobs = []
    for split, frac in fracs.items():
        for i in range(round(args.n * frac)):
            model, sid, key = rng.choice(pools[split])
            jobs.append({"split": split, "index": i, "model": model, "speaker_id": sid,
                         "speaker_key": key, "text": texts[rng.randrange(len(texts))],
                         "seed": rng.randrange(2**31)})
    jobs.sort(key=lambda j: (j["model"], j["speaker_id"] if j["speaker_id"] is not None else -1))

    rows = defaultdict(list)
    for job in jobs:
        rel = f"{job['split']}/none/none_{job['split']}_{job['index']:06d}.wav"
        duration = write_wav(outdir / rel,
                             render(job["text"], job["model"], job["speaker_id"], job["seed"]))
        rows[job["split"]].append({"file": rel, "split": job["split"], "intent": "none",
                                   "text": job["text"], "slots": {}, "model": job["model"],
                                   "speaker_id": job["speaker_id"], "speaker_key": job["speaker_key"],
                                   "duration_s": round(duration, 3), "length_scale": None})

    for split, records in rows.items():
        with open(outdir / f"manifest_{split}.jsonl", "a") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"  {split:6s} {len(records):6,d} synthetic negatives")

    # Real speech, held out as an eval set only. Duration-matched so the model
    # cannot separate it from commands on clip length.
    command_durations = [json.loads(l)["duration_s"]
                         for l in open(outdir / "manifest_test.jsonl")
                         if json.loads(l)["intent"] != "none"]
    speakers = sorted(p.name for p in src.iterdir() if p.is_dir())
    flacs = defaultdict(list)
    for path in src.rglob("*.flac"):
        flacs[path.parts[-3]].append(path)

    real = []
    for i in range(args.n_real):
        speaker = speakers[i % len(speakers)]
        audio, sr = sf.read(rng.choice(flacs[speaker]), dtype="int16")
        assert sr == SR, f"expected {SR} Hz, got {sr}"
        want = int(min(rng.choice(command_durations), MAXD) * SR)
        if len(audio) <= want:
            segment = np.pad(audio, (0, want - len(audio)))
        else:
            start = rng.randrange(0, len(audio) - want)
            segment = audio[start:start + want]
        rel = f"real_negatives/real_{i:06d}.wav"
        duration = write_wav(outdir / rel, segment.astype(np.int16))
        real.append({"file": rel, "split": "real_negatives", "intent": "none", "text": None,
                     "slots": {}, "model": "librispeech", "speaker_id": speaker,
                     "speaker_key": f"librispeech|{speaker}", "duration_s": round(duration, 3),
                     "length_scale": None})

    with open(outdir / "manifest_real_negatives.jsonl", "w") as handle:
        for record in real:
            handle.write(json.dumps(record) + "\n")
    print(f"  {'real':6s} {len(real):6,d} held-out real-speech negatives (eval only)")


if __name__ == "__main__":
    main()
