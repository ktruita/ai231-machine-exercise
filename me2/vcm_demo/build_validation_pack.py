"""Build a validation pack, the held-out clips validate_pi.py replays on the device.

evaluate_benchmark.py measures the model on the server. To show the figures
hold on the Pi, the pack carries a random sample of the same held-out sets,
small enough to copy, plus the server's own decode of every clip, made with the
same ONNX runtime the Pi uses:

    commands    commands from the class holdout and test splits, by speakers
                never heard in training
    negatives   their out-of-scope clips, and synthetic negatives' test clips
    wake        Speech Commands' "marvin" test clips, each set in one second of
                MUSAN noise either side at 15 dB, as evaluate_wakeword.py does

The presets' models and biases are read from run_demo.sh, where presets live.
Writes <pack>/ beside this file and ../vcm_demo_<pack>.tar.gz holding it and
validate_pi.py, to extract over the Pi package. --splits picks the clips from
any data directory, a count or all of each split; a clip labelled `none` is a
non-command. Reads the class data, so run it from the repo, not the Pi; a pack
holds the dataset's recordings, so it is never committed.

Usage:
    python build_validation_pack.py
"""
import argparse
import datetime
import json
import random
import re
import shutil
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from validate_pi import count_triggers, firing_mask, read_wav, window_scores
from vcm.runtime import CommandRecogniser, WakeWordDetector

REPO = Path(__file__).resolve().parent
ME2 = REPO.parent

# After this folder, so `vcm` stays the runtime's package
sys.path.append(str(ME2))
from evaluate_wakeword import embed_in_context  # noqa: E402

THRESHOLD = 0.6
WAKE = {"model": "deploy/wakeword_marvin", "threshold": 0.99, "n": 3, "m": 5, "hop_ms": 100,
        "refractory_ms": 1000}


def read_presets(names: list[str]) -> dict:
    """
    Read presets' models and `none` bias from run_demo.sh.

    Args:
        names: Preset names, as run_demo.sh's case labels

    Returns:
        Preset name to {"models": [...], "none_bias": float}
    """
    script = (REPO / "run_demo.sh").read_text()
    presets = {}
    for name in names:
        block = re.search(rf"^\s*{re.escape(name)}\)\n(.*?);;", script, re.S | re.M).group(1)
        presets[name] = {
            "models": re.search(r"MODELS=\(([^)]*)\)", block).group(1).split(),
            "none_bias": float(re.search(r"BIAS=(\S+)", block).group(1)),
        }

    return presets


def graded(record: dict, intent_slots: dict) -> dict:
    """
    The slots exact-match grading checks, as evaluate_benchmark.py grades.

    Args:
        record: Manifest row
        intent_slots: Intent to its slots, from the bundle's meta.json

    Returns:
        Slot to the value it must decode to, leaving out ungraded slots
    """
    skip = set(record.get("unsupervise", ()))

    return {s: record["slots"].get(s) for s in intent_slots[record["intent"]] if s not in skip}


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """Write float audio as 16-bit mono."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="validation_hf", help="folder beside this file to write the pack to")
    ap.add_argument("--splits", nargs="+", default=["c_holdout:all", "c_test:400", "c_test_synthneg:250"],
                    help="split:count, or split:all, from --data")
    ap.add_argument("--presets", nargs="+", default=["fallback_hf"])
    ap.add_argument("--data", default="class_data/v2/hf_only")
    ap.add_argument("--speech-commands", default="data/real_speech/speech_commands")
    ap.add_argument("--musan", default="data/augment/musan")
    args = ap.parse_args()

    rng = random.Random(231)
    data = ME2 / args.data
    PACK = REPO / args.pack
    presets = read_presets(args.presets)
    recognisers = {name: CommandRecogniser([REPO / m for m in p["models"]], none_bias=p["none_bias"])
                   for name, p in presets.items()}
    detector = WakeWordDetector(REPO / WAKE["model"])
    meta = next(iter(recognisers.values())).meta
    rate = meta["sample_rate"]
    hop = int(WAKE["hop_ms"] * rate / 1000)
    refractory = max(int(WAKE["refractory_ms"] / WAKE["hop_ms"]), 1)

    if PACK.exists():
        shutil.rmtree(PACK)
    clips = []

    for entry in args.splits:
        split, _, count = entry.partition(":")
        rows = [json.loads(line) for line in open(data / f"manifest_{split}.jsonl")]
        if count != "all":
            rows = rng.sample(rows, min(int(count), len(rows)))
        for i, record in enumerate(rows):
            kind = "negative" if record["intent"] == "none" else "command"
            rel = f"{split}/{i:04d}.wav"
            (PACK / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data / record["file"], PACK / rel)
            clips.append({"file": rel, "kind": kind, "split": split, "source": record.get("model", "?"),
                          "truth": {"intent": record["intent"], "slots": graded(record, meta["intent_slots"])}})

    root = ME2 / args.speech_commands
    positives = sorted(line for line in (root / "testing_list.txt").read_text().split() if line.startswith("marvin/"))
    noise_files = sorted((ME2 / args.musan / "noise").rglob("*.wav"))
    for i, name in enumerate(positives):
        clip, _ = sf.read(str(root / name), dtype="float32", always_2d=False)
        noise, _ = sf.read(str(rng.choice(noise_files)), dtype="float32", always_2d=False)
        if noise.ndim > 1:
            noise = noise[:, 0]
        rel = f"wake/{i:04d}.wav"
        write_wav(PACK / rel, embed_in_context(clip, noise, pad_samples=rate), rate)
        clips.append({"file": rel, "kind": "wake", "source": "speech_commands"})

    # The server's decode of each clip, read back from the pack as the Pi will read it
    for clip in clips:
        audio = read_wav(PACK / clip["file"])
        clip["server"] = {}
        if clip["kind"] != "wake":
            for name, recogniser in recognisers.items():
                result = recogniser(audio)
                clip["server"][name] = {"command": result["command"], "confidence": round(result["confidence"], 6)}
        fire = firing_mask(window_scores(detector, audio, hop)[0], WAKE["threshold"], WAKE["n"], WAKE["m"])
        if clip["kind"] == "negative":
            clip["server"]["wake_triggers"] = count_triggers(fire, refractory)
        if clip["kind"] == "wake":
            clip["server"]["wake_fired"] = bool(fire.any())

    pack = {"built": datetime.date.today().isoformat(), "threshold": THRESHOLD, "presets": presets,
            "wake": WAKE, "clips": clips}
    (PACK / "manifest.json").write_text(json.dumps(pack, indent=1))

    out = ME2 / f"vcm_demo_{args.pack}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(REPO / "validate_pi.py", arcname="vcm_demo/validate_pi.py")
        tar.add(PACK, arcname=f"vcm_demo/{args.pack}")

    counts = {kind: sum(c["kind"] == kind for c in clips) for kind in ("command", "negative", "wake")}
    size = sum(f.stat().st_size for f in PACK.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {PACK}: {counts}, {size:.0f} MB; presets {', '.join(presets)}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
