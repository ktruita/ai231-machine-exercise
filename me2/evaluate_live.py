"""Evaluate a run the way the live demo actually feeds it.

`evaluate.py` scores every utterance starting at the command. The live loop in
`vcm_demo/demo.py` captured the last six seconds once the speaker stopped,
which also holds the wake word said just before - and that cost vcm_stop 7.4
points of intent accuracy, almost all of it to `none`. The model's only
real-voice "marvin" examples are Speech Commands clips labelled `none`, so a
real "Marvin" at the start of the window pulls the decision toward rejection.

Conditions, all built from the same held-out utterances:
    as_evaluated   left-aligned command, what evaluate.py measures
    wake_word      "marvin", a pause, the command, then the endpointer's 0.8 s
                   of silence, right-aligned - the capture before the fix
    trimmed_Nms    the capture from the arm point on, still holding the last
                   N ms of "marvin", because the detector can arm before the
                   word ends

Usage:
    python evaluate_live.py modelstore/vcm_stop
"""
import argparse
import random
import wave
from pathlib import Path

import numpy as np
import torch

from dataloaders.vcm_dataloader import VoiceCommandDataset
from evaluate import load_run

REPO = Path(__file__).resolve().parent
SR = 16000
ENDPOINT_TAIL_S = 0.8
PAUSE_RANGE_S = (0.2, 0.6)
RESIDUALS_MS = (0, 100, 200, 300)
REAL_SOURCES = {"slurp", "speech_commands", "timers_and_such", "stop"}
SPEECH_COMMANDS = {"speech_commands", "speech_commands_slot"}


def load_clip(path: Path) -> np.ndarray:
    """
    Read a 16-bit mono wav as float32 in [-1, 1].

    Args:
        path: Wav file

    Returns:
        Waveform of shape (num_samples,)
    """
    with wave.open(str(path)) as handle:
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")

    return audio.astype(np.float32) / 32768


def trim_silence(waveform: np.ndarray, threshold: float = 0.02, margin: int = 800) -> np.ndarray:
    """
    Cut leading and trailing silence, keeping a short margin.

    Args:
        waveform: Audio of shape (num_samples,)
        threshold: Absolute level counted as sound (default: 0.02)
        margin: Samples kept either side of the sound (default: 800)

    Returns:
        The trimmed waveform, or the input if nothing crosses the threshold
    """
    loud = np.where(np.abs(waveform) > threshold)[0]
    if loud.size == 0:
        return waveform

    return waveform[max(loud[0] - margin, 0): loud[-1] + margin]


def held_out_wake_words(data: Path, root: Path) -> list[np.ndarray]:
    """
    Real "marvin" recordings from speakers the command model never trained on.

    Drawing the prefix from training speakers would let the model recognise
    the voice rather than cope with the word, and flatter the result.

    Args:
        data: Dataset directory holding manifest_train.jsonl
        root: Speech Commands "marvin" directory

    Returns:
        Trimmed waveforms, one per held-out clip
    """
    import json

    seen = set()
    for line in open(data / "manifest_train.jsonl"):
        record = json.loads(line)
        if record.get("model") in SPEECH_COMMANDS:
            seen.add(str(record.get("speaker_id")))

    clips = [p for p in sorted(root.glob("*.wav")) if p.name.split("_")[0] not in seen]

    return [trim_silence(load_clip(p)) for p in clips]


def right_align(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    """Keep the last num_samples, padding on the left: a buffer that ends now."""
    waveform = waveform[-num_samples:]

    return np.pad(waveform, (num_samples - len(waveform), 0)).astype(np.float32)


def left_align(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    """Pad on the right, as CommandRecogniser.fit_length does to a short capture."""
    waveform = waveform[:num_samples]

    return np.pad(waveform, (0, num_samples - len(waveform))).astype(np.float32)


def conditions(waveform: np.ndarray, marvin: np.ndarray, pause_s: float) -> dict[str, np.ndarray]:
    """
    Build every condition from one test utterance.

    The wake word and pause are drawn once per utterance and shared, so the
    conditions differ only in what the capture holds.

    Args:
        waveform: Test-split waveform, left-aligned and zero padded
        marvin: Trimmed "marvin" recording
        pause_s: Gap between the wake word and the command in seconds

    Returns:
        Condition name mapped to a waveform of the model's window length
    """
    num_samples = len(waveform)
    command = np.trim_zeros(waveform, "b")
    pause = np.zeros(int(pause_s * SR), np.float32)
    tail = np.zeros(int(ENDPOINT_TAIL_S * SR), np.float32)

    built = {
        "as_evaluated": waveform.astype(np.float32),
        "wake_word": right_align(np.concatenate([marvin, pause, command, tail]), num_samples),
    }
    for ms in RESIDUALS_MS:
        residual = marvin[len(marvin) - int(ms * SR / 1000):] if ms else marvin[:0]
        built[f"trimmed_{ms}ms"] = left_align(np.concatenate([residual, pause, command, tail]), num_samples)

    return built


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory, e.g. modelstore/vcm_stop")
    ap.add_argument("--step", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--wake-words", default="data/real_speech/speech_commands/marvin")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    data = REPO / args.data
    dataset = VoiceCommandDataset(data_dir=str(data), spec_path=str(REPO / "commands.yaml"), split="test")
    module, _, _ = load_run(args.run, args.step, args.device)
    none = dataset.intent_names.index("none")

    commands = [i for i, r in enumerate(dataset.records)
                if r.get("model") in REAL_SOURCES and r["intent"] != "none"]
    timers = {i for i in commands
              if dataset.records[i].get("model") == "stop" and dataset.records[i]["intent"] == "timer.set"}

    marvins = held_out_wake_words(data, REPO / args.wake_words)
    rng = random.Random(args.seed)
    print(f"{Path(args.run).name}: {len(commands):,} held-out real commands, "
          f"{len(timers):,} STOP timers, {len(marvins):,} held-out 'marvin' clips\n")

    names = None
    intent_ok, rejected, number_ok = {}, {}, {}
    with torch.inference_mode():
        for start in range(0, len(commands), 64):
            chunk = commands[start:start + 64]
            items = [dataset[i] for i in chunk]
            built = [conditions(item["waveform"].numpy(), rng.choice(marvins),
                                rng.uniform(*PAUSE_RANGE_S)) for item in items]
            names = names or list(built[0])

            for name in names:
                logits = module(torch.from_numpy(np.stack([b[name] for b in built])).to(args.device))
                intent = logits["intent"].argmax(-1).cpu()
                tens = logits["number_tens"].argmax(-1).cpu()
                ones = logits["number_ones"].argmax(-1).cpu()

                for k, (i, item) in enumerate(zip(chunk, items)):
                    intent_ok[name] = intent_ok.get(name, 0) + int(intent[k] == item["intent"])
                    rejected[name] = rejected.get(name, 0) + int(intent[k] == none)
                    if i in timers:
                        number_ok[name] = number_ok.get(name, 0) + int(
                            tens[k] == item["target_number_tens"] and ones[k] == item["target_number_ones"])

    base = intent_ok["as_evaluated"] / len(commands)
    print(f"{'condition':16s} {'intent':>8s} {'vs eval':>9s} {'as none':>9s} {'timer num':>10s}")
    for name in names:
        acc = intent_ok[name] / len(commands)
        print(f"{name:16s} {acc:8.3f} {acc - base:+9.3f} {rejected[name] / len(commands):9.3f} "
              f"{number_ok.get(name, 0) / max(len(timers), 1):10.3f}")


if __name__ == "__main__":
    main()
