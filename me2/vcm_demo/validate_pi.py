"""Validate the deployed models on the device itself, from the validation pack.

The accuracy and false-accept figures were measured on the server
(evaluate_benchmark.py). The Pi runs the same ONNX models in the same runtime,
so they should hold here; this replays a sample of the held-out sets on the
device and says whether they do. build_validation_pack.py made the pack
(validation_hf/) with the server's own decode of every clip, so each decode
here is checked against the server as well as against the truth.

Reported:
    commands      intent and exact command accuracy on held-out commands
                  from speakers never heard in training, at the demo
                  threshold and with none, and exact accuracy by source -
                  some packs mix synthetic voices with real recordings
    non-commands  how often speech that is not a command is acted on
    wake word     "marvin" clips set in noise that fire the demo's 3-of-5
                  rule, and false wakes on the non-command speech
    latency       per command and per wake word window, on this device
    parity        decodes identical to the server's

numpy and onnxruntime only, like demo.py.

Usage:
    python validate_pi.py                     # fallback_hf, from validation_hf
    python validate_pi.py --limit 50          # a quick check
"""
import argparse
import json
import platform
import resource
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from vcm.runtime import CommandRecogniser, WakeWordDetector

REPO = Path(__file__).resolve().parent

# A decode matches the server's when the command is the same and the
# confidence agrees this closely; onnxruntime builds differ in the last digits
PARITY_TOLERANCE = 0.01


def read_wav(path: Path) -> np.ndarray:
    """Read a 16-bit mono wav as float32 in [-1, 1]."""
    with wave.open(str(path)) as handle:
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")

    return audio.astype(np.float32) / 32768


def window_scores(detector: WakeWordDetector, audio: np.ndarray, hop: int) -> tuple[np.ndarray, list[float]]:
    """
    Score every one-second window, `hop` samples apart, as the live loop does.

    Args:
        detector: Loaded wake word model
        audio: Waveform, float32 in [-1, 1]
        hop: Samples between windows

    Returns:
        Tuple of (wake word probability per window, milliseconds per window)
    """
    window = detector.num_samples
    scores, times = [], []
    for start in range(0, len(audio) - window + 1, hop):
        began = time.perf_counter()
        scores.append(detector(audio[start:start + window]))
        times.append((time.perf_counter() - began) * 1000)

    return np.array(scores, dtype=np.float32), times


def firing_mask(scores: np.ndarray, threshold: float, n: int, m: int) -> np.ndarray:
    """
    Positions where n of the last m windows are over threshold, the demo's rule.

    Args:
        scores: Wake word probability per window
        threshold: Firing threshold
        n: Windows over threshold required
        m: Length of the history considered

    Returns:
        Boolean array marking firing positions
    """
    above = (scores >= threshold).astype(np.int32)
    if len(above) < m:
        return np.zeros(0, dtype=bool)
    cumulative = np.cumsum(np.concatenate([[0], above]))

    return (cumulative[m:] - cumulative[:-m]) >= n


def count_triggers(fire: np.ndarray, refractory: int) -> int:
    """Count firing events, ignoring `refractory` windows after each one."""
    triggers, i = 0, 0
    while i < len(fire):
        if fire[i]:
            triggers += 1
            i += refractory
        else:
            i += 1

    return triggers


def device_line() -> str:
    """This device's model, architecture, runtime and, on a Pi, temperature and throttling."""
    tree = Path("/proc/device-tree/model")
    name = tree.read_text().strip("\x00\n ") if tree.exists() else platform.processor() or platform.platform()
    line = f"{name} · {platform.machine()} · onnxruntime {ort.__version__}"
    for query in ("measure_temp", "get_throttled"):
        try:
            out = subprocess.run(["vcgencmd", query], capture_output=True, text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            break
        if out:
            line += f" · {out}"

    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="validation_hf", help="the pack's folder beside this file")
    ap.add_argument("--preset", default="fallback_hf", help="a preset the pack holds the server's decodes for")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="clips of each kind, for a quick check")
    args = ap.parse_args()

    PACK = REPO / args.pack
    pack = json.loads((PACK / "manifest.json").read_text())
    if args.preset not in pack["presets"]:
        raise SystemExit(f"no preset {args.preset!r} in the pack; it has {sorted(pack['presets'])}")
    preset, wake, threshold = pack["presets"][args.preset], pack["wake"], pack["threshold"]
    recogniser = CommandRecogniser([REPO / m for m in preset["models"]], threads=args.threads,
                                   none_bias=preset["none_bias"])
    detector = WakeWordDetector(REPO / wake["model"], threads=args.threads)
    hop = int(wake["hop_ms"] * recogniser.meta["sample_rate"] / 1000)
    refractory = max(int(wake["refractory_ms"] / wake["hop_ms"]), 1)

    clips = {kind: [c for c in pack["clips"] if c["kind"] == kind][:args.limit]
             for kind in ("command", "negative", "wake")}
    print(f"device   {device_line()}")
    print(f"preset   {args.preset}: {' + '.join(preset['models'])}  (none bias {preset['none_bias']}, "
          f"threshold {threshold}, {args.threads} thread)")
    print(f"pack     {len(clips['command'])} commands, {len(clips['negative'])} non-commands, "
          f"{len(clips['wake'])} wake word clips, built {pack['built']}\n")

    decode_ms, window_ms = [], []
    stats = {"intent": 0, "exact": 0, "wrong": 0, "rejected": 0, "intent_any": 0, "exact_any": 0}
    by_source = {}
    acted_on, matches, worst = 0, 0, 0.0
    negative_seconds, false_wakes, wake_matches = 0.0, 0, 0
    mismatches = []

    for clip in clips["command"] + clips["negative"]:
        audio = read_wav(PACK / clip["file"])
        began = time.perf_counter()
        result = recogniser(audio)
        decode_ms.append((time.perf_counter() - began) * 1000)

        command, truth = result["command"], clip["truth"]
        acts = command["intent"] != "none" and result["confidence"] >= threshold
        server = clip["server"][args.preset]
        drift = abs(result["confidence"] - server["confidence"])
        same = command == server["command"] and drift <= PARITY_TOLERANCE
        matches += same
        worst = max(worst, drift)
        if not same:
            mismatches.append((clip["file"], command, server["command"]))

        if clip["kind"] == "negative":
            acted_on += acts
            negative_seconds += len(audio) / recogniser.meta["sample_rate"]
            scores, times = window_scores(detector, audio, hop)
            window_ms += times
            triggers = count_triggers(firing_mask(scores, wake["threshold"], wake["n"], wake["m"]), refractory)
            false_wakes += triggers
            wake_matches += triggers == clip["server"]["wake_triggers"]
            continue

        right_intent = command["intent"] == truth["intent"]
        exact = right_intent and all(command.get(k) == v for k, v in truth["slots"].items())
        stats["intent_any"] += right_intent
        stats["exact_any"] += exact
        stats["intent"] += right_intent and acts
        stats["exact"] += exact and acts
        stats["wrong"] += acts and not exact
        stats["rejected"] += not acts
        tally = by_source.setdefault(clip.get("source", "?"), [0, 0])
        tally[0] += 1
        tally[1] += exact and acts

    hits = 0
    for clip in clips["wake"]:
        scores, times = window_scores(detector, read_wav(PACK / clip["file"]), hop)
        window_ms += times
        fired = bool(firing_mask(scores, wake["threshold"], wake["n"], wake["m"]).any())
        hits += fired
        wake_matches += fired == clip["server"]["wake_fired"]

    n_cmd, n_neg, n_wake = len(clips["command"]), len(clips["negative"]), len(clips["wake"])
    pct = lambda k: 100 * stats[k] / max(n_cmd, 1)
    hours = negative_seconds / 3600
    window_s = recogniser.meta["window_s"]
    decode_p50, decode_p95 = np.percentile(decode_ms, 50), np.percentile(decode_ms, 95)
    window_p50, window_p95 = np.percentile(window_ms, 50), np.percentile(window_ms, 95)

    print(f"commands      {n_cmd} from speakers never heard in training")
    print(f"  at {threshold}      intent {pct('intent'):5.1f} %   exact {pct('exact'):5.1f} %   "
          f"wrong action {pct('wrong'):4.1f} %   didn't catch that {pct('rejected'):4.1f} %")
    print(f"  no threshold  intent {pct('intent_any'):5.1f} %   exact {pct('exact_any'):5.1f} %")
    print(f"  exact at {threshold} by source")
    for source, (n, right) in sorted(by_source.items(), key=lambda item: -item[1][0]):
        print(f"    {source:22s} {n:4d}  {100 * right / n:5.1f} %")
    print(f"non-commands  {n_neg}: acted on {100 * acted_on / max(n_neg, 1):.1f} % at {threshold}")
    print(f"wake word     {n_wake} \"marvin\" clips in noise: {100 * hits / max(n_wake, 1):.1f} % fire "
          f"({wake['n']} of {wake['m']} windows over {wake['threshold']})")
    print(f"              false wakes on the non-commands: {false_wakes} in {60 * hours:.1f} min "
          f"({false_wakes / max(hours, 1e-9):.1f} /h)")
    print(f"latency       command p50 {decode_p50:.1f} ms  p95 {decode_p95:.1f} ms  "
          f"(real-time factor {decode_p50 / (window_s * 1000):.4f})")
    print(f"              wake word window p50 {window_p50:.2f} ms  p95 {window_p95:.2f} ms  "
          f"({100 * window_p50 / wake['hop_ms']:.1f} % of a core, one window every {wake['hop_ms']} ms)")
    print(f"              peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    print(f"parity        {matches} / {n_cmd + n_neg} decodes identical to the server's "
          f"(largest confidence difference {worst:.4f}); {wake_matches} / {n_neg + n_wake} wake decisions")
    for file, here, there in mismatches[:5]:
        print(f"              differs: {file}  here {here}  server {there}")

    print("\nfor the slide")
    print(f"  keyword / intent acc   {100 * hits / max(n_wake, 1):.1f} % / {pct('intent'):.1f} % (p >= {threshold})")
    print(f"  false-accept rate      {100 * acted_on / max(n_neg, 1):.1f} % of non-commands (p >= {threshold})")
    print(f"  latency p95 / RTF      {decode_p95:.1f} ms / {decode_p50 / (window_s * 1000):.4f}")
    print(f"  runtime                onnxruntime {ort.__version__} · {args.threads} thread · "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
