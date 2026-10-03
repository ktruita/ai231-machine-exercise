"""Measure footprint and latency on the device.

Footprint on a Raspberry Pi is model size and p95 latency. Numbers from a
workstation say nothing about a Pi, so this is written to run there unchanged -
numpy and onnxruntime only, the same two dependencies as demo.py.

Protocol:
    one intra-op thread, 10 warmup runs, 100 timed runs
    p50 and p95 latency, real-time factor, peak resident memory

Each run is split into the stages a command actually passes through, since
the model is a small part of the wait the user perceives:
    features   numpy log-mel of the 6 s window
    inference  onnxruntime forward pass
    decode     logits to a command
The endpointer's wait for silence (0.8 s in demo.py) is added to give
end-to-end time from the end of speech to a decoded command.

Without the recording named by --clip, it times six seconds of low noise
instead: the work done is the same whatever the audio holds.

Usage:
    python bench_device.py                          # fallback_hf's two models
    python bench_device.py --model deploy/vcm_hf    # one of them
"""
import argparse
import platform
import resource
import time
import wave
from pathlib import Path

import numpy as np

from vcm.runtime import CommandRecogniser

REPO = Path(__file__).resolve().parent
WARMUP_RUNS = 10
TIMED_RUNS = 100
ENDPOINT_WAIT_MS = 800          # SpeechEndpointer end_blocks * hop in demo.py


def load_clip(path: Path) -> np.ndarray:
    """Read a 16-bit mono wav as float32 in [-1, 1]."""
    with wave.open(str(path)) as handle:
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")

    return audio.astype(np.float32) / 32768


def peak_rss_mb() -> float:
    """Peak resident set size of this process. ru_maxrss is KB on Linux, bytes on macOS."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return peak / 1024 / 1024 if platform.system() == "Darwin" else peak / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["deploy/vcm_hf", "deploy/vcm_hf_s2"],
                    help="one bundle, or several timed as an ensemble")
    ap.add_argument("--int8", action="store_true", help="time model_int8.onnx instead of model.onnx")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--clip", default="samples/Set_timer_for_15_min.wav")
    args = ap.parse_args()

    bundles = [REPO / m for m in args.model]
    model_file = "model_int8.onnx" if args.int8 else "model.onnx"
    rec = CommandRecogniser(bundles, threads=args.threads, model_file=model_file)
    clip = REPO / args.clip
    audio = load_clip(clip) if clip.exists() else \
        np.random.default_rng(0).normal(0.0, 0.01, rec.num_samples).astype(np.float32)
    waveform = rec.fit_length(audio)
    window_ms = rec.meta["window_s"] * 1000

    stages = {"features": [], "inference": [], "decode": []}
    for run in range(WARMUP_RUNS + TIMED_RUNS):
        t0 = time.perf_counter()
        mel = rec.features(waveform)
        t1 = time.perf_counter()
        logits = rec.combine(rec.run_sessions(mel))
        t2 = time.perf_counter()
        result = rec.decode(logits)
        t3 = time.perf_counter()

        if run >= WARMUP_RUNS:
            stages["features"].append((t1 - t0) * 1000)
            stages["inference"].append((t2 - t1) * 1000)
            stages["decode"].append((t3 - t2) * 1000)

    total = np.sum([stages[k] for k in stages], axis=0)
    size_kb = sum((b / model_file).stat().st_size for b in bundles) / 1024

    print(f"device    {platform.machine()}  {platform.processor() or platform.platform()}")
    print(f"model     {' + '.join(args.model)} ({model_file})  {size_kb:.0f} KB  threads {args.threads}")
    print(f"clip      {args.clip if clip.exists() else 'none, timing on noise'} -> "
          f"{result['command']}  p={result['confidence']:.3f}")
    print(f"protocol  {WARMUP_RUNS} warmup, {TIMED_RUNS} timed\n")
    print(f"{'stage':12s} {'p50 ms':>9s} {'p95 ms':>9s}")
    for name, values in list(stages.items()) + [("model total", total)]:
        print(f"{name:12s} {np.percentile(values, 50):9.2f} {np.percentile(values, 95):9.2f}")

    p95 = float(np.percentile(total, 95))
    print(f"\nreal-time factor     {np.percentile(total, 50) / window_ms:.4f}  (p50 total / {window_ms:.0f} ms window)")
    print(f"end-to-end p95       {ENDPOINT_WAIT_MS + p95:.0f} ms  "
          f"({ENDPOINT_WAIT_MS} ms endpointer wait + {p95:.1f} ms compute)")
    print(f"peak RSS             {peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
