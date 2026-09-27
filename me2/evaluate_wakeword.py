"""Measure the wake word false-accept rate on continuous audio.

Scoring isolated one-second clips flatters an always-on detector. A deployed
detector slides a window every hop over hours of speech and noise, which is
tens of thousands of decisions an hour rather than a few hundred in total, and
that is where false accepts actually accumulate.

Consecutive windows over threshold are collapsed into one trigger. Without
that, a single detection spanning five windows would be counted as five false
accepts and the rate would be inflated by the window hop.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils import find_latest_checkpoint

REPO = Path(__file__).resolve().parent


def load_model(run_dir: Path, device: str = "cuda"):
    """Load a trained wake word module in eval mode."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    OmegaConf.resolve(cfg.module)

    module = instantiate(cfg.module.module, cfg.module)
    ckpt = find_latest_checkpoint(run_dir / "checkpoints")
    module.load_state_dict(torch.load(ckpt, weights_only=False, map_location="cpu")["state_dict"])

    return module.to(device).eval()


def window_scores(module, audio: np.ndarray, window: int, hop: int,
                  batch_size: int = 512, device: str = "cuda") -> np.ndarray:
    """
    Score every sliding window of one recording.

    Args:
        module: Trained wake word module
        audio: Waveform, float32 in [-1, 1]
        window: Window length in samples
        hop: Hop between windows in samples
        batch_size: Windows per forward pass (default: 512)
        device: Device to run on (default: 'cuda')

    Returns:
        Wake word probability per window
    """
    if len(audio) < window:
        return np.zeros(0, dtype=np.float32)

    starts = range(0, len(audio) - window + 1, hop)
    windows = np.stack([audio[s:s + window] for s in starts])

    scores = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            chunk = torch.from_numpy(windows[i:i + batch_size]).to(device)
            scores.append(torch.softmax(module(chunk), dim=-1)[:, 1].cpu().numpy())

    return np.concatenate(scores)


def firing_mask(scores: np.ndarray, threshold: float, n: int, m: int) -> np.ndarray:
    """
    Decide which positions fire, requiring n of the last m windows over threshold.

    A spurious detection is usually one isolated window that happened to look
    like the target. A real wake word sits under the sliding window for several
    hops in a row, so requiring agreement across a short history suppresses the
    former while keeping the latter. n = m = 1 is the plain per-window rule.

    Args:
        scores: Wake word probability per window
        threshold: Firing threshold
        n: Windows over threshold required
        m: Length of the history considered

    Returns:
        Boolean array marking firing positions
    """
    above = (scores >= threshold).astype(np.int32)

    if n <= 1 and m <= 1:
        return above.astype(bool)
    if len(above) < m:
        return np.zeros(0, dtype=bool)

    # rolling count of hits over each window of m consecutive decisions
    cumulative = np.cumsum(np.concatenate([[0], above]))

    return (cumulative[m:] - cumulative[:-m]) >= n


def count_triggers(scores: np.ndarray, threshold: float, refractory: int,
                   n: int = 1, m: int = 1) -> int:
    """
    Count detection events, collapsing runs of consecutive windows.

    Args:
        scores: Wake word probability per window
        threshold: Firing threshold
        refractory: Windows to suppress after a trigger
        n: Windows over threshold required within the history (default: 1)
        m: Length of the history considered (default: 1)

    Returns:
        Number of distinct triggers
    """
    fire = firing_mask(scores, threshold, n, m)

    triggers = 0
    i = 0
    while i < len(fire):
        if fire[i]:
            triggers += 1
            i += refractory
        else:
            i += 1

    return triggers


def embed_in_context(clip: np.ndarray, noise: np.ndarray, pad_samples: int,
                     snr_db: float = 15.0) -> np.ndarray:
    """
    Put a wake word clip inside continuous background, as the device would hear it.

    False rejects measured on bare one-second clips are optimistic: the clip is
    exactly one window, so the n-of-m rule has no history to work with. Embedding
    the clip gives the detector the same sliding-window view it gets in use.

    Args:
        clip: Wake word waveform
        noise: Background waveform, tiled if short
        pad_samples: Background samples before and after the clip
        snr_db: Level of the clip over the background (default: 15.0)

    Returns:
        Waveform of length pad_samples * 2 + len(clip)
    """
    total = pad_samples * 2 + len(clip)
    if len(noise) < total:
        noise = np.tile(noise, int(np.ceil(total / max(len(noise), 1))))
    background = noise[:total].copy()

    clip_power = np.mean(clip ** 2)
    noise_power = np.mean(background ** 2)
    if clip_power > 0 and noise_power > 0:
        background *= np.sqrt(clip_power / (noise_power * 10 ** (snr_db / 10)))

    background[pad_samples:pad_samples + len(clip)] += clip

    return np.clip(background, -1.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="modelstore/wakeword_marvin")
    ap.add_argument("--librispeech", default="data/negatives/LibriSpeech/dev-clean")
    ap.add_argument("--musan", default="data/augment/musan")
    ap.add_argument("--max-files", type=int, default=600)
    ap.add_argument("--hop-ms", type=int, default=100)
    ap.add_argument("--refractory-ms", type=int, default=1000)
    ap.add_argument("--n", type=int, default=1, help="windows over threshold required")
    ap.add_argument("--m", type=int, default=1, help="history length for the n-of-m rule")
    args = ap.parse_args()

    module = load_model(REPO / args.run)
    sr = module.features.sample_rate
    window = sr
    hop = int(args.hop_ms * sr / 1000)
    refractory = max(int(args.refractory_ms / args.hop_ms), 1)

    rng = random.Random(231)
    sources = {
        "librispeech (real speech)": sorted((REPO / args.librispeech).rglob("*.flac")),
        "musan music": sorted((REPO / args.musan / "music").rglob("*.wav")),
        "musan noise": sorted((REPO / args.musan / "noise").rglob("*.wav")),
    }

    thresholds = (0.5, 0.9, 0.95, 0.99, 0.999)
    print(f"sliding window: {window/sr:.1f}s every {args.hop_ms}ms, "
          f"{args.refractory_ms}ms refractory, rule: {args.n}-of-{args.m}\n")
    print(f"{'source':26s} {'hours':>7s} {'windows':>10s}  " +
          "  ".join(f"FA/h@{t}" for t in thresholds))
    print("-" * 86)

    totals = {t: 0 for t in thresholds}
    total_hours = 0.0

    for name, files in sources.items():
        rng.shuffle(files)
        files = files[:args.max_files]

        counts = {t: 0 for t in thresholds}
        seconds = 0.0
        n_windows = 0

        for path in files:
            audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio[:, 0]
            if file_sr != sr:
                continue

            scores = window_scores(module, audio, window, hop)
            if not len(scores):
                continue

            seconds += len(audio) / sr
            n_windows += len(scores)
            for t in thresholds:
                counts[t] += count_triggers(scores, t, refractory, args.n, args.m)

        hours = seconds / 3600
        total_hours += hours
        for t in thresholds:
            totals[t] += counts[t]

        rates = "  ".join(f"{counts[t]/max(hours,1e-9):8.1f}" for t in thresholds)
        print(f"{name:26s} {hours:7.2f} {n_windows:10,d}  {rates}")

    print("-" * 86)
    rates = "  ".join(f"{totals[t]/max(total_hours,1e-9):8.1f}" for t in thresholds)
    print(f"{'ALL':26s} {total_hours:7.2f} {'':10s}  {rates}")

    # False rejects under the same sliding rule, on wake words embedded in noise
    from dataloaders import WakeWordDataset

    dataset = WakeWordDataset(REPO / "data/real_speech/speech_commands", split="test")
    positives = [p for p, label in dataset.items if label == 1]
    noise_files = sorted((REPO / args.musan / "noise").rglob("*.wav"))

    misses = {t: 0 for t in thresholds}
    for path in positives:
        clip = np.zeros(0, dtype=np.float32)
        with sf.SoundFile(str(path)) as handle:
            clip = handle.read(dtype="float32", always_2d=False)
        if clip.ndim > 1:
            clip = clip[:, 0]

        noise, _ = sf.read(str(rng.choice(noise_files)), dtype="float32", always_2d=False)
        if noise.ndim > 1:
            noise = noise[:, 0]

        audio = embed_in_context(clip, noise, pad_samples=sr)
        scores = window_scores(module, audio, window, hop)
        for t in thresholds:
            if not firing_mask(scores, t, args.n, args.m).any():
                misses[t] += 1

    print()
    print(f"{'false rejects':26s} {'':7s} {len(positives):10,d}  " +
          "  ".join(f"{misses[t]/max(len(positives),1):8.4f}" for t in thresholds))


if __name__ == "__main__":
    main()
