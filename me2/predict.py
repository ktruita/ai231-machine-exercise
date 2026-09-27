"""Run a trained voice command model on WAV files.

Takes the same path a deployed device would: read audio, run the model, print
the structured command. No labels involved, so this also works on audio you
record yourself.
"""
import argparse
from pathlib import Path

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dataloaders.vcm_dataloader import read_wav, fit_length
from utils import find_latest_checkpoint


def load_model(run_dir: str | Path, device: str = "cpu"):
    """
    Load a trained module from a run directory.

    Args:
        run_dir: Directory holding config.yaml and checkpoints/
        device: Device to run on (default: 'cpu')

    Returns:
        The module in eval mode
    """
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    OmegaConf.resolve(cfg.module)

    module = instantiate(cfg.module.module, cfg.module)
    ckpt_path = find_latest_checkpoint(run_dir / "checkpoints")
    state = torch.load(ckpt_path, weights_only=False, map_location="cpu")["state_dict"]
    module.load_state_dict(state)

    return module.to(device).eval()


def predict(module, paths: list[str], device: str = "cpu") -> list[dict]:
    """
    Decode a batch of WAV files into commands.

    Args:
        module: Trained module
        paths: WAV file paths
        device: Device to run on (default: 'cpu')

    Returns:
        One result dictionary per file, with the command and its confidence
    """
    # Read the window from the spec rather than assuming it. It has changed
    # once already, and a stale constant here would silently truncate audio.
    audio_spec = yaml.safe_load(open(Path(__file__).resolve().parent / "commands.yaml"))["audio"]
    num_samples = int(audio_spec["max_duration_s"] * audio_spec["sample_rate"])

    waveforms = torch.stack([
        torch.from_numpy(fit_length(read_wav(p), num_samples)) for p in paths
    ]).to(device)

    with torch.no_grad():
        logits = module(waveforms)
        commands = module.backbone.decode(logits)
        confidence = torch.softmax(logits["intent"], dim=-1).max(dim=-1).values

    return [
        {"file": Path(p).name, "command": c, "confidence": round(s.item(), 3)}
        for p, c, s in zip(paths, commands, confidence)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+", help="WAV files to decode")
    ap.add_argument("--run", default="modelstore/vcm_masked_loss")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    module = load_model(args.run, args.device)

    for result in predict(module, args.wavs, args.device):
        command = result["command"]
        slots = " ".join(f"{k}={v}" for k, v in command.items() if k != "intent")
        print(f"{result['file']:44s} -> {command['intent']:14s} {slots:34s} p={result['confidence']:.3f}")


if __name__ == "__main__":
    main()
