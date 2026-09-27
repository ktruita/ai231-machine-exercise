import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def find_latest_checkpoint(ckpt_dir: Path, filename_match: str = None) -> Path | None:
    """
    Find the most recently written checkpoint in a directory.

    Args:
        ckpt_dir: Directory containing .ckpt files
        filename_match: Only consider checkpoints whose name contains this string

    Returns:
        Path to the newest matching checkpoint, or None if there is none
    """
    if not ckpt_dir.exists():
        return None

    paths = list(ckpt_dir.glob("*.ckpt"))

    if filename_match is not None:
        paths = [p for p in paths if filename_match in p.name]

    if not paths:
        return None

    return sorted(paths, key=lambda x: x.stat().st_mtime)[-1]


def load_or_resume_config(config: DictConfig) -> tuple[DictConfig, Path | None]:
    """
    Load existing config for resuming or return new config.

    Args:
        config: New config from Hydra

    Returns:
        Tuple of (final_config, checkpoint_path)
    """
    new_exp_dir = Path(config.exp_dir)
    resume_from = Path(config.get("resume_from", config.exp_dir))
    ckpt_dir = resume_from / "checkpoints"

    if not ckpt_dir.exists():
        return config, None

    new_exp_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, new_exp_dir / "config.yaml")

    ckpt_path = find_latest_checkpoint(
        ckpt_dir,
        filename_match=getattr(config, "ckpt_filename_match", None)
    )

    if ckpt_path:
        print(f"Resuming model from checkpoint: {ckpt_path}")
    else:
        print(f"No checkpoint found in {ckpt_dir}, starting fresh")

    if config.mode == "test":
        ckpt_config_path = resume_from / "config.yaml"
        if ckpt_config_path.exists():
            config = OmegaConf.load(ckpt_config_path)

    return config, ckpt_path


def count_parameters(model) -> dict[str, int]:
    """
    Count parameters per top-level submodule.

    Args:
        model: Module to inspect

    Returns:
        Mapping of submodule name to parameter count, plus a 'total' entry
    """
    counts = {name: sum(p.numel() for p in child.parameters())
              for name, child in model.named_children()}
    counts["total"] = sum(p.numel() for p in model.parameters())

    return counts
