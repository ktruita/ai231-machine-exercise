from pathlib import Path

import diffusers
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def load_module(
    path: str,
    device: str = "auto",
    dotlist: list = None,
    return_config: bool = True,
    ckpt_fname: str = None,
    **kwargs,
):
    """
    Load a Lightning module from a saved checkpoint.

    Args:
        path: Path to model directory containing config.yaml and checkpoints/
        device: Device to load model on ('auto', 'cuda', or 'cpu')
        dotlist: List of config overrides
        return_config: Whether to return (module, config) or just module
        ckpt_fname: Specific checkpoint filename, otherwise uses most recent
        **kwargs: Additional arguments passed to module instantiation

    Returns:
        Lightning module, or (module, config) if return_config=True
    """
    dotlist = dotlist or []

    # Resolve path
    if Path("modelstore").joinpath(path).exists():
        path = Path("modelstore").joinpath(path)
    else:
        path = Path(path)

    # Load and merge config
    cfg = OmegaConf.load(path / "config.yaml")
    cfg.merge_with_dotlist(dotlist)

    # Instantiate module and load checkpoint
    module = instantiate(cfg.module.module, cfg.module, **kwargs)
    module.init_from_ckpt(path, ckpt_fname=ckpt_fname, missing_warning=False)

    # Move to device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device).eval()

    return (module, cfg) if return_config else module


class BaseLightningModule(L.LightningModule):
    """Base Lightning module with common utilities for training and checkpointing."""

    def __init__(self):
        super().__init__()

    def mylog(self, dct=None, mode="auto", **kwargs):
        """
        Log metrics with automatic train/val prefix.

        Args:
            dct: Dictionary of metrics to log
            mode: Logging mode ('auto', 'train_', 'val_', etc.)
            **kwargs: Additional metrics to log
        """
        dct = dct or {}
        if mode == "auto":
            mode = "train_" if self.training else "val_"

        dct.update(kwargs)
        for k, v in dct.items():
            self.log(
                mode + k,
                v,
                prog_bar=True,
                sync_dist=True,
                add_dataloader_idx=True
            )

    def init_from_ckpt(
        self,
        path: str,
        ckpt_fname: str = None,
        ignore_keys: list = None,
        missing_warning: bool = True,
    ):
        """
        Initialize module from checkpoint.

        Args:
            path: Directory containing checkpoints/ folder or direct checkpoint path
            ckpt_fname: Specific checkpoint filename (optional)
            ignore_keys: List of key prefixes to ignore when loading
            missing_warning: Whether to warn about missing keys
        """
        ignore_keys = ignore_keys or []

        # Resolve checkpoint path
        if Path(path).is_dir():
            ckpt_dir = Path(path) / "checkpoints"
            paths = list(ckpt_dir.glob("*.ckpt"))

            if ckpt_fname is not None:
                paths = [p for p in paths if ckpt_fname in p.name]

            if not paths:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

            # Use most recent checkpoint
            path = sorted(paths, key=lambda x: x.stat().st_mtime)[-1]

        # Load state dict
        sd = torch.load(path, weights_only=True, map_location="cpu")["state_dict"]

        # Remove ignored keys
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict")
                    del sd[k]

        # Load state dict
        self.load_state_dict(sd, strict=False)

        # Report missing keys
        missing_keys = set(
            ".".join(k.split(".")[:2])
            for k in self.state_dict().keys()
            if k not in sd.keys()
        )

        if missing_keys and missing_warning:
            print(f"Missing keys: {missing_keys}")

        print(f"Restored from {path}")

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine schedule and warmup."""
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        sched = diffusers.optimization.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles,
        )

        return [opt], [{
            "scheduler": sched,
            "interval": "step",
            "frequency": 1,
        }]
