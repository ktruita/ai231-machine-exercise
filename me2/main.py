import shutil
import signal
import warnings
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.pytorch.callbacks import TQDMProgressBar
from omegaconf import DictConfig, OmegaConf

from utils import load_or_resume_config, count_parameters


class SafeCheckpointIO(TorchCheckpointIO):
    def load_checkpoint(self, path, map_location=None, weights_only=True):
        return torch.load(path, map_location=map_location, weights_only=False)


class FixedStepProgressBar(TQDMProgressBar):
    """Progress bar that shows progress relative to steps within the current epoch."""

    def __init__(self, max_steps: int, steps_per_epoch: int, refresh_rate: int = 100):
        super().__init__(refresh_rate=refresh_rate)
        self.max_steps = max_steps
        self.steps_per_epoch = steps_per_epoch
        self._last_step_logged = 0

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.total = self.steps_per_epoch
        return bar

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        # Reset bar total and position at the start of each epoch
        resumed_offset = trainer.global_step % self.steps_per_epoch
        self.train_progress_bar.total = self.steps_per_epoch
        self.train_progress_bar.n = resumed_offset
        self.train_progress_bar.refresh()
        self._last_step_logged = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only update display every refresh_rate steps, suppressing 1,2,3...
        if trainer.global_step - self._last_step_logged >= self.refresh_rate:
            steps_in_epoch = trainer.global_step % self.steps_per_epoch or self.steps_per_epoch
            self.train_progress_bar.n = steps_in_epoch
            self.train_progress_bar.total = self.steps_per_epoch
            self.train_progress_bar.refresh()
            self._last_step_logged = trainer.global_step


class StepCheckpoint(L.Callback):
    """Save checkpoint every N training steps."""

    def __init__(
        self,
        save_dir: str | Path,
        save_frequency: int = 2000,
        prefix: str = "checkpoint",
    ) -> None:
        """
        Initialize step-based checkpoint callback.

        Args:
            save_dir: Directory to save checkpoints
            save_frequency: Save every N steps
            prefix: Checkpoint filename prefix
        """
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.prefix = prefix

    def on_train_batch_end(self, trainer: L.Trainer, *args, **kwargs) -> None:
        """Save checkpoint if step matches frequency."""
        if trainer.global_step % self.save_frequency == 0 and trainer.global_step > 0:
            self._save(trainer)

    def on_train_end(self, trainer: L.Trainer, *args, **kwargs) -> None:
        """Always save at the end so the final weights are never lost."""
        self._save(trainer)

    def _save(self, trainer: L.Trainer) -> None:
        """Save checkpoint to disk."""
        ckpt_dir = self.save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.prefix}_step_{trainer.global_step}.ckpt"
        ckpt_path = ckpt_dir / filename

        trainer.save_checkpoint(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


def setup_logger(config: DictConfig) -> L.pytorch.loggers.MLFlowLogger | None:
    """
    Setup MLflow logger if logging is enabled.

    Args:
        config: Hydra config

    Returns:
        MLflow logger or None
    """
    if not config.log:
        return None

    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name=config.project,
        run_name=config.name,
        tracking_uri=getattr(config.cluster, "mlflow_tracking_uri", "mlruns"),
    )

    # Log hyperparameters and freeze the config next to the checkpoints
    if not (Path(config.exp_dir) / "checkpoints").exists():
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

        exp_dir = Path(config.exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, exp_dir / "config.yaml")

        # The spec defines the label space, so a run is only reloadable
        # alongside the spec it was trained against. config.yaml records the
        # hyperparameters but not this, which left an older checkpoint
        # unloadable once the label space changed.
        spec_path = Path(config.module.spec_path)
        if spec_path.exists():
            shutil.copy(spec_path, exp_dir / "commands.yaml")

    return logger


def create_dataloaders(config: DictConfig) -> tuple:
    """
    Create train/val or test dataloaders based on mode.

    Args:
        config: Hydra config

    Returns:
        Tuple of dataloaders (train, val) or (test,)
    """
    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.cluster.cpus,
        "pin_memory": True,
        "persistent_workers": config.cluster.cpus > 0,
    }

    if config.mode in ("train", "all"):
        train_dataset = instantiate(config.dataloader.dataset)
        val_dataset = instantiate(
            config.dataloader.dataset,
            **getattr(config.dataloader, "validation_args", {})
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **dataloader_kwargs
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            **dataloader_kwargs
        )

        return train_loader, val_loader

    else:  # test mode
        test_dataset = instantiate(
            config.dataloader.dataset,
            **getattr(config.dataloader, "test_args", {})
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,  # No shuffle for reproducible test results
            **dataloader_kwargs
        )

        return (test_loader, )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training/testing entry point."""

    # Register eval resolver for config
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except:
        pass

    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Reset signal handler (for SLURM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # Load or resume config
    config, ckpt_path = load_or_resume_config(config)

    # Seed before anything random is built. Seeded after the model, as it once
    # was, the initial weights came from an unseeded generator, and two runs
    # with the same seed started from different weights
    L.seed_everything(config.seed)

    # Setup logging
    logger = setup_logger(config)

    # Create dataloaders
    dataloaders = create_dataloaders(config)

    # Initialize model
    OmegaConf.resolve(config.module)
    model = instantiate(config.module.module, config.module)

    counts = count_parameters(model)
    print(f"Parameters: {counts}")

    # Only the command module has a loss-masking switch
    if hasattr(model, "masked_loss"):
        print(f"masked_loss: {model.masked_loss}")

    # Setup checkpoint callback
    checkpoint_callback = StepCheckpoint(
        save_dir=config.exp_dir,
        save_frequency=config.save_step_frequency,
    )

    steps_per_epoch = len(dataloaders[0])

    # Configure trainer
    torch.set_float32_matmul_precision("medium")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=config.cluster.gpus,
        precision=config.cluster.precision,
        max_steps=config.max_steps,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_freq,
        limit_train_batches=getattr(config, "limit_train_batches", None),
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=getattr(config, "limit_test_batches", config.limit_val_batches),
        plugins=[SafeCheckpointIO()],
        callbacks=[
            FixedStepProgressBar(
                max_steps=config.max_steps,
                steps_per_epoch=steps_per_epoch,
                refresh_rate=config.log_freq
            ),
            checkpoint_callback
            ],
        logger=logger,
        enable_checkpointing=False,  # Use custom checkpoint callback
        profiler=getattr(config, "profiler", None),
    )

    # Debug mode
    if config.get("debug", False):
        breakpoint()

    # Run training or testing
    if config.mode in ("train", "all"):
        train_loader, val_loader = dataloaders
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    else:
        test_loader = dataloaders
        trainer.test(model, test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
