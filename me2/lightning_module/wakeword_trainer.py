import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import Tensor

from .base_module import BaseLightningModule


class WakeWordModule(BaseLightningModule):
    """
    Training module for the wake word detector.

    Reports false rejects and false accepts separately rather than plain
    accuracy. An always-on detector is judged by how often it fires when nobody
    spoke to it, and a single accuracy figure hides that completely.
    """

    def __init__(
        self,
        cfg,
        name="wakeword",
        lr=3e-4,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        num_warmup_steps=300,
        num_training_steps=6000,
        num_cycles=0.5,
        label_smoothing=0.0,
        **kwargs,
    ):
        """
        Initialize the wake word module.

        Args:
            cfg: Hydra config containing features and backbone specifications
            name: Model name for saving outputs
            lr: Learning rate
            betas: AdamW betas
            weight_decay: AdamW weight decay
            num_warmup_steps: Learning rate warmup steps
            num_training_steps: Total training steps
            num_cycles: Cosine schedule cycles
            label_smoothing: Cross-entropy label smoothing (default: 0.0)
        """
        super().__init__()
        self.__dict__.update(locals())

        self.features = instantiate(cfg.features, _convert_="all")
        self.backbone = instantiate(cfg.backbone, _convert_="all")

        self.save_hyperparameters(ignore=["cfg"])

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform: Audio of shape (B, num_samples), float in [-1, 1]

        Returns:
            Logits of shape (B, 2)
        """
        return self.backbone(self.features(waveform))

    def reset_counts(self) -> None:
        """Zero the running counts at the start of an epoch."""
        self.counts = {"positives": 0, "negatives": 0, "missed": 0, "fired": 0}

    def update_counts(self, logits: Tensor, labels: Tensor) -> None:
        """
        Accumulate false rejects and false accepts over the epoch.

        Args:
            logits: Logits of shape (B, 2)
            labels: Binary targets of shape (B,)
        """
        predicted = logits.argmax(dim=-1)
        positive = labels == 1

        self.counts["positives"] += positive.sum().item()
        self.counts["negatives"] += (~positive).sum().item()
        self.counts["missed"] += ((predicted == 0) & positive).sum().item()
        self.counts["fired"] += ((predicted == 1) & ~positive).sum().item()

    def compute_rates(self) -> dict[str, float]:
        """
        Turn the accumulated counts into error rates.

        Returns:
            False reject rate, false accept rate and accuracy
        """
        positives = max(self.counts["positives"], 1)
        negatives = max(self.counts["negatives"], 1)
        total = positives + negatives

        return {
            "frr": self.counts["missed"] / positives,
            "far": self.counts["fired"] / negatives,
            "acc": 1 - (self.counts["missed"] + self.counts["fired"]) / total,
        }

    def training_step(self, batch, batch_nb):
        """Training step over one batch of windows."""
        logits = self.forward(batch["waveform"])
        loss = F.cross_entropy(logits, batch["label"], label_smoothing=self.label_smoothing)

        self.mylog(loss=loss)

        return loss

    def on_validation_epoch_start(self):
        """Reset counts at epoch start."""
        self.reset_counts()

    def validation_step(self, batch, batch_nb):
        """Validation step over one batch of windows."""
        logits = self.forward(batch["waveform"])
        loss = F.cross_entropy(logits, batch["label"], label_smoothing=self.label_smoothing)

        self.update_counts(logits, batch["label"])
        self.mylog(loss=loss)

        return loss

    def on_validation_epoch_end(self):
        """Compute and log error rates from the whole epoch."""
        self.mylog(**self.compute_rates(), mode="val_")
