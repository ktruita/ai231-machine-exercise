import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import Tensor

from vcm.features import LogMelSpectrogram
from vcm.spec import load_command_spec
from .base_module import BaseLightningModule


class VoiceCommandModule(BaseLightningModule):
    """
    Training module for the voice command model.

    Wraps the log-mel front-end and the backbone, sums one cross-entropy per
    head, and reports exact command accuracy, which requires the intent and
    every slot the intent uses to be right at once.
    """

    def __init__(
        self,
        cfg,
        name="vcm",
        lr=3e-4,
        betas=(0.9, 0.98),
        weight_decay=0.05,
        num_warmup_steps=500,
        num_training_steps=20000,
        num_cycles=0.5,
        masked_loss=True,
        label_smoothing=0.0,
        class_weight_power=0.5,
        **kwargs,
    ):
        """
        Initialize the voice command module.

        Args:
            cfg: Hydra config containing spec_path, features and backbone specifications
            name: Model name for saving outputs
            lr: Learning rate
            betas: AdamW betas
            weight_decay: AdamW weight decay
            num_warmup_steps: Learning rate warmup steps
            num_training_steps: Total training steps
            num_cycles: Cosine schedule cycles
            masked_loss: Grade a slot head only on utterances whose intent uses that
                slot. False grades every head on every utterance, which lets the head
                score well by always answering N/A (default: True)
            label_smoothing: Cross-entropy label smoothing (default: 0.0)
            class_weight_power: Exponent on inverse class frequency. The training
                mix is uneven - `none` is 36% of utterances against 7% for
                timer.set, and `play` is 49% of media actions against 9% for
                `stop` - so an unweighted loss lets the model win by following
                the prior. 0 disables weighting, 1 is full inverse frequency,
                0.5 softens it so rare classes are helped without being
                over-weighted into noise (default: 0.5)
        """
        super().__init__()
        self.__dict__.update(locals())

        spec = load_command_spec(cfg.spec_path)

        # Front-end is part of the module so the device runs the same mel the
        # model was trained on, and so it exports with the graph
        self.features = instantiate(cfg.features, _convert_="all")
        self.backbone = instantiate(cfg.backbone, spec=spec, _convert_="all")

        # Training-only, and deliberately not part of the front-end: the export
        # takes the backbone alone, so this can never leak into the device graph
        self.spec_augment = instantiate(cfg.spec_augment, _convert_="all") \
            if cfg.get("spec_augment") else None

        self.slot_names = self.backbone.slot_names
        self.intent_names = self.backbone.intent_names

        self.save_hyperparameters(ignore=["cfg"])

    def on_train_start(self):
        """
        Derive class weights from the training distribution.

        Done here rather than in on_fit_start: Lightning has not attached the
        dataloaders that early, so trainer.train_dataloader is still None.
        """
        self.class_weights = {}
        if self.class_weight_power <= 0 or self.trainer.train_dataloader is None:
            return

        dataset = self.trainer.train_dataloader.dataset
        counts = {"intent": torch.zeros(len(self.intent_names))}
        counts.update({
            name: torch.zeros(len(dataset.slot_classes[name])) for name in self.slot_names
        })

        for record in dataset.records:
            if record.get("supervise_intent", True):
                counts["intent"][dataset.intent_index[record["intent"]]] += 1

            # Count N/A as its own class, and only over the utterances where the
            # slot is active. Skipping it left N/A at zero, which clamped to one
            # and so outweighed a room with 1,600 examples by roughly forty to
            # one - the model was being trained to answer "no room".
            active = set(dataset.intent_slots[record["intent"]]) | dataset.expand_supervised(record)
            for name in self.slot_names:
                if name not in active:
                    continue

                if name in dataset.digit_source:
                    base, divisor = dataset.digit_source[name]
                    raw = record["slots"].get(base)
                    value = None if raw is None else (raw % 10 if divisor == 1 else raw // 10)
                else:
                    value = record["slots"].get(name)

                key = "N/A" if value is None else value
                counts[name][dataset.slot_index[name][key]] += 1

        for head, count in counts.items():
            # Unseen classes would give an infinite weight, so floor at one
            weight = count.clamp(min=1.0).pow(-self.class_weight_power)
            self.class_weights[head] = (weight / weight.mean()).to(self.device)

    def forward(self, waveform: Tensor) -> dict[str, Tensor]:
        """
        Forward pass from raw audio to logits.

        Args:
            waveform: Audio of shape (B, num_samples), float in [-1, 1]

        Returns:
            Logits keyed by head name
        """
        mel = self.features(waveform)
        if self.spec_augment is not None:
            mel = self.spec_augment(mel)

        logits = self.backbone(mel)

        return logits

    def loss(self, logits: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Sum one cross-entropy per head.

        The slot heads are the whole reason masked_loss exists. A slot is N/A on
        roughly 80% of utterances, so grading every head on every utterance lets
        a head reach high accuracy by never predicting a value at all.

        Args:
            logits: Logits keyed by head name
            batch: Batch holding the intent target, per-slot targets and masks

        Returns:
            Tuple of (total loss, per-head losses for logging)
        """
        weights = getattr(self, "class_weights", {})

        intent_mask = batch["mask_intent"]
        per_intent = F.cross_entropy(
            logits["intent"], batch["intent"], label_smoothing=self.label_smoothing,
            weight=weights.get("intent"), reduction="none"
        )
        total = (per_intent * intent_mask).sum() / intent_mask.sum().clamp(min=1)
        parts = {"loss_intent": total.detach()}

        for name in self.slot_names:
            per_item = F.cross_entropy(
                logits[name],
                batch[f"target_{name}"],
                label_smoothing=self.label_smoothing,
                weight=weights.get(name),
                reduction="none"
            )

            if self.masked_loss:
                # Average over only the utterances whose intent uses this slot
                mask = batch[f"mask_{name}"]
                slot_loss = (per_item * mask).sum() / mask.sum().clamp(min=1)
            else:
                slot_loss = per_item.mean()

            total = total + slot_loss
            parts[f"loss_{name}"] = slot_loss.detach()

        return total, parts

    def reset_counts(self) -> None:
        """Zero the running accuracy counts at the start of an epoch."""
        self.counts = {"total": 0, "intent": 0, "exact": 0}
        self.counts.update({f"ok_{name}": 0 for name in self.slot_names})
        self.counts.update({f"active_{name}": 0 for name in self.slot_names})

    def update_counts(self, logits: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
        """
        Accumulate correct counts over the epoch.

        Counts are accumulated rather than averaged per batch because a slot is
        active on roughly a sixth of the data, and the manifests are ordered by
        intent. An unshuffled validation batch can therefore contain no active
        utterance for a slot at all, and averaging per-batch ratios averages in
        those empty batches as zeros.

        Args:
            logits: Logits keyed by head name
            batch: Batch holding the intent target, per-slot targets and masks

        Returns:
            Exact command accuracy for this batch, for progress reporting
        """
        # Rows excluded from the intent loss are excluded from the metric too,
        # so the reported accuracy describes the task the model is graded on
        graded = batch["mask_intent"]
        intent_correct = (logits["intent"].argmax(dim=-1) == batch["intent"]) & graded

        # A command counts only if the intent and every slot it uses are right.
        # Using the target mask is exact: when the intent is wrong the utterance
        # has already failed, and when it is right the masks agree.
        exact = intent_correct.clone()

        for name in self.slot_names:
            mask = batch[f"mask_{name}"]
            correct = logits[name].argmax(dim=-1) == batch[f"target_{name}"]

            self.counts[f"ok_{name}"] += (correct & mask).sum().item()
            self.counts[f"active_{name}"] += mask.sum().item()
            exact = exact & (correct | ~mask)

        self.counts["total"] += graded.sum().item()
        self.counts["intent"] += intent_correct.sum().item()
        self.counts["exact"] += (exact & graded).sum().item()

        return (exact & graded).float().mean()

    def compute_accuracy(self) -> dict[str, float]:
        """
        Turn the accumulated counts into accuracies.

        Per-slot accuracy is over only the utterances whose intent uses that
        slot. Scoring over all of them would let a head look accurate while
        never predicting a value, since the slot is N/A on most utterances.

        Returns:
            Accuracies keyed by metric name
        """
        total = max(self.counts["total"], 1)

        metrics = {
            "acc_intent": self.counts["intent"] / total,
            "acc_exact": self.counts["exact"] / total,
        }
        for name in self.slot_names:
            active = self.counts[f"active_{name}"]
            metrics[f"acc_{name}"] = self.counts[f"ok_{name}"] / active if active else float("nan")

        return metrics

    def on_train_epoch_start(self):
        """Reset accuracy counts at epoch start."""
        self.reset_counts()

    def training_step(self, batch, batch_nb):
        """Training step over one batch of utterances."""
        logits = self.forward(batch["waveform"])
        loss, parts = self.loss(logits, batch)

        exact = self.update_counts(logits, batch)
        self.mylog(loss=loss, acc_exact=exact)

        return loss

    def on_validation_epoch_start(self):
        """Reset accuracy counts at epoch start."""
        self.reset_counts()

    def validation_step(self, batch, batch_nb):
        """Validation step over one batch of utterances."""
        logits = self.forward(batch["waveform"])
        loss, parts = self.loss(logits, batch)

        self.update_counts(logits, batch)
        self.mylog(loss=loss)
        self.mylog(**parts)

        return loss

    def on_validation_epoch_end(self):
        """Compute and log accuracies from the whole epoch."""
        self.mylog(**self.compute_accuracy(), mode="val_")

    def on_test_epoch_start(self):
        """Reset accuracy counts at epoch start."""
        self.reset_counts()

    def test_step(self, batch, batch_nb):
        """Test step over one batch of utterances."""
        logits = self.forward(batch["waveform"])
        loss, parts = self.loss(logits, batch)

        self.update_counts(logits, batch)
        self.mylog(loss=loss, mode="test_")

        return loss

    def on_test_epoch_end(self):
        """Compute and log accuracies from the whole epoch."""
        self.mylog(**self.compute_accuracy(), mode="test_")
