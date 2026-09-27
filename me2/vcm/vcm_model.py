import torch.nn as nn
from torch import Tensor

from .encoder import ConvEncoder
from .heads import ClassificationHeads
from .spec import build_label_space, active_slots, digit_slots


class VoiceCommandModel(nn.Module):
    """
    Voice Command Model for on-device spoken command understanding.

    Maps a log-mel spectrogram straight to a structured command without an
    intermediate transcript. A single convolutional encoder is shared by an
    intent classifier and one classifier per slot, which is what keeps the
    footprint small enough for a Raspberry Pi.
    """

    def __init__(
        self,
        spec: dict,
        num_mels: int = 40,
        dims: tuple[int, ...] = (64, 64, 96, 128),
        kernel_sizes: tuple[int, ...] = (13, 15, 17),
        dilations: tuple[int, ...] = (1, 2, 4),
        strides: tuple[int, ...] = (1, 2, 2),
    ) -> None:
        """
        Initialize the Voice Command Model.

        Args:
            spec: Parsed command specification, defines the whole label space
            num_mels: Number of log-mel filterbank channels (default: 40)
            dims: Channel width of the stem followed by each residual block (default: (64, 64, 96, 128))
            kernel_sizes: Temporal kernel width per residual block (default: (13, 15, 17))
            dilations: Dilation factor per residual block (default: (1, 2, 4))
            strides: Temporal stride per residual block (default: (1, 2, 2))
        """
        super().__init__()

        intent_names, slot_classes = build_label_space(spec)
        slot_sizes = {name: len(values) for name, values in slot_classes.items()}

        self.encoder = ConvEncoder(
            num_mels=num_mels,
            dims=dims,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            strides=strides
        )
        self.heads = ClassificationHeads(
            dim=self.encoder.output_dim,
            num_intents=len(intent_names),
            slot_sizes=slot_sizes
        )

        # Which heads each intent uses, so training can mask the loss and
        # inference can ignore heads the predicted intent has no slot for
        self.intent_names = intent_names
        self.slot_classes = slot_classes
        self.slot_names = list(slot_classes.keys())
        self.intent_slots = {name: active_slots(spec, name) for name in intent_names}
        self.digit_slots = digit_slots(spec)

    def forward(self, mel: Tensor) -> dict[str, Tensor]:
        """
        Forward pass of the Voice Command Model.

        Args:
            mel: Log-mel spectrogram of shape (B, num_mels, T)

        Returns:
            Logits keyed by head name, "intent" plus one entry per slot
        """
        x = self.encoder(mel)
        logits = self.heads(x)

        return logits

    def decode(self, logits: dict[str, Tensor]) -> list[dict]:
        """
        Turn logits into structured commands.

        The intent decides which slot heads are read, so a light command never
        reports a timer number even if that head produced a confident value.

        Digit-decomposed slots are recombined here, so callers see the single
        value the spec describes rather than the two heads that produced it.

        Args:
            logits: Logits keyed by head name

        Returns:
            One command dictionary per item in the batch
        """
        intent_ids = logits["intent"].argmax(dim=-1)

        commands = []
        for i, intent_id in enumerate(intent_ids.tolist()):
            intent_name = self.intent_names[intent_id]
            command = {"intent": intent_name}

            for slot_name in self.intent_slots[intent_name]:
                class_id = logits[slot_name][i].argmax(dim=-1).item()
                value = self.slot_classes[slot_name][class_id]
                command[slot_name] = None if value == "N/A" else value

            for base, (tens, ones) in self.digit_slots.items():
                if tens not in command:
                    continue
                high, low = command.pop(tens), command.pop(ones)
                command[base] = None if high is None or low is None else high * 10 + low

            commands.append(command)

        return commands
