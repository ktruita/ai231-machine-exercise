import torch.nn as nn
from torch import Tensor

from .encoder import ConvEncoder


class WakeWordModel(nn.Module):
    """
    Always-on wake word detector.

    A binary classifier over a one second window. It runs continuously while the
    command model stays idle, so it is deliberately smaller than the command
    model: it decides one acoustic pattern against everything else, rather than
    five command families and their slots.

    Splitting the two also moves the rejection burden here. The command model
    only ever sees audio that followed a trigger, so it no longer has to reject
    conversation, television or passing speech on its own.

    Flow: (B, num_mels, T) -> (B, 2)
    """

    def __init__(
        self,
        num_mels: int = 40,
        dims: tuple[int, ...] = (32, 32, 48, 64),
        kernel_sizes: tuple[int, ...] = (9, 11, 13),
        dilations: tuple[int, ...] = (1, 2, 4),
        strides: tuple[int, ...] = (1, 2, 2),
    ) -> None:
        """
        Initialize the wake word model.

        Args:
            num_mels: Number of log-mel filterbank channels (default: 40)
            dims: Channel width of the stem followed by each residual block (default: (32, 32, 48, 64))
            kernel_sizes: Temporal kernel width per residual block (default: (9, 11, 13))
            dilations: Dilation factor per residual block (default: (1, 2, 4))
            strides: Temporal stride per residual block (default: (1, 2, 2))
        """
        super().__init__()

        self.encoder = ConvEncoder(
            num_mels=num_mels,
            dims=dims,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            strides=strides
        )
        self.head = nn.Linear(self.encoder.output_dim, 2)

    def forward(self, mel: Tensor) -> Tensor:
        """
        Args:
            mel: Log-mel spectrogram of shape (B, num_mels, T)

        Returns:
            Logits of shape (B, 2), index 1 is the wake word
        """
        x = self.encoder(mel)

        return self.head(x.mean(dim=-1))
