import torch.nn as nn
from torch import Tensor


class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise separable 1D convolution: one filter per channel followed by a
    pointwise projection that mixes channels.

    Flow: (B, input_channels, T) -> (B, output_channels, T // stride)
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
    ) -> None:
        """
        Initialize the depthwise separable convolution.

        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            kernel_size: Width of the temporal kernel
            dilation: Spacing between kernel taps, widens the receptive field (default: 1)
            stride: Temporal downsampling factor (default: 1)
        """
        super().__init__()

        # Padding keeps the time axis length predictable so residuals line up
        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=input_channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            bias=False
        )
        self.norm = nn.BatchNorm1d(output_channels)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class ResidualBlock(nn.Module):
    """
    Two depthwise separable convolutions with a residual connection. The first
    convolution carries the stride and the channel change, the second refines
    at the new resolution.

    Flow: (B, input_channels, T) -> (B, output_channels, T // stride)
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
    ) -> None:
        """
        Initialize the residual block.

        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            kernel_size: Width of the temporal kernel
            dilation: Spacing between kernel taps (default: 1)
            stride: Temporal downsampling factor applied by the first conv (default: 1)
        """
        super().__init__()

        self.conv1 = DepthwiseSeparableConv1D(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride
        )
        self.conv2 = DepthwiseSeparableConv1D(
            input_channels=output_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )

        # Projection is only needed when the block changes shape
        if input_channels != output_channels or stride != 1:
            self.shortcut = nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv2(x)

        return x + residual


class ConvEncoder(nn.Module):
    """
    Convolutional encoder shared by every classification head. Turns a log-mel
    spectrogram into a short sequence of frame embeddings, widening the
    receptive field through stride and dilation rather than through depth.

    Flow: (B, num_mels, T) -> (B, dims[-1], T // 8)
    """

    def __init__(
        self,
        num_mels: int = 40,
        dims: tuple[int, ...] = (64, 64, 96, 128),
        kernel_sizes: tuple[int, ...] = (13, 15, 17),
        dilations: tuple[int, ...] = (1, 2, 4),
        strides: tuple[int, ...] = (1, 2, 2),
    ) -> None:
        """
        Initialize the convolutional encoder.

        Args:
            num_mels: Number of log-mel filterbank channels (default: 40)
            dims: Channel width of the stem followed by each residual block (default: (64, 64, 96, 128))
            kernel_sizes: Temporal kernel width per residual block (default: (13, 15, 17))
            dilations: Dilation factor per residual block (default: (1, 2, 4))
            strides: Temporal stride per residual block (default: (1, 2, 2))
        """
        super().__init__()

        # Validation
        if not len(dims) == len(kernel_sizes) + 1 == len(dilations) + 1 == len(strides) + 1:
            raise ValueError(
                f"dims ({len(dims)}) must be exactly one longer than kernel_sizes "
                f"({len(kernel_sizes)}), dilations ({len(dilations)}) and strides ({len(strides)})"
            )

        # Stem halves the time axis before the residual stack sees it
        self.stem = nn.Sequential(
            nn.Conv1d(num_mels, dims[0], kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i, (kernel_size, dilation, stride) in enumerate(zip(kernel_sizes, dilations, strides)):
            block = ResidualBlock(
                input_channels=dims[i],
                output_channels=dims[i + 1],
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride
            )
            self.blocks.append(block)

        # Pointwise mixing after the residual stack
        self.head = nn.Sequential(
            nn.Conv1d(dims[-1], dims[-1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dims[-1]),
            nn.ReLU()
        )

        self.output_dim = dims[-1]
        self.total_stride = 2
        for stride in strides:
            self.total_stride *= stride

    def forward(self, mel: Tensor) -> Tensor:
        """
        Args:
            mel: Log-mel spectrogram of shape (B, num_mels, T)

        Returns:
            Frame embeddings of shape (B, output_dim, T // total_stride)
        """
        # (B, num_mels, T) -> (B, dims[0], T//2)
        x = self.stem(mel)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)

        return x
