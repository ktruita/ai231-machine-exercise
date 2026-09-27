import torch
import torch.nn as nn
from torch import Tensor


def hz_to_mel(hz: Tensor | float) -> Tensor | float:
    """Convert frequency in Hz to the Slaney mel scale."""
    return 2595.0 * torch.log10(torch.as_tensor(hz) / 700.0 + 1.0)


def mel_to_hz(mel: Tensor) -> Tensor:
    """Convert the Slaney mel scale back to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    num_mels: int,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> Tensor:
    """
    Build the triangular mel filterbank matrix.

    Written out rather than taken from torchaudio so the Raspberry Pi only has
    to carry torch, and so the front-end is one matrix multiply that can be
    folded into an exported graph.

    Args:
        sample_rate: Audio sample rate in Hz
        n_fft: FFT size, gives n_fft // 2 + 1 frequency bins
        num_mels: Number of mel filters
        f_min: Lowest frequency covered by the bank (default: 20.0)
        f_max: Highest frequency covered, defaults to the Nyquist rate (default: None)

    Returns:
        Filterbank of shape (num_mels, n_fft // 2 + 1)
    """
    f_max = f_max or sample_rate / 2.0

    # Filter centres are evenly spaced on the mel scale, which is why low
    # frequencies get narrow filters and high frequencies get wide ones
    mel_points = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), num_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)

    # (num_mels + 2, 1) - (1, n_fft//2 + 1) -> (num_mels + 2, n_fft//2 + 1)
    slopes = hz_points.unsqueeze(1) - fft_freqs.unsqueeze(0)
    steps = hz_points[1:] - hz_points[:-1]

    # Each filter rises across its lower band and falls across its upper band
    lower = -slopes[:-2] / steps[:-1].unsqueeze(1)
    upper = slopes[2:] / steps[1:].unsqueeze(1)

    filterbank = torch.clamp(torch.minimum(lower, upper), min=0.0)

    return filterbank


class LogMelSpectrogram(nn.Module):
    """
    Log-mel filterbank front-end.

    Converts raw audio into the time-frequency representation the encoder
    expects. Kept as an nn.Module so it exports alongside the model and the
    device runs exactly the same front-end the model was trained on.

    Flow: (B, num_samples) -> (B, num_mels, num_samples // hop_length + 1)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        num_mels: int = 40,
        f_min: float = 20.0,
        f_max: float | None = None,
        log_offset: float = 1e-6,
    ) -> None:
        """
        Initialize the log-mel front-end.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            n_fft: FFT size, 400 samples is a 25 ms window at 16 kHz (default: 400)
            hop_length: Samples between frames, 160 is a 10 ms hop (default: 160)
            num_mels: Number of mel filters (default: 40)
            f_min: Lowest frequency covered by the bank (default: 20.0)
            f_max: Highest frequency covered, defaults to Nyquist (default: None)
            log_offset: Added before the log to keep silence finite (default: 1e-6)
        """
        super().__init__()

        filterbank = build_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            num_mels=num_mels,
            f_min=f_min,
            f_max=f_max
        )

        # Buffers so the filterbank and window move with .to(device) and are
        # saved in the checkpoint, but are not trained
        self.register_buffer("filterbank", filterbank)
        self.register_buffer("window", torch.hann_window(n_fft))

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.log_offset = log_offset

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform: Audio of shape (B, num_samples), float in [-1, 1]

        Returns:
            Log-mel spectrogram of shape (B, num_mels, num_frames)
        """
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True
        )

        # (B, n_fft//2 + 1, T) -> power spectrogram
        power = spectrum.real ** 2 + spectrum.imag ** 2

        # (num_mels, n_fft//2 + 1) x (B, n_fft//2 + 1, T) -> (B, num_mels, T)
        mel = torch.matmul(self.filterbank, power)

        return torch.log(mel + self.log_offset)


class SpecAugment(nn.Module):
    """
    Frequency and time masking on the log-mel spectrogram.

    Every other augmentation in this project acts on the waveform, which makes
    the recording sound different but leaves the spectrogram's structure intact.
    Masking bands and frames instead removes information outright, so the model
    cannot lean on any single formant region or any single instant to decide -
    it has to read the whole word. Training only, and applied after the
    front-end, so it never reaches the exported graph.

    Flow: (B, num_mels, T) -> (B, num_mels, T), masked in place of a copy

    Note the window is fixed at 6 s while a command occupies one to three of
    them, so a uniformly placed time mask often lands on padding and does
    nothing. Frequency masking carries most of the effect here.
    """

    def __init__(
        self,
        num_freq_masks: int = 2,
        freq_width: int = 6,
        num_time_masks: int = 2,
        time_width: int = 20,
    ) -> None:
        """
        Initialize the masking augmentation.

        Args:
            num_freq_masks: Number of frequency bands to mask per utterance (default: 2)
            freq_width: Largest band width in mel channels, sampled per mask (default: 6)
            num_time_masks: Number of time spans to mask per utterance (default: 2)
            time_width: Largest span in frames, 20 frames is 200 ms (default: 20)
        """
        super().__init__()

        self.num_freq_masks = num_freq_masks
        self.freq_width = freq_width
        self.num_time_masks = num_time_masks
        self.time_width = time_width

    def forward(self, mel: Tensor) -> Tensor:
        """
        Args:
            mel: Log-mel spectrogram of shape (B, num_mels, T)

        Returns:
            Spectrogram with masked regions replaced by the per-utterance mean
        """
        if not self.training:
            return mel

        mel = mel.clone()
        batch, num_mels, num_frames = mel.shape

        # The mel is an unnormalized log, so silence sits near log(1e-6) and
        # zero would be a loud value. Filling with the utterance mean is the
        # neutral choice the SpecAugment paper makes for normalized inputs
        fill = mel.mean(dim=(1, 2), keepdim=True)

        for axis, count, width, size in (
            (1, self.num_freq_masks, self.freq_width, num_mels),
            (2, self.num_time_masks, self.time_width, num_frames),
        ):
            limit = min(width, size)
            if count <= 0 or limit <= 0:
                continue

            for _ in range(count):
                # One width and one start per utterance, so the batch does not
                # share a mask and see the same band removed every time
                widths = torch.randint(0, limit + 1, (batch, 1), device=mel.device)
                starts = (torch.rand(batch, 1, device=mel.device) * (size - widths)).long()

                index = torch.arange(size, device=mel.device).unsqueeze(0)
                mask = (index >= starts) & (index < starts + widths)

                mask = mask.unsqueeze(2 if axis == 1 else 1)
                mel = torch.where(mask, fill, mel)

        return mel
