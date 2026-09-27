"""Log-mel front-end without torch, for deployment.

torch.stft does not export to ONNX - "STFT does not currently support complex
types" - so the front-end cannot live inside the exported graph. Keeping it
outside turns out better for the device anyway: with the mel computed here, the
Pi needs only numpy and onnxruntime, and never installs torch at all.

This must stay numerically identical to vcm/features.py. Any drift is a silent
train/deploy skew that would show up as unexplained accuracy loss on device.
"""
import numpy as np


class LogMelSpectrogram:
    """
    Log-mel filterbank front-end, numpy implementation.

    Mirrors vcm.features.LogMelSpectrogram exactly: 25 ms Hann window, 10 ms
    hop, reflect padding so the first frame is centred on sample 0, a triangular
    mel bank applied as one matrix multiply, and a floor before the log so
    silence stays finite.

    Flow: (num_samples,) -> (1, num_mels, num_frames)
    """

    def __init__(
        self,
        filterbank: np.ndarray,
        n_fft: int = 400,
        hop_length: int = 160,
        log_offset: float = 1e-6,
    ) -> None:
        """
        Initialize the front-end.

        Args:
            filterbank: Mel filterbank of shape (num_mels, n_fft // 2 + 1), as
                exported alongside the model so both were built from one spec
            n_fft: FFT size, 400 samples is 25 ms at 16 kHz (default: 400)
            hop_length: Samples between frames (default: 160)
            log_offset: Added before the log to keep silence finite (default: 1e-6)
        """
        self.filterbank = filterbank.astype(np.float32)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_offset = log_offset
        self.window = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """
        Args:
            waveform: Audio of shape (num_samples,), float in [-1, 1]

        Returns:
            Log-mel spectrogram of shape (1, num_mels, num_frames)
        """
        pad = self.n_fft // 2
        padded = np.pad(waveform.astype(np.float32), (pad, pad), mode="reflect")

        # One frame per hop, matching torch.stft with center=True
        num_frames = 1 + (len(waveform)) // self.hop_length
        indices = np.arange(self.n_fft)[None, :] + \
            self.hop_length * np.arange(num_frames)[:, None]
        frames = padded[indices] * self.window

        power = np.abs(np.fft.rfft(frames, n=self.n_fft, axis=-1)) ** 2

        # (num_mels, bins) x (frames, bins) -> (num_mels, frames)
        mel = self.filterbank @ power.T

        return np.log(mel + self.log_offset)[None].astype(np.float32)
