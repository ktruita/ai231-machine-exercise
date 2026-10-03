"""Waveform augmentation: room reverb, additive noise and gain.

Applied identically to commands and to the `none` class. Augmenting only one of
them would hand the model a channel cue that separates the classes, which is the
same mistake that made the first model reject all real speech.

Pools are split so the conditions used at evaluation are never seen in training.
Reverb trains on simulated impulse responses and evaluates on real measured
ones, which makes the far-field test genuinely harder rather than a replay of
the training distribution.
"""
import hashlib
import io
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

# MUSAN splits into three sources. `speech` is deliberately excluded - babble
# over a command would change what was said, not just how it sounds.
MUSAN_SOURCES = ("noise", "music")

# RIRS_NOISES ships simulated and real impulse responses. Real ones are measured
# in actual rooms and are the harder case, so they are reserved for evaluation.
RIR_TRAIN_DIR = "simulated_rirs"
RIR_EVAL_DIR = "real_rirs_isotropic_noises"

# Share of the noise pool held out for evaluation, so a noise-robustness score
# does not measure noise the model trained on. evaluate_benchmark.py goes
# further and mixes in RIRS_NOISES/pointsource_noises, which no run trains on.
NOISE_EVAL_FRACTION = 0.2


def list_noise_files(
    root: str | Path,
    sources: tuple[str, ...] = MUSAN_SOURCES,
    split: str | None = None,
) -> list[Path]:
    """
    List the MUSAN files usable as additive noise.

    Args:
        root: MUSAN directory
        sources: Which MUSAN subsets to draw from (default: noise and music)
        split: 'train' for the training side of a fixed partition, anything else
            for the held-out side, None for every file (default: None)

    Returns:
        Sorted file paths
    """
    root = Path(root)
    files = []
    for source in sources:
        files += sorted((root / source).rglob("*.wav"))

    if split is None:
        return files

    # Hash the path under MUSAN so the partition is the same on every machine
    def held_out(path: Path) -> bool:
        digest = hashlib.md5(str(path.relative_to(root)).encode()).hexdigest()
        return int(digest, 16) % 100 < NOISE_EVAL_FRACTION * 100

    return [p for p in files if held_out(p) != (split == "train")]


def list_rir_files(root: str | Path, split: str = "train") -> list[Path]:
    """
    List room impulse responses for a split.

    Args:
        root: RIRS_NOISES directory
        split: 'train' for simulated responses, anything else for real measured ones

    Returns:
        Sorted file paths
    """
    subdir = RIR_TRAIN_DIR if split == "train" else RIR_EVAL_DIR

    return sorted((Path(root) / subdir).rglob("*.wav"))


def read_segment(path: Path, num_samples: int, rng: random.Random) -> np.ndarray:
    """
    Read a random span of a noise file without loading the whole thing.

    MUSAN files run to several minutes, so they are read by offset rather than
    in full. Short files are tiled to reach the requested length.

    Args:
        path: WAV file to read
        num_samples: Length to return
        rng: Random source

    Returns:
        Waveform of shape (num_samples,), float32
    """
    info = sf.info(str(path))

    if info.frames <= num_samples:
        audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
        audio = np.tile(audio, int(np.ceil(num_samples / max(len(audio), 1))))
        return audio[:num_samples]

    start = rng.randrange(0, info.frames - num_samples)
    audio, _ = sf.read(str(path), start=start, frames=num_samples,
                       dtype="float32", always_2d=False)

    return audio if audio.ndim == 1 else audio[:, 0]


class AudioAugment:
    """
    Waveform augmentation pipeline.

    Order matters and follows the physical chain: the speaker is in a room
    (reverb), other things in the room make noise (additive noise), and the
    microphone sits at some distance and gain (level). Reversing reverb and
    noise would reverberate the noise as if it came from the speaker's mouth.
    """

    def __init__(
        self,
        musan_dir: str | Path,
        rir_dir: str | Path,
        split: str = "train",
        snr_db: tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0),
        gain_db: tuple[float, float] = (-15.0, 2.0),
        reverb_probability: float = 0.5,
        noise_probability: float = 0.8,
        codec_probability: float = 0.15,
        response_probability: float = 0.6,
        sample_rate: int = 16000,
        seed: int = 231,
    ) -> None:
        """
        Initialize the augmentation pipeline.

        Args:
            musan_dir: MUSAN directory
            rir_dir: RIRS_NOISES directory
            split: Which pool to draw from, 'train' uses simulated reverb (default: 'train')
            snr_db: Signal-to-noise ratios to sample from (default: 20 down to 0)
            gain_db: Range of output gain in decibels, applied after normalising to
                -1 dBFS so the range means the same thing for every clip (default: (-15.0, 2.0))
            reverb_probability: Chance of applying reverb (default: 0.5)
            noise_probability: Chance of adding noise (default: 0.8)
            codec_probability: Chance of a lossy codec round-trip. Kept low: a
                Vorbis round-trip at 16 kHz measured almost neutral, so it adds
                little beyond a slight perturbation (default: 0.15)
            response_probability: Chance of a random microphone response, which
                is the mismatch that actually separates recorded audio from
                synthesised audio (default: 0.6)
            sample_rate: Audio sample rate (default: 16000)
            seed: Base seed for the per-worker random source (default: 231)
        """
        self.noise_files = list_noise_files(musan_dir, split=split)
        self.rir_files = list_rir_files(rir_dir, split)

        # Files are needed only for the effects that can fire: a run on the class
        # dataset alone has noise clips but no impulse responses, and no reverb
        if not self.noise_files and noise_probability > 0:
            raise ValueError(f"No MUSAN noise found under {musan_dir}")
        if not self.rir_files and reverb_probability > 0:
            raise ValueError(f"No impulse responses found under {rir_dir} for split '{split}'")

        self.snr_db = snr_db
        self.gain_db = gain_db
        self.reverb_probability = reverb_probability
        self.noise_probability = noise_probability
        self.codec_probability = codec_probability
        self.response_probability = response_probability
        self.sample_rate = sample_rate
        self.rng = random.Random(seed)

    def apply_reverb(self, waveform: np.ndarray) -> np.ndarray:
        """Convolve with a room impulse response, keeping the original length."""
        rir, _ = sf.read(str(self.rng.choice(self.rir_files)), dtype="float32", always_2d=False)
        if rir.ndim > 1:
            rir = rir[:, 0]

        # Normalising the response keeps reverb from also changing the level
        peak = np.abs(rir).max()
        if peak > 0:
            rir = rir / peak

        # Trim the tail back to the input length so the window stays fixed
        return fftconvolve(waveform, rir)[:len(waveform)]

    def apply_noise(self, waveform: np.ndarray) -> np.ndarray:
        """Mix in a noise segment scaled to a sampled signal-to-noise ratio."""
        noise = read_segment(self.rng.choice(self.noise_files), len(waveform), self.rng)

        speech_power = np.mean(waveform ** 2)
        noise_power = np.mean(noise ** 2)
        if speech_power <= 0 or noise_power <= 0:
            return waveform

        snr = self.rng.choice(self.snr_db)
        scale = np.sqrt(speech_power / (noise_power * 10 ** (snr / 10)))

        return waveform + scale * noise

    def apply_codec(self, waveform: np.ndarray) -> np.ndarray:
        """
        Round-trip through a lossy codec.

        Training audio is clean PCM straight from the synthesiser, but anything
        recorded on a phone arrives having been through AAC or Opus. The
        artefacts - a hard bandwidth limit and quantisation noise around
        transients - are a systematic difference between training and
        deployment that noise and reverb do not reproduce.

        Vorbis stands in for AAC here: soundfile can do it in-process in about
        20 ms, where an ffmpeg subprocess per utterance would dominate the
        dataloader.

        Args:
            waveform: Waveform of shape (num_samples,)

        Returns:
            Waveform of the same shape, after encode and decode
        """
        buffer = io.BytesIO()
        sf.write(buffer, waveform, self.sample_rate, format="OGG", subtype="VORBIS")
        buffer.seek(0)
        decoded, _ = sf.read(buffer, dtype="float32", always_2d=False)

        if len(decoded) < len(waveform):
            decoded = np.pad(decoded, (0, len(waveform) - len(decoded)))

        return decoded[:len(waveform)]

    def apply_response(self, waveform: np.ndarray) -> np.ndarray:
        """
        Impose a random microphone frequency response.

        Measured against the training audio, a phone microphone is not a
        bandwidth limit - it is an EQ curve. One measured here sits 4x above the
        synthetic voices at 6-8 kHz and 4x below them at 2-3 kHz, so filtering
        the top off training audio would move it further away, not closer. What
        the model needs is invariance to the curve, which means seeing many of
        them.

        Implemented as a random tilt plus two peaking filters, which spans the
        shapes cheap capsules actually produce.

        Args:
            waveform: Waveform of shape (num_samples,)

        Returns:
            Waveform with a random response applied
        """
        nyquist = self.sample_rate / 2

        # Broad tilt: shelve one end of the band up or down
        tilt_db = self.rng.uniform(-8.0, 8.0)
        corner = self.rng.uniform(1500.0, 4000.0)
        shelf = "high" if tilt_db >= 0 else "low"
        sos = butter(2, corner / nyquist, btype=shelf, output="sos")
        shelved = sosfilt(sos, waveform)
        waveform = waveform + shelved * (10 ** (abs(tilt_db) / 20) - 1.0)

        # Two resonances, the peaks and notches a capsule and its housing add
        for _ in range(2):
            centre = self.rng.uniform(300.0, 6500.0)
            gain_db = self.rng.uniform(-9.0, 9.0)
            width = self.rng.uniform(0.4, 1.2)
            low = max(centre * (1 - width / 2), 50.0) / nyquist
            high = min(centre * (1 + width / 2), nyquist - 50.0) / nyquist
            if not 0 < low < high < 1:
                continue
            sos = butter(2, [low, high], btype="band", output="sos")
            band = sosfilt(sos, waveform)
            waveform = waveform + band * (10 ** (gain_db / 20) - 1.0)

        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak * 0.95

        return waveform.astype(np.float32)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """
        Args:
            waveform: Clean waveform of shape (num_samples,), float in [-1, 1]

        Returns:
            Augmented waveform of the same shape
        """
        if self.rng.random() < self.reverb_probability:
            waveform = self.apply_reverb(waveform)

        if self.rng.random() < self.noise_probability:
            waveform = self.apply_noise(waveform)

        # Codec and bandwidth come last of the physical chain: the microphone
        # captured the room, then the device encoded what it captured.
        if self.rng.random() < self.response_probability:
            waveform = self.apply_response(waveform)

        if self.rng.random() < self.codec_probability:
            waveform = self.apply_codec(waveform)

        # Normalise to a fixed ceiling first. Reverb and noise both change the
        # level, so without this the gain range would mean something different
        # for every clip and most of them would clip.
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform * (10 ** (-1 / 20) / peak)

        waveform = waveform * 10 ** (self.rng.uniform(*self.gain_db) / 20)

        # Clip rather than rescale: a real microphone clips too, and the upper
        # gain bound is set so this happens occasionally rather than routinely
        return np.clip(waveform, -1.0, 1.0).astype(np.float32)
