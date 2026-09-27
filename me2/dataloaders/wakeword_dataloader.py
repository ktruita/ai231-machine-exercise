import random
import wave
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Google Speech Commands ships its own speaker-disjoint split lists. Using them
# rather than a fresh random split keeps the numbers comparable with published
# keyword spotting baselines on the same corpus.
VALIDATION_LIST = "validation_list.txt"
TESTING_LIST = "testing_list.txt"

BACKGROUND_DIR = "_background_noise_"

INT16_SCALE = 32768.0


def read_wav(path: str | Path) -> np.ndarray:
    """Read a 16-bit mono WAV as float32 in [-1, 1]."""
    with wave.open(str(path)) as handle:
        frames = handle.readframes(handle.getnframes())

    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / INT16_SCALE


class WakeWordDataset(Dataset):
    """
    Speech Commands, relabelled as wake word against everything else.

    Negatives are subsampled rather than taken whole. The corpus holds roughly
    a hundred thousand non-target clips against two thousand of the target, and
    training on that ratio produces a detector that never fires.
    """

    def __init__(
        self,
        root: str | Path,
        wake_word: str = "marvin",
        split: str = "train",
        negative_ratio: float = 3.0,
        duration_s: float = 1.0,
        sample_rate: int = 16000,
        augment=None,
        seed: int = 231,
    ) -> None:
        """
        Initialize the wake word dataset.

        Args:
            root: Speech Commands directory
            wake_word: Target word (default: 'marvin')
            split: 'train', 'dev' or 'test' (default: 'train')
            negative_ratio: Negatives sampled per positive (default: 3.0)
            duration_s: Fixed window length in seconds (default: 1.0)
            sample_rate: Audio sample rate (default: 16000)
            augment: Callable applied to the waveform, or None (default: None)
            seed: Seed for negative subsampling (default: 231)
        """
        root = Path(root)
        validation = set(open(root / VALIDATION_LIST).read().split())
        testing = set(open(root / TESTING_LIST).read().split())

        def split_of(key: str) -> str:
            return "val" if key in validation else "test" if key in testing else "train"

        positives, negatives = [], []
        for word_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            word = word_dir.name
            if word == BACKGROUND_DIR:
                continue

            for path in sorted(word_dir.glob("*.wav")):
                if split_of(f"{word}/{path.name}") != split:
                    continue
                (positives if word == wake_word else negatives).append(path)

        rng = random.Random(seed)
        rng.shuffle(negatives)
        negatives = negatives[:int(len(positives) * negative_ratio)]

        self.items = [(p, 1) for p in positives] + [(p, 0) for p in negatives]
        rng.shuffle(self.items)

        self.num_samples = int(duration_s * sample_rate)
        self.wake_word = wake_word
        self.split = split
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Args:
            index: Item index

        Returns:
            Dictionary with the waveform and a binary label, 1 for the wake word
        """
        path, label = self.items[index]

        waveform = read_wav(path)
        if len(waveform) < self.num_samples:
            waveform = np.pad(waveform, (0, self.num_samples - len(waveform)))
        waveform = waveform[:self.num_samples]

        if self.augment is not None:
            waveform = self.augment(waveform)

        return {
            "waveform": torch.from_numpy(np.ascontiguousarray(waveform)),
            "label": torch.tensor(label, dtype=torch.long),
        }
