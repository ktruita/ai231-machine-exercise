import json
import wave
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from vcm.spec import load_command_spec, build_label_space, active_slots, digit_slots


# Manifests written by add_class_dataset.py and build_hf_only.py
MANIFEST_TEMPLATE = "manifest_{split}.jsonl"

INT16_SCALE = 32768.0


def read_wav(path: str | Path) -> np.ndarray:
    """
    Read a 16-bit mono WAV file as float32 in [-1, 1].

    Args:
        path: Path to the WAV file

    Returns:
        Waveform array of shape (num_samples,)
    """
    with wave.open(str(path)) as handle:
        frames = handle.readframes(handle.getnframes())

    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / INT16_SCALE


def fit_length(waveform: np.ndarray, num_samples: int, random_offset: bool = False) -> np.ndarray:
    """
    Pad or crop a waveform to a fixed length.

    A fixed length is what lets the batch be one dense tensor and the exported
    graph have a static input shape.

    Args:
        waveform: Waveform of shape (num_samples,)
        num_samples: Target length in samples
        random_offset: Place a short clip at a random offset instead of the start,
            which teaches the model not to expect speech at a fixed position (default: False)

    Returns:
        Waveform of shape (num_samples,)
    """
    length = len(waveform)

    if length >= num_samples:
        return waveform[:num_samples]

    pad = num_samples - length
    left = np.random.randint(0, pad + 1) if random_offset else 0

    return np.pad(waveform, (left, pad - left))


class VoiceCommandDataset(Dataset):
    """
    Voice command dataset backed by the JSONL manifests.

    Returns raw waveform rather than features, so augmentation stays in the
    time domain and the log-mel front-end can run batched on the GPU.

    Each item also carries a per-slot active mask taken from the command spec,
    which is what lets the training loss ignore heads the intent does not use.
    """

    def __init__(
        self,
        data_dir: str | Path,
        spec_path: str | Path,
        split: str = "train",
        random_offset: bool = False,
        augment=None,
        exclude_negative_models: tuple[str, ...] = (),
        repeat_sources: dict | None = None,
    ) -> None:
        """
        Initialize the voice command dataset.

        Args:
            data_dir: Directory holding the split folders and manifests
            spec_path: Path to commands.yaml
            split: Which manifest to read, e.g. 'c_train', 'c_val' or 'c_test' (default: 'train')
            random_offset: Randomise where a short clip sits in the fixed window (default: False)
            augment: Callable applied to the waveform, or None for clean audio. Applied
                to every class alike - augmenting only commands would give the model a
                channel cue that separates them from the reject class (default: None)
            exclude_negative_models: Sources whose `none` rows are dropped. Negatives
                that far outnumber a source's commands teach the model to reject
                anything that sounds like that source; with a wake word gating the
                input, the command model never hears ambient speech (default: ())
            repeat_sources: How many times to count rows from each source, e.g.
                {"real_voice": 3}, so a scarce kind of recording is not drowned
                out by a plentiful one (default: None)
        """
        self.data_dir = Path(data_dir)
        self.spec = load_command_spec(spec_path)

        manifest_path = self.data_dir / MANIFEST_TEMPLATE.format(split=split)
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest for split '{split}' at {manifest_path}")

        self.records = [json.loads(line) for line in open(manifest_path)]

        if exclude_negative_models:
            self.records = [
                record for record in self.records
                if not (record["intent"] == "none" and record["model"] in exclude_negative_models)
            ]

        if repeat_sources:
            # A key is either a source, "real_voice", or a source and intent,
            # "real_voice:TIMER", so a source's commands can be repeated
            # without repeating its negatives with them.
            def repeats(record: dict) -> int:
                pair = f"{record['model']}:{record['intent']}"
                return repeat_sources.get(pair, repeat_sources.get(record["model"], 1))

            self.records += [
                record for record in self.records for _ in range(repeats(record) - 1)
            ]

        # Label space and the index each class sits at, shared with the model
        self.intent_names, self.slot_classes = build_label_space(self.spec)
        self.slot_names = list(self.slot_classes.keys())
        self.intent_index = {name: i for i, name in enumerate(self.intent_names)}
        self.slot_index = {
            name: {value: i for i, value in enumerate(values)}
            for name, values in self.slot_classes.items()
        }

        # Which heads each intent uses, resolved once rather than per item
        self.intent_slots = {name: active_slots(self.spec, name) for name in self.intent_names}

        # A digit-decomposed slot is stored once in the manifest but read by two
        # heads, so the head name has to be mapped back to the value it splits
        self.digit_source = {}
        for base, (tens, ones) in digit_slots(self.spec).items():
            self.digit_source[tens] = (base, 10)
            self.digit_source[ones] = (base, 1)

        self.num_samples = int(self.spec["audio"]["max_duration_s"] * self.spec["audio"]["sample_rate"])
        self.split = split
        self.random_offset = random_offset
        self.augment = augment
        self.exclude_negative_models = tuple(exclude_negative_models)
        self.repeat_sources = dict(repeat_sources or {})

    def __len__(self) -> int:
        return len(self.records)

    def expand_supervised(self, record: dict) -> set[str]:
        """
        Slots a record supervises beyond the ones its intent uses.

        A `none` utterance can still say a slot's value, and grading that slot
        head on it teaches the value without teaching a command.

        Named at spec level in the manifest and expanded to head names here, so
        the manifest never has to know a slot is split across two heads.

        Args:
            record: Manifest record, optionally carrying a `supervise` list

        Returns:
            Head names to grade for this record
        """
        names = set()
        for slot in record.get("supervise", ()):
            parts = [head for head, (base, _) in self.digit_source.items() if base == slot]
            names.update(parts or [slot])

        return names

    def expand_unsupervised(self, record: dict) -> set[str]:
        """
        Slots a record's intent uses but whose value it cannot give.

        A recording can carry a command whose value lies outside a closed slot -
        an alarm at seven, against a schema that lists six, eight and nine. Its
        label is still right, so it trains the intent head; the slot head is
        left ungraded rather than taught a wrong value. add_class_dataset.py
        marks these rows; named at spec level, like `supervise`.

        Args:
            record: Manifest record, optionally carrying an `unsupervise` list

        Returns:
            Head names to leave ungraded for this record
        """
        names = set()
        for slot in record.get("unsupervise", ()):
            parts = [head for head, (base, _) in self.digit_source.items() if base == slot]
            names.update(parts or [slot])

        return names

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Args:
            index: Record index

        Returns:
            Dictionary with the waveform, the intent target, one target per slot
            and a boolean mask marking which slots the intent uses
        """
        record = self.records[index]

        waveform = read_wav(self.data_dir / record["file"])
        waveform = fit_length(waveform, self.num_samples, self.random_offset)

        if self.augment is not None:
            waveform = self.augment(waveform)

        intent_name = record["intent"]
        used = (set(self.intent_slots[intent_name]) | self.expand_supervised(record)) \
            - self.expand_unsupervised(record)

        # Slots the intent does not use are labelled N/A and masked out, so the
        # naive and masked loss runs read the same batch
        targets = {}
        mask = {}
        for name in self.slot_names:
            if name in self.digit_source:
                base, divisor = self.digit_source[name]
                raw = record["slots"].get(base)
                value = "N/A" if raw is None else (raw // divisor) % 10 if divisor == 1 else raw // 10
            else:
                value = record["slots"].get(name)
                value = "N/A" if value is None else value
            targets[name] = torch.tensor(self.slot_index[name][value], dtype=torch.long)
            mask[name] = torch.tensor(name in used, dtype=torch.bool)

        # A row marked supervise_intent: false trains only the slot heads it
        # supervises, and is left out of the intent loss
        item = {
            "waveform": torch.from_numpy(waveform),
            "intent": torch.tensor(self.intent_index[intent_name], dtype=torch.long),
            "mask_intent": torch.tensor(record.get("supervise_intent", True), dtype=torch.bool),
        }
        item.update({f"target_{name}": target for name, target in targets.items()})
        item.update({f"mask_{name}": flag for name, flag in mask.items()})

        return item
