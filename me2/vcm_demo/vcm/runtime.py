"""Deployable inference, without torch.

Everything here runs from an exported bundle - model.onnx, filterbank.npy and
meta.json - so the device needs only numpy and onnxruntime. The label space and
the intent-to-slot mapping travel in the metadata, so commands.yaml is not read
at runtime and cannot drift from the model that was trained against it.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features_numpy import LogMelSpectrogram

# What must match for bundles to be averaged: the same label space read from
# the same front-end. Anything else would decode one model's classes against
# another's indices without any error
ENSEMBLE_KEYS = ("intent_names", "slot_classes", "head_names", "digit_slots",
                 "n_fft", "hop_length", "num_mels", "sample_rate", "window_s")

# How often people say each timer number, and how much that breaks ties.
# Written by evaluate_numbers.py --write-prior; without it, numbers decode as
# the two digit heads choose separately
PRIOR_FILE = Path(__file__).resolve().parent / "number_prior.json"
NUMBERS = np.arange(1, 61)


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax over the last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))

    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Log-softmax over the last axis."""
    shifted = x - x.max(axis=-1, keepdims=True)

    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class CommandRecogniser:
    """
    Runs an exported command model on a waveform.

    Flow: (num_samples,) -> {"intent": str, <slot>: value}
    """

    def __init__(
        self,
        bundle: str | Path | list,
        threads: int = 1,
        model_file: str = "model.onnx",
        none_bias: float = 0.0,
        number_prior: str | Path | None = PRIOR_FILE,
    ) -> None:
        """
        Initialize the recogniser.

        Args:
            bundle: Directory holding model.onnx, filterbank.npy and meta.json,
                or a list of them to run as an ensemble. Members' probabilities
                are averaged; they must share a label space and front-end
            threads: onnxruntime intra-op threads. One is the measured optimum
                for a model this size - more threads cost synchronisation and
                win nothing, and on a Pi they take cores away from audio
                capture (default: 1)
            model_file: Graph to load from the bundle, model_int8.onnx for the
                quantised one (default: 'model.onnx')
            none_bias: Added to the `none` intent before decoding. Negative makes
                "didn't catch that" rarer. Measured on the two-seed
                vcm_mined_fsc ensemble at threshold 0.6, -0.5 raised correct
                commands from 68.7% to 72.2% for 0.5 points more wrong ones,
                and left 93.3% of out-of-scope requests alone against 95.3%
                (default: 0.0)
            number_prior: JSON of real timer-number counts and a weight. The
                timer number is then read from both digit heads together plus
                weight x log-prior, so a near tie goes to the number people
                say more often. None, or no such file, reads each digit head
                separately (default: vcm/number_prior.json)
        """
        bundles = [Path(b) for b in (bundle if isinstance(bundle, (list, tuple)) else [bundle])]
        self.meta = json.loads((bundles[0] / "meta.json").read_text())
        filterbank = np.load(bundles[0] / "filterbank.npy")

        # Validation
        for other in bundles[1:]:
            meta = json.loads((other / "meta.json").read_text())
            for key in ENSEMBLE_KEYS:
                if meta.get(key) != self.meta.get(key):
                    raise ValueError(f"{other} differs from {bundles[0]} in '{key}'; "
                                     f"an ensemble needs one label space and front-end")
            if not np.array_equal(np.load(other / "filterbank.npy"), filterbank):
                raise ValueError(f"{other} has a different mel filterbank from {bundles[0]}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        self.sessions = [
            ort.InferenceSession(str(b / model_file), options, providers=["CPUExecutionProvider"])
            for b in bundles
        ]
        self.session = self.sessions[0]
        self.bundles = bundles
        self.none_bias = none_bias

        self.number_prior = None
        if number_prior is not None and Path(number_prior).exists() and "number" in self.meta["digit_slots"]:
            spec = json.loads(Path(number_prior).read_text())
            counts = np.array([spec["counts"].get(str(n), 0) for n in NUMBERS], dtype=np.float64) + spec["smoothing"]
            tens, ones = self.meta["digit_slots"]["number"]
            self.number_prior = spec["weight"] * np.log(counts / counts.sum())                   # (60,)
            self.number_columns = ([self.meta["slot_classes"][tens].index(int(n) // 10) for n in NUMBERS],
                                   [self.meta["slot_classes"][ones].index(int(n) % 10) for n in NUMBERS])

        self.features = LogMelSpectrogram(
            filterbank,
            n_fft=self.meta["n_fft"],
            hop_length=self.meta["hop_length"],
        )
        self.num_samples = int(self.meta["window_s"] * self.meta["sample_rate"])

    def fit_length(self, waveform: np.ndarray) -> np.ndarray:
        """Pad or crop to the window the model was exported for."""
        if len(waveform) >= self.num_samples:
            return waveform[:self.num_samples]

        return np.pad(waveform, (0, self.num_samples - len(waveform)))

    def run_sessions(self, mel: np.ndarray) -> list[dict[str, np.ndarray]]:
        """Run every member on one mel spectrogram; the front-end is computed once."""
        return [dict(zip(self.meta["head_names"], session.run(None, {"mel": mel})))
                for session in self.sessions]

    def combine(self, runs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """
        Merge members into one set of scores per head.

        A single member passes through untouched. Several are averaged as
        probabilities rather than logits: each member's softmax is what its
        confidence means, and the mean is still a distribution.
        """
        if len(runs) == 1:
            return runs[0]

        return {head: np.log(np.mean([softmax(r[head]) for r in runs], axis=0) + 1e-12)
                for head in runs[0]}

    def logits(self, waveform: np.ndarray) -> dict[str, np.ndarray]:
        """
        Args:
            waveform: Audio of shape (num_samples,), float in [-1, 1]

        Returns:
            Scores keyed by head name - logits for one bundle, averaged
            log-probabilities for an ensemble
        """
        mel = self.features(self.fit_length(waveform))

        return self.combine(self.run_sessions(mel))

    def decode(self, logits: dict[str, np.ndarray]) -> dict:
        """
        Turn logits into a command.

        Mirrors VoiceCommandModel.decode: the intent selects which slot heads
        are read, and digit-decomposed slots are recombined so the caller sees
        the single value the spec describes.

        Args:
            logits: Logits keyed by head name

        Returns:
            The decoded command, plus the intent confidence
        """
        intent_scores = logits["intent"]
        if self.none_bias:
            intent_scores = intent_scores.copy()
            intent_scores[..., self.meta["intent_names"].index("none")] += self.none_bias

        intent_id = int(np.argmax(intent_scores))
        intent = self.meta["intent_names"][intent_id]

        scores = np.exp(intent_scores[0] - intent_scores[0].max())
        confidence = float(scores[intent_id] / scores.sum())

        command = {"intent": intent}
        for slot in self.meta["intent_slots"][intent]:
            value = self.meta["slot_classes"][slot][int(np.argmax(logits[slot]))]
            command[slot] = None if value == "N/A" else value

        for base, (tens, ones) in self.meta["digit_slots"].items():
            if tens not in command:
                continue
            high, low = command.pop(tens), command.pop(ones)
            command[base] = None if high is None or low is None else high * 10 + low
            if base == "number" and self.number_prior is not None:
                command[base] = self.read_number(logits)

        return {"command": command, "confidence": confidence}

    def read_number(self, logits: dict[str, np.ndarray]) -> int:
        """
        Read the timer number from both digit heads together, plus the prior.

        Read separately, the heads put a digit in the wrong place or in both -
        "fifty" as 55, "one" as 11. Scoring every valid number instead, with
        how often people say it, sends a near tie to the likelier number.
        """
        tens, ones = self.meta["digit_slots"]["number"]
        scores = log_softmax(logits[tens][0])[self.number_columns[0]] \
            + log_softmax(logits[ones][0])[self.number_columns[1]]                              # (60,)

        return int(NUMBERS[np.argmax(scores + self.number_prior)])

    def __call__(self, waveform: np.ndarray) -> dict:
        """Recognise a command in one call."""
        return self.decode(self.logits(waveform))


class WakeWordDetector:
    """
    Scores a one second window for the wake word.

    Holds no state: the n-of-m smoothing and refractory logic live in the demo
    loop, so this stays usable for offline sweeps as well.

    Flow: (num_samples,) -> probability
    """

    def __init__(
        self,
        bundle: str | Path,
        threads: int = 1,
    ) -> None:
        """
        Initialize the detector.

        Args:
            bundle: Directory holding model.onnx, filterbank.npy and meta.json
            threads: onnxruntime intra-op threads (default: 1)
        """
        bundle = Path(bundle)
        self.meta = json.loads((bundle / "meta.json").read_text())

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(bundle / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

        self.features = LogMelSpectrogram(
            np.load(bundle / "filterbank.npy"),
            n_fft=self.meta["n_fft"],
            hop_length=self.meta["hop_length"],
        )
        self.num_samples = int(self.meta["window_s"] * self.meta["sample_rate"])

    def __call__(self, waveform: np.ndarray) -> float:
        """
        Args:
            waveform: One second of audio, float in [-1, 1]

        Returns:
            Probability that the window contains the wake word
        """
        if len(waveform) < self.num_samples:
            waveform = np.pad(waveform, (0, self.num_samples - len(waveform)))

        logits = self.session.run(None, {"mel": self.features(waveform[:self.num_samples])})[0]
        scores = np.exp(logits[0] - logits[0].max())

        return float(scores[1] / scores.sum())
