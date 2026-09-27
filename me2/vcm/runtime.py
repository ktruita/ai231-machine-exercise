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


class CommandRecogniser:
    """
    Runs an exported command model on a waveform.

    Flow: (num_samples,) -> {"intent": str, <slot>: value}
    """

    def __init__(
        self,
        bundle: str | Path,
        threads: int = 1,
    ) -> None:
        """
        Initialize the recogniser.

        Args:
            bundle: Directory holding model.onnx, filterbank.npy and meta.json
            threads: onnxruntime intra-op threads. One is the measured optimum
                for a model this size - more threads cost synchronisation and
                win nothing, and on a Pi they take cores away from audio
                capture (default: 1)
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

    def fit_length(self, waveform: np.ndarray) -> np.ndarray:
        """Pad or crop to the window the model was exported for."""
        if len(waveform) >= self.num_samples:
            return waveform[:self.num_samples]

        return np.pad(waveform, (0, self.num_samples - len(waveform)))

    def logits(self, waveform: np.ndarray) -> dict[str, np.ndarray]:
        """
        Args:
            waveform: Audio of shape (num_samples,), float in [-1, 1]

        Returns:
            Logits keyed by head name
        """
        mel = self.features(self.fit_length(waveform))

        return dict(zip(self.meta["head_names"], self.session.run(None, {"mel": mel})))

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
        intent_id = int(np.argmax(logits["intent"]))
        intent = self.meta["intent_names"][intent_id]

        scores = np.exp(logits["intent"][0] - logits["intent"][0].max())
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

        return {"command": command, "confidence": confidence}

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
