"""Real-time voice command demo.

Two modes from one code path. `--wav` replays files through exactly the
pipeline the live loop uses, which is how a decode bug gets separated from a
microphone bug; without it the microphone opens and runs continuously.

Deliberately torch-free - it imports only numpy, onnxruntime and sounddevice,
so the same file runs on a laptop and on a Pi with nothing else installed.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from vcm.runtime import PRIOR_FILE, CommandRecogniser, WakeWordDetector

REPO = Path(__file__).resolve().parent

# Wake word firing rule, from the sweep in evaluate_wakeword.py: 3 of the last
# 5 windows over 0.99 gave 0.5 false accepts/hour against 20 for a bare
# per-window threshold, for 6 points of false rejects.
WAKE_THRESHOLD = 0.99
WAKE_N = 3
WAKE_M = 5

# Block RMS that counts as speech, tuned in a quiet room; --speech-level raises
# it for a noisy one. Setting it from the room's background at start-up was
# tried and dropped: in simulate_live.py it gained nothing at twice the
# background, and at three times it cost far-mic speech in a noisy room 9
# points, because the bar rose above quiet voices
SPEECH_LEVEL = 0.015

LEVEL_BLOCKS = " ▁▂▃▄▅▆▇█"


def level_meter(rms: float, width: int = 12) -> str:
    """Render a block's level as a bar, so a silent microphone is obvious."""
    filled = min(int(rms * 40 * width), width)

    return "█" * filled + "·" * (width - filled)


def format_command(result: dict) -> str:
    """Render a decoded command as one readable line."""
    command = result["command"]
    slots = "  ".join(f"{k}={v}" for k, v in command.items() if k != "intent" and v is not None)

    return f"{command['intent']:14s} {slots:34s} p={result['confidence']:.3f}"


def run_files(recogniser: CommandRecogniser, paths: list[str], threshold: float = 0.0) -> None:
    """
    Replay WAV files through the live pipeline.

    Verifies the deployed path offline. If this disagrees with predict.py, the
    bundle or the numpy front-end is wrong; if it agrees and the microphone
    still fails, the problem is audio, not inference.

    Args:
        recogniser: Loaded command model
        paths: WAV files to decode
        threshold: Confidence below which a decode is marked as rejected, as
            the live loop would answer "didn't catch that" (default: 0.0)
    """
    import wave

    for path in paths:
        with wave.open(path) as handle:
            audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        waveform = audio.astype(np.float32) / 32768.0

        start = time.perf_counter()
        result = recogniser(waveform)
        elapsed = (time.perf_counter() - start) * 1000

        mark = "  (below threshold)" if result["confidence"] < threshold else ""
        print(f"  {Path(path).stem[:36]:38s} -> {format_command(result)}  [{elapsed:.1f} ms]{mark}")


class SpeechEndpointer:
    """
    Marks where an utterance starts and stops, from frame energy.

    Without this the loop has to guess when someone has finished speaking.
    Decoding on a rolling window instead would fire continuously and print a
    different answer every hop as the window fills.

    Flow: per-block RMS -> None while speaking, then the captured utterance
    """

    def __init__(
        self,
        threshold: float = SPEECH_LEVEL,
        start_blocks: int = 2,
        end_blocks: int = 8,
        max_blocks: int = 60,
    ) -> None:
        """
        Initialize the endpointer.

        Args:
            threshold: RMS above which a block counts as speech (default: 0.015)
            start_blocks: Consecutive loud blocks needed to begin (default: 2)
            end_blocks: Consecutive quiet blocks that end it, 8 at 100 ms is
                0.8 s of silence - long enough to survive the pause between
                "turn off the" and "kitchen lights" (default: 8)
            max_blocks: Hard stop, so a noisy room cannot capture forever (default: 60)
        """
        self.threshold = threshold
        self.start_blocks = start_blocks
        self.end_blocks = end_blocks
        self.max_blocks = max_blocks
        self.reset()

    def reset(self) -> None:
        """Return to waiting for speech."""
        self.speaking = False
        self.loud = 0
        self.quiet = 0
        self.blocks = 0

    def update(self, rms: float) -> bool:
        """
        Args:
            rms: Root-mean-square level of the latest block

        Returns:
            True when an utterance has just ended
        """
        if not self.speaking:
            self.loud = self.loud + 1 if rms >= self.threshold else 0
            if self.loud >= self.start_blocks:
                self.speaking = True
                self.blocks = self.loud
            return False

        self.blocks += 1
        self.quiet = self.quiet + 1 if rms < self.threshold else 0

        if self.quiet >= self.end_blocks or self.blocks >= self.max_blocks:
            self.reset()
            return True

        return False


class CommandListener:
    """
    The live loop's decisions, one block of audio at a time.

    Holds the recent audio, the wake word's n-of-m smoothing, the endpointer
    and the timers around them, and reports what happened as events. The
    microphone loop and an offline replay (simulate_live.py) drive the same
    object, so what is tested is what runs.

    Two rules guard the command after the wake word:
        tail   Sound already under way when the wake word fires is the word
               itself, not the command. On 485 held-out "marvin" clips at a
               close-mic level, 1 wake in 4 fired with 0.2 s or more of the
               word still sounding - enough to start the endpointer, so a
               pause after "Marvin" ended a command that had not begun. Up to
               `skip_s` of it is ignored; if the sound runs on, it is the
               command, and the capture still starts at the wake word.
        wait   The command may start up to `wait_s` after the wake word; then
               the listener goes back to waiting for the wake word rather than
               taking whatever is said next as a command.

    Replayed through simulate_live.py - "marvin", a pause, a real command -
    the two rules cut decodes that came back before the command started from
    7.9% to 1.4%, and raised commands decoded right from 69% to 74%: at a close
    mic from 74% to 86% in a quiet room and 70% to 78% in a noisy one, and at
    a far mic 65% against 66% and 63% against 62%, within a stream of before.
    Three other changes were tried there and dropped: a
    minimum length for every utterance, which would drop a quarter of real
    "stop"s; capturing from just before the command's onset, which clipped
    quiet openings; and setting the speech level from the room, which lifted
    it above quiet voices.

    Flow: (hop,) block -> list of events
    """

    def __init__(
        self,
        recogniser: CommandRecogniser,
        detector: WakeWordDetector | None,
        hop_ms: int = 100,
        threshold: float = 0.0,
        speech_level: float = SPEECH_LEVEL,
        skip_s: float = 0.3,
        wait_s: float | None = 5.0,
    ) -> None:
        """
        Initialize the listener.

        Args:
            recogniser: Loaded command model
            detector: Loaded wake word model, or None to decode every utterance
            hop_ms: Block length in milliseconds (default: 100)
            threshold: Confidence below which a decode is reported as rejected.
                Showing "didn't catch that" is a better failure than confidently
                doing the wrong thing (default: 0.0, report everything)
            speech_level: Block RMS that counts as speech (default: 0.015)
            skip_s: Sound under way when the wake word fires that is taken as
                the word's own tail, 0 to count all of it (default: 0.3)
            wait_s: How long to wait for a command after the wake word, None to
                wait indefinitely (default: 5.0)
        """
        self.rate = recogniser.meta["sample_rate"]
        self.hop = int(self.rate * hop_ms / 1000)
        self.buffer = np.zeros(recogniser.num_samples + self.rate, dtype=np.float32)
        self.endpointer = SpeechEndpointer(threshold=speech_level, end_blocks=int(800 / hop_ms))

        self.recent = []
        self.armed = detector is None
        self.cooldown = 0
        self.clock = 0
        self.arm_at = 0
        self.skip = 0
        self.waited = 0

        self.recogniser = recogniser
        self.detector = detector
        self.threshold = threshold
        self.skip_blocks = int(round(skip_s * 1000 / hop_ms))
        self.wait_blocks = int(wait_s * 1000 / hop_ms) if wait_s else None
        self.cooldown_blocks = int(1500 / hop_ms)

    def feed(self, block: np.ndarray) -> list[dict]:
        """
        Take the next block of audio.

        Args:
            block: The newest `hop` samples, float in [-1, 1]

        Returns:
            What happened, in order: "wake", a "level" event for every block,
            then "result" or "timeout"
        """
        self.buffer = np.concatenate([self.buffer[len(block):], block])
        self.clock += len(block)
        rms = float(np.sqrt((block ** 2).mean())) if block.size else 0.0
        self.cooldown = max(self.cooldown - 1, 0)
        events = []

        wake = None
        if self.detector is not None and not self.armed and not self.cooldown:
            wake = self.detector(self.buffer[-self.rate:])
            self.recent = (self.recent + [wake >= WAKE_THRESHOLD])[-WAKE_M:]
            if sum(self.recent) >= WAKE_N:
                self.arm(self.clock - len(block))
                events.append({"type": "wake"})
        events.append(self.level(rms, wake))

        if not self.armed:
            return events

        if self.skip and not self.endpointer.speaking:
            if rms >= self.endpointer.threshold:
                self.skip -= 1
                return events
            self.skip = 0

        if self.endpointer.update(rms):
            events.append(self.decode())
            if self.detector is not None:
                self.armed = False
                self.cooldown = self.cooldown_blocks
        elif not self.endpointer.speaking and self.detector is not None and self.wait_blocks:
            self.waited += 1
            if self.waited >= self.wait_blocks:
                self.armed = False
                events.append({"type": "timeout"})

        return events

    def arm(self, at: int) -> None:
        """The wake word fired: await a command, counting the block at sample `at` as after it."""
        self.recent = []
        self.endpointer.reset()
        self.armed = True
        self.arm_at = at
        self.skip = self.skip_blocks
        self.waited = 0

    def level(self, rms: float, wake: float | None) -> dict:
        """A level event: the block's RMS, the wake word score and whether a command is awaited."""
        return {"type": "level", "rms": rms, "armed": self.armed,
                "wake": None if self.detector is None else (wake or 0.0)}

    def decode(self) -> dict:
        """
        Recognise the utterance that has just ended.

        The capture holds only what arrived after the wake word: the full
        window would also hold "Marvin", and the model's only real-voice
        examples of that word are labelled `none` - on held-out commands it
        cost 6.7 points of intent, recovered in full by this trim even with
        300 ms of the word left in (evaluate_live.py).

        Returns:
            A "result" event
        """
        since = self.recogniser.num_samples if self.detector is None else self.clock - self.arm_at
        take = min(since, self.recogniser.num_samples)

        began = time.perf_counter()
        result = self.recogniser(self.buffer[-take:])
        elapsed = (time.perf_counter() - began) * 1000
        rejected = result["confidence"] < self.threshold

        return {"type": "result", "command": result["command"], "confidence": result["confidence"],
                "latency_ms": elapsed, "rtf": elapsed / 1000 / self.recogniser.meta["window_s"],
                "rejected": rejected,
                "reason": f"confidence {result['confidence']:.0%} below {self.threshold:.0%}" if rejected else ""}


def show(event: dict, emit=None) -> None:
    """Print one listener event to the terminal, and pass it on to the browser console."""
    if emit is not None:
        emit(event)

    kind = event["type"]
    if kind == "level":
        state = "armed  " if event["armed"] else "listen "
        wake = "" if event["wake"] is None else f"  wake {event['wake']:5.3f}"
        print(f"\r[{state}] {level_meter(event['rms'])}  rms {event['rms']:5.3f}{wake}  ", end="", flush=True)
    elif kind == "wake":
        print(f"\n*** wake word ({WAKE_N}-of-{WAKE_M} @ {WAKE_THRESHOLD}) — speak now")
    elif kind == "timeout":
        print("\r  -> no command heard; listening for the wake word again")
    elif kind == "result":
        mark = "  (below threshold)" if event["rejected"] else ""
        print(f"\r  -> {format_command(event)}{mark}")
        print(f"     {event['latency_ms']:.1f} ms   RTF {event['rtf']:.4f}")


def run_live(
    recogniser: CommandRecogniser,
    detector: WakeWordDetector | None,
    device: int | None,
    hop_ms: int = 100,
    emit=None,
    threshold: float = 0.0,
    speech_level: float = SPEECH_LEVEL,
) -> None:
    """
    Listen continuously and decode commands.

    The audio callback only queues each block, and the main loop hands them to
    a CommandListener in order. Inference in the callback would block the
    audio device and drop frames; polling the newest samples instead would
    skip or repeat a block whenever the loop ran late.

    Args:
        recogniser: Loaded command model
        detector: Loaded wake word model, or None to decode every utterance
        device: Input device index, or None for the system default
        hop_ms: Block length in milliseconds (default: 100)
        emit: Optional callback taking one event dict, used by the browser UI
        threshold: Confidence below which a decode is reported as rejected (default: 0.0)
        speech_level: Block RMS that counts as speech (default: 0.015)
    """
    import queue

    import sounddevice as sd

    listener = CommandListener(recogniser, detector, hop_ms=hop_ms, threshold=threshold,
                               speech_level=speech_level)
    blocks = queue.Queue()

    def callback(indata, frames, time_info, status):
        blocks.put(indata[:, 0].copy())

    print(f"listening at {listener.rate} Hz on device {device if device is not None else 'default'}, "
          f"speech level {speech_level:.3f}")
    print("say the wake word first" if detector else "just speak a command")
    print("ctrl-c to stop\n")

    with sd.InputStream(samplerate=listener.rate, channels=1, dtype="float32",
                        blocksize=listener.hop, device=device, callback=callback):
        while True:
            try:
                block = blocks.get(timeout=0.5)     # a timeout keeps ctrl-c working on Windows
            except queue.Empty:
                continue
            for event in listener.feed(block):
                show(event, emit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["deploy/vcm_mined_fsc"],
                    help="one bundle, or several to run as an ensemble")
    ap.add_argument("--none-bias", type=float, default=0.0,
                    help="added to the `none` intent; negative says \"didn't catch that\" less often")
    ap.add_argument("--no-number-prior", action="store_true",
                    help="read timer numbers digit by digit, without vcm/number_prior.json")
    ap.add_argument("--wakeword", default="deploy/wakeword_marvin")
    ap.add_argument("--wav", nargs="*", help="replay files instead of listening")
    ap.add_argument("--no-wakeword", action="store_true",
                    help="recognise continuously, without gating on the wake word")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ui", action="store_true",
                    help="serve the browser console instead of printing only")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="reject decodes below this confidence, e.g. 0.6")
    ap.add_argument("--speech-level", type=float, default=SPEECH_LEVEL,
                    help="block RMS that counts as speech; raise it if the rms readout sits above it "
                         "while nobody is talking")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    recogniser = CommandRecogniser([REPO / m for m in args.model], threads=args.threads,
                                   none_bias=args.none_bias,
                                   number_prior=None if args.no_number_prior else PRIOR_FILE)
    print(f"model    {' + '.join(args.model)}  ({recogniser.meta['window_s']}s window, "
          f"{len(recogniser.meta['head_names'])} heads)")
    print(f"numbers  {'digit by digit' if recogniser.number_prior is None else 'with ' + PRIOR_FILE.name}")

    if args.wav:
        run_files(recogniser, args.wav, threshold=args.threshold)
        return

    detector = None
    if not args.no_wakeword:
        detector = WakeWordDetector(REPO / args.wakeword, threads=args.threads)
        print(f"wakeword {args.wakeword}")

    emit, server = None, None
    if args.ui:
        from vcm.server import EventServer

        server = EventServer()
        url = server.start(args.port)
        emit = server.emit

        size = sum(f.stat().st_size for m in args.model for f in (REPO / m).rglob("*") if f.is_file())
        server.latest = None
        server.emit({"type": "hello", "size": f"{size / 1e6:.1f} MB"})
        print(f"console  {url}   (also reachable on this machine's LAN address)")

    try:
        run_live(recogniser, detector, args.device, emit=emit, threshold=args.threshold,
                 speech_level=args.speech_level)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
