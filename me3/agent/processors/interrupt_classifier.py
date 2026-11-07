"""Backchannel-aware barge-in: "mhm" / "yeah" while the bot talks must NOT interrupt it;
anything else (a question, "wait", "stop", a correction) is steering and interrupts.

Decision rules while the bot is speaking:
  0. Faint speech (quieter than `faint_below_dbfs`: a TV, someone across the room talking to
     somebody else, background chatter) -> ignore: the bot keeps talking and the words are
     dropped from the conversation. Loudness is measured live (LevelMeter), so this needs
     no ASR and costs no delay.
  1. User keeps talking longer than `steer_after_s` -> steering (interrupt now, no need
     to wait for ASR). Backchannels are short, so long speech is never one.
     Borderline loudness (within `uncertain_db` of the faint level) -> keep talking until
     the transcript arrives; the STT's whole-utterance level decides (faint -> ignore).
  2. Short burst -> wait for its transcript: backchannel lexicon -> ignore; else interrupt.
When the bot is silent, any speech starts a user turn as usual.

No model inference: timing + loudness + lexicon.
"""

from __future__ import annotations

import array
import asyncio
import math
import re
import time
from collections import deque
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

BACKCHANNELS = {
    "mhm", "mm", "mmm", "hmm", "uh huh", "uh-huh", "yeah", "yes", "yep", "yup", "ok", "okay",
    "right", "sure", "i see", "got it", "cool", "nice", "wow", "oh", "ah", "alright",
    "all right", "true", "exactly", "totally", "go on", "really", "oo", "sige", "ah okay",
}
STEERING_WORDS = {"wait", "stop", "no", "but", "actually", "hold", "hey", "what", "why", "how"}


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z']+(?:-[a-z]+)?", text.lower()))


def is_backchannel(text: str, max_words: int = 3) -> bool:
    """True for short acknowledgements ("mhm", "yeah okay", "uh-huh right")."""
    norm = normalize(text)
    if not norm:
        return True  # noise / unintelligible blip
    words = norm.replace("-", " ").split()
    if len(words) > max_words or any(w in STEERING_WORDS for w in words):
        return False
    if norm in BACKCHANNELS or norm.replace("-", " ") in BACKCHANNELS:
        return True
    # every word (or adjacent pair) is itself a backchannel: "yeah okay", "mhm right"
    i = 0
    while i < len(words):
        if i + 1 < len(words) and f"{words[i]} {words[i + 1]}" in BACKCHANNELS:
            i += 2
        elif words[i] in BACKCHANNELS:
            i += 1
        else:
            return False
    return True


def chunk_dbfs(pcm: bytes) -> float:
    a = array.array("h", pcm)
    if not a:
        return -120.0
    return 20 * math.log10(max(math.sqrt(sum(x * x for x in a) / len(a)), 1.0) / 32768)


class LevelMeter(FrameProcessor):
    """Place right after the VAD. Measures how loud the current utterance is, live, so the
    turn-start strategy can tell a nearby user from a TV or someone across the room within
    a fraction of a second (no ASR round trip)."""

    def __init__(self, *, report_every_s: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._recent: deque[float] = deque(maxlen=15)  # ~0.3-0.5 s before VAD fired
        self._speech: list[float] | None = None
        # optional live readout (web UI mic meter): on_level(dbfs, vad_speaking), ~10x per second
        self.on_level = None
        self._report_every = report_every_s
        self._last_report = 0.0
        self._window_max = -120.0
        self._vad_speaking = False

    def level_dbfs(self) -> float | None:
        """Loudness of the loudest half of the current (or last) utterance."""
        levels = sorted(self._speech if self._speech else self._recent)
        if not levels:
            return None
        top = levels[len(levels) // 2:]
        return sum(top) / len(top)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            level = chunk_dbfs(frame.audio)
            self._recent.append(level)
            if self._speech is not None:
                self._speech.append(level)
            self._report(level)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speech = list(self._recent)  # VAD fires after ~0.2 s of speech
            self._vad_speaking = True
            self._report(None, force=True)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_speaking = False
            self._report(None, force=True)
        await self.push_frame(frame, direction)

    def _report(self, level: float | None, force: bool = False) -> None:
        if self.on_level is None:
            return
        if level is not None:
            self._window_max = max(self._window_max, level)
        now = time.time()
        if force or now - self._last_report >= self._report_every:
            try:
                self.on_level(self._window_max, self._vad_speaking)
            except Exception:  # noqa: BLE001  (UI problems must never touch the audio path)
                pass
            self._last_report, self._window_max = now, -120.0


@dataclass
class InterruptDecision:
    ts: float
    kind: str  # "backchannel" | "faint_ignored" | "steering_long" | "steering_transcript"
    text: str
    speech_s: float


class BackchannelAwareTurnStartStrategy(BaseUserTurnStartStrategy):
    def __init__(self, *, steer_after_s: float = 0.8, level_meter: LevelMeter | None = None,
                 faint_below_dbfs: float | None = None, uncertain_db: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        self.uncertain_db = uncertain_db
        self.steer_after_s = steer_after_s
        self.level_meter = level_meter
        self.faint_below_dbfs = faint_below_dbfs
        self._faint_burst = False
        self._uncertain = False
        self.decisions: list[InterruptDecision] = []
        self.on_decision = None  # optional callback(InterruptDecision), e.g. the web UI
        self._bot_speaking = False
        self._burst_started: float | None = None   # VAD start of a burst during bot speech
        self._awaiting_transcript = False
        self._timer: asyncio.Task | None = None

    async def handle_user_turn_started(self):
        self._reset_burst()

    def _reset_burst(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None
        self._burst_started = None
        self._awaiting_transcript = False
        self._faint_burst = False
        self._uncertain = False

    def _is_faint(self, margin: float = 0.0) -> bool:
        if self.level_meter is None or self.faint_below_dbfs is None:
            return False
        level = self.level_meter.level_dbfs()
        return level is not None and level < self.faint_below_dbfs + margin

    def _decide(self, kind: str, text: str = "") -> None:
        speech = time.time() - self._burst_started if self._burst_started else 0.0
        d = InterruptDecision(time.time(), kind, text, round(speech, 3))
        self.decisions.append(d)
        if self.on_decision:
            self.on_decision(d)
        logger.info(f"Barge-in decision: {kind} text={text!r} speech={d.speech_s}s")

    async def _steer_timer(self):
        await asyncio.sleep(self.steer_after_s)
        self._timer = None
        if self._is_faint():
            # not the user: keep talking; its transcript is dropped when it arrives
            self._faint_burst = True
            self._decide("faint_ignored", f"{self.level_meter.level_dbfs():.1f} dBFS")
            return
        if self._is_faint(self.uncertain_db):
            # borderline: keep talking, let the transcript (and its full-utterance level) decide
            self._uncertain = True
            self._decide("uncertain_wait", f"{self.level_meter.level_dbfs():.1f} dBFS")
            return
        self._decide("steering_long")
        self._reset_burst()
        await self.report_user_turn_started()

    async def process_frame(self, frame: Frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._bot_speaking:
                await self.report_user_turn_started()
                return
            self._burst_started = time.time()
            self._awaiting_transcript = False
            self._timer = asyncio.create_task(self._steer_timer())
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._timer:  # burst ended before it counted as steering -> judge by words
                self._timer.cancel()
                self._timer = None
                self._awaiting_transcript = True
                self._faint_burst = self._is_faint()
            elif self._faint_burst or self._uncertain:
                self._awaiting_transcript = True
        elif isinstance(frame, TranscriptionFrame):
            if self._awaiting_transcript and self._uncertain and not self._faint_burst:
                if frame.text.startswith("(faint"):  # the STT measured the whole utterance
                    self._decide("faint_ignored", frame.text)
                    self._reset_burst()
                    await self.trigger_reset_aggregation()
                else:
                    self._decide("steering_transcript", frame.text)
                    self._reset_burst()
                    await self.report_user_turn_started()
            elif self._awaiting_transcript and self._faint_burst:
                self._decide("faint_ignored", frame.text)
                self._reset_burst()
                await self.trigger_reset_aggregation()  # never answer the TV / side talk
            elif self._awaiting_transcript:
                if is_backchannel(frame.text):
                    self._decide("backchannel", frame.text)
                    self._reset_burst()
                    await self.trigger_reset_aggregation()  # drop "mhm" from the context
                else:
                    self._decide("steering_transcript", frame.text)
                    self._reset_burst()
                    await self.report_user_turn_started()
            elif not self._bot_speaking and self._burst_started is None:
                # VAD missed the start (e.g. very soft speech): transcript still opens a turn
                await self.report_user_turn_started()
