import pytest

from agent.processors.interrupt_classifier import is_backchannel


@pytest.mark.parametrize("text", [
    "Mhm.", "yeah", "Uh-huh", "okay", "Yeah, okay.", "mhm right", "I see", "got it", "", "Oh wow",
])
def test_backchannels(text):
    assert is_backchannel(text)


@pytest.mark.parametrize("text", [
    "Wait", "stop", "No, tomorrow", "What about tomorrow?", "yeah but what about Friday",
    "Turn off the lights", "okay play music", "Actually never mind", "yeah tell me more about it",
])
def test_steering(text):
    assert not is_backchannel(text)


# --- live loudness: faint overlap (TV, side talk) must not interrupt -------------------
import array
import asyncio
import math

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from agent.processors.interrupt_classifier import (
    BackchannelAwareTurnStartStrategy,
    LevelMeter,
    chunk_dbfs,
)


def tone(dbfs: float, ms: int = 20) -> bytes:
    amp = 32768 * 10 ** (dbfs / 20) * math.sqrt(2)
    return array.array("h", [int(amp * math.sin(i / 3)) for i in range(16 * ms)]).tobytes()


def test_chunk_dbfs_matches_level():
    assert abs(chunk_dbfs(tone(-23)) - -23) < 0.5


class _Probe(BackchannelAwareTurnStartStrategy):
    """Records what the strategy would do instead of talking to a real aggregator."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.started = 0
        self.resets = 0

    async def report_user_turn_started(self):
        self.started += 1

    async def trigger_reset_aggregation(self):
        self.resets += 1


async def _overlap(level_db: float, seconds: float, text: str):
    meter = LevelMeter()
    s = _Probe(steer_after_s=0.2, level_meter=meter, faint_below_dbfs=-26)
    await s.process_frame(BotStartedSpeakingFrame())
    for _ in range(10):  # audio before VAD fires
        meter._recent.append(chunk_dbfs(tone(level_db)))
    meter._speech = list(meter._recent)  # what LevelMeter does on VADUserStartedSpeakingFrame
    await s.process_frame(VADUserStartedSpeakingFrame())
    t_end = asyncio.get_event_loop().time() + seconds
    while asyncio.get_event_loop().time() < t_end:
        meter._speech.append(chunk_dbfs(tone(level_db)))
        await asyncio.sleep(0.02)
    await s.process_frame(VADUserStoppedSpeakingFrame())
    await s.process_frame(TranscriptionFrame(text, "u", "t"))
    return s


def test_faint_long_overlap_keeps_talking():
    s = asyncio.run(_overlap(-38, 0.5, "Roomie, fridge is empty again"))
    assert s.started == 0 and s.resets == 1
    assert [d.kind for d in s.decisions] == ["faint_ignored", "faint_ignored"]


def test_loud_long_overlap_interrupts():
    s = asyncio.run(_overlap(-22, 0.5, "Actually, let's talk about plants"))
    assert s.started == 1 and s.decisions[0].kind == "steering_long"


def test_faint_short_burst_ignored_even_if_words_look_like_steering():
    s = asyncio.run(_overlap(-38, 0.05, "wait what"))
    assert s.started == 0 and s.resets == 1


def test_borderline_level_waits_for_transcript_then_ignores_faint():
    # live level -25 (just above the -26 cut-off): keep talking; full utterance says faint
    s = asyncio.run(_overlap(-25, 0.5, "(faint, distant voice) Roomie, groceries?"))
    assert s.started == 0 and s.resets == 1
    assert [d.kind for d in s.decisions] == ["uncertain_wait", "faint_ignored"]


def test_borderline_level_then_real_words_interrupts():
    s = asyncio.run(_overlap(-25, 0.5, "Actually, tell me about plants"))
    assert s.started == 1 and s.decisions[-1].kind == "steering_transcript"


def test_level_meter_reports_for_ui():
    from pipecat.tests.utils import run_test

    meter = LevelMeter(report_every_s=0)
    seen = []
    meter.on_level = lambda db, speaking: seen.append((round(db), speaking))
    frames = [InputAudioRawFrame(tone(-20), 16000, 1), VADUserStartedSpeakingFrame(),
              InputAudioRawFrame(tone(-20), 16000, 1), VADUserStoppedSpeakingFrame()]
    asyncio.run(run_test(meter, frames_to_send=frames))
    assert (-20, False) in seen and (-20, True) in seen and seen[-1][1] is False
