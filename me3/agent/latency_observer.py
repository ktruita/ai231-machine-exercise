"""Standardized latency instrumentation: one JSONL record per user->bot turn and per barge-in.

All stamps are wall-clock (time.time()) so they line up with benchmark-side clocks.

Turn record (ms deltas from `user_vad_stop`, the moment VAD declared end of speech):
  asr_ms            transcript available
  endpoint_ms       end-of-turn decided (silence timeout; the LLM is triggered here)
  llm_ttft_ms       first LLM text token
  tts_first_ms      first synthesized audio chunk
  voice_to_voice_ms bot actually starts speaking (headline metric)
  + `vad_stop_secs` (the VAD's silence window, add it for true end-of-speech->audio)
Barge-in record: user speech start while bot talking -> bot stopped (barge_in_ms).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


class LatencyObserver(BaseObserver):
    def __init__(self, out_path: Path, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self.out_path = out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.turn: dict | None = None
        self.barge: dict | None = None
        self.bot_speaking = False
        self.last_bot_start = 0.0
        self.last_bot_stop = 0.0
        self.label: str | None = None  # benchmark item id currently being played
        self.on_record = None  # optional callback(record), e.g. the web UI

    def _write(self, record: dict) -> None:
        with open(self.out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.on_record:
            self.on_record(record)

    def _ms(self, key: str) -> float | None:
        t0 = self.turn.get("user_vad_stop") if self.turn else None
        t = self.turn.get(key) if self.turn else None
        return round((t - t0) * 1000, 1) if t and t0 else None

    def _finish_turn(self) -> None:
        t = self.turn
        self._write({
            "type": "turn", "label": self.label, "ts": t.get("user_vad_stop"),
            "transcript": t.get("transcript"), "tools": t.get("tools", []),
            "first_speech": t.get("first_speech"),
            "vad_stop_secs": t.get("vad_stop_secs"),
            "asr_ms": self._ms("asr_done"), "endpoint_ms": self._ms("turn_end"),
            "llm_ttft_ms": self._ms("llm_first"),
            "tts_first_ms": self._ms("tts_first"), "voice_to_voice_ms": self._ms("bot_start"),
        })
        self.turn = None

    async def on_push_frame(self, data: FramePushed):
        f, now = data.frame, time.time()

        if isinstance(f, VADUserStartedSpeakingFrame):
            if self.bot_speaking:
                self.barge = {"user_vad_start": now}
        elif isinstance(f, VADUserStoppedSpeakingFrame):
            if not self.bot_speaking:
                self.turn = {"user_vad_stop": now, "vad_stop_secs": f.stop_secs}
        elif isinstance(f, TranscriptionFrame) and self.turn and "asr_done" not in self.turn:
            self.turn.update(asr_done=now, transcript=f.text)
        elif isinstance(f, UserStoppedSpeakingFrame) and self.turn and "turn_end" not in self.turn:
            self.turn["turn_end"] = now
        elif isinstance(f, LLMTextFrame) and self.turn and "llm_first" not in self.turn:
            self.turn["llm_first"] = now
        elif isinstance(f, TTSSpeakFrame) and self.turn and "first_speech" not in self.turn:
            self.turn["first_speech"] = f.text  # instant backchannel/ack, not LLM text
        elif isinstance(f, FunctionCallInProgressFrame) and self.turn:
            # broadcast up- and downstream: count each tool call once
            seen = self.turn.setdefault("tool_call_ids", set())
            if f.tool_call_id not in seen:
                seen.add(f.tool_call_id)
                self.turn.setdefault("tools", []).append(f.function_name)
        elif isinstance(f, TTSAudioRawFrame) and self.turn and "tts_first" not in self.turn:
            self.turn["tts_first"] = now
        elif isinstance(f, BotStartedSpeakingFrame):
            self.bot_speaking, self.last_bot_start = True, now
            if self.turn and "bot_start" not in self.turn:
                self.turn["bot_start"] = now
                self._finish_turn()
        elif isinstance(f, InterruptionFrame):
            if self.barge:
                self.barge.setdefault("interrupted", now)
        elif isinstance(f, BotStoppedSpeakingFrame):
            self.bot_speaking, self.last_bot_stop = False, now
            if self.barge:
                start = self.barge["user_vad_start"]
                if "interrupted" in self.barge:  # bot was cut off by the user
                    self._write({"type": "barge_in", "label": self.label, "ts": start,
                                 "barge_in_ms": round((now - start) * 1000, 1)})
                else:  # user overlapped (backchannel) and the bot kept talking
                    self._write({"type": "overlap_ignored", "label": self.label, "ts": start})
                self.barge = None
