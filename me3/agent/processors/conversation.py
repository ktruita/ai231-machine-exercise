"""Conversation-feel processors: instant backchannels, tool acks, background-result announcements."""

from __future__ import annotations

import asyncio
import re
import time

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from common.mcp_session import PersistentMCP

# Phrases spoken without the LLM; pre-synthesized in the TTS cache.
ADDRESS_REPLY = "Yes?"
CONTINUE_REPLY = "Mhm?"
ACK_DELEGATE = "Okay, I'll work on that in the background."
ACK_LOOKUP = "Let me check that."
CACHED_PHRASES = [ADDRESS_REPLY, CONTINUE_REPLY, ACK_DELEGATE, ACK_LOOKUP]

ADDRESS_ONLY = {"hey", "hi", "hello", "hey there", "are you there", "you there", "excuse me", "assistant"}
# Only words that almost never end a complete request ("what's the weather like" is complete).
TRAILING_INCOMPLETE = {"and", "but", "um", "uh", "umm", "because", "or"}
SLOW_TOOLS = {"get_weather", "web_search"}  # tools that hit the network and deserve a "let me check"


# Words people put first when talking to SOMEONE ELSE in the room ("Roomie, ...", "Hey Mark, ...").
VOCATIVES = {"roomie", "roomy", "roommate", "bro", "dude", "mom", "mum", "mama", "dad", "papa", "babe",
             "honey", "sweetie", "darling", "guys", "kids", "buddy", "pal", "sis", "lola", "lolo", "tita",
             "tito", "ate", "kuya"}
# Common first names. Names that are also common verbs/words (mark, bill, will, rob, grant, ...)
# only count after "hey/yo" or before a comma, so "Mark this as done" is still a command.
FIRST_NAMES = set("""
aaron adam alex alice amanda amy andrea andrew angela anna anne anthony ashley ben benjamin beth
brian brandon brittany carla carlos carol caroline catherine charles charlie chloe chris christian
christina christine claire daniel danny dana david derek diana diego donna dylan edward elena eli
elizabeth ella emily emma eric erica ethan eva evan fiona frank gabriel gary george gianna grace greg
hannah harry heather helen henry ian isaac isabel isabella jacob james jamie jane jason jen jenna
jennifer jeremy jerry jess jessica jim joe joel john johnny jon jonathan jordan jose joseph josh
joshua julia julie justin karen kate katie kelly kevin kim kyle laura lauren leah lisa liz lucas
lucy luis luke maria marie marco maria mary matt matthew megan melissa mia michael michelle mike
miguel monica nancy natalie nathan nicole noah nora olivia oscar paul paula peter rachel rebecca
rita ryan sam samantha sara sarah scott sean sofia sophia steph steve steven susan taylor tom tony
tyler vanessa victor zoe rumi juan ana carmen pedro rosa jojo
""".split())
AMBIGUOUS_NAMES = {"mark", "bill", "will", "rob", "pat", "sue", "grant", "chase", "drew", "jack",
                   "ray", "max", "rose", "dawn", "joy", "faith", "hope", "summer", "april", "may", "june"}


def addressed_to_someone_else(text: str) -> bool:
    """'Derek, traffic was insane.' / 'Yo Mark, meeting moved.' / 'Roomie, groceries?'
    The assistant has no name, so speech that opens by naming a person is for that person."""
    text = re.sub(r"^\(faint, distant voice\)\s*", "", text.strip())
    m = re.match(r"(?:((?i:hey|hi|yo|oi|okay|ok))[,!.]?\s+)?([A-Za-z']+)([,.!?]?)\s+(\S.*)", text)
    if not m:
        return False
    greet, word, punct, rest = m.group(1), m.group(2), m.group(3), m.group(4)
    w = word.lower()
    if len(rest.split()) < 2 or w in {"assistant", "there", "you", "everyone"}:
        return False
    if w in VOCATIVES:
        return True
    if w in FIRST_NAMES:
        return word[0].isupper() or bool(greet)
    if w in AMBIGUOUS_NAMES:
        return word[0].isupper() and (bool(greet) or punct == ",")
    return False


SIDE_TALK_RESUME = ("(Someone nearby said this to another person, not to you, and it cut you off: "
                    "\"{text}\". Ignore it completely. Continue your previous answer from where you "
                    "stopped, in one or two sentences, without mentioning the interruption.)")
SIDE_TALK_IGNORED = "(Someone nearby said this to another person, not to you: \"{text}\". Ignored.)"
CUT_OFF_WINDOW_S = 15.0


def _last_user_text(context) -> str:
    for msg in reversed(context.get_messages()):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content or ""
    return ""


class BackchannelGate(FrameProcessor):
    """Sits before the LLM. Answers turns that don't need the LLM:

    - "hey" / "are you there?"           -> "Yes?"
    - utterance trailing off ("... and") -> "Mhm?" and wait; the next utterance joins the
      same context, so the LLM later sees the whole thought.
    """

    def __init__(self, *, side_talk_check: bool = False, timezone: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.side_talk_check = side_talk_check
        self.timezone = timezone  # set -> the system prompt always carries the current local time
        self.counts = {"address": 0, "continue": 0, "side_talk": 0}
        self._bot_speaking = False
        self._cut_off_at = 0.0  # when the bot was last interrupted mid-speech

    _CLOCK = re.compile(r"\n\nCurrent local time: [^\n]*$")  # matched before TOOL_RULE is re-added

    def _stamp_time(self, context) -> None:
        """Keep a 'Current local time' line at the end of the system prompt, so the LLM never
        has to guess the time or date (it used to invent one instead of calling get_time)."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        messages = list(context.get_messages())
        if not messages or not isinstance(messages[0], dict) or messages[0].get("role") != "system":
            return
        now = datetime.now(ZoneInfo(self.timezone))
        line = f"\n\nCurrent local time: {now.strftime('%-I:%M %p, %A %B %-d, %Y')} ({self.timezone})."
        content = messages[0]["content"]
        rule = TOOL_RULE if content.endswith(TOOL_RULE) else ""   # the rule stays the very last words
        core = self._CLOCK.sub("", content[: len(content) - len(rule)])
        messages[0] = {**messages[0], "content": core + line + rule}
        context.set_messages(messages)

    @staticmethod
    def _rewrite_last_user(context, new_text: str) -> None:
        messages = list(context.get_messages())
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                messages[i] = {**messages[i], "content": new_text}
                break
        context.set_messages(messages)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, InterruptionFrame) and self._bot_speaking:
            self._cut_off_at = time.time()
        if isinstance(frame, LLMContextFrame) and not getattr(frame, "speculation", False):
            if self.timezone:
                self._stamp_time(frame.context)
            raw = _last_user_text(frame.context).strip()
            if self.side_talk_check and addressed_to_someone_else(raw):
                # deterministic: the LLM never sees it as a request it might act on
                self.counts["side_talk"] += 1
                cut_off = time.time() - self._cut_off_at < CUT_OFF_WINDOW_S
                self._cut_off_at = 0.0
                logger.info(f"Side talk (named someone else, {'resume' if cut_off else 'ignore'}): {raw!r}")
                if cut_off:  # it stopped us mid-answer: pick the answer back up
                    self._rewrite_last_user(frame.context, SIDE_TALK_RESUME.format(text=raw))
                    await self.push_frame(frame, direction)
                else:        # we were idle: no reply at all, no LLM call
                    self._rewrite_last_user(frame.context, SIDE_TALK_IGNORED.format(text=raw))
                return
            words = re.findall(r"[a-z']+", raw.lower())
            text = " ".join(words)
            if text in ADDRESS_ONLY:
                self.counts["address"] += 1
                await self.push_frame(TTSSpeakFrame(ADDRESS_REPLY))
                return
            if words and words[-1] in TRAILING_INCOMPLETE and not raw.endswith("?"):
                self.counts["continue"] += 1
                await self.push_frame(TTSSpeakFrame(CONTINUE_REPLY))
                return
        await self.push_frame(frame, direction)


class ToolAck(FrameProcessor):
    """Sits between LLM and TTS. Speaks an instant ack when a slow tool or delegation starts."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, FunctionCallsStartedFrame) and direction == FrameDirection.DOWNSTREAM:
            names = {c.function_name for c in frame.function_calls}
            if names & {"delegate_task", "continue_task"}:
                await self.push_frame(TTSSpeakFrame(ACK_DELEGATE))
            elif names & SLOW_TOOLS:
                await self.push_frame(TTSSpeakFrame(ACK_LOOKUP))


# Last words of the system prompt (after the time line). With 21 tools, Qwen3-30B often replied
# "Turning the lights on" WITHOUT calling the tool; this rule at the very end fixed it
# (benchmark/replay_request.py, 2026-11-07: 0/6 -> 6/6).
TOOL_RULE = ("\n\nTOOLS ARE MANDATORY: for any device or assistant command (lights, music, volume, "
             "temperature, time, timers, alarms, reminders, calls, messages, weather) you MUST call the "
             "matching tool before replying. Never say you did something without calling its tool.")

SILENT_MARKER = "<silent>"

ADDRESSED_PROMPT = f"""
Not everything you hear is meant for you:
- A single person speaking conversationally IS talking to you, even if it's casual, off-topic,
  or sounds like part of a chat. Answer it normally.
- Commands and requests ("Mute the TV.", "Open the garage.", "Order a pizza.") are ALWAYS for
  you, even when you have no tool for them: then say in one sentence that you can't do that yet.
- Stay silent when the speech is not for you:
  * It starts by calling someone by name. You have no name, so "Maria, can you pass the salt?"
    or "Tom, the trash is full again." is said to another person.
  * It is marked {"(faint, distant voice)"}: it came from a TV, radio or someone across the room,
    even if it sounds like a question ("(faint, distant voice) What should I cook tonight?").
  * TV/radio/broadcast or announcement style speech, several people talking with each other,
    one side of a phone call, or noise / meaningless fragments.
- To stay silent, reply with exactly {SILENT_MARKER} and nothing else. Otherwise, if unsure, answer.
"""


class AddressFilter(FrameProcessor):
    """Sits between LLM and TTS. Swallows responses the LLM marked as not-addressed-to-me
    ("<silent>"), so side talk / TV / background speech gets no spoken reply.

    The marker can be split across streamed tokens ("<", "sil", "ent>"), so only the first
    few characters of each response are buffered until we can tell; normal answers lose
    nothing but those few characters of delay.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.silenced = 0
        self._reset()

    def _reset(self):
        self._decided: bool | None = None  # None = undecided, True = silent, False = speak
        self._buffer: list[LLMTextFrame] = []

    def _prefix(self) -> str:
        return "".join(f.text for f in self._buffer).lstrip().lower()

    async def _flush(self, direction: FrameDirection):
        for f in self._buffer:
            await self.push_frame(f, direction)
        self._buffer = []

    async def _silence(self):
        self._decided, self._buffer = True, []
        self.silenced += 1
        logger.info("Address check: not addressed to the assistant -> staying silent")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
            self._reset()
        elif isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._decided is True:
                return
            if self._decided is None:
                self._buffer.append(frame)
                prefix = self._prefix()
                if not prefix:
                    return
                if prefix.startswith(SILENT_MARKER):
                    await self._silence()
                elif not SILENT_MARKER.startswith(prefix):
                    self._decided = False
                    await self._flush(direction)
                return  # still a possible marker prefix: keep buffering
        elif isinstance(frame, LLMFullResponseEndFrame) and self._decided is None and self._buffer:
            prefix = self._prefix()
            if len(prefix) >= 2 and SILENT_MARKER.startswith(prefix):
                await self._silence()  # truncated marker, e.g. "<silent"
            else:
                await self._flush(direction)
        await self.push_frame(frame, direction)


class ShortSentenceMerger(FrameProcessor):
    """Sits between LLM and TTS. Joins a very short sentence onto the next one
    ("Sure! Try Atomic Habits." -> "Sure, Try Atomic Habits.").

    TTS synthesizes sentence by sentence. A lone "Sure!" is 0.5 s of audio, and the next,
    longer sentence often isn't ready when it ends, so the reply comes out as a tiny
    fragment, a gap, then the rest. Listeners (and Full-Duplex-Bench, which needs >= 1 s
    or > 3 words) hear that as a non-answer. Questions ("Really?") are left alone.
    """

    _END = re.compile(r"[.!]+(?=\s)")
    _LIST = re.compile(r"(^|\n)\s*(?:\d+[.)]|[-•])\s+")  # "1. ", "- " at a line start
    _MARKUP = re.compile(r"[*_#`\"“”]+")

    @classmethod
    def _speakable(cls, text: str) -> str:
        """Markdown the prompt forbids but the model sometimes writes anyway."""
        return cls._MARKUP.sub("", cls._LIST.sub(r"\1", text)).replace("\n", " ")

    def __init__(self, min_words: int = 4, max_words: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.min_words = min_words
        self.max_words = max_words  # stop after the sentence that reaches this many words
        self.merged = 0
        self.truncated = 0
        self._buf = ""
        self._spoken = 0

    async def _emit(self, text: str, direction: FrameDirection):
        if not text.strip():
            return
        if self.max_words and self._spoken >= self.max_words:
            if self._spoken < 10**6:
                self.truncated += 1
                self._spoken = 10**6  # count once per response
                logger.debug("Reply length cap reached: rest of the response not spoken")
            return
        self._spoken += len(re.findall(r"[\w']+", text))
        await self.push_frame(LLMTextFrame(text), direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
            self._buf, self._spoken = "", 0
        elif isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._buf += frame.text
            if "\n" in self._buf[:-4] or self._MARKUP.search(self._buf):
                keep = self._buf[-4:] if not self._buf.endswith(("\n", " ")) else ""
                self._buf = self._speakable(self._buf[:len(self._buf) - len(keep)]) + keep
            start = 0
            while m := self._END.search(self._buf, start):
                sentence = self._buf[:m.start()]
                if len(re.findall(r"[\w']+", sentence)) < self.min_words:
                    self._buf = sentence + "," + self._buf[m.end():]
                    self.merged += 1
                    start = m.start() + 1
                    continue
                await self._emit(self._buf[:m.end()], direction)
                self._buf, start = self._buf[m.end():], 0
            if "?" in self._buf:  # never hold back a question
                cut = self._buf.rindex("?") + 1
                await self._emit(self._buf[:cut], direction)
                self._buf = self._buf[cut:]
            return
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._emit(self._speakable(self._buf), direction)
            self._buf = ""
        await self.push_frame(frame, direction)


async def reset_context_when_idle(context, activity: "ActivityTracker", idle_s: float) -> None:
    """Benchmark mode: forget the conversation after `idle_s` of quiet, so independent test
    trials don't share history (which would also slow down and bias later trials)."""
    system = context.get_messages()[:1]
    while True:
        await asyncio.sleep(0.5)
        if activity.idle_for() >= idle_s and len(context.get_messages()) > len(system):
            context.set_messages(list(system))
            logger.debug("Idle: conversation context reset")


class ActivityTracker(BaseObserver):
    """Pipeline-wide "is anything happening?" signal, built from every frame (an observer,
    so it doesn't depend on where a processor sits). Busy while the user or bot speaks,
    an LLM response is streaming, or a tool call is running."""

    def __init__(self, **kwargs):
        super().__init__(observe_every_push=False, **kwargs)
        self.bot_speaking = False
        self.user_speaking = False
        self.llm_running = 0
        self.tools_running: set[str] = set()
        self.awaiting_until = 0.0  # we triggered the LLM and wait for it to start
        self.unspoken_until = 0.0  # text produced (LLM/ack) but not yet spoken: TTS is working
        self.last_activity = time.time()

    def expect_response(self, timeout_s: float = 10.0) -> None:
        self.awaiting_until = time.time() + timeout_s
        self.last_activity = time.time()

    @property
    def busy(self) -> bool:
        return (self.bot_speaking or self.user_speaking or self.llm_running > 0
                or bool(self.tools_running) or time.time() < self.awaiting_until
                or time.time() < self.unspoken_until)

    def idle_for(self) -> float:
        return 0.0 if self.busy else time.time() - self.last_activity

    async def on_push_frame(self, data: FramePushed):
        f = data.frame
        if isinstance(f, BotStartedSpeakingFrame):
            self.bot_speaking = True
        elif isinstance(f, BotStoppedSpeakingFrame):
            self.bot_speaking = False
            self.unspoken_until = 0.0
        elif isinstance(f, (LLMTextFrame, TTSSpeakFrame)) and f.text.strip():
            if not self.bot_speaking:
                self.unspoken_until = time.time() + 20.0  # safety expiry
        elif isinstance(f, UserStartedSpeakingFrame):
            self.user_speaking = True
        elif isinstance(f, UserStoppedSpeakingFrame):
            self.user_speaking = False
        elif isinstance(f, LLMFullResponseStartFrame):
            self.llm_running += 1
            self.awaiting_until = 0.0
        elif isinstance(f, LLMFullResponseEndFrame):
            self.llm_running = max(0, self.llm_running - 1)
        elif isinstance(f, FunctionCallInProgressFrame):
            self.tools_running.add(f.tool_call_id)
        elif isinstance(f, FunctionCallResultFrame):
            self.tools_running.discard(f.tool_call_id)
        elif isinstance(f, InterruptionFrame):
            self.llm_running, self.tools_running = 0, set()
            self.awaiting_until = self.unspoken_until = 0.0
        else:
            return
        self.last_activity = time.time()


class EventAnnouncer(FrameProcessor):
    """Sits before the LLM. Polls home/delegate servers for events (timer fired, task
    finished) and injects them into the conversation once nobody is speaking."""

    def __init__(self, sources: dict[str, PersistentMCP], activity: ActivityTracker,
                 poll_s: float = 1.0, quiet_s: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._sources = sources
        self._activity = activity
        self._poll_s = poll_s
        self._quiet_s = quiet_s
        self._pending: list[dict] = []
        self._task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._task is None:
            self._task = self.create_task(self._loop())
        await self.push_frame(frame, direction)

    def _idle(self) -> bool:
        return self._activity.idle_for() >= self._quiet_s

    async def _loop(self):
        while True:
            for name, client in self._sources.items():
                try:
                    self._pending.extend(await client.call("poll_events"))
                except Exception as e:
                    logger.debug(f"poll_events on {name} failed: {e}")
            if self._pending and self._idle():
                events, self._pending = self._pending, []
                lines = []
                for e in events:
                    if e["kind"].startswith("task_"):
                        lines.append(f"Background task {e['task_id']} ({e['goal']!r}) {e['status']}: {e['result']}")
                    else:
                        lines.append(e["message"])
                logger.info(f"Announcing events: {lines}")
                await self.push_frame(LLMMessagesAppendFrame(
                    messages=[{"role": "system", "content":
                               "[Background update — tell the user briefly and naturally]\n" + "\n".join(lines)}],
                    run_llm=True,
                ))
                self._activity.expect_response()
            await asyncio.sleep(self._poll_s)

    async def cleanup(self):
        await super().cleanup()
        if self._task:
            await self.cancel_task(self._task)
