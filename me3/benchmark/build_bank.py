"""Build the benchmark command bank.

1) Option B sheet of `Dataset Schema(1).xlsx` -> 93 labeled intent utterances
   (13 fixed intents x 3 variations + 6 slotted intents x 3 variations x 3 values).
2) Hand-written conversational items: delegation, backchannel-overlap, steering
   interruptions, addressing ("hey"), trailing-off turns, multi-turn scripts.

Run:  python -m benchmark.build_bank  ->  benchmark/bank/commands.jsonl
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import openpyxl

from common.config import ROOT

SHEET = "Option B"
XLSX = ROOT / "Dataset Schema(1).xlsx"
OUT = ROOT / "benchmark/bank/commands.jsonl"

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "five": 5, "ten": 10, "fifteen": 15, "thirty": 30}


def _seconds(value: str) -> int:
    n, unit = value.split()[:2]
    n = int(n) if n.isdigit() else NUM_WORDS[n]
    return n * {"second": 1, "minute": 60, "hour": 3600}[unit.rstrip("s")]


def _spoken_time(t: dt.time) -> str:
    h = t.hour % 12 or 12
    suffix = "AM" if t.hour < 12 else "PM"
    return f"{h} {suffix}" if t.minute == 0 else f"{h}:{t.minute:02d} {suffix}"


def slot_for(intent: str, value) -> tuple[str, dict]:
    """Spoken form of a slot value + the expected tool slot (matches home-mcp's intent log)."""
    if intent == "TIMER":
        return value, {"duration_seconds": _seconds(value)}
    if intent == "ALARM":
        return _spoken_time(value), {"time": value.strftime("%H:%M")}
    if intent == "TEMPERATURE":
        return value, {"degrees": int(re.match(r"\d+", value).group())}
    if intent == "BRIGHTNESS":
        return value, {"percent": int(re.match(r"\d+", value).group())}
    if intent == "COLOR":
        return value.lower(), {"color": value.lower()}
    if intent == "CREATE_REMINDER":
        return value.lower(), {"task": value.lower()}
    raise ValueError(intent)


def option_b_items() -> list[dict]:
    ws = openpyxl.load_workbook(XLSX, data_only=True)[SHEET]
    rows = [r for r in ws.iter_rows(values_only=True) if r and r[0]]
    header, rows = rows[0], rows[1:]
    var_cols = [i for i, h in enumerate(header) if h and str(h).startswith("Variation")]
    val_cols = [i for i, h in enumerate(header) if h and str(h).startswith("Value")]
    items = []
    for row in rows:
        intent, kind = row[0], row[1]
        variations = [row[i] for i in var_cols if row[i]]
        if kind == "Fixed":
            for vi, text in enumerate(variations, 1):
                items.append({"id": f"{intent}-v{vi}", "kind": "intent", "text": text,
                              "intent": intent, "slots": {}})
        else:
            values = [row[i] for i in val_cols if row[i] not in (None, "-")]
            for vi, template in enumerate(variations, 1):
                for ki, value in enumerate(values, 1):
                    spoken, slots = slot_for(intent, value)
                    text = re.sub(r"\{\w+\}", spoken, template)
                    items.append({"id": f"{intent}-v{vi}-x{ki}", "kind": "intent", "text": text,
                                  "intent": intent, "slots": slots})
    return items


LONG_ANSWER = "Tell me a short story about a cat who learns to fly."

CONVERSATIONAL = [
    # Delegation: must ack instantly, return immediately, and announce result later.
    {"id": "DELEGATE-website", "kind": "delegate",
     "text": "Can you make me a simple one page website about my dog Max? He's a golden retriever who loves the beach."},
    {"id": "DELEGATE-report", "kind": "delegate",
     "text": "Write me a short report comparing three cheap ways to learn Japanese and save it as a file."},
    {"id": "DELEGATE-script", "kind": "delegate",
     "text": "Write a Python script that renames all the photos in a folder by the date they were taken."},
    # Talk while the harness works (run right after a delegate item).
    {"id": "CHAT-while-busy", "kind": "intent", "text": "While that's going, what's the weather like?",
     "intent": "WEATHER", "slots": {}},
    # Backchannel overlap: "mhm" during a long answer must NOT stop the bot.
    {"id": "BC-mhm", "kind": "overlap", "setup": LONG_ANSWER, "overlay": "Mhm.", "expect": "continue"},
    {"id": "BC-yeah", "kind": "overlap", "setup": LONG_ANSWER, "overlay": "Yeah.", "expect": "continue"},
    {"id": "BC-okay", "kind": "overlap", "setup": LONG_ANSWER, "overlay": "Okay.", "expect": "continue"},
    {"id": "BC-uhhuh", "kind": "overlap", "setup": LONG_ANSWER, "overlay": "Uh huh.", "expect": "continue"},
    # Steering: genuine interruptions must stop the bot and be answered.
    {"id": "STEER-stop", "kind": "overlap", "setup": LONG_ANSWER,
     "overlay": "Wait, stop. Actually, what time is it?", "expect": "interrupt"},
    {"id": "STEER-change", "kind": "overlap", "setup": LONG_ANSWER,
     "overlay": "Hold on, can you turn off the lights first?", "expect": "interrupt"},
    {"id": "STEER-short", "kind": "overlap", "setup": LONG_ANSWER, "overlay": "No, stop.", "expect": "interrupt"},
    {"id": "STEER-question", "kind": "overlap", "setup": LONG_ANSWER,
     "overlay": "Sorry, what was the cat's name?", "expect": "interrupt"},
    # Agent-side backchannels without the LLM.
    {"id": "ADDR-hey", "kind": "address", "text": "Hey.", "expect_reply": "Yes?"},
    {"id": "ADDR-there", "kind": "address", "text": "Are you there?", "expect_reply": "Yes?"},
    {"id": "TRAIL-and", "kind": "trailing", "text": "I was thinking about tomorrow and",
     "followup": "set an alarm for 6 AM.", "expect_reply": "Mhm?", "intent": "ALARM",
     "slots": {"time": "06:00"}},
    # Memory round trip.
    {"id": "MEM-tell", "kind": "chat", "text": "By the way, my sister's name is Ana."},
    {"id": "MEM-ask", "kind": "chat", "text": "What's my sister's name?", "expect_contains": "Ana"},
]


def main() -> None:
    items = option_b_items()
    n_intent = len(items)
    items += CONVERSATIONAL
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(items)} items ({n_intent} Option-B intent items) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
