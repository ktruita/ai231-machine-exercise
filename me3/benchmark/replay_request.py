"""Replay the last request captured by benchmark.llm_proxy, changing one thing at a time, to find
what stops the model from calling tools. 3 tries per variant, non-streamed (cheap, < $0.01).

Run on the Pi (MCP servers running):  .venv/bin/python -m benchmark.replay_request
"""

from __future__ import annotations

import asyncio
import copy
import json

import httpx

from benchmark.probe_latency import tool_schemas
from common.config import ROOT, env

LOG = ROOT / "data" / "llm_requests.jsonl"


async def run(client, body, tries=3):
    body = copy.deepcopy(body)
    body.pop("stream", None)
    body.pop("stream_options", None)
    out = []
    for _ in range(tries):
        r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}"}, json=body)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        out.append("TOOL " + msg["tool_calls"][0]["function"]["name"] if msg.get("tool_calls") else "text")
    return out


async def main():
    lines = [json.loads(l) for l in open(LOG)]
    base = next(l["body"] for l in reversed(lines) if l["summary"]["last_user"] == "Lights on.") \
        if any(l["summary"]["last_user"] == "Lights on." for l in lines) else lines[-1]["body"]
    user = next(m["content"] for m in reversed(base["messages"]) if m["role"] == "user")
    print(f"replaying: {user!r}, {len(base['tools'])} tools\n")

    raw = await tool_schemas()  # our MCP schemas, as the working probe sends them
    raw_by_name = {t["function"]["name"]: t for t in raw}

    variants = {}
    variants["1 as captured (should fail)"] = base
    v = copy.deepcopy(base)
    v["tools"] = [t for t in v["tools"] if t["function"]["name"] not in
                  {"delegate_task", "continue_task", "task_status", "cancel_task"}]
    variants["2 without the 4 background-task tools"] = v
    v = copy.deepcopy(base)
    v["tools"] = [raw_by_name.get(t["function"]["name"], t) for t in v["tools"]]
    variants["3 tool schemas as MCP sends them"] = v
    v = copy.deepcopy(base)
    sysmsg = v["messages"][0]["content"]
    cut = sysmsg.find("\nNot available in this session")
    if cut > 0:
        v["messages"][0]["content"] = sysmsg[:cut]
    variants["4 without the 'not available' note"] = v

    RULE = ("\n\nTOOLS ARE MANDATORY: for any device or assistant command (lights, music, volume, "
            "temperature, time, timers, alarms, reminders, calls, messages, weather) you MUST call the "
            "matching tool before replying. Never say you did something without calling its tool.")
    v = copy.deepcopy(base)
    v["messages"][0]["content"] += RULE
    variants["5 + firm tool rule at end of prompt"] = v
    for model in ("openai/gpt-4.1-mini", "google/gemini-2.5-flash-lite"):
        v = copy.deepcopy(base)
        v["model"] = model
        v.pop("provider", None)
        variants[f"6 model {model}"] = v

    def with_user(body, text):
        b = copy.deepcopy(body)
        for m in reversed(b["messages"]):
            if m["role"] == "user":
                m["content"] = text
                break
        return b

    print("first tool as captured:\n" + json.dumps(base["tools"][0], indent=1)[:700] + "\n")
    print("system prompt tail:\n" + base["messages"][0]["content"][-400:] + "\n")
    async with httpx.AsyncClient(timeout=60) as client:
        for name, body in variants.items():
            try:
                res = await run(client, body) + await run(client, with_user(body, "Play music."))
                print(f"{name:42s} {sum(r.startswith('TOOL') for r in res)}/6  {res}")
            except Exception as e:  # noqa: BLE001
                print(f"{name:42s} error {e}")


if __name__ == "__main__":
    asyncio.run(main())
