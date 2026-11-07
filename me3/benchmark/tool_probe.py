"""Does the voice LLM call tools? Sends the agent's exact system prompt + MCP tool list to
OpenRouter with a few Option B commands, in four variants, and prints which provider answered
and whether a tool was called. Cost: a few small requests (well under $0.01).

Needs the MCP servers running.  Run:  .venv/bin/python -m benchmark.tool_probe
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from agent.processors.conversation import ADDRESSED_PROMPT
from benchmark.probe_latency import tool_schemas
from common.config import ROOT, env, load_config

COMMANDS = ["Turn on the lights.", "Play some music.", "Set the temperature to 22 degrees."]


async def ask(client, model, system, tools, text, provider):
    body = {"model": model, "temperature": 0.4, "max_tokens": 150, "tools": tools,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}
    if provider:
        body["provider"] = provider
    r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                          headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}"}, json=body)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    calls = [c["function"]["name"] + c["function"]["arguments"] for c in (msg.get("tool_calls") or [])]
    return data.get("provider", "?"), calls, (msg.get("content") or "")[:50]


async def ask_stream(client, model, system, tools, text, provider):
    """Same request the way Pipecat sends it: streamed, max_completion_tokens."""
    body = {"model": model, "temperature": 0.4, "max_completion_tokens": 150, "tools": tools,
            "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}
    if provider:
        body["provider"] = provider
    names, content, prov = [], "", "?"
    async with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                             headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}"}, json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            prov = chunk.get("provider", prov)
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                content += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    name = (tc.get("function") or {}).get("name")
                    if name:
                        names.append(name)
    return prov, names, content[:50]


async def main():
    cfg = load_config()
    model = cfg["llm"]["model"]
    base = (ROOT / "agent/prompts/system.md").read_text() + ADDRESSED_PROMPT
    tz = cfg["location"]["timezone"]
    stamped = base + f"\n\nCurrent local time: {datetime.now(ZoneInfo(tz)).strftime('%-I:%M %p, %A %B %-d, %Y')} ({tz})."
    tools = await tool_schemas()
    print(f"model {model} · {len(tools)} tools: {', '.join(t['function']['name'] for t in tools)}\n")
    variants = [("prompt + time stamp, fastest provider", stamped, {"sort": "latency"}),
                ("prompt only,         fastest provider", base, {"sort": "latency"}),
                ("prompt + time stamp, default routing ", stamped, None),
                ("prompt only,         default routing ", base, None)]
    variants += [("STREAMED (like Pipecat) + stamp, fastest", stamped, {"sort": "latency"}),
                 ("STREAMED (like Pipecat), default       ", stamped, None)]
    async with httpx.AsyncClient(timeout=60) as client:
        for name, system, provider in variants:
            print(name)
            fn = ask_stream if name.startswith("STREAMED") else ask
            for text in COMMANDS:
                try:
                    prov, calls, content = await fn(client, model, system, tools, text, provider)
                    print(f"   {text:38s} {prov:14s} {'TOOL ' + calls[0] if calls else 'no tool: ' + repr(content)}")
                except Exception as e:  # noqa: BLE001
                    print(f"   {text:38s} error {e}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
