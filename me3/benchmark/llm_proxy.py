"""Recording proxy for the voice LLM: logs every chat-completions request the agent sends
(tools, messages, parameters), forwards it unchanged to OpenRouter, and notes whether the
streamed answer contained a tool call. For debugging "the agent stopped calling tools".

Run on the Pi (terminal 1):   .venv/bin/python -m benchmark.llm_proxy
Then (terminal 2):            ME3_LLM_BASE_URL=http://127.0.0.1:8199/v1 .venv/bin/python -m benchmark.run_bench \
                                 --kinds intent --ids PLAY_MUSIC-v1 LIGHT_ON-v1 TIME-v2 VOLUME_UP-v2
Requests are written to data/llm_requests.jsonl and summarised on screen.
"""

from __future__ import annotations

import json
import time

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

from common.config import ROOT, env

UPSTREAM = "https://openrouter.ai/api/v1/chat/completions"
LOG = ROOT / "data" / "llm_requests.jsonl"
client = httpx.AsyncClient(timeout=120)


async def chat(request: Request):
    body = await request.json()
    msgs = body.get("messages", [])
    tools = body.get("tools") or []
    summary = {
        "t": time.strftime("%H:%M:%S"),
        "n_tools": len(tools),
        "tool_names": [t.get("function", {}).get("name") for t in tools],
        "tool_choice": body.get("tool_choice"),
        "roles": [m.get("role") for m in msgs],
        "last_user": next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), None),
        "system_tail": (msgs[0].get("content") or "")[-160:] if msgs and msgs[0].get("role") == "system" else None,
        "params": {k: v for k, v in body.items() if k not in ("messages", "tools")},
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"summary": summary, "body": body}) + "\n")

    up = client.build_request("POST", UPSTREAM, json=body,
                              headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}"})
    resp = await client.send(up, stream=True)

    async def relay():
        tool_calls, text = [], ""
        async for line in resp.aiter_lines():
            yield (line + "\n").encode()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    for ch in json.loads(line[6:]).get("choices") or []:
                        d = ch.get("delta") or {}
                        text += d.get("content") or ""
                        tool_calls += [(tc.get("function") or {}).get("name") for tc in d.get("tool_calls") or []
                                       if (tc.get("function") or {}).get("name")]
                except ValueError:
                    pass
        await resp.aclose()
        print(f"[{summary['t']}] user={summary['last_user']!r} tools_sent={summary['n_tools']} "
              f"tool_choice={summary['tool_choice']} roles={summary['roles']}\n"
              f"      -> {'TOOL ' + ','.join(tool_calls) if tool_calls else 'no tool: ' + repr(text[:60])}\n"
              f"      params={summary['params']}", flush=True)

    return StreamingResponse(relay(), status_code=resp.status_code,
                             media_type=resp.headers.get("content-type", "text/event-stream"))


app = Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])

if __name__ == "__main__":
    print(f"LLM recording proxy on http://127.0.0.1:8199/v1 -> {UPSTREAM}; log: {LOG}")
    uvicorn.run(app, host="127.0.0.1", port=8199, log_level="warning")
