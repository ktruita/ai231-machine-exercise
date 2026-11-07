"""LLM-as-judge for conversational naturalness (via OpenRouter; runs off the critical path).

Scores each benchmarked exchange 1-5 on a fixed rubric and writes judge.jsonl next to
items.jsonl. Use a model *different* from the agent's to limit self-preference bias.

Run:  python -m benchmark.judge benchmark/results/<stamp>/<config> [--model openai/gpt-4.1-mini]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from common.config import env

BANK = {json.loads(l)["id"]: json.loads(l) for l in open(Path(__file__).parent / "bank/commands.jsonl")}

RUBRIC = """You are evaluating a VOICE assistant. Its reply was spoken aloud by TTS.
Score each criterion from 1 (bad) to 5 (excellent):
- naturalness: sounds like a person talking, not a document; no markdown/lists/symbols.
- brevity: appropriately short for voice; no rambling or needless preamble.
- task_success: did what was asked (or asked one sensible clarifying question).
- turn_taking: handled interruptions/backchannels appropriately (if the scenario has an
  interruption, the reply should address the new request; if the user only said "mhm",
  the assistant should simply have continued). Give 5 if not applicable.
Return ONLY JSON: {"naturalness":n,"brevity":n,"task_success":n,"turn_taking":n,"comment":"<one sentence>"}"""


def scenario(item: dict, result: dict) -> str:
    b = BANK[item["id"]]
    user = " ".join(b[k] for k in ("text", "setup", "overlay", "followup") if k in b)
    extra = ""
    if result["kind"] == "overlap":
        extra = (f"\n(The user said {b['overlay']!r} while the assistant was talking; "
                 f"the assistant {'stopped' if result.get('interrupted') else 'kept talking'}.)")
    return f"User said: {user}{extra}\nAssistant's final reply: {result.get('reply') or '(no reply)'}"


async def judge_one(client: httpx.AsyncClient, model: str, item: dict) -> dict:
    r = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}"},
        json={"model": model, "temperature": 0, "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": RUBRIC},
                           {"role": "user", "content": scenario(item, item)}]},
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    try:
        scores = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except json.JSONDecodeError:
        scores = {"error": text[:200]}
    return {"id": item["id"], "run": item.get("run", 0), "kind": item["kind"], **scores}


async def main(run_dir: Path, model: str) -> None:
    items = [json.loads(l) for l in open(run_dir / "items.jsonl")]
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=60) as client:
        async def one(it):
            async with sem:
                return await judge_one(client, model, it)
        scored = await asyncio.gather(*(one(i) for i in items))
    with open(run_dir / "judge.jsonl", "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")
    print(f"Judged {len(scored)} items -> {run_dir / 'judge.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    a = ap.parse_args()
    asyncio.run(main(a.run_dir, a.model))
