"""Benchmark runner: drives the real agent pipeline with the recorded command bank.

For each configuration (LLM model x ASR backend x TTS backend) every bank item is played
into the pipeline in real time; we record per-item outcomes plus the observer's per-turn
latency records. MCP servers must be running (scripts/run_servers.sh).

Run:
  python -m benchmark.run_bench                                  # config.yaml defaults
  python -m benchmark.run_bench --models google/gemini-2.5-flash-lite openai/gpt-4.1-nano
  python -m benchmark.run_bench --kinds intent --limit 20 --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
import wave

from loguru import logger
from pipecat.pipeline.runner import PipelineRunner

from agent.main import build_agent
from benchmark.file_transport import FileAudioInput, RecordingOutput
from benchmark.render_audio import audio_path
from common.config import ROOT, load_config, path, server_url
from common.mcp_session import PersistentMCP

BANK = ROOT / "benchmark/bank/commands.jsonl"
RESULTS = ROOT / "benchmark/results"
CHAINED = {"CHAT-while-busy", "MEM-ask"}  # keep context from the previous item
ORDER = ["address", "trailing", "intent", "overlap", "chat", "delegate"]  # delegate last


def load_pcm(item_id: str, field: str = "text") -> bytes:
    with wave.open(str(audio_path(item_id, field))) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return w.readframes(w.getnframes())


class Bench:
    def __init__(self, agent, mic: FileAudioInput, home: PersistentMCP, delegate: PersistentMCP):
        self.agent, self.mic, self.home, self.delegate = agent, mic, home, delegate
        self.obs = agent.observer
        self.system = agent.context.get_messages()[:1]

    async def wait_bot_done(self, since: float, timeout: float = 25.0, quiet: float = 1.5,
                            require_start: bool = True) -> bool:
        """Wait until the bot has spoken (started after `since` = start of the user's utterance,
        unless require_start=False, e.g. it was already talking) and then stayed quiet."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            o = self.obs
            spoke = o.last_bot_start > since or (not require_start and o.last_bot_stop > since)
            if spoke and not o.bot_speaking and time.time() - o.last_bot_stop >= quiet:
                return True
            await asyncio.sleep(0.1)
        return False

    def replies_since(self, n_messages: int) -> str:
        """All assistant text added to the context after message index `n_messages`."""
        msgs = self.agent.context.get_messages()[n_messages:]
        return " ".join(m["content"] for m in msgs if isinstance(m, dict)
                        and m.get("role") == "assistant" and isinstance(m.get("content"), str))

    async def run_item(self, item: dict) -> dict:
        if item["id"] not in CHAINED:
            self.agent.context.set_messages(list(self.system))
        self.obs.label = item["id"]
        kind, t0 = item["kind"], time.time()
        n_ctx = len(self.agent.context.get_messages())
        out = {"id": item["id"], "kind": kind, "t0": t0}

        if kind in ("intent", "chat", "address", "delegate"):
            start, end = await self.mic.say(load_pcm(item["id"]))
            out["user_speech_end"] = end
            out["responded"] = await self.wait_bot_done(start)

        elif kind == "trailing":
            start, _ = await self.mic.say(load_pcm(item["id"]))
            out["responded"] = await self.wait_bot_done(start, quiet=0.5)
            start, _ = await self.mic.say(load_pcm(item["id"], "followup"))
            await self.wait_bot_done(start)

        elif kind == "overlap":
            start, _ = await self.mic.say(load_pcm(item["id"], "setup"))
            # wait for the long answer to start, let it run a bit, then overlap
            for _ in range(200):
                if self.obs.last_bot_start > start:
                    break
                await asyncio.sleep(0.1)
            await asyncio.sleep(1.5)
            n_dec = len(self.agent.turn_start.decisions)
            t_overlay, _ = await self.mic.say(load_pcm(item["id"], "overlay"))
            decisions = self.agent.turn_start.decisions[n_dec:]
            interrupted = any(d.kind.startswith("steering") for d in decisions)
            if interrupted:  # the bot must now answer the new request
                out["responded"] = await self.wait_bot_done(t_overlay)
            else:            # the bot should simply finish what it was saying
                await self.wait_bot_done(t_overlay, timeout=60, require_start=False)
            decisions = self.agent.turn_start.decisions[n_dec:]
            interrupted = any(d.kind.startswith("steering") for d in decisions)
            out.update(expect=item["expect"], interrupted=interrupted,
                       decisions=[d.kind for d in decisions],
                       correct=(interrupted == (item["expect"] == "interrupt")))

        # --- scoring --------------------------------------------------------------
        out["reply"] = self.replies_since(n_ctx)
        if "intent" in item:
            logged = await self.home.call("intent_log_since", {"ts": t0})
            got = logged[0] if logged else None
            out.update(expected_intent=item["intent"], expected_slots=item["slots"],
                       got_intent=got and got["intent"], got_slots=got and got["slots"],
                       intent_ok=bool(got and got["intent"] == item["intent"]),
                       slots_ok=bool(got and all(got["slots"].get(k) == v for k, v in item["slots"].items())))
        if "expect_reply" in item:
            turns = [json.loads(l) for l in open(self.obs.out_path)] if self.obs.out_path.exists() else []
            first = next((t for t in turns if t.get("label") == item["id"] and t["type"] == "turn"), {})
            out["backchannel_ok"] = first.get("first_speech") == item["expect_reply"]
        if "expect_contains" in item:
            out["recall_ok"] = item["expect_contains"].lower() in out["reply"].lower()
        if kind == "delegate":
            # newest task created during this item (task_status is newest-first)
            tasks = await self.delegate.call("task_status", {})
            fresh = [t for t in tasks if t["created"] >= t0]
            out["task_id"] = fresh[0]["task_id"] if fresh else None
            out["delegated"] = bool(fresh)
        await asyncio.sleep(0.5)
        return out

    async def harness_turnaround(self, task_ids: list[str], max_wait: float) -> list[dict]:
        """Blackbox harness metric: delegate -> finished (seconds), waits up to max_wait."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            tasks = [t for t in await self.delegate.call("task_status", {}) if t["task_id"] in task_ids]
            if tasks and all(t["status"] != "running" for t in tasks):
                break
            await asyncio.sleep(5)
        return [{"task_id": t["task_id"], "harness": t["harness"], "status": t["status"],
                 "turnaround_s": t["elapsed_s"]} for t in tasks]


async def run_config(items: list[dict], overrides: dict, runs: int, harness_wait: float) -> None:
    name = "__".join(f"{k}={v}".replace("/", "_") for k, v in overrides.items()) or "default"
    run_dir = RESULTS / time.strftime("%Y%m%d-%H%M%S") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    mic = FileAudioInput()
    speaker = RecordingOutput(cfg["tts"]["sample_rate"])
    agent = await build_agent(mic, speaker, session=f"bench-{name}", overrides=overrides)
    agent.observer.out_path = run_dir / "latency.jsonl"
    pipeline_runner = PipelineRunner(handle_sigint=False)
    await pipeline_runner.add_workers(agent.task)
    runner = asyncio.create_task(pipeline_runner.run())
    home = await PersistentMCP(server_url("home")).start()
    delegate = await PersistentMCP(server_url("delegate")).start()
    bench = Bench(agent, mic, home, delegate)
    await asyncio.sleep(2.0)  # let VAD/transport settle

    results = []
    try:
        for run_i in range(runs):
            for item in items:
                logger.info(f"[{name} run {run_i + 1}/{runs}] {item['id']}")
                r = await bench.run_item(item)
                r["run"] = run_i
                results.append(r)
                with open(run_dir / "items.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
        task_ids = [r["task_id"] for r in results if r.get("task_id")]
        if task_ids:
            logger.info(f"Waiting up to {harness_wait}s for {len(task_ids)} delegated tasks...")
            for t in await bench.harness_turnaround(task_ids, harness_wait):
                with open(run_dir / "harness.jsonl", "a") as f:
                    f.write(json.dumps(t) + "\n")
        (run_dir / "config.json").write_text(json.dumps(
            {"overrides": overrides, "llm": cfg["llm"], "asr": cfg["asr"], "tts": cfg["tts"],
             "delegate": {k: v for k, v in cfg["delegate"].items() if k != "harnesses"}}, indent=2))
        with wave.open(str(run_dir / "bot_audio.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(cfg["tts"]["sample_rate"])
            w.writeframes(bytes(speaker.recording))
    finally:
        await agent.task.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        for c in [home, delegate, *agent.pollers]:
            await c.close()
    print(f"Results: {run_dir.relative_to(ROOT)}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[None], help="OpenRouter model ids")
    ap.add_argument("--asr", nargs="*", default=[None], help="deepgram groq_whisper")
    ap.add_argument("--tts", nargs="*", default=[None], help="deepgram_aura cartesia edge_tts")
    ap.add_argument("--kinds", nargs="*", default=ORDER)
    ap.add_argument("--ids", nargs="*", help="only these item ids")
    ap.add_argument("--limit", type=int, help="max items per kind")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--harness-wait", type=float, default=600)
    a = ap.parse_args()

    bank = [json.loads(l) for l in open(BANK)]
    items = []
    for kind in ORDER:
        if kind in a.kinds:
            of_kind = [i for i in bank if i["kind"] == kind and (not a.ids or i["id"] in a.ids)]
            items += of_kind[: a.limit] if a.limit else of_kind
    # "talk while the harness works" must directly follow a delegation
    busy = [i for i in items if i["id"] == "CHAT-while-busy"]
    items = [i for i in items if i["id"] != "CHAT-while-busy"]
    last_delegate = max((n for n, i in enumerate(items) if i["kind"] == "delegate"), default=len(items) - 1)
    items[last_delegate + 1:last_delegate + 1] = busy

    for model, asr, tts in itertools.product(a.models, a.asr, a.tts):
        overrides = {k: v for k, v in
                     {"llm_model": model, "asr_backend": asr, "tts_backend": tts}.items() if v}
        await run_config(items, overrides, a.runs, a.harness_wait)


if __name__ == "__main__":
    path("data_dir")
    asyncio.run(main())
