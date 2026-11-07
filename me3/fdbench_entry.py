"""Entry point for the class benchmark (fd-benchmark / fdbench, Full-Duplex-Bench based).

Their launcher imports this file and hands us a Pipecat transport that carries 16 kHz
audio to/from fdbench on the laptop. Our agent is unchanged: build_agent() already takes
any input/output transport.

On the Pi (MCP servers must be running: scripts/run_servers.sh):
  FDBENCH_ENTRY=$PWD/fdbench_entry.py .venv/bin/python fd-benchmark/launch_assistant.py --check
  FDBENCH_ENTRY=$PWD/fdbench_entry.py .venv/bin/python fd-benchmark/launch_assistant.py \
      --url ws://<laptop-ip>:8791 \
      --stats-procs asr_mcp=mcp_servers.asr,tts_mcp=mcp_servers.tts,tools_mcp=mcp_servers.home,delegate_mcp=mcp_servers.delegate

Optional env overrides for experiments:
  ME3_SPEECH_TIMEOUT=0.8   end-of-turn silence after VAD stop (default: config turn.user_speech_timeout)
  ME3_RESET_IDLE_S=3       forget the conversation after N s of quiet between trials
  ME3_FAINT_BELOW_DBFS=-31 quieter speech is tagged faint (TV / across the room)
  ME3_LLM_MODEL=...        OpenRouter model id
  ME3_ASR_BACKEND=groq_whisper / ME3_TTS_BACKEND=...   speech backend override

Benchmark mode excludes the delegation and memory tools (see build()).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

PROJECT_MARKER = "agent/main.py"


def audio_rates() -> tuple[int, int]:
    from common.config import load_config

    return 16000, load_config()["tts"]["sample_rate"]


async def build(transport, log_dir: Path):
    from pipecat.pipeline.runner import PipelineRunner

    from agent.main import build_agent
    from agent.processors.conversation import reset_context_when_idle
    from common.config import load_config

    turn = load_config().get("turn", {})
    # Full-Duplex-Bench tests conversation only: no background tasks (they'd cost credit and
    # replace answers with "I'll work on that") and no memory lookups (slower, irrelevant).
    overrides = {"exclude_tools": ["delegate_task", "continue_task", "task_status", "cancel_task",
                                   "remember", "recall", "forget", "web_search"],
                 "cancel_on_unusable": True,
                 # measured on the Pi (full set): users -14..-26 dBFS, TV/background -35.7..-39.7
                 "faint_below_dbfs": float(os.environ.get("ME3_FAINT_BELOW_DBFS", -31))}
    if os.environ.get("ME3_SPEECH_TIMEOUT"):
        overrides["speech_timeout"] = float(os.environ["ME3_SPEECH_TIMEOUT"])
    if os.environ.get("ME3_LLM_MODEL"):
        overrides["llm_model"] = os.environ["ME3_LLM_MODEL"]
    if os.environ.get("ME3_ASR_BACKEND"):
        overrides["asr_backend"] = os.environ["ME3_ASR_BACKEND"]
    if os.environ.get("ME3_TTS_BACKEND"):
        overrides["tts_backend"] = os.environ["ME3_TTS_BACKEND"]
    reset_idle_s = float(os.environ.get("ME3_RESET_IDLE_S")
                         or turn.get("reset_context_after_idle_s") or 3.0)

    agent = await build_agent(transport.input(), transport.output(),
                              session="fdbench", overrides=overrides)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    agent.observer.out_path = log_dir / "latency.jsonl"  # our per-stage latency, next to theirs

    async def run():
        resetter = asyncio.create_task(
            reset_context_when_idle(agent.context, agent.activity, reset_idle_s))
        try:
            runner = PipelineRunner(handle_sigint=False)
            await runner.add_workers(agent.task)
            await runner.run()
        finally:
            resetter.cancel()
            for p in agent.pollers:
                await p.close()

    async def cancel():
        await agent.task.cancel()

    return run(), cancel
