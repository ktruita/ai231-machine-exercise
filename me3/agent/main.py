"""Full-duplex voice agent (runs on the Pi; all model inference is remote except Silero VAD).

Pipeline:
  mic -> VAD -> asr-mcp -> user turn (backchannel-aware barge-in) -> BackchannelGate
      -> EventAnnouncer -> LLM (OpenRouter, MCP tools) -> ToolAck -> tts-mcp -> speaker

Run:  python -m agent.main            mic on the Pi; voice + UI in the laptop browser
                                      (open http://<pi-ip>:8080, see config.yaml `ui`)
      python -m agent.main --no-mic   type-only (UI testing without a microphone)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass

from loguru import logger
from mcp.client.session_group import StreamableHttpParameters
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.worker import PipelineParams, ProcessorUnusablePolicy
from pipecat.frames.frames import InterruptionFrame, LLMMessagesAppendFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.mcp_service import MCPClient
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from agent.latency_observer import LatencyObserver
from agent.processors.conversation import (
    TOOL_RULE,
    ShortSentenceMerger,
    ADDRESSED_PROMPT,
    CACHED_PHRASES,
    ActivityTracker,
    AddressFilter,
    BackchannelGate,
    EventAnnouncer,
    ToolAck,
)
from agent.processors.interrupt_classifier import BackchannelAwareTurnStartStrategy, LevelMeter
from agent.processors.mcp_speech import MCPSegmentedSTTService, MCPTTSService
from common.config import ROOT, env, load_config, path, server_url
from common.mcp_session import PersistentMCP

LLM_TOOLS = {
    "home": ["music", "get_weather", "get_time", "lights", "set_temperature", "call",
             "send_message", "set_timer", "set_alarm", "create_reminder", "list_reminders",
             "end_call", "dismiss_alarm", "web_search"],
    "memory": ["remember", "recall", "forget"],
    "delegate": ["delegate_task", "continue_task", "task_status", "cancel_task"],
}


class ToggleableMuteWhileSpeaking(AlwaysUserMuteStrategy):
    """Ignore the mic while the bot talks, only when enabled (UI toggle).

    Fallback for setups where the bot's voice leaks into the mic (e.g. laptop speakers
    near the headset): it stops self-interruption, at the cost of barge-in.
    """

    def __init__(self, enabled: bool = False):
        super().__init__()
        self.enabled = enabled

    async def process_frame(self, frame) -> bool:
        speaking = await super().process_frame(frame)
        return speaking and self.enabled


@dataclass
class Agent:
    task: PipelineTask
    observer: LatencyObserver
    turn_start: BackchannelAwareTurnStartStrategy
    gate: BackchannelGate
    tts: MCPTTSService
    context: LLMContext
    pollers: list[PersistentMCP]
    mute: ToggleableMuteWhileSpeaking
    activity: ActivityTracker
    address_filter: AddressFilter
    level_meter: LevelMeter


async def build_agent(transport_input, transport_output, *, session: str | None = None,
                      overrides: dict | None = None, extra_observers: list | None = None) -> Agent:
    """Assemble the pipeline around any input/output transport (mic or benchmark file feed).

    `overrides` may set llm_model / asr_backend / tts_backend / speech_timeout /
    exclude_tools for benchmark sweeps.
    """
    cfg = load_config()
    o = overrides or {}
    turn = cfg.get("turn", {})
    session = session or time.strftime("%Y%m%d-%H%M%S")

    # LLM-visible tools from our MCP servers (benchmarks can exclude some, e.g. delegation).
    excluded = set(o.get("exclude_tools", ()))
    if not cfg.get("search", {}).get("enabled", False):
        excluded.add("web_search")
    tools = []
    for name, allowed in LLM_TOOLS.items():
        allowed = [t for t in allowed if t not in excluded]
        if not allowed:
            continue
        client = MCPClient(StreamableHttpParameters(url=server_url(name)), tools_filter=allowed)
        tools.extend((await client.tools()).standard_tools)

    system_prompt = (ROOT / "agent/prompts/system.md").read_text()
    if turn.get("addressed_check", True):
        system_prompt += ADDRESSED_PROMPT
    # Only mention excluded tools the prompt itself talks about (e.g. delegation in the benchmark).
    # Never phrase it as "just answer directly": the model took that as "use no tools at all"
    # and stopped calling the home tools (Option B run 2026-11-07, 32% intent accuracy).
    mentioned = sorted(t for t in excluded if t in system_prompt)
    if mentioned:
        system_prompt += (f"\nNot available in this session: {', '.join(mentioned)}. If a request needs "
                          "one of these, say so briefly. All other tools work as usual: keep using them.\n")
    system_prompt += TOOL_RULE  # keep last: see TOOL_RULE in conversation.py
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}], tools=tools)

    faint_below = o.get("faint_below_dbfs", turn.get("faint_below_dbfs"))
    level_meter = LevelMeter()
    turn_start = BackchannelAwareTurnStartStrategy(
        steer_after_s=float(turn.get("steer_after_s", 0.8)),
        level_meter=level_meter, faint_below_dbfs=faint_below)
    mute = ToggleableMuteWhileSpeaking(enabled=cfg.get("ui", {}).get("mute_while_speaking", False))
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[turn_start],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=float(
                    o.get("speech_timeout", turn.get("user_speech_timeout", 0.4))))],
            ),
            user_mute_strategies=[mute],
            # A turn with no transcript (noise, a cough, an unrecognised "mhm" over the bot) gets
            # no answer. Pipecat's default makes the bot ask "could you repeat that?", which
            # is a false turn-take and cuts off our own reply.
            empty_user_turn=None,
        ),
    )

    # ME3_LLM_BASE_URL: point at any OpenAI-compatible endpoint (e.g. a local fake for tests).
    base_url = os.environ.get("ME3_LLM_BASE_URL")
    llm = OpenRouterLLMService(
        api_key=os.environ.get("OPENROUTER_API_KEY", "none") if base_url else env("OPENROUTER_API_KEY"),
        base_url=base_url or "https://openrouter.ai/api/v1",
        settings=OpenRouterLLMService.Settings(
            model=o.get("llm_model", cfg["llm"]["model"]),
            temperature=cfg["llm"]["temperature"],
            # a spoken reply is 2-3 sentences (~60 tokens); a cap stops runaway lists that
            # keep talking into the next turn
            max_completion_tokens=cfg["llm"].get("max_tokens") or None,
            provider={"sort": cfg["llm"]["provider_sort"]},
        ),
    )
    stt = MCPSegmentedSTTService(url=server_url("asr"), backend=o.get("asr_backend", "default"),
                                 faint_below_dbfs=faint_below)
    tts = MCPTTSService(url=server_url("tts"), backend=o.get("tts_backend", "default"),
                        cached_phrases=CACHED_PHRASES,
                        split_long=bool(cfg["tts"].get("split_long_sentences", False)))
    await tts.warm_cache()

    pollers = [await PersistentMCP(server_url("home")).start(),
               await PersistentMCP(server_url("delegate")).start()]
    gate = BackchannelGate(side_talk_check=bool(turn.get("addressed_check", True)),
                           timezone=cfg.get("location", {}).get("timezone"))
    activity = ActivityTracker()
    address_filter = AddressFilter()
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(
        stop_secs=turn.get("vad_stop_secs", 0.2),
        min_volume=cfg.get("vad", {}).get("min_volume", 0.6))))

    pipeline = Pipeline([
        transport_input,
        vad,
        level_meter,
        stt,
        user_agg,
        gate,
        EventAnnouncer({"home": pollers[0], "delegate": pollers[1]}, activity),
        llm,
        address_filter,
        ShortSentenceMerger(max_words=cfg["llm"].get("max_spoken_words")),
        ToolAck(),
        tts,
        transport_output,
        assistant_agg,
    ])

    observer = LatencyObserver(path("data_dir") / "latency" / f"{session}.jsonl")
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, audio_in_sample_rate=16000,
                              audio_out_sample_rate=cfg["tts"]["sample_rate"]),
        observers=[observer, activity, *(extra_observers or [])],
        idle_timeout_secs=None,
        # Benchmark: if the link to fdbench stalls and the output gives up, end the session so
        # the launcher reconnects, instead of carrying on with no way to be heard.
        processor_unusable_policy=(ProcessorUnusablePolicy.CANCEL if o.get("cancel_on_unusable")
                                   else ProcessorUnusablePolicy.CONTINUE),
    )
    return Agent(task, observer, turn_start, gate, tts, context, pollers, mute, activity,
                 address_filter, level_meter)


async def main(no_mic: bool = False):
    import uvicorn

    from agent.web.browser_output import BrowserAudioOutput
    from agent.web.hub import Hub
    from agent.web.server import UIControls, create_app, playback_latency_record
    from agent.web.ui_observer import UIObserver

    cfg = load_config()
    ui = cfg.get("ui", {})
    hub = Hub()

    # Input: headset mic on the Pi (or silence for --no-mic). Output: browser or Pi headset.
    local = None
    if not no_mic or cfg.get("audio_out") == "local":
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
        local = LocalAudioTransport(LocalAudioTransportParams(
            audio_in_enabled=not no_mic, audio_out_enabled=cfg.get("audio_out") == "local",
            input_device_index=_int_env("ME3_INPUT_DEVICE"),
            output_device_index=_int_env("ME3_OUTPUT_DEVICE"),
        ))
    if no_mic:
        from benchmark.file_transport import FileAudioInput
        transport_input = FileAudioInput()  # continuous silence; talk via the UI text box
    else:
        transport_input = local.input()
    if cfg.get("audio_out", "browser") == "browser":
        transport_output = BrowserAudioOutput(hub, cfg["tts"]["sample_rate"])
    else:
        transport_output = local.output()

    agent = await build_agent(transport_input, transport_output, extra_observers=[UIObserver(hub)])
    loop = asyncio.get_running_loop()
    agent.observer.on_record = lambda r: loop.create_task(hub.emit("latency", record=r))
    # live mic meter in the UI header (not kept in the replay history)
    agent.level_meter.on_level = lambda db, speaking: loop.create_task(
        hub.emit("mic", keep=False, db=round(db, 1), speaking=speaking))
    agent.turn_start.on_decision = lambda d: loop.create_task(
        hub.emit("decision", kind=d.kind, text=d.text, speech_s=d.speech_s))

    async def say(text: str) -> None:
        await hub.emit("user_text", text=text, typed=True)
        if agent.observer.bot_speaking:  # typing over the bot interrupts it, like speaking does
            await agent.task.queue_frame(InterruptionFrame())
        await agent.task.queue_frame(LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": text}], run_llm=True))

    def on_playback_started(burst: int, rtt_ms: float) -> None:
        if isinstance(transport_output, BrowserAudioOutput):
            record = playback_latency_record(transport_output.sent_at, burst, rtt_ms)
            if record:
                agent.observer._write(record)

    def set_mute(on: bool) -> None:
        agent.mute.enabled = on

    delegate = agent.pollers[1]
    controls = UIControls(
        say=say,
        cancel_task=lambda task_id: delegate.call("cancel_task", {"task_id": task_id}),
        set_mute_while_speaking=set_mute,
        on_playback_started=on_playback_started,
        mute_while_speaking=lambda: agent.mute.enabled,
    )
    app = create_app(hub, controls, path("workspaces_dir"), home=agent.pollers[0],
                     delegate=delegate, token=ui.get("token") or None,
                     sample_rate=cfg["tts"]["sample_rate"], music_dir=path("music_dir"))
    server = uvicorn.Server(uvicorn.Config(
        app, host=ui.get("host", "0.0.0.0"), port=ui.get("port", 8080), log_level="warning"))
    web = asyncio.create_task(server.serve())

    logger.info(f"Voice agent ready. UI: http://<this-device-ip>:{ui.get('port', 8080)}"
                + (f"/?token={ui['token']}" if ui.get("token") else ""))
    try:
        runner = PipelineRunner()
        await runner.add_workers(agent.task)
        await runner.run()
    finally:
        server.should_exit = True
        await asyncio.gather(web, return_exceptions=True)
        for p in agent.pollers:
            await p.close()


def _int_env(name: str) -> int | None:
    v = os.environ.get(name)
    return int(v) if v else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mic", action="store_true", help="no microphone; type in the web UI")
    asyncio.run(main(no_mic=ap.parse_args().no_mic))
