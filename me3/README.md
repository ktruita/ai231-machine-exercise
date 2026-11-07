# ME3: Full-Duplex Voice Agent on Raspberry Pi 5

A natural, interruptible voice assistant that runs on a Raspberry Pi 5. All model inference happens remotely; the only model on the Pi is Silero VAD.

- **Easy tasks** (the 19 intents in `Dataset Schema(1).xlsx` → Option B) run as fast MCP tool calls.
- **Hard tasks** (build a website, write a report) are delegated to an agent harness (OpenCode or Hermes) in the background. You can keep talking while they run, and the agent tells you when they're done.
- **Backchanneling**: "Mhm?" when you trail off, "Yes?" when you just say "hey", and instant acks like "Let me check that".
- **Barge-in**: saying "mhm" or "yeah" while the agent talks does *not* stop it; "wait", "stop" or a new question does.
- **Benchmark**: a standardized latency, accuracy and naturalness benchmark built from one shared command bank.

```
Headset mic ─► VAD (Silero, on Pi) ─► asr-mcp ─► backchannel-aware turn taking ─► BackchannelGate
   ─► EventAnnouncer ─► LLM (OpenRouter) ─► ToolAck ─► tts-mcp ─► headset
                         │ MCP tools
           home-mcp (19 intents) · memory-mcp · delegate-mcp ─► OpenCode / Hermes
```

All five MCP servers are our own and run on the Pi (streamable HTTP on localhost). Any MCP-capable harness can also connect to them.

| Server | Port | Wraps |
|---|---|---|
| `home` | 8101 | music (mpv), weather (Open-Meteo), time, lights, temperature, call/message (simulated), timers, alarms, reminders |
| `memory` | 8102 | remember / recall / forget (SQLite FTS, no embeddings) |
| `asr` | 8103 | Deepgram Nova-3 (default), Groq Whisper (fallback) |
| `tts` | 8104 | Deepgram Aura (default), Cartesia, Edge TTS (free, no key) |
| `delegate` | 8105 | background harness tasks: OpenCode, Hermes |

## Using this repo

This is the demo version of ME3. To run it you need your own keys:

1. `cp .env.example .env` and fill in `OPENROUTER_API_KEY` (paid, the only required one) and
   `DEEPGRAM_API_KEY` (free credit). `.env` is git-ignored: never commit it.
2. Follow **Setup** and **Run** below (Raspberry Pi 5, USB headset; see `scripts/setup_pi.sh`).
3. On a Pi, `scripts/install_service.sh` starts everything on boot.

Not included on purpose:
- `.env` (API keys), `data/` (databases, logs) and `workspaces/` (generated sites): created at run time.
- The class files `Dataset Schema(1).xlsx` and the slide template. The test bank built from the
  spreadsheet is included (`benchmark/bank/commands.jsonl`); put the spreadsheet in this folder only
  to rebuild it. The bank's voice clips are generated, not stored: `python -m benchmark.render_audio`.
- `fd-benchmark/`: clone the class benchmark yourself (see **Class benchmark** below).

Music in `music/` is public domain / CC0 (sources in `music/CREDITS.txt`).

## Setup

On the Pi:
```bash
scripts/setup_pi.sh          # apt deps, venv, OpenCode; lists audio devices
nano .env                    # API keys + headset device indexes
```

On a laptop for development (no mic needed):
```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
```

Keys: `OPENROUTER_API_KEY` is the only paid one. Deepgram (free credit) is used for ASR and TTS by default. To run with no speech keys at all, set `tts.backend: edge_tts` and `asr.backend: groq_whisper` (Groq has a free tier) in `config.yaml`.

Models are set in `config.yaml`. Switching LLM is a one-line change to any OpenRouter id:
- `llm.model` is the fast, cheap voice model.
- `delegate.harness_model` is the stronger model used only by harness tasks.

## Run

On the Pi (e.g. over `ssh pi@<pi-ip>`):
```bash
scripts/run_servers.sh       # start all MCP servers (logs in data/logs/)
.venv/bin/python -m agent.main
scripts/run_servers.sh stop
```

## Web UI: talk into the Pi's headset, hear the agent on your laptop

1. Find the Pi's address with `hostname -I` on the Pi.
2. On the laptop, open `http://<pi-ip>:8080`.
3. Click **▶ Start audio** once. Browsers block sound until you click.

The headset mic on the Pi listens; the agent's voice plays through the laptop speakers. The page shows:
- **Status**: listening, speaking, thinking.
- **Mic meter**: live headset level; green "hearing you" while the agent detects speech, red "no mic" if no audio arrives.
- **Conversation**: a live transcript. Instant backchannels are dashed; ignored "mhm" overlaps are struck through; tool calls and barge-in decisions are shown inline.
- **Background tasks**: harness, status, elapsed time, live progress (files made so far, current step) and summary, with **view files** (a generated website opens directly) and **cancel**. Follow-ups ("make it blue") are shown as ↳ and edit the same folder.
- **Home**: lights, brightness, color, temperature, music, timers/alarms, reminders.
- **Latency**: per-turn ASR, end-of-turn, LLM, TTS and voice→voice; median barge-in; Pi→laptop playback delay.
- **Text box**: type instead of speaking. Typing while the agent talks interrupts it.

Settings live in `config.yaml`:
- `audio_out: browser | local`. With `local`, audio plays on the Pi headset instead.
- `ui.port`
- `ui.token`: require `?token=…` in the URL if the network is shared.

**Echo:** if the laptop speakers are close to the headset mic, the agent can hear itself and keep interrupting. Two fixes:
- Tick **Mute mic while agent speaks** in the UI. This gives up barge-in.
- Raise `vad.min_volume` (e.g. 0.75).

**Start on boot (recommended for demos):** `scripts/install_service.sh` installs a system service
that runs `scripts/start_demo.sh` at every boot and restarts it if it crashes. Plug in the Pi, wait
~30 s, open the page. Log: `journalctl -u me3 -f`. Stop it (`sudo systemctl stop me3`) before running
a benchmark, since benchmarks restart the MCP servers; start it again afterwards.

No mic, e.g. for UI development on a laptop: `python -m agent.main --no-mic`, then type in the page.

**Harness commands:** check the `delegate.harnesses` command templates in `config.yaml` against your installed versions:
- OpenCode: `opencode run -m openrouter/<model> …`
- Hermes: `hermes chat --provider openrouter -q …`

Both run one-shot in a fresh `workspaces/<task>/` folder and are asked to write `SUMMARY.md`, which the agent reads aloud.

## Class benchmark (fd-benchmark, Full-Duplex-Bench)

`fdbench` runs on the **Windows laptop** (plays clips, records, scores). The Pi runs only the
agent through their launcher, using our entry file `fdbench_entry.py`. No mic/speaker involved.

**Laptop (Windows PowerShell, not WSL; the Pi must be able to connect in on port 8791):**
```powershell
git clone https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking.git fd-benchmark
cd fd-benchmark
scripts\install_windows.ps1
python tools/fetch_datasets.py
scripts\open_firewall_windows.ps1        # as administrator
copy .env.example .env                   # put OPENROUTER_API_KEY in it (for the judge)
```

**Pi (inside ~/me3):**
```bash
git clone https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking.git fd-benchmark
scripts/run_servers.sh
export FDBENCH_ENTRY=$PWD/fdbench_entry.py
.venv/bin/python fd-benchmark/launch_assistant.py --check      # must print READY
```

**Run one round:** start fdbench on the laptop first, then the launcher on the Pi:
```powershell
python -m fdbench run --student <your-student-no> --rounds 1 --connect direct --where remote `
  --judge-provider openrouter --judge-model openai/gpt-4.1-mini --recordings f
```
```bash
.venv/bin/python fd-benchmark/launch_assistant.py --once --url ws://<laptop-ip>:8791 \
  --stats-procs asr_mcp=mcp_servers.asr,tts_mcp=mcp_servers.tts,tools_mcp=mcp_servers.home
```
Results: `fd-benchmark\runs\<student>\<date-time>\results.html`. Our own per-stage latency log
is written next to theirs (`latency.jsonl` in the launcher's log dir).

Benchmark-mode behaviour (see `config.yaml` → `turn`):
- the conversation is forgotten after 3 s of quiet (`ME3_RESET_IDLE_S`), so trials are independent;
- `addressed_check` makes the agent stay silent for TV / side talk (false turn-taking);
- speech quieter than −31 dBFS (`ME3_FAINT_BELOW_DBFS`) is treated as TV / someone across the room:
  it is tagged "(faint, distant voice)" for the LLM, and while the agent talks it never interrupts
  (the agent keeps talking, measured live by `LevelMeter`, no ASR wait). Measured on the Pi:
  users −14 to −26 dBFS, TV/background −35.7 to −39.7;
- speech that opens by naming someone else ("Derek, …", "Yo Mark, …", "Roomie, …") is side talk:
  the LLM is told not to act on it (continue a cut-off answer, otherwise stay silent);
- loud speech over the agent for 0.8 s (`turn.steer_after_s`) interrupts it; shorter bursts are
  judged by their words ("mhm", "totally" = keep talking);
- the reply is spoken in whole sentences (very short ones are joined, lists/markdown stripped,
  `llm.max_tokens` caps the length);
- if the link to fdbench stalls, the session ends and the launcher reconnects;
- `ME3_SPEECH_TIMEOUT=0.8` (env) tries a longer end-of-turn wait: fewer replies during
  mid-sentence pauses, slower turn-taking.

## Our own benchmark (Option B intents, delegation, harness)

```bash
.venv/bin/python -m benchmark.build_bank       # 93 Option-B items + 17 conversational items
.venv/bin/python -m benchmark.render_audio     # user-voice WAVs (free Edge TTS); swap in your own recordings
scripts/run_servers.sh
.venv/bin/python -m benchmark.run_bench --models google/gemini-2.5-flash-lite openai/gpt-4.1-nano
.venv/bin/python -m benchmark.judge  benchmark/results/<stamp>/<config>
.venv/bin/python -m benchmark.report benchmark/results/<stamp>
```

The runner plays every bank item into the **real pipeline** in real time, using a file-audio transport with continuous background silence so VAD behaves as it does with a mic. It runs one pass per configuration (model × ASR × TTS). Useful flags: `--kinds intent overlap`, `--ids …`, `--limit N`, `--runs N`.

### Measurement method (standardized)

Every timestamp is wall-clock and comes from a single pipeline observer (`agent/latency_observer.py`). Each turn is anchored at **VAD end-of-speech** (`user_vad_stop`):

| Metric | Definition |
|---|---|
| **Voice→voice** (headline) | Bot audio starts − user end-of-speech. Includes the VAD silence window (`vad_stop_secs`). |
| ASR | Transcript available − VAD stop |
| End-of-turn | Turn-end decision − VAD stop (the silence timeout) |
| LLM TTFT | First LLM token − end-of-turn |
| TTS first | First audio chunk − first LLM token |
| Barge-in | User starts speaking while the bot talks → bot audio stops (steering only) |
| Delegate ack | Voice→voice for delegation requests (should be an instant cached ack) |
| Harness turnaround | `delegate_task` → harness exits (black box) |
| Intent / slot accuracy | First `home` tool call vs. Option-B label, compared on slot values |
| Backchannel vs. steering accuracy | "mhm/yeah/okay" overlaps must not interrupt; real interruptions must |
| Naturalness, brevity, task success, turn-taking | LLM-as-judge (1–5), using a different model from the agent's |

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Layout

```
agent/        main.py (pipeline + web UI), latency_observer.py, prompts/system.md,
              web/ server.py · hub.py · browser_output.py · ui_observer.py · static/index.html
              processors/ mcp_speech.py · interrupt_classifier.py · conversation.py
mcp_servers/  home · memory · asr · tts · delegate (+ adapters/)
benchmark/    build_bank · render_audio · file_transport · run_bench · judge · report
common/       config, MCP client helpers
config.yaml   all settings; secrets in .env
```
