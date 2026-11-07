# AI 231 — Machine Exercises

Coursework for AI 231. Each machine exercise is self-contained in its own folder.

| ME | Topic | Result |
|----|-------|--------|
| [ME1 - Einops/Einsum](me1/) | 3-layer CNN on MNIST, all layers built from `einops`/`einsum` | 99.09% test accuracy |
| [ME2 - Voice Command Model](me2/) | On-device voice commands for a Raspberry Pi: a 685k-parameter CNN trained from scratch on the class's Hugging Face dataset alone, 19 commands, behind a "Marvin" wake word | 91% of the class test's commands right at the demo threshold, 0.05 unwanted actions per hour |
| [ME3 - Full-Duplex Voice Agent](me3/) | Raspberry Pi 5 voice assistant you can interrupt: our own MCP servers (ASR, TTS, home, memory, delegate), cloud models via OpenRouter, backchannels and barge-in, a background coding harness (OpenCode) | Option B: 72% intent / 99% slot accuracy; Full-Duplex-Bench false turn-taking 50% → 20%; 3.2 s mean voice-to-voice |

ME1 is a notebook committed with its outputs, so its results are visible without
re-running it. ME2 is a training pipeline with a Raspberry Pi demo:
[me2/README.md](me2/README.md) shows how to run the trained model, and how to get the
data and train it again.

ME3 is a full-duplex voice agent on a Raspberry Pi: [me3/README.md](me3/README.md) shows how to
set it up with your own API keys and how to run the benchmarks.
