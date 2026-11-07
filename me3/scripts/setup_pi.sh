#!/usr/bin/env bash
# One-time setup on Raspberry Pi OS (Bookworm, 64-bit) for a Pi 5.
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y python3-dev portaudio19-dev mpv git curl

# uv (fast Python package manager)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt

# Harnesses (API-only, no local inference). Both use OpenRouter via OPENROUTER_API_KEY.
command -v node >/dev/null || { curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -; sudo apt-get install -y nodejs; }
export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"
command -v opencode >/dev/null || curl -fsSL https://opencode.ai/install | bash        # OpenCode 1.x
command -v hermes >/dev/null || curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-browser
echo "Then give both your OpenRouter key: 'opencode auth login' (choose OpenRouter) and"
echo "'hermes config set OPENROUTER_API_KEY <key>'."

[[ -f .env ]] || { cp .env.example .env; echo "Created .env — fill in your API keys."; }

echo
echo "Audio devices (set ME3_INPUT_DEVICE / ME3_OUTPUT_DEVICE in .env to your headset's index):"
.venv/bin/python - <<'PY'
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    print(f"  [{i}] {d['name']}  in={d['maxInputChannels']} out={d['maxOutputChannels']}")
PY
