#!/usr/bin/env bash
# One command for the demo: start the MCP servers + voice agent, print the UI address.
# Ctrl+C stops everything.   Usage: scripts/start_demo.sh [--no-mic]
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f .env ]] || { echo "No .env yet: cp .env.example .env and add your API keys."; exit 1; }
[[ -x .venv/bin/python ]] || { echo "Not installed yet: run  bash scripts/setup_pi.sh"; exit 1; }

scripts/run_servers.sh
trap 'echo; echo "Stopping..."; scripts/run_servers.sh stop' EXIT

port=$(.venv/bin/python -c "from common.config import load_config; print(load_config().get('ui', {}).get('port', 8080))")
ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "=============================================================="
echo "  Open on your laptop:  http://${ip:-<pi-ip>}:${port}"
echo "  Then click 'Start audio' and talk into the headset."
echo "  Ctrl+C here to stop."
echo "=============================================================="
echo

.venv/bin/python -m agent.main "$@"
