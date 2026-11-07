#!/usr/bin/env bash
# Start all MCP servers (home, memory, asr, tts, delegate) in the background on localhost.
# Logs: data/logs/<server>.log   Stop: scripts/run_servers.sh stop   Restart: ... restart
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
# harness CLIs: the OpenCode installer uses ~/.opencode/bin, Hermes uses ~/.local/bin
export PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH"
mkdir -p data/logs
SERVERS=(home memory asr tts delegate)

if [[ "${1:-}" == "stop" || "${1:-}" == "restart" ]]; then
  for s in "${SERVERS[@]}"; do pkill -f "mcp_servers.$s.server" && echo "stopping $s" || true; done
  # wait until they have really exited (and freed their ports), up to ~10 s, then force
  for _ in $(seq 50); do
    pgrep -f "mcp_servers\.(home|memory|asr|tts|delegate)\.server" >/dev/null || break
    sleep 0.2
  done
  pkill -9 -f "mcp_servers\.(home|memory|asr|tts|delegate)\.server" 2>/dev/null || true
  echo "all stopped"
  [[ "${1:-}" == "stop" ]] && exit 0
fi

for s in "${SERVERS[@]}"; do
  if pgrep -f "mcp_servers.$s.server" >/dev/null; then echo "$s already running"; continue; fi
  nohup "$PY" -m "mcp_servers.$s.server" >"data/logs/$s.log" 2>&1 &
  echo "started $s (pid $!)"
done

# wait until every server answers on its port
for s in "${SERVERS[@]}"; do
  port=$("$PY" -c "from common.config import load_config; print(load_config()['mcp_servers']['$s']['port'])")
  for _ in $(seq 50); do
    (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null && break
    sleep 0.2
  done
  (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null && echo "$s ready on :$port" || { echo "$s FAILED (see data/logs/$s.log)"; exit 1; }
done
