#!/usr/bin/env bash
# Start the whole demo automatically when the Pi boots: MCP servers + voice agent + web UI.
# It also restarts by itself if it crashes.
#
#   scripts/install_service.sh            install + start now (asks for your sudo password)
#   scripts/install_service.sh uninstall  remove it again
#
# Everyday commands once installed:
#   sudo systemctl status me3        is it running?
#   journalctl -u me3 -f             live log (Ctrl+C to leave)
#   sudo systemctl restart me3       after changing code/config (e.g. after an rsync)
#   sudo systemctl stop me3          BEFORE running a benchmark (they restart the servers)
#   sudo systemctl start me3         after the benchmark
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=$PWD
USER_NAME=$(id -un)

if [[ "${1:-}" == "uninstall" ]]; then
  sudo systemctl disable --now me3 2>/dev/null || true
  sudo rm -f /etc/systemd/system/me3.service
  sudo systemctl daemon-reload
  echo "me3 service removed."
  exit 0
fi

[[ -f .env ]] || { echo "No .env yet: create it first (see README)."; exit 1; }

sudo tee /etc/systemd/system/me3.service >/dev/null <<EOF
[Unit]
Description=ME3 voice agent (MCP servers + agent + web UI)
Wants=network-online.target
After=network-online.target sound.target

[Service]
User=$USER_NAME
WorkingDirectory=$DIR
ExecStart=$DIR/scripts/start_demo.sh
Restart=on-failure
RestartSec=5
KillMode=control-group
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now me3
sleep 8
systemctl --no-pager --lines=0 status me3 | head -5
ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "Installed. It now starts on every boot. Open http://${ip:-<pi-ip>}:8080 on the laptop."
echo "Live log: journalctl -u me3 -f"
