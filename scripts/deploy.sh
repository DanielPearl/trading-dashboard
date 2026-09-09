#!/usr/bin/env bash
# One-command droplet deploy — pulls EVERY repo the two dashboard
# services import from, restarts BOTH services, and sanity-checks the
# panes. Exists because partial deploys were the most common breakage
# class (2026-09-09 audit): one service restarted but not the other,
# or a bot repo pulled without kalshi_sdk.
set -euo pipefail

REPOS=(
  /root/kalshi_sdk
  /root/trading-dashboard
  /root/cpi
  /root/unemployment-claims
  /root/nba
  /root/baseball
  /root/tennis-forecast
  /root/table-tennis-forecast
  /root/darts-forecast
  /root/weather-forecast
  /root/port-forecast
  /root/gas-prices
  /root/billboard-charts
)

for r in "${REPOS[@]}"; do
  if [ -d "$r/.git" ]; then
    echo "pull $r"
    git -C "$r" pull -q || echo "  !! pull failed: $r"
  fi
done

systemctl restart trading-dashboard-live trading-dashboard-sim
systemctl is-active trading-dashboard-live trading-dashboard-sim

# Wait for both HTTP servers to actually listen (up to 90s) — the
# services report active before the socket binds, and check_panes
# against a half-started server flags every pane as broken.
for port in 8081 8080; do
  for _ in $(seq 1 45); do
    if curl -s --max-time 2 -o /dev/null "http://127.0.0.1:$port/"; then
      echo "port $port up"; break
    fi
    sleep 2
  done
done

# Post-deploy regression check — flags any bot pane that renders
# empty while its data says it shouldn't.
if [ -f /root/trading-dashboard/scripts/check_panes.py ]; then
  /root/trading-dashboard/.venv/bin/python \
    /root/trading-dashboard/scripts/check_panes.py || true
fi
