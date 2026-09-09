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
sleep 4
systemctl is-active trading-dashboard-live trading-dashboard-sim

# Post-deploy regression check — flags any bot pane that renders
# empty while its data says it shouldn't.
if [ -f /root/trading-dashboard/scripts/check_panes.py ]; then
  /root/trading-dashboard/.venv/bin/python \
    /root/trading-dashboard/scripts/check_panes.py || true
fi
