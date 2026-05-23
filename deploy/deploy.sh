#!/usr/bin/env bash
# Redeploy the trading dashboards on the Digital Ocean droplet.
#
#   ssh root@<droplet>
#   cd /root/trading-dashboard
#   bash deploy/deploy.sh
#
# Manages BOTH trading-dashboard-sim.service (port 8080, paper trading)
# and trading-dashboard-live.service (port 8081, real money — all bots
# disabled by default). Pulls latest main, installs deps if
# requirements.txt changed, syncs each systemd unit, and restarts
# both. Idempotent.
#
# To target just one service: bash deploy/deploy.sh sim
#                            bash deploy/deploy.sh live
set -euo pipefail

cd "$(dirname "$0")/.."

# Either "sim", "live", or "both" (default).
TARGET="${1:-both}"
case "$TARGET" in
  sim|live|both) ;;
  *) echo "usage: deploy.sh [sim|live|both]"; exit 2 ;;
esac

echo "→ git pull"
git fetch origin
git reset --hard origin/main

# Single shared venv for both services. Both processes import the
# same trading_dashboard package + bot modules; only the YAML they
# read at startup differs.
if [[ ! -d .venv ]]; then
  echo "→ creating .venv"
  python3 -m venv .venv
fi
echo "→ pip install -r requirements.txt"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

# Install or update each systemd unit only if its checked-in copy
# differs from /etc/systemd/system/. ``cmp -s`` returns 0 on
# identical, non-zero on different — the OR short-circuit means we
# only run the cp+daemon-reload+enable trio when the file actually
# changed.
sync_unit() {
  local svc="$1"
  local src="deploy/${svc}.service"
  local dst="/etc/systemd/system/${svc}.service"
  if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
    echo "→ installing/updating ${svc}.service"
    cp "$src" "$dst"
    systemctl daemon-reload
    systemctl enable "$svc"
  fi
}

# Pre-create the live data dir for bot_states_live.json so the
# dashboard's atomic-write path has somewhere to land on first
# toggle click. Cheap, idempotent.
mkdir -p data

restart_one() {
  local svc="$1"
  echo "→ restart ${svc}"
  systemctl restart "$svc"
  sleep 1
  systemctl --no-pager status "$svc" | head -10
  echo
}

# Migration: if the legacy trading-dashboard.service is still
# installed from before the sim/live split, stop + disable it so it
# doesn't compete for port 8080 with trading-dashboard-sim.service.
if [[ -f /etc/systemd/system/trading-dashboard.service ]]; then
  echo "→ stopping + disabling legacy trading-dashboard.service"
  systemctl stop trading-dashboard.service 2>/dev/null || true
  systemctl disable trading-dashboard.service 2>/dev/null || true
fi

if [[ "$TARGET" == "sim" || "$TARGET" == "both" ]]; then
  sync_unit trading-dashboard-sim
  restart_one trading-dashboard-sim
fi
if [[ "$TARGET" == "live" || "$TARGET" == "both" ]]; then
  sync_unit trading-dashboard-live
  restart_one trading-dashboard-live
fi

echo
echo "Both dashboards should now be live:"
echo "  sim  → http://$(hostname -I | awk '{print $1}'):8080"
echo "  live → http://$(hostname -I | awk '{print $1}'):8081"
echo
echo "Tail journals with:"
echo "  journalctl -fu trading-dashboard-sim"
echo "  journalctl -fu trading-dashboard-live"
