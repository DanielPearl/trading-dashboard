#!/usr/bin/env bash
# Redeploy the trading dashboard on the Digital Ocean droplet.
#
#   ssh root@<droplet>
#   cd /root/trading-dashboard
#   bash deploy/deploy.sh
#
# Pulls latest main, installs deps if requirements.txt changed, and
# restarts the systemd unit. Idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "→ git pull"
git fetch origin
git reset --hard origin/main

# Per-repo venv. Matches the convention used by the bot services on this
# droplet, and sidesteps the PEP 668 externally-managed warning that
# Bookworm's system python3 emits.
if [[ ! -d .venv ]]; then
  echo "→ creating .venv"
  python3 -m venv .venv
fi
echo "→ pip install -r requirements.txt"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

# First-time install: copy the unit file. On subsequent runs this is a no-op.
if [[ ! -f /etc/systemd/system/trading-dashboard.service ]]; then
  echo "→ installing systemd unit"
  cp deploy/trading-dashboard.service /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable trading-dashboard
fi

echo "→ restart trading-dashboard"
systemctl restart trading-dashboard
sleep 1
systemctl --no-pager status trading-dashboard | head -15

echo
echo "→ tail journal (^C to exit):"
journalctl -fu trading-dashboard -n 20
