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

# Copy the systemd unit if it's missing OR if our checked-in version
# differs from what's installed. The previous "first-install only"
# variant silently skipped MemoryMax / EnvironmentFile tweaks on
# redeploys; this version keeps the installed unit in sync.
if [[ ! -f /etc/systemd/system/trading-dashboard.service ]] || \
   ! cmp -s deploy/trading-dashboard.service /etc/systemd/system/trading-dashboard.service; then
  echo "→ installing/updating systemd unit"
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
