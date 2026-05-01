# Deploy notes

## First-time install on a fresh droplet

```bash
ssh root@<droplet>

# 1. Clone the repo. Requires the droplet's deploy key to be added to GitHub.
cd /root
git clone git@github.com:DanielPearl/trading-dashboard.git
cd trading-dashboard

# 2. Install Python deps. PyYAML is the only one — stdlib does the rest.
pip3 install -r requirements.txt

# 3. Install + start the systemd unit.
cp deploy/trading-dashboard.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now trading-dashboard

# 4. Confirm it's listening on :8080.
curl -s localhost:8080/healthz
journalctl -fu trading-dashboard
```

## Cutover from the old gas-bot dashboard

The old dashboard was packaged inside the gas-prices repo and ran out of
`/root/gas-prices/dashboard.py`. To switch:

```bash
# 1. Stop and disable the old unit (whatever its name is — likely `gas-dashboard`
#    or `dashboard`). Replace OLD_UNIT below with the actual name.
systemctl stop OLD_UNIT
systemctl disable OLD_UNIT

# 2. Install the new dashboard (steps 1–3 above).

# 3. Confirm the new dashboard sees every bot. The "registered bots" log
#    line lists each one and whether its DB exists:
journalctl -u trading-dashboard | grep "registered bots"
```

The two dashboards bind the same port, so the old one must be stopped
before the new one starts.

## Redeploy after a `git push`

```bash
ssh root@<droplet>
cd /root/trading-dashboard
bash deploy/deploy.sh
```

The script fast-forwards to `origin/main`, reinstalls deps, and
`systemctl restart`s. Tails the journal at the end — Ctrl-C to exit.
