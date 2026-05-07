# Trading Dashboard

Single web dashboard over the Kalshi paper-trading bot portfolio. Runs as
a separate process on Digital Ocean and reads each bot's SQLite DB
directly (sqlite supports concurrent readers, so this is safe alongside
the live bots).

Each bot in the portfolio writes the same schema:
- `model_snapshots` — what the model thinks right now
- `market_views` — per-ticker buy verdict and feature audit
- `positions` / `position_marks` — open and closed paper trades
- `training_pairs` — closed-bet feedback used by the live recalibrator

This dashboard reads those tables, rolls them up across bots, and serves
one HTML page on port 8080. Stdlib only (`http.server`) — no Flask/FastAPI.

## Bots currently registered

| Key                  | Name                | Source repo                        |
| -------------------- | ------------------- | ---------------------------------- |
| `gas-prices`         | Gas Prices          | `~/Documents/Kalshi/Retail Gas Prices` |
| `unemployment-claims`| Unemployment Claims | `~/Documents/Kalshi/Unemployment Claims` |
| `natural-gas`        | Natural Gas         | `~/Documents/Kalshi/Natural Gas Prices` |
| `whale-watcher`      | Whale Watcher       | `~/Documents/Kalshi/Whale Watcher` |

Add or remove bots by editing `config/dashboard.yaml` — no code change needed.

## Rules Intel tab

The dashboard also surfaces signals from the rules-parser daemon
(`~/Documents/Kalshi/Rules Parser`). The Rules Intel tab is a read-only
viewer over `rules_intel.db` and a `/api/rules-intel` JSON endpoint is
exposed for external consumers.

Configure in `config/dashboard.yaml`:

```yaml
rules_intel:
  enabled: true
  db_path: /root/rules-parser/data/rules_intel.db
```

The dashboard never imports the rules-parser trading code, so this
tab can never place an order.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# point db_path at local sim.db files for whichever bots you want to see
$EDITOR config/dashboard.yaml

python dashboard.py
# open http://localhost:8080
```

`db_path` entries that don't exist render a friendly stub — fine for
local dev where only one bot is populated.

## Layout

```
dashboard.py                          # entrypoint
config/
  dashboard.yaml                      # bot registry + display thresholds
src/trading_dashboard/
  dashboard.py                        # the server, HTML render, JSON snapshot
  config.py                           # YAML loader → typed dataclasses
  logging_setup.py
deploy/
  trading-dashboard.service           # systemd unit for Digital Ocean
  deploy.sh                           # one-shot redeploy script for the droplet
```

## Adding a new bot

1. Make sure the bot writes a `data/sim.db` with at minimum a
   `model_snapshots` table. Other tables (`positions`, `market_views`,
   `position_marks`, `training_pairs`) are all optional — the dashboard
   degrades gracefully via `_safe_query`.
2. Append an entry to `bots:` in `config/dashboard.yaml`:
   ```yaml
   - key: my-new-bot
     name: My New Bot
     db_path: /root/my-new-bot/data/sim.db
     decisions_path: /root/my-new-bot/data/decisions.jsonl
   ```
3. `systemctl restart trading-dashboard` on the droplet.

## Deploy (Digital Ocean)

See `deploy/README.md`. TL;DR:

```bash
ssh root@<droplet>
cd /root && git clone git@github.com:DanielPearl/trading-dashboard.git
cd trading-dashboard
pip3 install -r requirements.txt
cp deploy/trading-dashboard.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now trading-dashboard
```

For redeploys on top of an existing checkout: `bash deploy/deploy.sh`.
