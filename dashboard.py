"""Run the multi-bot trading dashboard.

    python dashboard.py                       # uses config/dashboard.yaml
    python dashboard.py --port 9090
    python dashboard.py --config /etc/dash.yaml

Reads each bot's sim.db directly. Safe to run alongside the bots —
sqlite supports concurrent readers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trading_dashboard.dashboard import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
