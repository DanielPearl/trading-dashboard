"""Live-dashboard data-file bootstrap.

The live config points every bot at a parallel set of data paths
(``data/live.db``, ``data/outputs-live/watchlist.json``, etc.) so
live runtime state is fully isolated from sim. The trade-off is
that on a fresh install — or just before the first live executor
for a given bot is written — none of those files exist, and the
dashboard's renderers (which were designed around "if the db
doesn't exist, render unavailable") show every bot as broken.

That hides what's actually true: the bot's MODEL ITSELF still
exists. It's the same trained model on disk, the same training
artifacts (metrics.json, feature_importance.csv, ROC curve data,
calibration data, coefficients) shared with sim. The user should
see the model card populated — only the live RUNTIME sections
(positions, P&L, current predictions) should show empty.

This module makes that happen by ensuring each live bot's data
files exist before serve() starts. Each file is:

  * Created empty (with the right schema for SQLite, with the
    empty-shell JSON the renderer expects for JSON files).
  * Left alone if it already exists — operator data is never
    clobbered.
  * Sourced FROM the sibling sim file when possible — for
    SQLite this means we copy sim.db's table definitions into
    live.db so the standard renderer's SELECTs return empty rows
    instead of erroring on missing tables.

Called once at startup from dashboard.py main() when mode='live'.
Idempotent and safe to call repeatedly.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


log = logging.getLogger("dashboard.live_bootstrap")


# Empty shells for the two JSON shapes the renderers consume. Keep
# the field set small — anything missing falls through to the
# renderer's existing graceful "no value" handling.
_EMPTY_WATCHLIST = {"generated_at": "", "rows": []}
_EMPTY_SIM_STATE = {
    "started_at": "",
    "last_tick_at": "",
    "open_positions": [],
    "closed_positions": [],
    "stats": {
        "total_opened": 0,
        "total_closed": 0,
        "open_count": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "total_realized_pnl": 0.0,
        "total_unrealized_pnl": 0.0,
        "total_staked": 0.0,
        "roi": None,
    },
    "last_settled_at_by_match_id": {},
}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _sim_path_for(live_path: Path) -> Path | None:
    """Guess the sibling sim path for a given live path. We follow
    the rewriting convention used by config/dashboard-live.yaml:

      .../data/live.db                → .../data/sim.db
      .../data/outputs-live/<file>    → .../data/outputs/<file>

    Returns None if no recognised live → sim transformation
    applies. The caller treats that as "don't try to mirror; just
    create an empty file".
    """
    s = str(live_path)
    if s.endswith("/live.db") or s.endswith("\\live.db"):
        return Path(s[: -len("live.db")] + "sim.db")
    if "/outputs-live/" in s:
        return Path(s.replace("/outputs-live/", "/outputs/"))
    return None


def _ensure_live_sqlite(live_path: Path) -> bool:
    """Create ``live_path`` if it doesn't exist. When a sibling
    sim.db exists, copy its CREATE TABLE / CREATE INDEX statements
    so the live file has the same empty schema — that lets the
    standard renderer's ``SELECT * FROM positions WHERE ...`` style
    queries return zero rows instead of erroring on a missing
    table. Returns True if anything was created.
    """
    if live_path.exists():
        return False
    live_path.parent.mkdir(parents=True, exist_ok=True)
    sim_path = _sim_path_for(live_path)
    # Always create the file, even if no sim sibling — at minimum
    # the file's existence flips the renderer past its early-exit.
    if sim_path is None or not sim_path.exists():
        # No schema to copy; just touch an empty SQLite file. The
        # renderer's reads will error per-query but the bot card
        # will at least render rather than getting bot_unavailable.
        sqlite3.connect(str(live_path)).close()
        log.info("created empty live SQLite at %s (no sim sibling to "
                 "mirror schema from)", live_path)
        return True
    try:
        src = sqlite3.connect(str(sim_path))
        dst = sqlite3.connect(str(live_path))
        try:
            # CREATE TABLE / CREATE INDEX statements live in
            # sqlite_master.sql. We replay them verbatim against
            # the empty live file. Skip the sqlite_sequence entry
            # SQLite injects automatically when AUTOINCREMENT is
            # used — that table gets recreated on demand.
            cur = src.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE sql IS NOT NULL AND name != 'sqlite_sequence'",
            )
            ddl_statements = [row[0] for row in cur.fetchall()]
            for ddl in ddl_statements:
                try:
                    dst.execute(ddl)
                except sqlite3.OperationalError as e:
                    log.warning("live_bootstrap: skipping DDL "
                                 "(%s): %s", e, ddl[:80])
            dst.commit()
            log.info("created live SQLite at %s mirroring schema "
                     "from %s (%d tables/indexes)",
                     live_path, sim_path, len(ddl_statements))
        finally:
            src.close()
            dst.close()
    except Exception:  # noqa: BLE001
        log.exception("live_bootstrap: failed to mirror schema "
                       "%s → %s", sim_path, live_path)
    return True


def _ensure_live_json(live_path: Path, shape: dict) -> bool:
    """Write an empty-shell JSON at ``live_path`` if it doesn't
    exist. ``shape`` is one of ``_EMPTY_WATCHLIST`` /
    ``_EMPTY_SIM_STATE`` so the dashboard's readers find the
    expected top-level keys (``rows``, ``open_positions``, etc.)
    and render empty sections instead of crashing on KeyError.
    """
    if live_path.exists():
        return False
    _atomic_write_json(live_path, shape)
    log.info("created live JSON placeholder at %s", live_path)
    return True


def bootstrap_live_data_files(bots: Iterable[dict]) -> int:
    """Walk the bot registry. For each bot, create whichever live
    data file(s) it expects but doesn't yet have. Returns the
    total number of files created (0 on every restart after the
    first one — bootstrap is purely a first-boot affordance).
    """
    created = 0
    for b in bots:
        db_path = b.get("db_path")
        watchlist = b.get("watchlist_json_path")
        sim_state = b.get("sim_state_path")
        # db_path can be either a .db (standard / billboard) or a
        # .json (tennis-shape sport bots — they treat the JSON as
        # their "DB" for the dashboard's availability check).
        if db_path:
            p = Path(db_path)
            if str(p).endswith(".db"):
                if _ensure_live_sqlite(p):
                    created += 1
            elif str(p).endswith(".json"):
                if _ensure_live_json(p, _EMPTY_WATCHLIST):
                    created += 1
        if watchlist and watchlist != db_path:
            if _ensure_live_json(Path(watchlist), _EMPTY_WATCHLIST):
                created += 1
        if sim_state:
            if _ensure_live_json(Path(sim_state), _EMPTY_SIM_STATE):
                created += 1
    if created:
        log.info("bootstrap_live_data_files — created %d placeholder "
                 "files; live dashboard cards will render with "
                 "training-artifact metadata + empty runtime sections "
                 "until each bot's live executor starts writing", created)
    return created
