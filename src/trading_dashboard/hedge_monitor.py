"""Auto-hedge daemon for the dashboard.

Scans every registered sim.db bot's ``positions`` table on a fixed
interval. For each open position whose unrealized P&L per contract
crosses the global hedge thresholds (profit-lock or stop-loss), the
daemon closes the position in place — sets ``status='closed'``,
stamps an exit price + timestamp + realized_pnl_cents, and tags it
with ``error_type='hedge_pl'`` / ``'hedge_sl'`` so the History tab
clearly attributes the close to the hedge engine.

The user-facing effect: a hedged position drops out of the Summary's
active-bets table and appears on the History tab on the next page
load. No partial closes — the daemon's policy is "exit the whole
position when the threshold fires", which matches what the toggle-
off semantics imply (the bot has stopped taking signal-driven risk).

JSON-source bots (tennis, survivor, whale) keep their own state
files outside sim.db and are skipped here — their sim engines run
their own close logic.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.hedge_monitor")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _latest_mark_cents(c: sqlite3.Connection, position_id: int,
                        side: str) -> int | None:
    """Most recent quote for the position from ``position_marks``.

    For a YES position we want the YES side's mid as the price we'd
    sell at — that's the cash we'd get back if we exited now. For
    NO it's the NO side's mid. Falls back to the cents we have on
    hand if either mid is missing.
    """
    row = c.execute(
        "SELECT yes_ask_cents, yes_bid_cents, no_ask_cents, mid_cents "
        "FROM position_marks WHERE position_id = ? "
        "ORDER BY updated_at DESC LIMIT 1",
        (position_id,),
    ).fetchone()
    if not row:
        return None
    side = (side or "").upper()
    if side == "YES":
        # Sell YES at the bid (= what someone else will pay).
        cand = row["yes_bid_cents"] or row["mid_cents"]
    else:
        # Sell NO ≈ buy YES, so the exit value is 100 − yes_ask.
        ya = row["yes_ask_cents"]
        cand = (100 - int(ya)) if ya is not None else row["mid_cents"]
    if cand is None:
        return None
    try:
        return int(cand)
    except (TypeError, ValueError):
        return None


def _unrealized_pnl_per_contract(side: str, entry: int,
                                   exit_mark: int) -> int:
    """Profit per contract in cents at the exit_mark.

    YES bought at entry: gain = exit − entry (exit is the YES price
    we'd receive on a sell).
    NO bought at entry: the entry was 100 − yes_ask; current value
    is also 100 − yes_ask. Gain = (100 − yes_ask_now) − entry, but
    we pass exit_mark already converted to "value of the NO side
    right now" so the same gain = exit_mark − entry formula holds.
    """
    return int(exit_mark) - int(entry)


def _close_position(c: sqlite3.Connection, *, position_id: int,
                     entry: int, contracts: int, exit_mark: int,
                     side: str, reason: str) -> int:
    """Mark a position closed in the bot's sim.db. Returns the
    realized P&L in cents (signed)."""
    pnl_per_contract = _unrealized_pnl_per_contract(side, entry, exit_mark)
    realized_cents = pnl_per_contract * max(1, int(contracts or 1))
    c.execute(
        "UPDATE positions SET "
        "  status = 'closed', "
        "  exit_price_cents = ?, "
        "  exited_at = ?, "
        "  realized_pnl_cents = ?, "
        "  error_type = ? "
        "WHERE id = ? AND status = 'open'",
        (int(exit_mark), _now_iso(), realized_cents, reason, position_id),
    )
    return realized_cents


def _check_db(db_path: str, bot_key: str, bot_name: str,
                profit_lock_cents: int, stop_loss_cents: int,
                ) -> List[Dict[str, Any]]:
    """Scan one bot's sim.db. Returns the list of closes applied."""
    p = Path(db_path)
    if not p.exists():
        return []
    closed: List[Dict[str, Any]] = []
    with closing(sqlite3.connect(db_path)) as c:
        c.row_factory = sqlite3.Row
        # ``hedge_id`` is the column the existing bots stamp when
        # they place an internal hedge — we leave it alone so we
        # don't shadow that mechanism. Our daemon only operates on
        # truly-open positions that haven't been hedged yet.
        rows = c.execute(
            "SELECT id, ticker, side, entry_price_cents, contracts "
            "FROM positions "
            "WHERE status = 'open' AND (hedge_id IS NULL OR hedge_id = 0)"
        ).fetchall()
        if not rows:
            return []
        for row in rows:
            pos_id = int(row["id"])
            side = (row["side"] or "").upper()
            try:
                entry = int(row["entry_price_cents"])
            except (TypeError, ValueError):
                continue
            contracts = int(row["contracts"] or 1)
            mark = _latest_mark_cents(c, pos_id, side)
            if mark is None:
                continue
            pnl = _unrealized_pnl_per_contract(side, entry, mark)
            reason = None
            if profit_lock_cents > 0 and pnl >= profit_lock_cents:
                reason = "hedge_pl"
            elif stop_loss_cents > 0 and pnl <= -stop_loss_cents:
                reason = "hedge_sl"
            if not reason:
                continue
            realized = _close_position(
                c, position_id=pos_id, entry=entry,
                contracts=contracts, exit_mark=mark,
                side=side, reason=reason,
            )
            c.commit()
            closed.append({
                "bot_key": bot_key, "bot_name": bot_name,
                "position_id": pos_id, "ticker": row["ticker"],
                "side": side, "entry": entry, "exit": mark,
                "contracts": contracts, "pnl_per_contract": pnl,
                "realized_cents": realized, "reason": reason,
            })
    return closed


def tick(bots: List[dict], hedge_cfg: dict) -> List[Dict[str, Any]]:
    """One scan across every sim.db-style bot. Returns the closes
    applied this tick so the caller can log / report them."""
    if not hedge_cfg or not hedge_cfg.get("enabled"):
        return []
    pl = int(hedge_cfg.get("profit_lock_cents", 0) or 0)
    sl = int(hedge_cfg.get("stop_loss_cents", 0) or 0)
    if pl <= 0 and sl <= 0:
        return []
    closed: List[Dict[str, Any]] = []
    for b in bots:
        # JSON-source bots use their own sim engines outside sim.db.
        if b.get("dashboard_type") in ("tennis", "survivor", "whale"):
            continue
        db = b.get("db_path") or ""
        if not db:
            continue
        try:
            results = _check_db(db, b.get("key", ""), b.get("name", ""),
                                  profit_lock_cents=pl, stop_loss_cents=sl)
        except Exception:  # noqa: BLE001
            log.exception("hedge tick failed for %s", b.get("key"))
            continue
        for r in results:
            log.info(
                "[hedge] %s %s pos=%d side=%s entry=%d exit=%d pnl=%+d "
                "contracts=%d reason=%s",
                r["bot_name"], r["ticker"], r["position_id"], r["side"],
                r["entry"], r["exit"], r["pnl_per_contract"],
                r["contracts"], r["reason"],
            )
        closed.extend(results)
    return closed


def start_daemon(bots: List[dict], hedge_cfg: dict,
                  interval_seconds: int = 30) -> threading.Thread:
    """Spawn the hedge-monitor background thread. Daemon = True so
    SIGINT on the dashboard tears it down cleanly. No-op (returns a
    dead thread) when hedge.enabled is false."""

    def _loop() -> None:
        log.info("hedge_monitor started (interval=%ds, "
                  "profit_lock=%d¢, stop_loss=%d¢)",
                  interval_seconds,
                  hedge_cfg.get("profit_lock_cents", 0),
                  hedge_cfg.get("stop_loss_cents", 0))
        while True:
            try:
                tick(bots, hedge_cfg)
            except Exception:  # noqa: BLE001
                log.exception("hedge_monitor loop iteration failed")
            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, daemon=True,
                           name="hedge-monitor")
    if hedge_cfg and hedge_cfg.get("enabled"):
        t.start()
    return t
