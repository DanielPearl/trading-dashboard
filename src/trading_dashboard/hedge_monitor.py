"""Auto-hedge + auto-settle daemon for the dashboard.

Scans every registered bot's open positions on a fixed interval. Two
trigger paths can close a position:

  1. Hedge: unrealized P&L per contract crosses the global
     profit-lock or stop-loss threshold.
  2. Stale: the position's Kalshi-encoded ticker date is more than
     1 hour past — the underlying contract has settled but the
     bot's own close path missed it (live state rotated away from
     the match before the bot saw winner_side). Close at the
     latest mark we have on file so the position lands in
     History instead of lingering on the active-bets table.

The closed position drops out of the Summary's active-bets table
and appears on the History tab on the next page load.

Per-bot adapters:
  - sim.db bots (gas, claims, nat-gas, CPI, NBA):
      UPDATE positions SET status='closed', exit_price_cents,
      exited_at, realized_pnl_cents, error_type='hedge_pl'|'hedge_sl'.
  - tennis / survivor (sim_state.json):
      Move position from open_positions → closed_positions with
      closed_at, exit_market_prob, realized_pnl, and
      exit_reason='hedge_pl'|'hedge_sl'. Atomic write via tempfile +
      os.replace so the bot's own tick never sees a torn file.
  - whale (orders.jsonl):
      Append a ``hedge_close`` order record. Whale doesn't track
      per-position P&L explicitly — the daemon mirrors the sim.db
      math against the latest market quote and appends to the log.

No partial closes — the daemon's policy is "exit the whole position
when the threshold fires."
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
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


def _ticker_is_stale(ticker: str | None, grace_minutes: int = 15) -> bool:
    """True when the Kalshi ticker's encoded settlement date is more
    than ``grace_minutes`` in the past — the contract has settled
    but the position is still ``status='open'``.

    Lives here (not in dashboard.py) to keep hedge_monitor self-
    contained. Falls back to importing from dashboard.py when
    available; otherwise re-implements the same parser inline.
    """
    if not ticker:
        return False
    try:
        from .dashboard import minutes_to_close_from_ticker
    except Exception:  # noqa: BLE001
        return False
    mtc = minutes_to_close_from_ticker(ticker)
    if mtc is None:
        return False
    return mtc < -grace_minutes


def _check_db(db_path: str, bot: Dict[str, Any],
                profit_lock_cents: int, stop_loss_cents: int,
                ) -> List[Dict[str, Any]]:
    """Scan one bot's sim.db. Returns the list of closes applied."""
    bot_key = bot.get("key", "")
    bot_name = bot.get("name", "")
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
        # Pull model_yes_prob_at_entry when the schema has it — the
        # in-game model uses it as the pre-game prior for the
        # divergence / reversion features. Tolerated absence via the
        # PRAGMA probe so older sim.dbs still work.
        col_names = {r["name"] for r in
                      c.execute("PRAGMA table_info(positions)").fetchall()}
        prior_col = ("model_yes_prob_at_entry"
                      if "model_yes_prob_at_entry" in col_names else "NULL")
        rows = c.execute(
            f"SELECT id, ticker, side, entry_price_cents, contracts, "
            f"  {prior_col} AS model_yes_prob_at_entry "
            f"FROM positions "
            f"WHERE status = 'open' AND (hedge_id IS NULL OR hedge_id = 0)"
        ).fetchall()
        if not rows:
            return []
        for row in rows:
            pos_id = int(row["id"])
            ticker = row["ticker"] or ""
            side = (row["side"] or "").upper()
            try:
                entry = int(row["entry_price_cents"])
            except (TypeError, ValueError):
                continue
            contracts = int(row["contracts"] or 1)
            mark = _latest_mark_cents(c, pos_id, side)
            reason: str | None = None
            pnl = 0
            # Stale path first — fires even when mark is None (use
            # entry as exit so the realized P&L is zero rather than
            # stranding the position forever).
            if _ticker_is_stale(ticker):
                exit_mark = mark if mark is not None else entry
                pnl = _unrealized_pnl_per_contract(side, entry, exit_mark)
                reason = "settled_auto"
            elif mark is not None:
                pnl = _unrealized_pnl_per_contract(side, entry, mark)
                if profit_lock_cents > 0 and pnl >= profit_lock_cents:
                    reason = "hedge_pl"
                elif stop_loss_cents > 0 and pnl <= -stop_loss_cents:
                    reason = "hedge_sl"
            if not reason:
                continue
            # Sport bots: pre-game stop-loss is blocked — pre-game
            # price drops are usually noise (injury rumors, lineup
            # leaks) that reverse by tip-off. Profit-lock is NOT
            # blocked: if the line moves heavily our way before the
            # game (we're sitting on real gains), lock them in.
            # Settled_auto exits proceed regardless.
            if reason == "hedge_sl":
                from . import sport_game_state
                allowed, gate_reason = sport_game_state.is_hedge_allowed(
                    bot, ticker,
                )
                if not allowed:
                    log.info(
                        "[hedge] skip %s %s reason=%s — sport gate: %s",
                        bot_name, ticker, reason, gate_reason,
                    )
                    continue

            # In-game model advisory layer. For sport bots that are
            # past the start-gate, consult the live model. It can:
            #   • pre-empt a close (ACTION_EXIT_NOW + confidence) when
            #     no threshold has fired yet — closes with reason
            #     ``ingame_exit``.
            #   • suppress a threshold close (ACTION_LET_RUN / HOLD)
            #     when the model expects the move to reverse or sees
            #     a market overreaction.
            # Non-sport bots get None from in_game.predict and skip
            # this block entirely. Settled_auto closes always proceed.
            from . import in_game as _in_game
            position_for_model = {
                "ticker": ticker,
                "side": side,
                "entry_price_cents": entry,
                "model_yes_prob_at_entry": row["model_yes_prob_at_entry"],
            }
            prediction = _in_game.predict(bot, position_for_model)
            if prediction is not None and prediction.confidence >= 0.5:
                from . import in_game as _ig  # noqa: F401
                from .in_game.base import (
                    ACTION_EXIT_NOW as _EXIT, ACTION_LET_RUN as _LET,
                    ACTION_HOLD as _HOLD,
                )
                if reason in ("hedge_pl", "hedge_sl") and \
                        prediction.recommended_action in (_LET, _HOLD):
                    log.info(
                        "[hedge] suppress %s %s reason=%s — in-game: %s",
                        bot_name, ticker, reason, prediction.reason,
                    )
                    continue
                if reason is None and \
                        prediction.recommended_action == _EXIT:
                    log.info(
                        "[hedge] pre-empt %s %s — in-game says exit: %s",
                        bot_name, ticker, prediction.reason,
                    )
                    reason = "ingame_exit"
                    if mark is None:
                        # If we don't have a mark, exit at entry — same
                        # idiom the stale path uses so the position lands
                        # in History with zero realized P&L rather than
                        # being stranded indefinitely.
                        pnl = 0
            exit_mark = mark if mark is not None else entry
            realized = _close_position(
                c, position_id=pos_id, entry=entry,
                contracts=contracts, exit_mark=exit_mark,
                side=side, reason=reason,
            )
            c.commit()
            closed.append({
                "bot_key": bot_key, "bot_name": bot_name,
                "position_id": pos_id, "ticker": ticker,
                "side": side, "entry": entry, "exit": exit_mark,
                "contracts": contracts, "pnl_per_contract": pnl,
                "realized_cents": realized, "reason": reason,
            })
    return closed


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically — concurrent readers (the bot's own tick)
    never see a torn file. Tempfile lands in the same directory so
    the os.replace stays on one filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent),
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    try:
        json.dump(payload, tmp, indent=2, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, path)


def _check_tennis_state(sim_state_path: str | None, bot: Dict[str, Any],
                         profit_lock_cents: int,
                         stop_loss_cents: int,
                         ) -> List[Dict[str, Any]]:
    """Hedge open positions in a tennis-shape sim_state.json (also
    works for the survivor bot — same schema).

    Position P&L in cents = (current_market_prob − entry_market_prob)
    × 100, applied to the bet's side. On a hedge close we move the
    position from ``open_positions`` to ``closed_positions`` with
    ``closed_at``, ``exit_market_prob``, ``realized_pnl``, and
    ``exit_reason='hedge_pl'|'hedge_sl'``. The bot's own ``_settle_
    position`` won't fire on these later because they're no longer
    in open_positions.

    SKIPPED in live mode. The tennis live executor in
    ``bots/tennis_live_executor.py`` is explicit that it holds
    positions to natural Kalshi expiry ("always settle, never
    partial-close") — it never sends a sell order. The hedge here
    only mutates the local JSON, which:
      (a) makes the dashboard's realized-P&L number on `hedge_sl` /
          `hedge_pl` exits diverge from the real Kalshi position, and
      (b) loops with the executor's ``_reconcile_positions`` catch-up
          branch, which restores the position from ``closed_positions``
          on the next tick because Kalshi still lists it open. Observed
          on 2026-05-26: KXATPMATCH-26MAY25QUICOM hit hedge_sl 30+
          times over 33 minutes before the underlying market finalized.
    Sim mode is left alone — the paper simulator has no Kalshi side
    to disagree with, so hedge_pl/sl exits there are correct.
    """
    bot_key = bot.get("key", "")
    bot_name = bot.get("name", "")
    if not sim_state_path:
        return []
    if "outputs-live" in str(sim_state_path):
        return []
    p = Path(sim_state_path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("hedge: failed to read %s: %s", p, exc)
        return []
    open_positions = state.get("open_positions") or []
    if not open_positions:
        return []
    still_open: List[Dict[str, Any]] = []
    newly_closed: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for pos in open_positions:
        entry = pos.get("entry_market_prob")
        mark = pos.get("current_market_prob")
        ticker = pos.get("match_id") or pos.get("ticker")
        if entry is None:
            still_open.append(pos)
            continue
        try:
            entry_f = float(entry)
        except (TypeError, ValueError):
            still_open.append(pos)
            continue
        # No mark → no hedge action. (We used to fall through to a
        # stale/settled_auto path here; that produced false-positive
        # closes on tennis positions whose encoded match date had
        # passed but whose contract was still live. Tennis-shape
        # bots manage their own settlement.)
        if mark is None:
            still_open.append(pos)
            continue
        try:
            mark_f = float(mark)
        except (TypeError, ValueError):
            still_open.append(pos)
            continue
        pnl_cents = (mark_f - entry_f) * 100.0
        reason: str | None = None
        # NOTE: the stale/settled_auto cleanup is intentionally
        # NOT applied to tennis-shape positions. The _ticker_is_stale
        # heuristic parses YYMMMDD from the ticker and treats that
        # date as the contract close — correct for sim.db bots but
        # wrong for KXATPMATCH / KXWTAMATCH where the encoded date
        # is the match DAY, not when the contract closes. Auto-
        # closing on that date created a feedback loop with the
        # tennis bot's own logic (bot re-opens the position every
        # tick, daemon closes it ~60s later, duplicate "lost"
        # entries pile up in History). Tennis-shape bots manage
        # their own settlement via watchlist status updates.
        if profit_lock_cents > 0 and pnl_cents >= profit_lock_cents:
            reason = "hedge_pl"
        elif stop_loss_cents > 0 and pnl_cents <= -stop_loss_cents:
            reason = "hedge_sl"
        if not reason:
            still_open.append(pos)
            continue
        # Sport gate: pre-match stop-loss is blocked — pre-match
        # price action is noisy (injury chatter, lineup news) and
        # often reverses. Profit-lock is NOT blocked: if the line
        # has moved heavily in our favour before the match, take
        # the win.
        if reason == "hedge_sl":
            from . import sport_game_state
            allowed, gate_reason = sport_game_state.is_hedge_allowed(
                bot, ticker,
            )
            if not allowed:
                log.info(
                    "[hedge] skip %s %s reason=%s — sport gate: %s",
                    bot_name, ticker, reason, gate_reason,
                )
                still_open.append(pos)
                continue
        # In-game model advisory layer. The tennis-shape bot already
        # produces a live probability per match (live_prob_a / _b);
        # the model here overlays market-state checks on top.
        from . import in_game as _in_game
        position_for_model = {
            "ticker": ticker,
            "match_id": pos.get("match_id"),
            "side": pos.get("side"),
            "entry_market_prob": pos.get("entry_market_prob"),
            "entry_price_cents": pos.get("entry_price_cents"),
        }
        prediction = _in_game.predict(bot, position_for_model)
        if prediction is not None and prediction.confidence >= 0.5:
            from .in_game.base import (
                ACTION_EXIT_NOW as _EXIT, ACTION_LET_RUN as _LET,
                ACTION_HOLD as _HOLD,
            )
            if reason in ("hedge_pl", "hedge_sl") and \
                    prediction.recommended_action in (_LET, _HOLD):
                log.info(
                    "[hedge] suppress %s %s reason=%s — in-game: %s",
                    bot_name, ticker, reason, prediction.reason,
                )
                still_open.append(pos)
                continue
            if reason is None and \
                    prediction.recommended_action == _EXIT:
                log.info(
                    "[hedge] pre-empt %s %s — in-game says exit: %s",
                    bot_name, ticker, prediction.reason,
                )
                reason = "ingame_exit"
        stake = float(pos.get("stake", 1.0) or 1.0)
        slippage = float(pos.get("slippage", 0.0) or 0.0)
        realized = stake * (mark_f - entry_f - slippage)
        closed_pos = {
            **pos,
            "closed_at": now,
            "exit_market_prob": round(mark_f, 4),
            "realized_pnl": round(realized, 4),
            "exit_reason": reason,
            # Tennis dashboard adapter inspects ``result`` to colour
            # the History row. Hedged exits aren't a clean win/loss,
            # so we mark them HEDGE so the adapter can highlight
            # them appropriately if it ever cares.
            "result": "HEDGE",
            "won": realized > 0,  # purely informational
        }
        newly_closed.append(closed_pos)
        actions.append({
            "bot_key": bot_key, "bot_name": bot_name,
            "position_id": pos.get("position_id"),
            "ticker": pos.get("match_id"),
            "side": pos.get("side"),
            "entry_cents": int(round(entry_f * 100)),
            "exit_cents": int(round(mark_f * 100)),
            "pnl_per_contract": int(round(pnl_cents)),
            "realized_pnl": realized,
            "reason": reason,
        })
    if not actions:
        return []
    # Persist updated state. Append to closed_positions, replace
    # open_positions with the remainder.
    state["open_positions"] = still_open
    state["closed_positions"] = (state.get("closed_positions") or []) + newly_closed
    # Refresh stats: total_closed, wins/losses (we mark wins by
    # realized_pnl > 0 since there's no settle outcome), total
    # realized P&L. Mirrors the tennis simulator's stats block.
    stats = state.setdefault("stats", {})
    closed_all = state["closed_positions"]
    wins = sum(1 for c in closed_all if c.get("won"))
    losses = sum(1 for c in closed_all if not c.get("won"))
    total_realized = sum(float(c.get("realized_pnl", 0) or 0) for c in closed_all)
    total_unrealized = sum(float(p.get("unrealized_pnl", 0) or 0)
                            for p in still_open)
    stats.update({
        "open_count": len(still_open),
        "total_closed": len(closed_all),
        "wins": wins, "losses": losses,
        "total_realized_pnl": round(total_realized, 4),
        "total_unrealized_pnl": round(total_unrealized, 4),
        "win_rate": (wins / (wins + losses)) if (wins + losses) else None,
        "total_staked": round(sum(float(c.get("stake", 0) or 0)
                                    for c in closed_all), 4),
    })
    _atomic_write_json(p, state)
    return actions


def _check_whale_orders(orders_path: str | None, bot_key: str,
                         bot_name: str, profit_lock_cents: int,
                         stop_loss_cents: int,
                         ) -> List[Dict[str, Any]]:
    """Hedge for the whale-watcher bot.

    Whale doesn't keep an open_positions ledger in the same shape as
    the other bots — its ``orders.jsonl`` is an append-only log of
    follow / hedge decisions. For each whale "follow" record without
    a matching "close" entry, the daemon computes the trade's
    current P&L vs the configured thresholds and appends a
    ``hedge_close`` row so the close lands in the History tab via
    the whale adapter.
    """
    if not orders_path:
        return []
    p = Path(orders_path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log.warning("hedge: failed to read %s: %s", p, exc)
        return []
    # Build a (ticker -> [follow_record, close_record?]) index.
    open_follows: Dict[str, Dict[str, Any]] = {}
    for ln in lines:
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        kind = (rec.get("kind") or rec.get("action") or "").lower()
        ticker = rec.get("ticker") or ""
        if kind in ("follow", "buy", "open") and ticker:
            open_follows[ticker] = rec
        elif kind in ("close", "hedge_close", "sell", "exit") and ticker:
            open_follows.pop(ticker, None)
    actions: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_lines: List[str] = []
    for ticker, rec in open_follows.items():
        entry = rec.get("entry_price_cents") or rec.get("price_cents")
        mark = rec.get("current_price_cents") or rec.get("mark_cents")
        side = (rec.get("side") or "").upper()
        if entry is None or mark is None:
            continue
        try:
            entry_i = int(entry)
            mark_i = int(mark)
        except (TypeError, ValueError):
            continue
        if side == "NO":
            pnl_cents = (100 - mark_i) - (100 - entry_i)
        else:
            pnl_cents = mark_i - entry_i
        reason: str | None = None
        if profit_lock_cents > 0 and pnl_cents >= profit_lock_cents:
            reason = "hedge_pl"
        elif stop_loss_cents > 0 and pnl_cents <= -stop_loss_cents:
            reason = "hedge_sl"
        if not reason:
            continue
        close_rec = {
            "kind": "hedge_close", "ticker": ticker,
            "side": side or "YES",
            "entry_price_cents": entry_i,
            "exit_price_cents": mark_i,
            "pnl_cents": pnl_cents,
            "reason": reason,
            "created_at": now,
        }
        new_lines.append(json.dumps(close_rec))
        actions.append({
            "bot_key": bot_key, "bot_name": bot_name,
            "ticker": ticker, "side": close_rec["side"],
            "entry_cents": entry_i, "exit_cents": mark_i,
            "pnl_per_contract": pnl_cents, "reason": reason,
        })
    if new_lines:
        # Append to the jsonl atomically (open + write under a lock).
        with p.open("a", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
    return actions


def tick(bots: List[dict], hedge_cfg: dict) -> List[Dict[str, Any]]:
    """One scan across every registered bot. Returns the closes
    applied this tick so the caller can log / report them."""
    if not hedge_cfg or not hedge_cfg.get("enabled"):
        return []
    pl = int(hedge_cfg.get("profit_lock_cents", 0) or 0)
    sl = int(hedge_cfg.get("stop_loss_cents", 0) or 0)
    if pl <= 0 and sl <= 0:
        return []
    closed: List[Dict[str, Any]] = []
    for b in bots:
        dt = b.get("dashboard_type") or "standard"
        bot_key = b.get("key", "")
        bot_name = b.get("name", "")
        try:
            if dt in ("sport", "survivor"):
                # Sport (tennis / table-tennis / darts / NBA) and
                # survivor all use sim_state.json. Survivor currently
                # has no open positions but the framework is in
                # place — as soon as the bot starts paper-trading
                # elimination markets, this path picks them up
                # automatically. NBA's sim_state.json is synthesized
                # by bots/_sport_adapter from sim.db after each tick.
                results = _check_tennis_state(
                    b.get("sim_state_path"), b,
                    profit_lock_cents=pl, stop_loss_cents=sl,
                )
            elif dt == "whale":
                results = _check_whale_orders(
                    b.get("orders_path") or b.get("signals_path"),
                    bot_key, bot_name,
                    profit_lock_cents=pl, stop_loss_cents=sl,
                )
            elif dt == "rules-parser":
                # Rules Parser books its own simulated P&L on settlement
                # via settle_simulated_positions in the bot service.
                # Its DB has no `positions` table — the hedge monitor
                # has nothing to do here.
                continue
            else:
                db = b.get("db_path") or ""
                if not db:
                    continue
                results = _check_db(
                    db, b,
                    profit_lock_cents=pl, stop_loss_cents=sl,
                )
        except Exception:  # noqa: BLE001
            log.exception("hedge tick failed for %s", bot_key)
            continue
        for r in results:
            log.info(
                "[hedge] %s %s side=%s entry=%s exit=%s pnl=%+d reason=%s",
                r["bot_name"], r.get("ticker"), r.get("side"),
                r.get("entry_cents") or r.get("entry"),
                r.get("exit_cents") or r.get("exit"),
                r.get("pnl_per_contract", 0), r.get("reason"),
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
