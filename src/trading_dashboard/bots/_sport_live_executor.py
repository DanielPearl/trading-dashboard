"""Generic sport live executor — places REAL Kalshi orders on real money.

One parameterized implementation of the per-sport executors (the MLB
one was the template; darts / table-tennis / NBA instantiate this class
directly and MLBLiveExecutor is now a thin subclass). Built on
bots/_live_core.py so buy/sell mechanics are identical across bots.

Works on tennis-shape watchlist rows: one row per match, each side
backed by its own binary market (ticker_a / ticker_b), always bought
as YES on the chosen side. The probability driving every entry is the
row's live_prob (for benchmark-driven sports that IS the devigged
Pinnacle line — rows without one are never buy-eligible upstream).

Safety model (same as tennis / world-cup / mlb):
  * dry_run defaults to TRUE — no real order until the operator arms
    the bot (live Home-tab toggle ON, or ``<bot>_trader.live.dry_run:
    false`` + restart).
  * On the first REAL tick, leftover dry-run paper positions are
    voided (VOID_DRY_RUN) so they don't block the real entries.
  * Hard caps in code clamp whatever the YAML says.
  * Inventory check before every profit-lock sell (shared-account
    discipline — never sell contracts we can't verify we hold).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from . import _live_core as core

# Hard ceilings shared by every sport unless a subclass/instance
# overrides them. A YAML typo can widen nothing past these.
HARD_CAPS = {
    "contracts_per_order": 1,
    "max_open_positions": 10,
    "max_orders_per_day": 15,
    "min_edge_pp": 0.05,
    "max_entry_price_cents": 80,
    "min_entry_price_cents": 15,
    "price_deviation_cents": 3,
    "min_profit_lock_bid": 90,
}

DEFAULTS = {
    "max_open_positions": 6,
    "max_orders_per_day": 10,
    "contracts_per_order": 1,
    "min_edge_pp": 0.05,
    "max_entry_price_cents": 80,
    "min_entry_price_cents": 15,
    "price_deviation_cents": 3,
    "prematch_buffer_minutes": 10,
    "profit_lock_yes_bid_cents": 95,
}


def snapshot_by_ticker(records: list[dict] | None) -> dict[str, dict]:
    """Flatten per-event records into {market_ticker: snapshot}.

    Handles both record shapes: MLB/world-cup records carry a
    ``markets`` dict of per-outcome snapshots (status / result /
    yes_bid); darts / table-tennis records are flat rows with only
    ticker_a / ticker_b and no per-market status — those yield no
    snapshot, and the executor falls back to direct Kalshi lookups
    per open position (a few extra reads, never wrong data).
    """
    out: dict[str, dict] = {}
    for rec in records or []:
        mkts = rec.get("markets")
        if isinstance(mkts, dict):
            for mkt in mkts.values():
                if isinstance(mkt, dict) and mkt.get("ticker"):
                    out[mkt["ticker"]] = mkt
    return out


class SportLiveExecutor:
    """Stateful executor wired into a sport bot module when its
    ``<bot>_trader.live`` config block exists. Runs in the LIVE
    dashboard process (the sim process keeps its own paper state;
    this one writes to data/outputs-live/)."""

    def __init__(self, cfg: dict, state_path: str, *,
                 bot_key: str,
                 tournament: str,
                 surface: str,
                 win_verb: str = "winning",
                 hard: dict | None = None,
                 defaults: dict | None = None) -> None:
        self.bot_key = bot_key
        self.tournament = tournament
        self.surface = surface
        self.win_verb = win_verb
        self._log = logging.getLogger(f"dashboard.{bot_key}-live-executor")
        h = {**HARD_CAPS, **(hard or {})}
        d = {**DEFAULTS, **(defaults or {})}

        def capped(key, hard_v, *, floor=False, cast=int):
            v = cast(cfg.get(key, d.get(key)) or d.get(key))
            return max(v, hard_v) if floor else min(v, hard_v)

        self.dry_run = bool(cfg.get("dry_run", True))
        self.max_open = capped("max_open_positions",
                               h["max_open_positions"])
        self.max_orders_per_day = capped("max_orders_per_day",
                                         h["max_orders_per_day"])
        self.contracts_per_order = capped("contracts_per_order",
                                          h["contracts_per_order"])
        self.min_edge = capped("min_edge_pp", h["min_edge_pp"],
                               floor=True, cast=float)
        self.max_entry_price_cents = capped(
            "max_entry_price_cents", h["max_entry_price_cents"])
        self.min_entry_price_cents = capped(
            "min_entry_price_cents", h["min_entry_price_cents"],
            floor=True)
        self.price_deviation_cents = capped(
            "price_deviation_cents", h["price_deviation_cents"])
        self.prematch_buffer_minutes = float(
            cfg.get("prematch_buffer_minutes",
                    d["prematch_buffer_minutes"])
            or d["prematch_buffer_minutes"])
        raw_pl = int(cfg.get("profit_lock_yes_bid_cents",
                             d["profit_lock_yes_bid_cents"]) or 0)
        self.profit_lock_yes_bid = (max(raw_pl, h["min_profit_lock_bid"])
                                    if raw_pl > 0 else 0)
        self.state_path = Path(state_path)
        self._config_dry_run = self.dry_run
        self._daily = core.DailyOrderCounter()
        self._session = core.KalshiSession(bot_key)

        self._log.info(
            "%s-live-executor configured (dry_run=%s, max_open=%d, "
            "orders/day=%d, contracts=%d, min_edge=%.2f, "
            "price=[%d¢,%d¢], profit_lock=%d¢, state=%s)",
            bot_key, self.dry_run, self.max_open, self.max_orders_per_day,
            self.contracts_per_order, self.min_edge,
            self.min_entry_price_cents, self.max_entry_price_cents,
            self.profit_lock_yes_bid, self.state_path)
        if not self.dry_run:
            self._log.warning(
                "%s-live-executor is in LIVE mode — real Kalshi orders "
                "will be placed. Flip the %s Home-tab toggle OFF or set "
                "%s_trader.live.dry_run back to true to stop.",
                bot_key, bot_key, bot_key)

    # ── tick ────────────────────────────────────────────────────────

    def tick(self, watchlist_rows: list[dict],
             records: list[dict], armed: bool = False) -> dict:
        """One executor tick. ``armed`` comes from the LIVE dashboard's
        Home-tab toggle for this bot (explicitly ON in bot_states_live)
        — flipping that toggle IS the live-trading switch: ON → real
        Kalshi orders from the next tick; OFF → no new orders (existing
        positions still settle). Config dry_run: false also arms,
        independent of the toggle."""
        was_dry = self.dry_run
        self.dry_run = self._config_dry_run and not armed
        if was_dry != self.dry_run:
            if self.dry_run:
                self._log.warning("%s-live DISARMED (toggle off) — no "
                                  "further real orders", self.bot_key)
            else:
                self._log.warning("%s-live ARMED via Home-tab toggle — "
                                  "real Kalshi orders will be placed "
                                  "from this tick on", self.bot_key)
        state = self._load_state()
        if not self.dry_run:
            voided = core.void_dry_run_positions(state)
            if voided:
                self._log.warning("%s-live voided %d leftover dry-run "
                                  "position(s) — real entries unblocked",
                                  self.bot_key, voided)
        mkt_by_ticker = snapshot_by_ticker(records)

        self._reconcile(state, mkt_by_ticker)
        self._maybe_profit_lock(state, mkt_by_ticker)
        self._mark_to_market(state, watchlist_rows)

        candidates = sorted(
            (r for r in watchlist_rows
             if r.get("buy_eligible") and r.get("buy_side") in ("A", "B")),
            key=lambda r: float(r.get("buy_score") or 0.0), reverse=True)
        for row in candidates:
            self._maybe_place(row, state)

        state["stats"] = core.compute_stats(state)
        state["last_tick_at"] = core.now_iso()
        core.atomic_write_json(self.state_path, state)
        return state

    # ── order placement ─────────────────────────────────────────────

    def _maybe_place(self, row: dict, state: dict) -> None:
        match_id = str(row.get("match_id") or "")
        side = row.get("buy_side")
        ticker = (row.get("ticker_a") if side == "A"
                  else row.get("ticker_b"))
        if not ticker:
            return
        if len(state.get("open_positions", [])) >= self.max_open:
            return
        if self._daily.current() >= self.max_orders_per_day:
            self._log.info("%s-live skip %s: max_orders_per_day (%d)",
                           self.bot_key, match_id, self.max_orders_per_day)
            return
        seen = {p.get("match_id") for p in state.get("open_positions", [])}
        seen |= set(state.get("last_settled_at_by_match_id", {}))
        if match_id in seen:
            return

        ask_cents = (row.get("yes_ask_cents_a") if side == "A"
                     else row.get("yes_ask_cents_b"))
        model_p = (row.get("live_prob_a") if side == "A"
                   else row.get("live_prob_b"))
        edge = float(row.get("buy_side_edge") or 0.0)
        if ask_cents is None or model_p is None:
            return
        ask_cents = int(ask_cents)
        if not (self.min_entry_price_cents <= ask_cents
                <= self.max_entry_price_cents):
            self._log.info("%s-live skip %s: ask %d¢ outside [%d, %d]",
                           self.bot_key, ticker, ask_cents,
                           self.min_entry_price_cents,
                           self.max_entry_price_cents)
            return
        if edge < self.min_edge:
            return
        # No favorites-only gate here (the model-driven executors keep
        # theirs): the probability IS the sharp line, so buying an
        # underdog below its fair value — e.g. a 45% team at 40¢ — is
        # the whole strategy, and the sim's paper trader takes those.
        # Removed 2026-07-09 after it blocked COL@SF while the sim
        # bought it; sim and live must evaluate the same gates. The
        # entry-price floor (≥15¢) still keeps longshot tails out.
        kickoff = row.get("kickoff")
        if kickoff:
            try:
                k = datetime.fromisoformat(
                    str(kickoff).replace("Z", "+00:00"))
                mins = (k - datetime.now(timezone.utc)
                        ).total_seconds() / 60.0
                if mins <= self.prematch_buffer_minutes:
                    self._log.info("%s-live skip %s: %.0f min to start "
                                   "(≤ %.0f buffer)", self.bot_key,
                                   ticker, mins,
                                   self.prematch_buffer_minutes)
                    return
            except ValueError:
                pass
        current_ask = self._session.yes_ask_cents(ticker,
                                                  fallback=ask_cents)
        if current_ask is None:
            return
        if abs(current_ask - ask_cents) > self.price_deviation_cents:
            self._log.info("%s-live skip %s: market moved %d¢ → %d¢",
                           self.bot_key, ticker, ask_cents, current_ask)
            return
        balance = self._session.balance_cents()
        cost = current_ask * self.contracts_per_order
        if balance is not None and balance < cost:
            self._log.warning("%s-live skip %s: balance %d¢ < cost %d¢",
                              self.bot_key, ticker, balance, cost)
            return

        self._daily.increment()
        side_team = (row.get("player_a") if side == "A"
                     else row.get("player_b"))
        if self.dry_run:
            order_id = (f"DRY-RUN-"
                        f"{int(datetime.now(timezone.utc).timestamp())}"
                        f"-{side}")
            status = "dry_run_simulated"
        else:
            order_id, status = self._session.submit_ioc(
                ticker=ticker, action="buy",
                count=self.contracts_per_order,
                yes_price_cents=current_ask, kind="buy")
            if order_id is None:
                return
        position = {
            "position_id": f"{ticker}-{side}-{int(datetime.now(timezone.utc).timestamp())}",
            "order_id": order_id,
            "order_status": status,
            "ticker": ticker,
            "match_id": match_id,
            "market_side": "YES",
            "event_ticker": row.get("event_ticker") or match_id,
            "tournament": row.get("tournament") or self.tournament,
            "surface": row.get("surface") or self.surface,
            "player_a": row.get("player_a") or "",
            "player_b": row.get("player_b") or "",
            "side": "PLAYER_A" if side == "A" else "PLAYER_B",
            "side_player": side_team or "?",
            "entry_market_prob": current_ask / 100.0,
            "entry_model_prob": float(model_p),
            "current_market_prob": current_ask / 100.0,
            "current_model_prob": float(model_p),
            "stake": cost / 100.0,
            "contracts": self.contracts_per_order,
            "slippage": 0.0,
            "unrealized_pnl": 0.0,
            "label_at_open": row.get("recommended_action") or "",
            "reason_at_open": row.get("reason_for_signal") or "",
            "opened_at": core.now_iso(),
            "event_title": row.get("event_title") or "",
        }
        state.setdefault("open_positions", []).append(position)
        self._log.info("%s-live %s %s — BUY %d YES on %s (%s %s) at "
                       "%d¢ edge=%.3f", self.bot_key,
                       "DRY-RUN OPENED" if self.dry_run else "PLACED order",
                       order_id, self.contracts_per_order, ticker,
                       side_team, self.win_verb, current_ask, edge)

    # ── settlement / exits ──────────────────────────────────────────

    def _reconcile(self, state: dict, mkt_by_ticker: dict) -> None:
        """Settle open positions whose market has finalized — positions
        here are always YES on their ticker, so the market's own result
        grades them directly."""
        still_open = []
        for pos in state.get("open_positions", []):
            ticker = pos.get("ticker") or ""
            result = self._session.market_result(
                ticker, snapshot=mkt_by_ticker.get(ticker))
            if result is not None:
                settle = 1.0 if result == "yes" else 0.0
                closed = core.build_closed_record(
                    pos, settle_prob=settle, reason="settlement",
                    result="SETTLED")
                state.setdefault("closed_positions", []).append(closed)
                state.setdefault("last_settled_at_by_match_id", {})[
                    pos.get("match_id", "")] = closed["closed_at"]
                self._log.info("%s-live SETTLED %s — %s, P&L %+.3f",
                               self.bot_key, ticker,
                               "WON" if closed["won"] else "LOST",
                               closed["realized_pnl"])
                continue
            still_open.append(pos)
        state["open_positions"] = still_open

    def _maybe_profit_lock(self, state: dict, mkt_by_ticker: dict) -> None:
        if self.profit_lock_yes_bid <= 0:
            return
        still_open, closed_now = [], []
        for pos in state.get("open_positions", []):
            ticker = pos.get("ticker") or ""
            mkt = mkt_by_ticker.get(ticker) or {}
            bid = mkt.get("yes_bid")
            bid_cents = (int(round(float(bid) * 100))
                         if bid is not None else
                         self._session.yes_bid_cents(ticker))
            if bid_cents is None or bid_cents < self.profit_lock_yes_bid:
                still_open.append(pos)
                continue
            contracts = int(pos.get("contracts") or 1)
            if self.dry_run:
                order_id = (f"DRY-RUN-LOCK-"
                            f"{int(datetime.now(timezone.utc).timestamp())}")
                status = "dry_run_simulated"
            else:
                # Inventory check BEFORE selling: another process (or a
                # manual trade) may have already disposed of the
                # contracts — selling without inventory opens a naked
                # short. Shared-account discipline: this executor only
                # ever sells what it can verify it holds.
                held = self._session.position_count(ticker)
                if held is None:
                    still_open.append(pos)  # can't verify — defer
                    continue
                if held <= 0:
                    closed = core.build_closed_record(
                        pos, settle_prob=bid_cents / 100.0,
                        reason="closed_externally",
                        result="EXTERNAL_CLOSE")
                    closed_now.append(closed)
                    self._log.warning(
                        "%s-live %s already flat at Kalshi — recording "
                        "external close at %d¢ (approx)", self.bot_key,
                        ticker, bid_cents)
                    continue
                order_id, status = self._session.submit_ioc(
                    ticker=ticker, action="sell",
                    count=min(contracts, held),
                    yes_price_cents=bid_cents, kind="pl")
                if not order_id or (status or "").lower() != "executed":
                    # Not filled — keep holding; idempotent coid makes
                    # the retry next tick safe.
                    still_open.append(pos)
                    continue
            closed = core.build_closed_record(
                pos, settle_prob=bid_cents / 100.0, reason="profit_lock",
                result="PROFIT_LOCK", exit_order_id=order_id,
                exit_order_status=status)
            closed_now.append(closed)
            self._log.info("%s-live PROFIT-LOCKED %s — sold @ %d¢, "
                           "realized %+.3f", self.bot_key, ticker,
                           bid_cents, closed["realized_pnl"])
        if closed_now:
            state["open_positions"] = still_open
            state.setdefault("closed_positions", []).extend(closed_now)
            for c in closed_now:
                state.setdefault("last_settled_at_by_match_id", {})[
                    c.get("match_id", "")] = c["closed_at"]

    def _mark_to_market(self, state: dict, rows: list[dict]) -> None:
        by_ticker = {}
        for r in rows:
            if r.get("ticker_a"):
                by_ticker[r["ticker_a"]] = (r, "a")
            if r.get("ticker_b"):
                by_ticker[r["ticker_b"]] = (r, "b")
        for pos in state.get("open_positions", []):
            entry = by_ticker.get(pos.get("ticker") or "")
            if not entry:
                continue
            row, side = entry
            price = (row.get("market_prob_a") if side == "a"
                     else row.get("market_prob_b"))
            model_p = (row.get("live_prob_a") if side == "a"
                       else row.get("live_prob_b"))
            if price is not None:
                contracts = int(pos.get("contracts") or 1)
                pos["current_market_prob"] = float(price)
                pos["unrealized_pnl"] = round(
                    (float(price) - float(pos.get("entry_market_prob")
                                          or 0.0)) * contracts, 4)
            if model_p is not None:
                pos["current_model_prob"] = float(model_p)

    # ── state ───────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                import json
                with self.state_path.open("r", encoding="utf-8") as f:
                    s = json.load(f)
                    if s and s.get("open_positions") is not None:
                        return s
            except (OSError, ValueError):
                self._log.exception("%s-live state load failed; fresh "
                                    "start", self.bot_key)
        return {
            "started_at": core.now_iso(),
            "last_tick_at": None,
            "open_positions": [],
            "closed_positions": [],
            "stats": {},
            "last_settled_at_by_match_id": {},
        }
