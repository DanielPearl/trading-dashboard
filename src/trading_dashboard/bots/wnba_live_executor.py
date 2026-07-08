"""WNBA live executor — places REAL Kalshi orders on real money.

Sibling of world_cup_live_executor.py. Built on _live_core so buy/sell
mechanics are identical across bots. Consumes the tennis-shape
watchlist rows emitted by _sport_adapter for WNBA — one row per match,
with per-side ticker_a / ticker_b and yes_ask_cents_a / yes_ask_cents_b.
The chosen side is bought as YES on its own ticker.

Safety model (same as tennis and world-cup):
  * dry_run defaults to TRUE — the executor never touches the order
    endpoint until the operator flips ``wnba_trader.live.dry_run`` to
    false in dashboard.yaml and restarts the sim service. Dry-run ticks
    log every order they WOULD place and accumulate paper positions in
    the live state file.
  * On the first REAL tick, leftover dry-run paper positions are
    voided (VOID_DRY_RUN) so they don't block the real entries.
  * Hard caps in code clamp whatever the YAML says.
  * The Home-tab WNBA toggle gates new orders every tick.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from . import _live_core as core

log = logging.getLogger("dashboard.wnba-live-executor")

_HARD_MAX_CONTRACTS_PER_ORDER = 1
_HARD_MAX_OPEN_POSITIONS = 8
_HARD_MAX_ORDERS_PER_DAY = 10
_HARD_MIN_EDGE_PP = 0.05
_HARD_MAX_ENTRY_PRICE_CENTS = 85
_HARD_MIN_ENTRY_PRICE_CENTS = 10
_HARD_PRICE_DEVIATION_CENTS = 3
_HARD_MIN_PROFIT_LOCK_BID = 90


class WNBALiveExecutor:
    """Stateful executor wired into bots/wnba.py when the
    ``wnba_trader.live`` config block exists. Runs ALONGSIDE the paper
    simulator: the WNBA Bot.tick() keeps writing to sim.db (the sim
    dashboard reads that), while this executor writes to a separate
    outputs-live/ state file that the live dashboard reads."""

    def __init__(self, cfg: dict, state_path: str) -> None:
        def capped(key, default, hard, *, floor=False, cast=int):
            v = cast(cfg.get(key, default) or default)
            return max(v, hard) if floor else min(v, hard)

        self.dry_run = bool(cfg.get("dry_run", True))
        self.max_open = capped("max_open_positions", 4,
                               _HARD_MAX_OPEN_POSITIONS)
        self.max_orders_per_day = capped("max_orders_per_day", 6,
                                         _HARD_MAX_ORDERS_PER_DAY)
        self.contracts_per_order = capped("contracts_per_order", 1,
                                          _HARD_MAX_CONTRACTS_PER_ORDER)
        self.min_edge = capped("min_edge_pp", 0.08, _HARD_MIN_EDGE_PP,
                               floor=True, cast=float)
        self.max_entry_price_cents = capped(
            "max_entry_price_cents", 70, _HARD_MAX_ENTRY_PRICE_CENTS)
        self.min_entry_price_cents = capped(
            "min_entry_price_cents", 15, _HARD_MIN_ENTRY_PRICE_CENTS,
            floor=True)
        self.price_deviation_cents = capped(
            "price_deviation_cents", 3, _HARD_PRICE_DEVIATION_CENTS)
        # Require Pinnacle to be listed before we buy — same "no sharp
        # reference, no trade" gate tennis enforces via
        # no_pinnacle_reference. Kept configurable so we can disable it
        # for late-tip games Pinnacle drops off.
        self.require_pinnacle = bool(cfg.get("require_pinnacle", True))
        raw_pl = int(cfg.get("profit_lock_yes_bid_cents", 95) or 0)
        self.profit_lock_yes_bid = (max(raw_pl, _HARD_MIN_PROFIT_LOCK_BID)
                                    if raw_pl > 0 else 0)
        self.state_path = Path(state_path)
        self._config_dry_run = self.dry_run
        self._daily = core.DailyOrderCounter()
        self._session = core.KalshiSession("wnba")

        log.info(
            "wnba-live-executor configured (dry_run=%s, max_open=%d, "
            "orders/day=%d, contracts=%d, min_edge=%.2f, "
            "price=[%d¢,%d¢], require_pinnacle=%s, profit_lock=%d¢, "
            "state=%s)",
            self.dry_run, self.max_open, self.max_orders_per_day,
            self.contracts_per_order, self.min_edge,
            self.min_entry_price_cents, self.max_entry_price_cents,
            self.require_pinnacle, self.profit_lock_yes_bid,
            self.state_path)
        if not self.dry_run:
            log.warning(
                "wnba-live-executor is in LIVE mode — real Kalshi "
                "orders will be placed. Flip the WNBA Home-tab toggle "
                "OFF or set wnba_trader.live.dry_run back to true to "
                "stop.")

    # ── tick ────────────────────────────────────────────────────────

    def tick(self, watchlist_rows: list[dict],
             armed: bool = False) -> dict:
        """One executor tick. ``armed`` mirrors the world-cup pattern:
        the LIVE dashboard's WNBA Home-tab toggle (explicitly ON in
        bot_states_live) arms; OFF disarms. Config ``dry_run: false``
        also arms, independent of the toggle."""
        was_dry = self.dry_run
        self.dry_run = self._config_dry_run and not armed
        if was_dry != self.dry_run:
            if self.dry_run:
                log.warning("wnba-live DISARMED (toggle off) — no "
                            "further real orders")
            else:
                log.warning("wnba-live ARMED via Home-tab toggle — real "
                            "Kalshi orders will be placed from this "
                            "tick on")
        state = self._load_state()
        if not self.dry_run:
            voided = core.void_dry_run_positions(state)
            if voided:
                log.warning("wnba-live voided %d leftover dry-run "
                            "position(s) — real entries unblocked",
                            voided)

        self._reconcile(state)
        self._maybe_profit_lock(state)
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
            log.info("wnba-live skip %s: max_orders_per_day (%d)",
                     match_id, self.max_orders_per_day)
            return
        seen = {p.get("match_id") for p in state.get("open_positions", [])}
        seen |= set(state.get("last_settled_at_by_match_id", {}))
        if match_id in seen:
            return

        ask_cents = (row.get("yes_ask_cents_a") if side == "A"
                     else row.get("yes_ask_cents_b"))
        model_p = (row.get("live_prob_a") if side == "A"
                   else row.get("live_prob_b"))
        pinnacle_p = (row.get("pinnacle_prob_a") if side == "A"
                       else row.get("pinnacle_prob_b"))
        edge = float(row.get("buy_side_edge") or 0.0)
        if ask_cents is None or model_p is None:
            return
        ask_cents = int(ask_cents)
        if not (self.min_entry_price_cents <= ask_cents
                <= self.max_entry_price_cents):
            log.info("wnba-live skip %s: ask %d¢ outside [%d, %d]",
                     ticker, ask_cents, self.min_entry_price_cents,
                     self.max_entry_price_cents)
            return
        if edge < self.min_edge:
            return
        if self.require_pinnacle and pinnacle_p is None:
            log.info("wnba-live skip %s: no Pinnacle reference "
                     "(require_pinnacle=true)", ticker)
            return
        # Tennis's model-favourite gate: never buy a side the reference
        # prob says is a loser. Uses Pinnacle when present, else the
        # blended model prob.
        favour_p = float(pinnacle_p) if pinnacle_p is not None else float(model_p)
        if favour_p <= 0.50:
            log.info("wnba-live skip %s: reference prob %.3f ≤ 0.50 — "
                     "refuse to buy a side the sharp line expects to "
                     "lose", ticker, favour_p)
            return
        current_ask = self._session.yes_ask_cents(ticker,
                                                  fallback=ask_cents)
        if current_ask is None:
            return
        if abs(current_ask - ask_cents) > self.price_deviation_cents:
            log.info("wnba-live skip %s: market moved %d¢ → %d¢",
                     ticker, ask_cents, current_ask)
            return
        balance = self._session.balance_cents()
        cost = current_ask * self.contracts_per_order
        if balance is not None and balance < cost:
            log.warning("wnba-live skip %s: balance %d¢ < cost %d¢",
                        ticker, balance, cost)
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
            "tournament": row.get("tournament") or "WNBA",
            "surface": row.get("surface") or "",
            "player_a": row.get("player_a") or "",
            "player_b": row.get("player_b") or "",
            "side": "PLAYER_A" if side == "A" else "PLAYER_B",
            "side_player": side_team or "?",
            "entry_market_prob": current_ask / 100.0,
            "entry_model_prob": float(model_p),
            "entry_pinnacle_prob": (float(pinnacle_p)
                                     if pinnacle_p is not None else None),
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
        log.info("wnba-live %s %s — BUY %d YES on %s (%s winning) at "
                 "%d¢ edge=%.3f pinnacle=%s",
                 "DRY-RUN OPENED" if self.dry_run else "PLACED order",
                 order_id, self.contracts_per_order, ticker, side_team,
                 current_ask, edge,
                 f"{pinnacle_p:.3f}" if pinnacle_p is not None else "—")

    # ── settlement / exits ──────────────────────────────────────────

    def _reconcile(self, state: dict) -> None:
        """Settle open positions whose market has finalized — positions
        here are always YES on their ticker, so the market's own result
        grades them directly. Unlike the world-cup executor we don't
        have a per-tick markets snapshot handed in (WNBA bot doesn't
        emit one at the executor boundary), so we rely on the Kalshi
        client to fetch fresh market status inside ``market_result``."""
        still_open = []
        for pos in state.get("open_positions", []):
            ticker = pos.get("ticker") or ""
            result = self._session.market_result(ticker, snapshot=None)
            if result is not None:
                settle = 1.0 if result == "yes" else 0.0
                closed = core.build_closed_record(
                    pos, settle_prob=settle, reason="settlement",
                    result="SETTLED")
                state.setdefault("closed_positions", []).append(closed)
                state.setdefault("last_settled_at_by_match_id", {})[
                    pos.get("match_id", "")] = closed["closed_at"]
                log.info("wnba-live SETTLED %s — %s, P&L %+.3f", ticker,
                         "WON" if closed["won"] else "LOST",
                         closed["realized_pnl"])
                continue
            still_open.append(pos)
        state["open_positions"] = still_open

    def _maybe_profit_lock(self, state: dict) -> None:
        if self.profit_lock_yes_bid <= 0:
            return
        still_open, closed_now = [], []
        for pos in state.get("open_positions", []):
            ticker = pos.get("ticker") or ""
            bid_cents = self._session.yes_bid_cents(ticker)
            if bid_cents is None or bid_cents < self.profit_lock_yes_bid:
                still_open.append(pos)
                continue
            contracts = int(pos.get("contracts") or 1)
            if self.dry_run:
                order_id = (f"DRY-RUN-LOCK-"
                            f"{int(datetime.now(timezone.utc).timestamp())}")
                status = "dry_run_simulated"
            else:
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
                    log.warning(
                        "wnba-live %s already flat at Kalshi — recording "
                        "external close at %d¢ (approx)", ticker,
                        bid_cents)
                    continue
                order_id, status = self._session.submit_ioc(
                    ticker=ticker, action="sell",
                    count=min(contracts, held),
                    yes_price_cents=bid_cents, kind="pl")
                if not order_id or (status or "").lower() != "executed":
                    still_open.append(pos)
                    continue
            closed = core.build_closed_record(
                pos, settle_prob=bid_cents / 100.0, reason="profit_lock",
                result="PROFIT_LOCK", exit_order_id=order_id,
                exit_order_status=status)
            closed_now.append(closed)
            log.info("wnba-live PROFIT-LOCKED %s — sold @ %d¢, realized "
                     "%+.3f", ticker, bid_cents, closed["realized_pnl"])
        if closed_now:
            state["open_positions"] = still_open
            state.setdefault("closed_positions", []).extend(closed_now)
            for c in closed_now:
                state.setdefault("last_settled_at_by_match_id", {})[
                    c.get("match_id", "")] = c["closed_at"]

    def _mark_to_market(self, state: dict, rows: list[dict]) -> None:
        by_ticker: dict[str, tuple[dict, str]] = {}
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
                log.exception("wnba-live state load failed; fresh start")
        return {
            "started_at": core.now_iso(),
            "last_tick_at": None,
            "open_positions": [],
            "closed_positions": [],
            "stats": {},
            "last_settled_at_by_match_id": {},
        }
