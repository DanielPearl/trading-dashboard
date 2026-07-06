"""World Cup live executor — places REAL Kalshi orders on real money.

Built on bots/_live_core.py (the shared live-trading machinery: order
submission, idempotency, daily budget, settle/close records, stats) so
buy/sell mechanics are identical across bots. Adapted to KXWCADVANCE
rows: one watchlist row per match, each side backed by its own binary
market (ticker_a / ticker_b), always bought as YES on the chosen side.

Safety model (same as tennis):
  * dry_run defaults to TRUE — the executor never touches the order
    endpoint until the operator flips ``world_cup_trader.live.dry_run``
    to false in dashboard.yaml and restarts the sim service. Dry-run
    ticks log every order they WOULD place and accumulate paper
    positions in the live state file.
  * On the first REAL tick, leftover dry-run paper positions are
    voided (VOID_DRY_RUN) so they don't block the real entries.
  * Hard caps in code clamp whatever the YAML says.
  * The Home-tab world-cup toggle gates new orders every tick.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from . import _live_core as core

log = logging.getLogger("dashboard.world-cup-live-executor")

_HARD_MAX_CONTRACTS_PER_ORDER = 1
_HARD_MAX_OPEN_POSITIONS = 8
_HARD_MAX_ORDERS_PER_DAY = 10
_HARD_MIN_EDGE_PP = 0.05
_HARD_MAX_ENTRY_PRICE_CENTS = 85
_HARD_MIN_ENTRY_PRICE_CENTS = 10
_HARD_PRICE_DEVIATION_CENTS = 3
_HARD_MIN_PROFIT_LOCK_BID = 90


class WorldCupLiveExecutor:
    """Stateful executor wired into bots/world_cup.py when the
    ``world_cup_trader.live`` config block exists. Runs ALONGSIDE the
    paper simulator (sim site keeps its own state; this one writes to
    data/outputs-live/)."""

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
            "max_entry_price_cents", 85, _HARD_MAX_ENTRY_PRICE_CENTS)
        self.min_entry_price_cents = capped(
            "min_entry_price_cents", 10, _HARD_MIN_ENTRY_PRICE_CENTS,
            floor=True)
        self.price_deviation_cents = capped(
            "price_deviation_cents", 3, _HARD_PRICE_DEVIATION_CENTS)
        self.prematch_buffer_minutes = float(
            cfg.get("prematch_buffer_minutes", 15) or 15)
        raw_pl = int(cfg.get("profit_lock_yes_bid_cents", 95) or 0)
        self.profit_lock_yes_bid = (max(raw_pl, _HARD_MIN_PROFIT_LOCK_BID)
                                    if raw_pl > 0 else 0)
        self.state_path = Path(state_path)
        self._daily = core.DailyOrderCounter()
        self._session = core.KalshiSession("world-cup")

        log.info(
            "world-cup-live-executor configured (dry_run=%s, max_open=%d, "
            "orders/day=%d, contracts=%d, min_edge=%.2f, "
            "price=[%d¢,%d¢], profit_lock=%d¢, state=%s)",
            self.dry_run, self.max_open, self.max_orders_per_day,
            self.contracts_per_order, self.min_edge,
            self.min_entry_price_cents, self.max_entry_price_cents,
            self.profit_lock_yes_bid, self.state_path)
        if not self.dry_run:
            log.warning(
                "world-cup-live-executor is in LIVE mode — real Kalshi "
                "orders will be placed. Flip the world-cup Home-tab "
                "toggle OFF or set world_cup_trader.live.dry_run back "
                "to true to stop.")

    # ── tick ────────────────────────────────────────────────────────

    def tick(self, watchlist_rows: list[dict],
             records: list[dict]) -> dict:
        state = self._load_state()
        if not self.dry_run:
            voided = core.void_dry_run_positions(state)
            if voided:
                log.warning("wc-live voided %d leftover dry-run "
                            "position(s) — real entries unblocked",
                            voided)
        mkt_by_ticker: dict[str, dict] = {}
        for rec in records:
            for mkt in (rec.get("markets") or {}).values():
                if mkt.get("ticker"):
                    mkt_by_ticker[mkt["ticker"]] = mkt

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
            log.info("wc-live skip %s: max_orders_per_day (%d)",
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
        edge = float(row.get("buy_side_edge") or 0.0)
        if ask_cents is None or model_p is None:
            return
        ask_cents = int(ask_cents)
        if not (self.min_entry_price_cents <= ask_cents
                <= self.max_entry_price_cents):
            log.info("wc-live skip %s: ask %d¢ outside [%d, %d]",
                     ticker, ask_cents, self.min_entry_price_cents,
                     self.max_entry_price_cents)
            return
        if edge < self.min_edge:
            return
        if float(model_p) <= 0.50:
            log.info("wc-live skip %s: model prob %.3f ≤ 0.50 — refuse "
                     "to buy a side the model expects to lose", ticker,
                     float(model_p))
            return
        kickoff = row.get("kickoff")
        if kickoff:
            try:
                k = datetime.fromisoformat(
                    str(kickoff).replace("Z", "+00:00"))
                mins = (k - datetime.now(timezone.utc)
                        ).total_seconds() / 60.0
                if mins <= self.prematch_buffer_minutes:
                    log.info("wc-live skip %s: %.0f min to kickoff "
                             "(≤ %.0f buffer)", ticker, mins,
                             self.prematch_buffer_minutes)
                    return
            except ValueError:
                pass
        current_ask = self._session.yes_ask_cents(ticker,
                                                  fallback=ask_cents)
        if current_ask is None:
            return
        if abs(current_ask - ask_cents) > self.price_deviation_cents:
            log.info("wc-live skip %s: market moved %d¢ → %d¢",
                     ticker, ask_cents, current_ask)
            return
        balance = self._session.balance_cents()
        cost = current_ask * self.contracts_per_order
        if balance is not None and balance < cost:
            log.warning("wc-live skip %s: balance %d¢ < cost %d¢",
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
            "tournament": row.get("tournament") or "FIFA World Cup 2026",
            "surface": row.get("surface") or "Soccer",
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
        log.info("wc-live %s %s — BUY %d YES on %s (%s advancing) at "
                 "%d¢ edge=%.3f",
                 "DRY-RUN OPENED" if self.dry_run else "PLACED order",
                 order_id, self.contracts_per_order, ticker, side_team,
                 current_ask, edge)

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
                log.info("wc-live SETTLED %s — %s, P&L %+.3f", ticker,
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
                order_id, status = self._session.submit_ioc(
                    ticker=ticker, action="sell", count=contracts,
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
            log.info("wc-live PROFIT-LOCKED %s — sold @ %d¢, realized "
                     "%+.3f", ticker, bid_cents, closed["realized_pnl"])
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
                log.exception("wc-live state load failed; fresh start")
        return {
            "started_at": core.now_iso(),
            "last_tick_at": None,
            "open_positions": [],
            "closed_positions": [],
            "stats": {},
            "last_settled_at_by_match_id": {},
        }
