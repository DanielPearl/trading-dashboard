"""World Cup live executor — places REAL Kalshi orders on real money.

Adapted from tennis_live_executor for KXWCADVANCE rows: one watchlist
row per match, each side backed by its own binary market (ticker_a /
ticker_b), always bought as YES on the chosen side's market.

Safety model (same as tennis):
  * dry_run defaults to TRUE — the executor never touches the order
    endpoint until the operator flips ``world_cup_trader.live.dry_run``
    to false in dashboard.yaml and restarts. Dry-run ticks log every
    order they WOULD place and accumulate paper positions in the live
    state file so the live page shows exactly what real mode would do.
  * Hard caps in code clamp whatever the YAML says.
  * Idempotent client_order_id per (market, day) so retries de-dupe
    server-side.
  * The Home-tab world-cup toggle gates new orders every tick.
  * Positions settle by reconciliation against Kalshi (finalized
    markets) plus a 95¢ profit-lock sell.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("dashboard.world-cup-live-executor")

_HARD_MAX_CONTRACTS_PER_ORDER = 1
_HARD_MAX_OPEN_POSITIONS = 8
_HARD_MAX_ORDERS_PER_DAY = 10
_HARD_MIN_EDGE_PP = 0.05
_HARD_MAX_ENTRY_PRICE_CENTS = 85
_HARD_MIN_ENTRY_PRICE_CENTS = 10
_HARD_PRICE_DEVIATION_CENTS = 3
_HARD_MIN_PROFIT_LOCK_BID = 90


class _DailyOrderCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._date = self._today()
        self._count = 0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def increment(self) -> int:
        with self._lock:
            t = self._today()
            if t != self._date:
                self._date, self._count = t, 0
            self._count += 1
            return self._count

    def current(self) -> int:
        with self._lock:
            return self._count if self._today() == self._date else 0


def _atomic_write_json(path: Path, payload: Any) -> None:
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


def _coid(prefix: str, ticker: str, cents: int) -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    seed = f"wc-{prefix}|{ticker}|{cents}|{today}"
    return f"{prefix}-" + hashlib.sha256(seed.encode()).hexdigest()[:28]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        self._daily = _DailyOrderCounter()
        self._client = None

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

        state["stats"] = self._compute_stats(state)
        state["last_tick_at"] = _now_iso()
        self._save_state(state)
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
        # One position per match, either side, open or already traded.
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
        # Pre-match only: re-check kickoff at execution time.
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
        # Re-check the live ask before committing.
        current_ask = self._current_yes_ask(ticker, fallback=ask_cents)
        if current_ask is None:
            return
        if abs(current_ask - ask_cents) > self.price_deviation_cents:
            log.info("wc-live skip %s: market moved %d¢ → %d¢",
                     ticker, ask_cents, current_ask)
            return
        balance = self._balance_cents()
        cost = current_ask * self.contracts_per_order
        if balance is not None and balance < cost:
            log.warning("wc-live skip %s: balance %d¢ < cost %d¢",
                        ticker, balance, cost)
            return

        self._daily.increment()
        side_team = (row.get("player_a") if side == "A"
                     else row.get("player_b"))
        if self.dry_run:
            order_id = f"DRY-RUN-{int(datetime.now(timezone.utc).timestamp())}-{side}"
            status = "dry_run_simulated"
        else:
            order_id, status = self._submit_buy(ticker, current_ask)
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
            "opened_at": _now_iso(),
            "event_title": row.get("event_title") or "",
        }
        state.setdefault("open_positions", []).append(position)
        log.info("wc-live %s %s — BUY %d YES on %s (%s advancing) at "
                 "%d¢ edge=%.3f",
                 "DRY-RUN OPENED" if self.dry_run else "PLACED order",
                 order_id, self.contracts_per_order, ticker, side_team,
                 current_ask, edge)

    # ── Kalshi wrappers ─────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from kalshi_sdk import KalshiClient
        except ImportError:
            log.exception("kalshi_sdk not importable")
            return None
        key = os.environ.get("KALSHI_API_KEY_ID", "").strip()
        pem = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if not key or not pem:
            log.error("Kalshi creds missing in env")
            return None
        try:
            self._client = KalshiClient(api_key_id=key,
                                        private_key_path=pem)
        except Exception:  # noqa: BLE001
            log.exception("KalshiClient init failed")
            return None
        return self._client

    def _submit_buy(self, ticker: str,
                    ask_cents: int) -> tuple[Optional[str], str]:
        client = self._get_client()
        if client is None:
            return None, "no_client"
        try:
            resp = client.place_order(
                ticker=ticker, side="yes", action="buy",
                count=self.contracts_per_order, order_type="limit",
                yes_price=ask_cents,
                time_in_force="immediate_or_cancel",
                client_order_id=_coid("buy", ticker, ask_cents))
        except Exception as exc:  # noqa: BLE001
            log.exception("wc-live place_order failed for %s: %s",
                          ticker, exc)
            return None, f"error: {exc!r}"
        order = (resp or {}).get("order") or {}
        return order.get("order_id"), order.get("status") or "submitted"

    def _market(self, ticker: str) -> dict:
        client = self._get_client()
        if client is None:
            return {}
        try:
            return (client.get_market(ticker) or {}).get("market") or {}
        except Exception:  # noqa: BLE001
            log.exception("get_market failed for %s", ticker)
            return {}

    def _current_yes_ask(self, ticker: str,
                         fallback: int) -> Optional[int]:
        mkt = self._market(ticker)
        for key, scale in (("yes_ask", 1), ("yes_ask_dollars", 100)):
            v = mkt.get(key)
            if v is not None:
                try:
                    return int(round(float(v) * scale))
                except (TypeError, ValueError):
                    continue
        return fallback

    def _current_yes_bid(self, ticker: str) -> Optional[int]:
        mkt = self._market(ticker)
        for key, scale in (("yes_bid", 1), ("yes_bid_dollars", 100)):
            v = mkt.get(key)
            if v is not None:
                try:
                    return int(round(float(v) * scale))
                except (TypeError, ValueError):
                    continue
        return None

    def _balance_cents(self) -> Optional[int]:
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client.get_balance() or {}
            for key in ("buying_power", "balance"):
                if resp.get(key) is not None:
                    return int(resp[key])
        except Exception:  # noqa: BLE001
            log.exception("get_balance failed (gate skipped)")
        return None

    # ── settlement / exits ──────────────────────────────────────────

    def _close(self, pos: dict, settle_prob: float, reason: str,
               result: str, **extra) -> dict:
        entry = float(pos.get("entry_market_prob") or 0.0)
        contracts = int(pos.get("contracts") or 1)
        realized = (settle_prob - entry) * contracts
        won = realized > 0
        side = pos.get("side") or "PLAYER_A"
        closed = dict(pos)
        closed.update({
            "closed_at": _now_iso(),
            "settle_market_prob": settle_prob,
            "realized_pnl": round(realized, 4),
            "won": won,
            "close_reason": reason,
            "result": result,
            "winner_side": side if won else (
                "PLAYER_B" if side == "PLAYER_A" else "PLAYER_A"),
            **extra,
        })
        return closed

    def _reconcile(self, state: dict, mkt_by_ticker: dict) -> None:
        """Settle open positions whose market has finalized. Uses the
        market's own result (yes/no) — positions here are always YES
        on their ticker."""
        still_open = []
        for pos in state.get("open_positions", []):
            ticker = pos.get("ticker") or ""
            mkt = mkt_by_ticker.get(ticker) or {}
            status = (mkt.get("status") or "").lower()
            result = (mkt.get("result") or "").lower()
            if not status:
                m = self._market(ticker)
                status = (m.get("status") or "").lower()
                result = (m.get("result") or "").lower()
            if status in ("finalized", "settled") and result in ("yes",
                                                                 "no"):
                settle = 1.0 if result == "yes" else 0.0
                closed = self._close(pos, settle, "settlement", "SETTLED")
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
                         self._current_yes_bid(ticker))
            if bid_cents is None or bid_cents < self.profit_lock_yes_bid:
                still_open.append(pos)
                continue
            contracts = int(pos.get("contracts") or 1)
            if self.dry_run:
                order_id, status = (
                    f"DRY-RUN-LOCK-{int(datetime.now(timezone.utc).timestamp())}",
                    "dry_run_simulated")
            else:
                client = self._get_client()
                if client is None:
                    still_open.append(pos)
                    continue
                try:
                    resp = client.place_order(
                        ticker=ticker, side="yes", action="sell",
                        count=contracts, order_type="limit",
                        yes_price=bid_cents,
                        time_in_force="immediate_or_cancel",
                        client_order_id=_coid("pl", ticker, bid_cents))
                except Exception:  # noqa: BLE001
                    log.exception("wc-live profit-lock failed for %s",
                                  ticker)
                    still_open.append(pos)
                    continue
                order = (resp or {}).get("order") or {}
                order_id = order.get("order_id")
                status = (order.get("status") or "").lower()
                if not order_id or status != "executed":
                    still_open.append(pos)
                    continue
            closed = self._close(pos, bid_cents / 100.0, "profit_lock",
                                 "PROFIT_LOCK", exit_order_id=order_id,
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
                with self.state_path.open("r", encoding="utf-8") as f:
                    s = json.load(f)
                    if s and s.get("open_positions") is not None:
                        return s
            except (OSError, json.JSONDecodeError):
                log.exception("wc-live state load failed; fresh start")
        return {
            "started_at": _now_iso(),
            "last_tick_at": None,
            "open_positions": [],
            "closed_positions": [],
            "stats": {},
            "last_settled_at_by_match_id": {},
        }

    def _save_state(self, state: dict) -> None:
        _atomic_write_json(self.state_path, state)

    @staticmethod
    def _compute_stats(state: dict) -> dict:
        open_p = state.get("open_positions", [])
        closed = state.get("closed_positions", [])
        wins = sum(1 for c in closed if c.get("won"))
        losses = sum(1 for c in closed if c.get("won") is False)
        realized = sum(float(c.get("realized_pnl") or 0) for c in closed)
        unreal = sum(float(p.get("unrealized_pnl") or 0) for p in open_p)
        staked = sum(float(p.get("stake") or 0) for p in open_p + closed)
        return {
            "total_opened": len(open_p) + len(closed),
            "total_closed": len(closed),
            "open_count": len(open_p),
            "wins": wins, "losses": losses,
            "win_rate": (wins / len(closed)) if closed else None,
            "total_realized_pnl": round(realized, 4),
            "total_unrealized_pnl": round(unreal, 4),
            "total_staked": round(staked, 4),
            "roi": round(realized / staked, 4) if staked else None,
        }
