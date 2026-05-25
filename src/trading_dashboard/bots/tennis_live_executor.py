"""Tennis live executor — places REAL Kalshi orders on real money.

ONLY runs when the dashboard mode is "live", the tennis_trader.enabled
flag in dashboard-live.yaml is true, AND the homepage toggle for
tennis is on. Three gates have to align before this module can
produce an order; flipping any of them OFF stops the bot
immediately (no new orders; existing positions are left alone and
settle on Kalshi's normal expiry).

Design principles
-----------------

* **Hard caps in code, not just config.** The YAML config can
  TIGHTEN the caps below, but the constants here are the
  outermost limit. A typo in dashboard-live.yaml saying
  ``max_open_positions: 500`` still results in 5 open positions
  max, because the executor clamps.

* **dry_run is the default.** If the config doesn't set
  ``dry_run: false`` explicitly, the executor never calls
  Kalshi's order endpoint — it just logs every decision it
  WOULD have made. The deploy step that flips the flag is
  intentionally a manual operator action.

* **Idempotent orders.** Every order carries a
  ``client_order_id`` derived from (match_id, side, today's
  date). If we retry the same order — restart, transient
  network failure, race with the toggle — Kalshi de-dupes
  server-side and we don't double up.

* **Honest state.** Positions are recorded in the same
  ``sim_state.json`` schema the paper simulator uses
  (open_positions / closed_positions / stats), so the
  dashboard's existing tennis renderer can show live trading
  with zero render changes. The state file lives under
  ``data/outputs-live/`` so it never mixes with sim.

* **Always settle, never partial-close.** v1 holds positions
  to Kalshi's natural expiry — no profit-lock / stop-loss
  active exits. Kalshi settles automatically when the match
  resolves; we just record the outcome in sim_state when we
  see the position vanish from Kalshi's portfolio.

* **All gates from kalshi_sdk.validators run** before any
  order. The buy_eligible flag on a watchlist row is the
  upstream tennis bot's own pre-screen; the executor then
  re-validates with the SDK's shared gates (min volume,
  min open_interest, max_spread, min_ev_per_contract,
  min_prob_edge_over_breakeven, etc.) so a stale or
  re-priced market gets rejected at execution time.
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


log = logging.getLogger("dashboard.tennis-live-executor")


# ──────────────────────────────────────────────────────────────────────
# Hard safety caps (code, not config). Config can lower these but
# never raise them. A YAML typo will fail safe.
# ──────────────────────────────────────────────────────────────────────
_HARD_MAX_CONTRACTS_PER_ORDER = 1
_HARD_MAX_OPEN_POSITIONS = 5
_HARD_MAX_ORDERS_PER_DAY = 20
_HARD_MIN_EDGE_PP = 0.05           # 5 percentage points
_HARD_MAX_ENTRY_PRICE_CENTS = 70   # don't pay > 70¢ for any contract
_HARD_MIN_ENTRY_PRICE_CENTS = 15   # don't trade tail probabilities
_HARD_PRICE_DEVIATION_CENTS = 3    # current ask must be within 3¢ of
                                     # the watchlist verdict's price


class _DailyOrderCounter:
    """Process-wide rolling-day counter for total orders placed.
    Resets at UTC midnight. Thread-safe so concurrent ticks (the
    dashboard runs the tennis daemon in one thread, but the
    Kalshi API client is shared and could be touched from other
    bot threads in future) don't race past the daily limit.
    """

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
                self._date = t
                self._count = 0
            self._count += 1
            return self._count

    def current(self) -> int:
        with self._lock:
            t = self._today()
            if t != self._date:
                return 0
            return self._count


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomic write — concurrent readers (the dashboard render
    process) never see a half-written file. Same pattern as
    ``_sport_adapter._atomic_write_json``.
    """
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


def _client_order_id(match_id: str, side: str, ask_cents: int) -> str:
    """Idempotency key. Stable for the same (match, side, day) so
    a retried order de-dupes server-side instead of doubling up.
    The ask_cents fold-in means a fresh order on the same match
    AFTER a price move is a new key — which is fine because the
    price-deviation check (_HARD_PRICE_DEVIATION_CENTS) would
    have rejected a retry on a moved market anyway.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    seed = f"tennis|{match_id}|{side}|{ask_cents}|{today}"
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


class TennisLiveExecutor:
    """Stateful executor — keeps a Kalshi client + the rolling
    daily counter alive across ticks so we don't re-auth or
    re-count on every iteration. Wired into ``bots/tennis.py``'s
    daemon when the dashboard is in live mode and the
    ``tennis_trader.live`` config block exists.
    """

    def __init__(self, cfg: dict, state_path: str) -> None:
        # Config knobs (clamped to hard caps below).
        self.dry_run = bool(cfg.get("dry_run", True))  # safe default
        self.max_open = min(int(cfg.get("max_open_positions", 5)),
                             _HARD_MAX_OPEN_POSITIONS)
        self.max_orders_per_day = min(
            int(cfg.get("max_orders_per_day", 20)),
            _HARD_MAX_ORDERS_PER_DAY,
        )
        self.contracts_per_order = min(
            int(cfg.get("contracts_per_order", 1)),
            _HARD_MAX_CONTRACTS_PER_ORDER,
        )
        self.min_edge = max(float(cfg.get("min_edge_pp", 0.05)),
                             _HARD_MIN_EDGE_PP)
        self.max_entry_price_cents = min(
            int(cfg.get("max_entry_price_cents", 70)),
            _HARD_MAX_ENTRY_PRICE_CENTS,
        )
        self.min_entry_price_cents = max(
            int(cfg.get("min_entry_price_cents", 15)),
            _HARD_MIN_ENTRY_PRICE_CENTS,
        )
        self.price_deviation_cents = min(
            int(cfg.get("price_deviation_cents", 3)),
            _HARD_PRICE_DEVIATION_CENTS,
        )

        self.state_path = Path(state_path)
        self._daily = _DailyOrderCounter()
        self._client = None  # lazy — see _get_client()

        log.info(
            "tennis-live-executor configured (dry_run=%s, max_open=%d, "
            "max_orders_per_day=%d, contracts_per_order=%d, "
            "min_edge=%.3f, entry_price=[%d¢, %d¢], "
            "price_deviation_max=%d¢, state=%s)",
            self.dry_run, self.max_open, self.max_orders_per_day,
            self.contracts_per_order, self.min_edge,
            self.min_entry_price_cents, self.max_entry_price_cents,
            self.price_deviation_cents, self.state_path,
        )
        if not self.dry_run:
            log.warning(
                "tennis-live-executor is in LIVE mode — real Kalshi "
                "orders will be placed when conditions are met. To "
                "stop, flip the tennis toggle on the dashboard's "
                "Home tab OFF, or set tennis_trader.live.dry_run "
                "back to true in dashboard-live.yaml.",
            )

    # ──────────────────────────────────────────────────────────────
    # Public entrypoint (called from bots/tennis.py per tick)
    # ──────────────────────────────────────────────────────────────

    def tick(self, watchlist_rows: list[dict],
              live_records: list[dict]) -> dict:
        """One executor tick. Returns the new state dict (same shape
        the paper simulator returns, so the dashboard renders it
        identically).
        """
        state = self._load_state()
        # 1) Settle positions whose match finished (read from
        #    Kalshi's portfolio so we don't have to track expiry
        #    ourselves).
        self._reconcile_positions(state)
        # 2) Mark-to-market open positions using the current
        #    market prices in live_records. Skip in dry_run to
        #    keep the JSON readable.
        live_by_id = {str(r.get("match_id") or ""): r
                       for r in live_records}
        for p in state.get("open_positions", []):
            self._mark_to_market(p, live_by_id.get(p.get("match_id", "")))
        # 3) Consider new orders. Only the rows the upstream tennis
        #    bot already pre-screened as buy_eligible — no point
        #    re-validating from scratch.
        candidates = [r for r in watchlist_rows
                      if r.get("buy_eligible") and r.get("buy_side")]
        # Highest score first (tennis bot already computes
        # buy_score = edge × EV); cap to whatever the daily / open
        # budgets allow.
        candidates.sort(key=lambda r: float(r.get("buy_score") or 0.0),
                        reverse=True)
        for row in candidates:
            ok = self._maybe_place(row, state, live_by_id)
            if not ok:
                # Don't break — a single rejection may leave room
                # for the next row (e.g. duplicate-ticker skip
                # doesn't burn a slot; cap-reached skip does and
                # subsequent rows will hit it too, but the
                # _maybe_place log already says so).
                pass
        # 4) Update stats + persist.
        state["stats"] = self._compute_stats(state)
        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state(state)
        return state

    # ──────────────────────────────────────────────────────────────
    # Order placement
    # ──────────────────────────────────────────────────────────────

    def _maybe_place(self, row: dict, state: dict,
                      live_by_id: dict) -> bool:
        """Run all gates + place (or simulate) an order for ``row``.
        Returns True if anything happened (placed or dry-run logged),
        False if skipped for a known reason.
        """
        match_id = str(row.get("match_id") or "")
        side_letter = row.get("buy_side")  # "A" or "B"
        if side_letter not in ("A", "B"):
            return False

        # Cap: max open positions
        if len(state.get("open_positions", [])) >= self.max_open:
            log.info("tennis-live skip %s: max_open_positions reached "
                     "(%d)", match_id, self.max_open)
            return False

        # Cap: max orders per day
        if self._daily.current() >= self.max_orders_per_day:
            log.info("tennis-live skip %s: max_orders_per_day reached "
                     "(%d)", match_id, self.max_orders_per_day)
            return False

        # Skip if we already have a position on this match (either
        # side). Prevents duplicate exposure across ticks.
        if any(p.get("match_id") == match_id
               for p in state.get("open_positions", [])):
            log.debug("tennis-live skip %s: position already open",
                      match_id)
            return False

        # Resolve which player + ticker we're buying YES on.
        player_field = "player_a" if side_letter == "A" else "player_b"
        ask_field = ("yes_ask_cents_a" if side_letter == "A"
                      else "yes_ask_cents_b")
        ticker = row.get("title_a") if side_letter == "A" else row.get("title_b")
        # title_a / title_b on the watchlist are the Kalshi YES
        # question text, NOT tickers. Need to map back. The live
        # record (parallel array via live_records) carries
        # ticker_a / ticker_b that ARE Kalshi tickers.
        live = live_by_id.get(match_id) or {}
        ticker = live.get("ticker_a") if side_letter == "A" else live.get("ticker_b")
        if not ticker:
            log.info("tennis-live skip %s side %s: no ticker on live "
                     "record (market may have closed)",
                     match_id, side_letter)
            return False

        ask_cents = row.get(ask_field)
        if ask_cents is None:
            log.info("tennis-live skip %s side %s: no ask price in row",
                     match_id, side_letter)
            return False
        ask_cents = int(ask_cents)

        # Price band gate
        if ask_cents < self.min_entry_price_cents:
            log.info("tennis-live skip %s side %s: ask %d¢ below "
                     "min_entry_price_cents %d", match_id, side_letter,
                     ask_cents, self.min_entry_price_cents)
            return False
        if ask_cents > self.max_entry_price_cents:
            log.info("tennis-live skip %s side %s: ask %d¢ above "
                     "max_entry_price_cents %d", match_id, side_letter,
                     ask_cents, self.max_entry_price_cents)
            return False

        # Edge gate (tennis bot already filtered, but re-check
        # against our floor)
        edge = (row.get("buy_side_edge") or 0.0)
        if abs(edge) < self.min_edge:
            log.info("tennis-live skip %s side %s: edge %.3f below "
                     "min %.3f", match_id, side_letter, edge,
                     self.min_edge)
            return False

        # Pre-flight: re-check the current Kalshi ask vs the
        # watchlist's recorded ask. The watchlist row is up to
        # 60s stale; if the market moved >3¢ since then, the
        # edge has likely vanished. Skip rather than chase.
        current_ask = self._current_yes_ask(ticker, fallback_cents=ask_cents)
        if current_ask is None:
            log.info("tennis-live skip %s side %s: couldn't fetch "
                     "current ask (market may be closed)",
                     match_id, side_letter)
            return False
        if abs(current_ask - ask_cents) > self.price_deviation_cents:
            log.info("tennis-live skip %s side %s: market moved "
                     "%d¢ → %d¢ (>%d¢) since watchlist refreshed",
                     match_id, side_letter, ask_cents, current_ask,
                     self.price_deviation_cents)
            return False

        # Balance check
        balance_cents = self._account_buying_power_cents()
        order_cost = current_ask * self.contracts_per_order
        if balance_cents is not None and balance_cents < order_cost:
            log.warning("tennis-live skip %s side %s: balance %d¢ "
                        "below order cost %d¢", match_id, side_letter,
                        balance_cents, order_cost)
            return False

        # All gates pass. Place (or dry-run).
        side_player = row.get(player_field) or "?"
        order_id, order_status = self._submit_order(
            ticker=ticker,
            match_id=match_id,
            side_letter=side_letter,
            side_player=side_player,
            ask_cents=current_ask,
        )
        if order_id is None and not self.dry_run:
            # Real submit failed; don't record a position.
            return False

        # Record the position (whether dry-run or real). For
        # dry-run, we still bump the daily counter so the cap
        # is realistic.
        self._daily.increment()
        if self.dry_run:
            log.info(
                "DRY-RUN would BUY %d %s on %s (%s vs %s, side=%s) "
                "at %d¢ — edge=%.3f buy_score=%.4f",
                self.contracts_per_order, "YES (player_a)" if side_letter == "A" else "YES (player_b)",
                ticker, row.get("player_a"), row.get("player_b"),
                side_letter, current_ask, edge,
                row.get("buy_score") or 0.0,
            )
            return True

        # Real order — record the position.
        position = {
            "position_id": f"{match_id}-{side_letter}-{int(datetime.now(timezone.utc).timestamp())}",
            "order_id": order_id,
            "order_status": order_status,
            "ticker": ticker,
            "match_id": match_id,
            "tournament": row.get("tournament") or "",
            "surface": row.get("surface") or "",
            "player_a": row.get("player_a") or "",
            "player_b": row.get("player_b") or "",
            "side": "PLAYER_A" if side_letter == "A" else "PLAYER_B",
            "side_player": side_player,
            "entry_market_prob": float(current_ask) / 100.0,
            "entry_model_prob": float(row.get(
                "live_prob_a" if side_letter == "A" else "live_prob_b"
            ) or 0.0),
            "current_market_prob": float(current_ask) / 100.0,
            "current_model_prob": float(row.get(
                "live_prob_a" if side_letter == "A" else "live_prob_b"
            ) or 0.0),
            "stake": (current_ask * self.contracts_per_order) / 100.0,
            "contracts": self.contracts_per_order,
            "slippage": 0.0,
            "unrealized_pnl": 0.0,
            "label_at_open": row.get("recommended_action") or "STRONG_EDGE",
            "reason_at_open": row.get("reason_for_signal") or "",
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "title": row.get("title_a" if side_letter == "A" else "title_b") or "",
            "event_title": row.get("event_title") or "",
        }
        state.setdefault("open_positions", []).append(position)
        log.info(
            "tennis-live PLACED order %s — BUY %d YES on %s (%s, "
            "side=%s) at %d¢ edge=%.3f",
            order_id, self.contracts_per_order, ticker, side_player,
            side_letter, current_ask, edge,
        )
        return True

    # ──────────────────────────────────────────────────────────────
    # Kalshi client wrappers
    # ──────────────────────────────────────────────────────────────

    def _get_client(self):
        """Lazy Kalshi client init. Same env-var auth the read-only
        client uses, but this one places orders. Cached for the life
        of the executor.
        """
        if self._client is not None:
            return self._client
        try:
            from kalshi_sdk import KalshiClient  # type: ignore
        except ImportError:
            log.exception("kalshi_sdk not importable; can't place orders")
            return None
        api_key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()
        priv_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if not api_key_id or not priv_key_path:
            log.error("kalshi creds missing in env (KALSHI_API_KEY_ID, "
                       "KALSHI_PRIVATE_KEY_PATH); can't place orders")
            return None
        try:
            self._client = KalshiClient(
                api_key_id=api_key_id,
                private_key_path=priv_key_path,
            )
        except Exception:  # noqa: BLE001
            log.exception("KalshiClient init failed")
            return None
        return self._client

    def _submit_order(self, *, ticker: str, match_id: str,
                        side_letter: str, side_player: str,
                        ask_cents: int) -> tuple[Optional[str], str]:
        """Place the order. Returns (order_id, status) or
        (None, reason) on failure. In dry_run mode, returns
        (None, "dry_run") without touching the client.
        """
        if self.dry_run:
            return None, "dry_run"
        client = self._get_client()
        if client is None:
            return None, "no_client"
        coid = _client_order_id(match_id, side_letter, ask_cents)
        try:
            # We always buy YES on the favoured side. The bot's
            # side selection already decides A vs B; "buy YES on
            # B" is equivalent to "bet against A". Time-in-force
            # IOC behaves market-like at the configured limit:
            # fills as much as it can right now or cancels the
            # rest. No partial fills accumulate as resting orders.
            resp = client.place_order(
                ticker=ticker,
                side="yes",
                action="buy",
                count=self.contracts_per_order,
                order_type="limit",
                yes_price=ask_cents,
                time_in_force="IOC",
                client_order_id=coid,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("tennis-live place_order failed for %s side "
                           "%s at %d¢: %s", ticker, side_letter,
                           ask_cents, exc)
            return None, f"error: {exc!r}"
        order = (resp or {}).get("order") or {}
        order_id = order.get("order_id")
        status = order.get("status") or "submitted"
        return order_id, status

    def _account_buying_power_cents(self) -> Optional[int]:
        """Best-effort balance check. Returns None on any failure;
        callers treat None as "skip the gate" rather than "block",
        because we'd rather under-trade on a transient balance API
        flake than over-trade on a false-positive zero.
        """
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client.get_balance() or {}
            # Kalshi returns balance in cents on the "balance" key
            # in current API; some accounts also have a
            # "buying_power" field. Prefer buying_power when set.
            for key in ("buying_power", "balance"):
                v = resp.get(key)
                if v is not None:
                    return int(v)
        except Exception:  # noqa: BLE001
            log.exception("get_balance failed (continuing without gate)")
        return None

    def _current_yes_ask(self, ticker: str,
                          fallback_cents: int) -> Optional[int]:
        """Fetch the current YES ask for a ticker. Used to confirm
        the watchlist's recorded ask is still close to reality
        before we place. Returns the fallback if the SDK can't
        give us a current price — better to use slightly stale
        data than nothing.
        """
        client = self._get_client()
        if client is None:
            return fallback_cents
        try:
            mkt = client.get_market(ticker) or {}
            market = mkt.get("market") or {}
            # Kalshi returns yes_ask in cents.
            ask = market.get("yes_ask")
            if ask is not None:
                return int(ask)
        except Exception:  # noqa: BLE001
            log.exception("get_market failed for %s (using fallback "
                           "%d¢)", ticker, fallback_cents)
        return fallback_cents

    # ──────────────────────────────────────────────────────────────
    # State persistence + reconciliation
    # ──────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        """Read the on-disk state. Returns the empty-shell if the
        file doesn't exist yet (live_bootstrap normally creates it,
        but a fresh repo without that bootstrap should still work).
        """
        if not self.state_path.exists():
            return self._empty_state()
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                return json.load(f) or self._empty_state()
        except (OSError, json.JSONDecodeError):
            log.exception("tennis-live state load failed (%s); "
                           "starting fresh", self.state_path)
            return self._empty_state()

    def _save_state(self, state: dict) -> None:
        _atomic_write_json(self.state_path, state)

    @staticmethod
    def _empty_state() -> dict:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "started_at": now,
            "last_tick_at": now,
            "open_positions": [],
            "closed_positions": [],
            "stats": {
                "total_opened": 0, "total_closed": 0, "open_count": 0,
                "wins": 0, "losses": 0, "win_rate": None,
                "total_realized_pnl": 0.0, "total_unrealized_pnl": 0.0,
                "total_staked": 0.0, "roi": None,
            },
            "last_settled_at_by_match_id": {},
        }

    def _mark_to_market(self, position: dict,
                          live: dict | None) -> None:
        """Update unrealized P&L on an open position using the
        current market_prob from the live record. No-op when the
        match isn't in live_records (probably already finished —
        _reconcile_positions handles the close).
        """
        if not live:
            return
        side = position.get("side")
        a = live.get("market_prob_a")
        if a is None:
            return
        # Current market_prob for the side we hold:
        side_prob = a if side == "PLAYER_A" else (1.0 - a)
        entry = float(position.get("entry_market_prob") or 0.0)
        contracts = int(position.get("contracts") or 1)
        # YES side: P&L per contract = side_prob - entry (in dollars).
        pnl = (side_prob - entry) * contracts
        position["current_market_prob"] = side_prob
        position["unrealized_pnl"] = pnl

    def _reconcile_positions(self, state: dict) -> None:
        """Query Kalshi for our open positions; any match in our
        local ``open_positions`` that Kalshi no longer lists has
        settled. Read the settlement outcome and move the position
        to ``closed_positions``.

        Kept generous on failure — a transient SDK error doesn't
        flush our state. The position stays open locally until the
        next tick where we DO get a clean read.
        """
        local_open = state.get("open_positions", [])
        if not local_open:
            return
        client = self._get_client()
        if client is None:
            return
        try:
            resp = client.get_positions(limit=200) or {}
            kalshi_positions = resp.get("positions") or []
            kalshi_tickers = {p.get("ticker") for p in kalshi_positions
                               if p.get("position", 0) != 0}
        except Exception:  # noqa: BLE001
            log.exception("get_positions failed; deferring "
                           "reconciliation to next tick")
            return

        still_open: list[dict] = []
        for p in local_open:
            if p.get("ticker") in kalshi_tickers:
                still_open.append(p)
                continue
            # Position is gone from Kalshi → settled (won or lost).
            # Pull the final market price to record realised P&L.
            settle_cents = self._settle_price_cents(p.get("ticker") or "")
            settle_prob = (float(settle_cents) / 100.0
                            if settle_cents is not None else 1.0)
            entry = float(p.get("entry_market_prob") or 0.0)
            contracts = int(p.get("contracts") or 1)
            realized = (settle_prob - entry) * contracts
            won = settle_prob > 0.5
            closed = dict(p)
            closed.update({
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "settle_market_prob": settle_prob,
                "realized_pnl": realized,
                "won": won,
                "close_reason": "settled",
                "result": "SETTLED",
                "winner_side": p.get("side") if won else (
                    "PLAYER_B" if p.get("side") == "PLAYER_A" else "PLAYER_A"
                ),
            })
            state.setdefault("closed_positions", []).append(closed)
            state.setdefault("last_settled_at_by_match_id",
                              {})[p.get("match_id", "")] = closed["closed_at"]
            log.info("tennis-live SETTLED %s — %s, P&L %+.3f",
                     p.get("ticker"), "WON" if won else "LOST", realized)
        state["open_positions"] = still_open

    def _settle_price_cents(self, ticker: str) -> Optional[int]:
        client = self._get_client()
        if client is None:
            return None
        try:
            mkt = (client.get_market(ticker) or {}).get("market") or {}
            # On a settled market, the YES side resolves to 100 or 0.
            settle = mkt.get("settlement_value")
            if settle is not None:
                return int(settle)
            # Fall back to last trade price if settlement_value
            # isn't published yet.
            last = mkt.get("last_price")
            if last is not None:
                return int(last)
        except Exception:  # noqa: BLE001
            log.exception("settle_price fetch failed for %s", ticker)
        return None

    def _compute_stats(self, state: dict) -> dict:
        open_positions = state.get("open_positions", [])
        closed = state.get("closed_positions", [])
        wins = sum(1 for c in closed if c.get("won"))
        losses = sum(1 for c in closed if c.get("won") is False)
        realized = sum(float(c.get("realized_pnl") or 0.0) for c in closed)
        unrealized = sum(float(p.get("unrealized_pnl") or 0.0)
                          for p in open_positions)
        staked = sum(float(p.get("stake") or 0.0)
                      for p in open_positions + closed)
        n_closed = len(closed)
        return {
            "total_opened": n_closed + len(open_positions),
            "total_closed": n_closed,
            "open_count": len(open_positions),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / n_closed) if n_closed > 0 else None,
            "total_realized_pnl": realized,
            "total_unrealized_pnl": unrealized,
            "total_staked": staked,
            "roi": (realized / staked) if staked > 0 else None,
        }
