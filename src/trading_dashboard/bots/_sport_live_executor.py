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
    # NOT a policy limit — the trading constraint is concurrency
    # (max_open_positions, "N at a time"), per user 2026-07-10. The
    # daily counter exists only as a runaway backstop (a bug rapidly
    # opening/closing positions burns out here instead of at Kalshi).
    "max_orders_per_day": 100,
    "min_edge_pp": 0.05,
    "max_entry_price_cents": 70,
    "min_entry_price_cents": 15,
    "price_deviation_cents": 3,
    "min_profit_lock_bid": 90,
    # Tennis / darts extras — enforced whenever the subclass exposes them.
    "min_profit_lock_gain": 5,
    "min_stop_loss_cents": 15,
    "max_stop_loss_cents": 40,
    "max_hours_to_close": 72.0,
}

DEFAULTS = {
    "max_open_positions": 6,
    "max_orders_per_day": 50,
    "contracts_per_order": 1,
    "min_edge_pp": 0.05,
    "max_entry_price_cents": 70,
    "min_entry_price_cents": 15,
    "price_deviation_cents": 3,
    "prematch_buffer_minutes": 10,
    "profit_lock_yes_bid_cents": 95,
    # Extras — 0 / None disables the branch (default behaviour).
    "profit_lock_gain_cents": 0,
    "stop_loss_cents": 0,
    "max_hours_to_close_for_open": None,
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
                 defaults: dict | None = None,
                 orphan_ticker_prefixes: tuple[str, ...] | None = None) -> None:
        """``orphan_ticker_prefixes`` is the safety net for the tennis-style
        catch-up-from-Kalshi flow — when the executor boots into a fresh
        state file but Kalshi already lists open positions the operator
        opened out-of-band (or from a prior process), those positions get
        re-adopted into ``open_positions`` so profit-lock / stop-loss /
        settlement all track them. The prefix filter keeps a shared-
        account catch-up from bleeding OTHER bots' positions into this
        one (see the comment in tennis's old executor around line 1282).
        Passing ``None`` disables the whole orphan-adoption path.
        """
        self.bot_key = bot_key
        self.tournament = tournament
        self.surface = surface
        self.win_verb = win_verb
        self.orphan_ticker_prefixes = tuple(orphan_ticker_prefixes or ())
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
        # Extras (tennis + darts). 0 disables. The gain / stop-loss
        # thresholds are clamped INTO the safe band so a config typo
        # can't fire the exit on 1-2¢ chatter or ride a loser to 0¢.
        raw_gn = int(cfg.get("profit_lock_gain_cents",
                              d["profit_lock_gain_cents"]) or 0)
        self.profit_lock_gain = (max(raw_gn, h["min_profit_lock_gain"])
                                  if raw_gn > 0 else 0)
        raw_sl = int(cfg.get("stop_loss_cents",
                              d["stop_loss_cents"]) or 0)
        if raw_sl <= 0:
            self.stop_loss = 0
        else:
            self.stop_loss = max(h["min_stop_loss_cents"],
                                   min(h["max_stop_loss_cents"], raw_sl))
        raw_hrs = cfg.get("max_hours_to_close_for_open",
                          d["max_hours_to_close_for_open"])
        if raw_hrs is None:
            self.max_hours_to_close = None
        else:
            self.max_hours_to_close = min(float(raw_hrs),
                                            h["max_hours_to_close"])
        # Skip buys where the sharp benchmark (Pinnacle guest feed or
        # The Odds API cascade) isn't quoting the match. Added
        # 2026-07-11 after the "no edge" audits on WNBA + tennis
        # traced back to the internal-model fallback firing on
        # matches Pinnacle hadn't posted a line for yet. The config
        # key existed in dashboard-live.yaml before this — the
        # executor just wasn't reading it.
        self.require_pinnacle = bool(cfg.get("require_pinnacle", False))
        # Strong-edge bypass retired 2026-07-15 (was a patch over the
        # 30¢ min_entry_price floor). Now that the floor is at 15¢
        # (SDK hard floor), the true-edge gate handles filtering
        # directly — no separate bypass layer needed.
        self.state_path = Path(state_path)
        self._config_dry_run = self.dry_run
        self._daily = core.DailyOrderCounter()
        self._session = core.KalshiSession(bot_key)
        # In-memory state — hydrated from disk on first tick, written
        # atomically at the end of every tick. Avoids re-reading the
        # JSON file on every render / poll.
        self._state_cache: dict | None = None
        # Rate-limit the SETTLE DEFERRED warning per ticker so a slow
        # Kalshi result-populate doesn't spam the journal.
        self._defer_log_at: dict[str, float] = {}

        self._log.info(
            "%s-live-executor configured (dry_run=%s, max_open=%d, "
            "orders/day=%d, contracts=%d, min_edge=%.2f, "
            "price=[%d¢,%d¢], "
            "profit_lock=%d¢, gain=%s, stop_loss=%s, "
            "hours_to_close=%s, require_pinnacle=%s, "
            "orphan_prefixes=%s, state=%s)",
            bot_key, self.dry_run, self.max_open, self.max_orders_per_day,
            self.contracts_per_order, self.min_edge,
            self.min_entry_price_cents, self.max_entry_price_cents,
            self.profit_lock_yes_bid,
            f"+{self.profit_lock_gain}¢" if self.profit_lock_gain else "off",
            f"-{self.stop_loss}¢" if self.stop_loss else "off",
            f"{self.max_hours_to_close}h" if self.max_hours_to_close else "off",
            self.require_pinnacle,
            self.orphan_ticker_prefixes or "off",
            self.state_path)
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
            # Orphan adoption: reconstruct sim_state positions Kalshi
            # already knows about (operator opened out-of-band, or a
            # prior process crashed after placing the order). Runs only
            # in real-money mode — dry-run positions never touched
            # Kalshi so there's nothing to recover. Watchlist_rows are
            # passed through so the stub can be enriched with player
            # names / event_title from the current watchlist — leaving
            # them empty produces "? vs — Model 50%" ghost rows on
            # the Active bets display.
            if self.orphan_ticker_prefixes:
                self._adopt_orphans(state, watchlist_rows)
        mkt_by_ticker = snapshot_by_ticker(records)

        self._reconcile(state, mkt_by_ticker)
        # Exit triggers, cheapest-to-evaluate first. Any exit that fires
        # removes the position from open_positions so the next trigger
        # can't double-close.
        self._maybe_profit_lock(state, mkt_by_ticker)
        if self.profit_lock_gain > 0:
            self._maybe_profit_lock_gain(state, mkt_by_ticker)
        if self.stop_loss > 0:
            self._maybe_stop_loss(state, mkt_by_ticker)
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
        self._state_cache = state
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
        # Sharp-benchmark gate. When ``require_pinnacle`` is true in the
        # config, refuse to trade any row where the row's
        # ``pinnacle_prob_yes`` (populated by the tennis / WNBA / MLB
        # exporter from the Pinnacle guest feed + Odds API cascade)
        # isn't set. This prevents the exact failure mode the
        # 2026-07-11 tennis audit surfaced: rows without a sharp line
        # fall back to the internal Sackmann-trained model, which
        # emits ~50% on unseen fixtures and clears the 9pp gate on
        # deep-underdog Kalshi asks — a phantom edge that dissolves
        # once Pinnacle actually posts.
        if self.require_pinnacle and row.get("pinnacle_prob_yes") is None:
            self._log.info(
                "%s-live skip %s: no Pinnacle line (require_pinnacle)",
                self.bot_key, ticker)
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
        # Hours-to-close gate — tennis wants to skip overnight bets on
        # tomorrow's match where a full day of injury / withdrawal news
        # has room to move the line. Off by default (None) so bots that
        # want every listed match (MLB / world-cup / WNBA) keep taking
        # them.
        if self.max_hours_to_close is not None:
            expiry = row.get("expected_expiration_time")
            if expiry:
                try:
                    exp_ts = datetime.fromisoformat(
                        str(expiry).replace("Z", "+00:00"))
                    hrs = (exp_ts - datetime.now(timezone.utc)
                           ).total_seconds() / 3600.0
                    if hrs > self.max_hours_to_close:
                        self._log.info(
                            "%s-live skip %s: %.1fh to close (> %.1fh cap)",
                            self.bot_key, ticker, hrs,
                            self.max_hours_to_close)
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
        # Re-verify the edge against the price we're actually about to
        # pay. ``buy_side_edge`` was computed off the exporter-time ask;
        # the deviation gate above tolerates a 3¢ adverse move, which
        # silently turns a 9pp row into a ~6pp fill (2026-07-14 audit:
        # "bets are being bought when the pp difference is much less
        # than 9pp"). Only adverse moves shrink the edge — a cheaper
        # ask than the exporter saw makes the trade better, not worse.
        edge_at_order = edge - max(0, current_ask - ask_cents) / 100.0
        if edge_at_order < self.min_edge:
            self._log.info(
                "%s-live skip %s: edge %.1fpp at ask %d¢ < %.0fpp floor "
                "(row had %.1fpp at %d¢)", self.bot_key, ticker,
                edge_at_order * 100, current_ask, self.min_edge * 100,
                edge * 100, ask_cents)
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

    def _sell_at_bid(self, state: dict, mkt_by_ticker: dict, *,
                     trigger: str,
                     bid_matches: callable) -> None:
        """Shared exit-trigger loop for the two bid-driven exits
        (``profit_lock_gain`` and ``stop_loss``). ``bid_matches`` is a
        callable ``(pos, bid_cents) -> (bool, close_reason_suffix)`` that
        returns True when this trigger should fire on the position.
        """
        still_open, closed_now = [], []
        for pos in state.get("open_positions", []):
            ticker = pos.get("ticker") or ""
            mkt = mkt_by_ticker.get(ticker) or {}
            bid = mkt.get("yes_bid")
            bid_cents = (int(round(float(bid) * 100))
                         if bid is not None else
                         self._session.yes_bid_cents(ticker))
            if bid_cents is None:
                still_open.append(pos)
                continue
            fire, suffix = bid_matches(pos, bid_cents)
            if not fire:
                still_open.append(pos)
                continue
            contracts = int(pos.get("contracts") or 1)
            if self.dry_run:
                order_id = (f"DRY-RUN-{trigger.upper()}-"
                            f"{int(datetime.now(timezone.utc).timestamp())}")
                status = "dry_run_simulated"
            else:
                held = self._session.position_count(ticker)
                if held is None:
                    still_open.append(pos)
                    continue
                if held <= 0:
                    closed_now.append(core.build_closed_record(
                        pos, settle_prob=bid_cents / 100.0,
                        reason="closed_externally",
                        result="EXTERNAL_CLOSE"))
                    self._log.warning(
                        "%s-live %s already flat at Kalshi — recording "
                        "external close at %d¢", self.bot_key,
                        ticker, bid_cents)
                    continue
                order_id, status = self._session.submit_ioc(
                    ticker=ticker, action="sell",
                    count=min(contracts, held),
                    yes_price_cents=bid_cents, kind=trigger[:2])
                if not order_id or (status or "").lower() != "executed":
                    still_open.append(pos)
                    continue
            reason = f"{trigger}_{suffix}" if suffix else trigger
            result = ("PROFIT_LOCK_GAIN" if trigger == "profit_lock_gain"
                      else "STOP_LOSS" if trigger == "stop_loss"
                      else "PROFIT_LOCK")
            closed = core.build_closed_record(
                pos, settle_prob=bid_cents / 100.0, reason=reason,
                result=result, exit_order_id=order_id,
                exit_order_status=status)
            closed_now.append(closed)
            self._log.info("%s-live %s %s — sold @ %d¢, realized %+.3f",
                           self.bot_key, trigger.upper(), ticker,
                           bid_cents, closed["realized_pnl"])
        if closed_now:
            state["open_positions"] = still_open
            state.setdefault("closed_positions", []).extend(closed_now)
            for c in closed_now:
                state.setdefault("last_settled_at_by_match_id", {})[
                    c.get("match_id", "")] = c["closed_at"]

    def _maybe_profit_lock_gain(self, state: dict,
                                  mkt_by_ticker: dict) -> None:
        """Sell when bid − entry >= ``profit_lock_gain`` — captures
        favourable mid-match line moves before they reverse."""
        def _fire(pos: dict, bid_cents: int) -> tuple[bool, str]:
            entry_cents = int(round(
                float(pos.get("entry_market_prob") or 0.0) * 100))
            gain = bid_cents - entry_cents
            return gain >= self.profit_lock_gain, f"gain{gain}c"
        self._sell_at_bid(state, mkt_by_ticker,
                          trigger="profit_lock_gain",
                          bid_matches=_fire)

    def _maybe_stop_loss(self, state: dict,
                          mkt_by_ticker: dict) -> None:
        """Sell when entry − bid >= ``stop_loss`` — cuts the tail of
        positions that would ride to expiry at 0."""
        def _fire(pos: dict, bid_cents: int) -> tuple[bool, str]:
            entry_cents = int(round(
                float(pos.get("entry_market_prob") or 0.0) * 100))
            drop = entry_cents - bid_cents
            return drop >= self.stop_loss, f"drop{drop}c"
        self._sell_at_bid(state, mkt_by_ticker,
                          trigger="stop_loss",
                          bid_matches=_fire)

    # ── orphan adoption ─────────────────────────────────────────────

    def _adopt_orphans(self, state: dict,
                        watchlist_rows: list[dict] | None = None) -> None:
        """Re-adopt Kalshi positions the executor doesn't know about.

        Two source patterns motivate this:
          1. First real tick on a fresh state file, with a position
             already open from a prior process / manual click.
          2. The executor voided a position as dry-run leftover but the
             actual Kalshi order had filled real contracts.

        Watchlist_rows (when supplied) is looked up by ticker_a /
        ticker_b to pull player names, event_title, and the
        event-ticker parent onto the orphan record — otherwise the
        adopted stub reaches the dashboard with empty fields and
        renders as "? vs — Model 50%" (2026-07-11 regression).

        The prefix filter is a hard safety: a shared Kalshi account can
        list positions from every bot in the fleet, and adopting the
        wrong one silently profit-locks another bot's win.
        """
        try:
            positions = self._session.list_positions(limit=200) or []
        except Exception:  # noqa: BLE001
            self._log.exception(
                "%s-live orphan discovery failed; deferring", self.bot_key)
            return
        known = {p.get("ticker") for p in state.get("open_positions", [])
                 if p.get("ticker")}
        recently_settled = set(state.get("last_settled_at_by_match_id", {}))
        # Ticker → (row, "a"|"b") lookup for enrichment. Empty when
        # the caller doesn't pass rows (upstream bots that never
        # populate them).
        rows_by_ticker: dict[str, tuple[dict, str]] = {}
        for r in watchlist_rows or []:
            ta = r.get("ticker_a")
            tb = r.get("ticker_b")
            if ta:
                rows_by_ticker[ta] = (r, "a")
            if tb:
                rows_by_ticker[tb] = (r, "b")
        for kp in positions:
            ticker = kp.get("ticker") or ""
            if not ticker or ticker in known:
                continue
            try:
                held = float(kp.get("position_fp") or kp.get("position") or 0)
            except (TypeError, ValueError):
                continue
            if abs(held) <= 0:
                continue
            if not any(ticker.startswith(p)
                       for p in self.orphan_ticker_prefixes):
                continue
            # Enrich from the watchlist row when we have one. Kalshi's
            # position record doesn't carry player names — the
            # watchlist row is the only place they live.
            wl_entry = rows_by_ticker.get(ticker)
            if wl_entry is not None:
                row, side_key = wl_entry
                player_a = row.get("player_a") or ""
                player_b = row.get("player_b") or ""
                side_player = (player_a if side_key == "a" else player_b)
                side_code = "PLAYER_A" if side_key == "a" else "PLAYER_B"
                event_title = (row.get("event_title") or row.get("title")
                               or "")
                match_id = row.get("match_id") or ticker.rsplit("-", 1)[0]
                mp = (row.get("live_prob_a") if side_key == "a"
                      else row.get("live_prob_b"))
                model_prob = float(mp) if mp is not None else None
            else:
                player_a = ""
                player_b = ""
                side_player = ""
                side_code = "PLAYER_A"
                event_title = ""
                # Fall back to stripping the ticker's final segment; if
                # that isn't the event ticker Kalshi returned, the
                # display-side ``kalshi_positions_to_active_bets``
                # helper will re-derive from the current watchlist on
                # its next tick.
                match_id = (kp.get("event_ticker")
                           or ticker.rsplit("-", 1)[0])
                model_prob = None
            if match_id in recently_settled:
                continue
            entry_price = (float(kp.get("avg_price_cents") or 0.0) / 100.0
                           if kp.get("avg_price_cents") is not None
                           else float(kp.get("entry_market_prob") or 0.5))
            pos = {
                "position_id": f"orphan-{ticker}",
                "order_id": f"recovered-{ticker}",
                "order_status": "recovered",
                "ticker": ticker,
                "match_id": match_id,
                "market_side": "YES",
                "event_ticker": kp.get("event_ticker") or match_id,
                "tournament": self.tournament,
                "surface": self.surface,
                "player_a": player_a,
                "player_b": player_b,
                "side": side_code,
                "side_player": side_player,
                "entry_market_prob": entry_price,
                "entry_model_prob": (model_prob if model_prob is not None
                                     else entry_price),
                "current_market_prob": entry_price,
                "current_model_prob": (model_prob if model_prob is not None
                                       else entry_price),
                "stake": entry_price * abs(held),
                "contracts": int(abs(held)),
                "slippage": 0.0,
                "unrealized_pnl": 0.0,
                "label_at_open": "RECOVERED",
                "reason_at_open": "orphan adoption from Kalshi",
                "opened_at": core.now_iso(),
                "event_title": event_title,
            }
            state.setdefault("open_positions", []).append(pos)
            enrichment = ("enriched" if wl_entry is not None
                           else "stub — no watchlist row")
            self._log.info(
                "%s-live RECOVERED %s as orphan (%d contracts @ %.2f, %s)",
                self.bot_key, ticker, int(abs(held)), entry_price,
                enrichment)

    # ── state ───────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if self._state_cache is not None:
            return self._state_cache
        if self.state_path.exists():
            try:
                import json
                with self.state_path.open("r", encoding="utf-8") as f:
                    s = json.load(f)
                    if s and s.get("open_positions") is not None:
                        self._state_cache = s
                        return s
            except (OSError, ValueError):
                self._log.exception("%s-live state load failed; fresh "
                                    "start", self.bot_key)
        fresh = {
            "started_at": core.now_iso(),
            "last_tick_at": None,
            "open_positions": [],
            "closed_positions": [],
            "stats": {},
            "last_settled_at_by_match_id": {},
        }
        self._state_cache = fresh
        return fresh
