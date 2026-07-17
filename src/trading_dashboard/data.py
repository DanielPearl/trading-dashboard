"""Data access — SQLite readers, Kalshi portfolio history, snapshots."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from .fmt import (
    kalshi_fee_cents,
    minutes_to_close_from_ticker,
    unrealized_pnl_cents,
)

import logging
log = logging.getLogger("dashboard")


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #

def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def _sibling_sim_db(db_path: str) -> str | None:
    """For a live.db path, return the sibling sim.db path. None when
    the path doesn't look like a live database (sim itself never
    needs this fallback). Lets the model-snapshot fetch fall through
    to sim's data when live.db is empty — the same trained model
    runs in both modes, so sim's most-recent snapshot is an honest
    proxy for the live model's current state until a live executor
    starts writing its own snapshots.
    """
    if db_path.endswith("/live.db") or db_path.endswith("\\live.db"):
        return db_path[: -len("live.db")] + "sim.db"
    return None


def fetch_latest_model(db_path: str) -> dict | None:
    if not Path(db_path).exists():
        return None
    try:
        with closing(_conn(db_path)) as c:
            row = c.execute(
                "SELECT * FROM model_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            out = dict(row) if row else None
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return None
    # Fall back to the sibling sim.db when the live DB has no
    # snapshot yet. This is what makes the live dashboard's model
    # cards show the current state of each bot's model — they're
    # the same trained models in both modes, so reading sim's
    # latest snapshot is honest (and the alternative is showing
    # an empty card forever, since no live bot exists yet to
    # write its own snapshots).
    if out is None:
        sim_path = _sibling_sim_db(db_path)
        if sim_path and Path(sim_path).exists():
            try:
                with closing(_conn(sim_path)) as c:
                    row = c.execute(
                        "SELECT * FROM model_snapshots "
                        "ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    out = dict(row) if row else None
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                pass
    if out is None:
        return None
    # Augment with closed-bet feedback stats — separate try so a missing
    # training_pairs table on older bot DBs doesn't blank the model card.
    try:
        with closing(_conn(db_path)) as c:
            pairs = c.execute(
                "SELECT COUNT(*) AS n, "
                "       COALESCE(SUM(MIN(MAX(horizon_weeks, 0), 1)), 0) AS w "
                "FROM training_pairs"
            ).fetchone()
            if pairs:
                out["training_pairs_count"] = int(pairs["n"] or 0)
                out["training_pairs_total_weight"] = float(pairs["w"] or 0.0)
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        pass
    # Realized-bet win rate. Wins / (wins + losses) over all closed
    # positions for this bot. Distinct from the training-holdout
    # directional accuracy — this is the model's actual performance on
    # live Kalshi outcomes, the ultimate ground truth.
    try:
        with closing(_conn(db_path)) as c:
            wl = c.execute(
                "SELECT "
                "  SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) wins, "
                "  SUM(CASE WHEN realized_pnl_cents < 0 THEN 1 ELSE 0 END) losses "
                "FROM positions WHERE status = 'closed'"
            ).fetchone()
            if wl:
                out["actual_wins"] = int(wl["wins"] or 0)
                out["actual_losses"] = int(wl["losses"] or 0)
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        pass
    return out


def _safe_query(db_path: str, query: str, params: tuple = ()) -> List[dict]:
    """Single helper that all fetch_* functions use to query a bot's
    SQLite DB. Returns [] if the file is missing OR the table doesn't
    exist on this bot's schema. Bot DBs aren't required to have every
    table the gas-prices schema defines (e.g. natural-gas doesn't have
    `position_marks` since it's a daily-cadence bot, not continuous).

    Also tolerates files that aren't SQLite at all (DatabaseError) —
    e.g. if a registry entry's db_path ever points at a JSONL.
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            return [dict(r) for r in c.execute(query, params).fetchall()]
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []


def fetch_bot_effective_config(db_path: str) -> dict | None:
    """Read ``<data_dir>/effective_config.json`` next to the bot's sim.db.

    Bots that use ``kalshi_sdk.write_effective_config`` at startup emit
    this file with their *resolved* gates (post env-overrides, post
    per-bot widens). When present, the dashboard's buy-criteria panel
    renders these instead of the dashboard YAML's display defaults —
    so what the panel claims is what the bot actually applies.

    Returns ``None`` when the file is missing or unreadable; the caller
    falls back to the dashboard YAML and marks the panel as "showing
    dashboard defaults — bot has not reported its live config".

    Tolerant of the bot writing JSON elsewhere — we also try the
    sibling ``effective_config.json`` alongside the watchlist.json that
    sport bots use as their data anchor.
    """
    if not db_path:
        return None
    candidates = [Path(db_path).parent / "effective_config.json"]
    try:
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def resolve_bot_thresholds(
    bot: dict,
    fallback_edge: dict,
    fallback_validators: dict,
    fallback_risk: dict,
    fallback_hedge: dict,
) -> Tuple[dict, dict, dict, dict, dict, dict]:
    """Build the per-bot edge/validators/risk/hedge dicts the renderer uses.

    Priority is the bot's live ``effective_config.json`` (written at
    startup by ``kalshi_sdk.write_effective_config``); missing fields
    fall back to the dashboard YAML so a partially-reported config
    still looks complete to the user.

    Returns ``(edge, validators, risk, hedge, extra, source_meta)`` where
    ``extra`` is the bot's own ``extra`` block (used by the modal to
    render bot-specific rules that don't fit the shared schema — e.g.
    the tennis bot's Pinnacle-as-reference framing) and ``source_meta``
    carries provenance the rules modal renders:

        {"source": "live" | "fallback",
         "captured_at": "..." | None,
         "missing_keys": [...]}    # fields the bot didn't report
    """
    db_path = bot.get("db_path") or bot.get("watchlist_json_path") or ""
    live = fetch_bot_effective_config(db_path)
    if not live:
        return (
            dict(fallback_edge or {}),
            dict(fallback_validators or {}),
            dict(fallback_risk or {}),
            dict(fallback_hedge or {}),
            {},
            {"source": "fallback", "captured_at": None, "missing_keys": []},
        )
    missing: List[str] = []

    def _merge(live_section: Any, fallback: dict, section_label: str) -> dict:
        merged = dict(fallback or {})
        live_dict = live_section if isinstance(live_section, dict) else {}
        for k, v in live_dict.items():
            if v is not None:
                merged[k] = v
        # Track keys the dashboard expected but the bot didn't report —
        # surfaced in the modal so the user knows the panel is mixed.
        for k in (fallback or {}):
            if k not in live_dict or live_dict.get(k) is None:
                missing.append(f"{section_label}.{k}")
        return merged

    edge = _merge(live.get("edge"), fallback_edge, "edge")
    validators = _merge(live.get("validators"), fallback_validators, "validators")
    risk = _merge(live.get("risk"), fallback_risk, "risk")
    hedge = _merge(live.get("hedge"), fallback_hedge, "hedge")
    extra = live.get("extra") if isinstance(live.get("extra"), dict) else {}
    return (
        edge, validators, risk, hedge, extra,
        {
            "source": "live",
            "captured_at": live.get("captured_at"),
            "missing_keys": missing,
        },
    )


def fetch_summary(db_path: str, period_days: int | None = None) -> dict:
    """Lifetime + recent stats used by the Summary section.

    ``period_days`` filters the period-scoped fields (period_bets_made,
    period_net_pnl_cents, period_wins, period_losses) to bets that
    opened (for bets_made) or closed (for P&L/wins/losses) within the
    last N days. None → lifetime.
    """
    empty = {
        "total_bets": 0, "open_count": 0, "exposure_cents": 0,
        "active_contracts": 0,
        "closed_count": 0, "realized_pnl_cents": 0,
        "wins_lifetime": 0, "losses_lifetime": 0,
        "avg_win_cents": 0, "avg_loss_cents": 0,
        "bets_today": 0, "this_week_pnl_cents": 0,
        "biggest_win_cents": 0, "biggest_loss_cents": 0,
        "period_bets_made": 0, "period_net_pnl_cents": 0,
        "period_wins": 0, "period_losses": 0,
        "period_money_spent_cents": 0,
        "period_money_gained_cents": 0,
        "period_contracts_bought": 0,
        "potential_gain_cents": 0,
        "active_money_spent_cents": 0,
    }
    if not Path(db_path).exists():
        return empty
    try:
        with closing(_conn(db_path)) as c:
            total = c.execute(
                "SELECT COUNT(*) n FROM positions"
            ).fetchone()
            open_row = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(entry_price_cents * contracts), 0) exp "
                "FROM positions WHERE status = 'open'"
            ).fetchone()
            # Active-bets count + active-contracts sum: filter out
            # zombie open positions whose ticker has already settled
            # (>60 min past close). Mirrors the same guard the
            # active-bets table applies (``hide_settled``) so the
            # headline cards agree with what's rendered in the table.
            open_rows = c.execute(
                "SELECT p.ticker, p.contracts, p.entry_price_cents, "
                "  (SELECT mv.minutes_to_close FROM market_views mv "
                "     WHERE mv.ticker = p.ticker "
                "     ORDER BY mv.id DESC LIMIT 1) AS mtc "
                "FROM positions p WHERE p.status = 'open'"
            ).fetchall()
            active_count = 0
            active_contracts = 0
            active_money_spent_cents = 0
            active_potential_gain_cents = 0
            for r in open_rows:
                mtc = r["mtc"]
                if mtc is None:
                    mtc = minutes_to_close_from_ticker(r["ticker"])
                if (mtc if mtc is not None else 0) >= -60:
                    active_count += 1
                    ctr = r["contracts"] or 0
                    entry_c = r["entry_price_cents"] or 0
                    active_contracts += ctr
                    fee_c = kalshi_fee_cents(entry_c, ctr)
                    active_money_spent_cents += entry_c * ctr + fee_c
                    active_potential_gain_cents += (100 - entry_c) * ctr - fee_c
            closed_row = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(realized_pnl_cents), 0) pnl, "
                "SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) wins, "
                "SUM(CASE WHEN realized_pnl_cents < 0 THEN 1 ELSE 0 END) losses "
                "FROM positions WHERE status = 'closed'"
            ).fetchone()
            avg_win = c.execute(
                "SELECT COALESCE(AVG(realized_pnl_cents), 0) v FROM positions "
                "WHERE status = 'closed' AND realized_pnl_cents > 0"
            ).fetchone()
            avg_loss = c.execute(
                "SELECT COALESCE(AVG(realized_pnl_cents), 0) v FROM positions "
                "WHERE status = 'closed' AND realized_pnl_cents < 0"
            ).fetchone()
            biggest_win = c.execute(
                "SELECT COALESCE(MAX(realized_pnl_cents), 0) v FROM positions WHERE status='closed'"
            ).fetchone()
            biggest_loss = c.execute(
                "SELECT COALESCE(MIN(realized_pnl_cents), 0) v FROM positions WHERE status='closed'"
            ).fetchone()
            this_week_pnl = c.execute(
                "SELECT COALESCE(SUM(realized_pnl_cents), 0) v FROM positions "
                "WHERE status = 'closed' "
                "  AND date(exited_at) >= date('now', '-6 days')"
            ).fetchone()
            this_week_wl = c.execute(
                "SELECT "
                "  SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) wins, "
                "  SUM(CASE WHEN realized_pnl_cents < 0 THEN 1 ELSE 0 END) losses "
                "FROM positions WHERE status = 'closed' "
                "  AND date(exited_at) >= date('now', '-6 days')"
            ).fetchone()
            bets_today = c.execute(
                "SELECT COUNT(*) n FROM trades "
                "WHERE kind = 'entry' AND substr(created_at, 1, 10) = date('now')"
            ).fetchone()
            # Potential gain + active money-spent are computed inline
            # from the open-rows loop above (matches the active-bets
            # table renderer cell-for-cell: subtracts Kalshi entry fee
            # on potential, includes entry fee in cost, and applies the
            # same hide-settled mtc≥−60 filter).
            # Period-filtered values: when period_days is None, scope
            # to lifetime; otherwise restrict to the rolling window.
            if period_days is None:
                period_bets_made = total["n"]
                period_net_pnl = closed_row["pnl"] or 0
                period_wins = closed_row["wins"] or 0
                period_losses = closed_row["losses"] or 0
                spent_row = c.execute(
                    "SELECT COALESCE(SUM(entry_price_cents * contracts), 0) v "
                    "FROM positions"
                ).fetchone()
                gained_row = c.execute(
                    "SELECT COALESCE("
                    "  SUM(entry_price_cents * contracts + realized_pnl_cents), 0"
                    ") v FROM positions WHERE status = 'closed'"
                ).fetchone()
                contracts_row = c.execute(
                    "SELECT COALESCE(SUM(contracts), 0) v "
                    "FROM positions WHERE status = 'closed'"
                ).fetchone()
                period_money_spent = spent_row["v"] or 0
                period_money_gained = gained_row["v"] or 0
                period_contracts_bought = contracts_row["v"] or 0
            else:
                # bets_made = opened in the window (open or closed)
                period_window = f"-{int(period_days)} days"
                pmade = c.execute(
                    "SELECT COUNT(*) n FROM positions "
                    "WHERE date(opened_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                pclosed = c.execute(
                    "SELECT COALESCE(SUM(realized_pnl_cents), 0) pnl, "
                    "  SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) wins, "
                    "  SUM(CASE WHEN realized_pnl_cents < 0 THEN 1 ELSE 0 END) losses "
                    "FROM positions WHERE status = 'closed' "
                    "  AND date(exited_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                period_bets_made = pmade["n"]
                period_net_pnl = pclosed["pnl"] or 0
                period_wins = pclosed["wins"] or 0
                period_losses = pclosed["losses"] or 0
                # Money spent = cost basis of every position opened in
                # the period (open + closed). Bets still live count.
                spent_row = c.execute(
                    "SELECT COALESCE(SUM(entry_price_cents * contracts), 0) v "
                    "FROM positions WHERE date(opened_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                # Money gained = cash returned from positions closed
                # in the period (entry × contracts + realized_pnl ≥ 0).
                gained_row = c.execute(
                    "SELECT COALESCE("
                    "  SUM(entry_price_cents * contracts + realized_pnl_cents), 0"
                    ") v FROM positions WHERE status = 'closed' "
                    "  AND date(exited_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                contracts_row = c.execute(
                    "SELECT COALESCE(SUM(contracts), 0) v "
                    "FROM positions WHERE status = 'closed' "
                    "  AND date(exited_at) >= date('now', ?)",
                    (period_window,),
                ).fetchone()
                period_money_spent = spent_row["v"] or 0
                period_money_gained = gained_row["v"] or 0
                period_contracts_bought = contracts_row["v"] or 0
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return empty
    return {
        "total_bets": int(total["n"] or 0),
        "open_count": int(active_count),
        "exposure_cents": int(open_row["exp"] or 0),
        "active_contracts": int(active_contracts),
        "closed_count": int(closed_row["n"] or 0),
        "realized_pnl_cents": int(closed_row["pnl"] or 0),
        "wins_lifetime": int(closed_row["wins"] or 0),
        "losses_lifetime": int(closed_row["losses"] or 0),
        "avg_win_cents": int(round(float(avg_win["v"] or 0))),
        "avg_loss_cents": int(round(float(avg_loss["v"] or 0))),
        "biggest_win_cents": int(biggest_win["v"] or 0),
        "biggest_loss_cents": int(biggest_loss["v"] or 0),
        "this_week_pnl_cents": int(this_week_pnl["v"] or 0),
        "this_week_wins": int(this_week_wl["wins"] or 0),
        "this_week_losses": int(this_week_wl["losses"] or 0),
        "bets_today": int(bets_today["n"] or 0),
        "period_bets_made": int(period_bets_made or 0),
        "period_net_pnl_cents": int(period_net_pnl or 0),
        "period_wins": int(period_wins or 0),
        "period_losses": int(period_losses or 0),
        "period_money_spent_cents": int(period_money_spent or 0),
        "period_money_gained_cents": int(period_money_gained or 0),
        "period_contracts_bought": int(period_contracts_bought or 0),
        "potential_gain_cents": int(active_potential_gain_cents),
        "active_money_spent_cents": int(active_money_spent_cents),
    }


def fetch_latest_open_position(db_path: str) -> dict | None:
    """The single most-recently opened position (with mark info).

    Tolerates DBs that don't have a positions/position_marks table at
    all — the natural-gas bot writes only model_snapshots + market_views
    since it produces signals, not simulated trades.
    """
    if not Path(db_path).exists():
        return None
    try:
        with closing(_conn(db_path)) as c:
            row = c.execute(
                "SELECT p.*, m.yes_ask_cents AS mark_yes_ask, m.no_ask_cents AS mark_no_ask, "
                "m.yes_bid_cents AS mark_yes_bid, m.mid_cents AS mark_mid, "
                "m.spread_cents AS mark_spread, m.updated_at AS mark_updated_at "
                "FROM positions p LEFT JOIN position_marks m ON p.id = m.position_id "
                "WHERE p.status = 'open' ORDER BY p.opened_at DESC LIMIT 1"
            ).fetchone()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return None
    return dict(row) if row else None


def fetch_active_bets_with_marks(db_path: str) -> List[dict]:
    """Open positions joined with their latest mark + latest mtc + the
    most recent market_views row's strike bounds (so the summary table
    can render the human Question for each row). The local schema
    uses strike_low / strike_high (not floor_strike / cap_strike — the
    Kalshi API names).

    For ``mark_yes_ask`` / ``mark_no_ask`` we COALESCE position_marks
    onto market_views so bots that don't write the position_marks
    table (e.g. natural-gas, which only emits model_snapshots +
    market_views) still show a live "Current" cell on the Home tab's
    active-bets table.
    """
    return _safe_query(
        db_path,
        "SELECT p.*, "
        "       COALESCE(m.yes_ask_cents, "
        "         (SELECT mv.yes_ask_cents FROM market_views mv "
        "            WHERE mv.ticker = p.ticker "
        "            ORDER BY mv.id DESC LIMIT 1)"
        "       ) AS mark_yes_ask, "
        "       COALESCE(m.no_ask_cents, "
        "         (SELECT mv.no_ask_cents FROM market_views mv "
        "            WHERE mv.ticker = p.ticker "
        "            ORDER BY mv.id DESC LIMIT 1)"
        "       ) AS mark_no_ask, "
        "       (SELECT mv.minutes_to_close FROM market_views mv "
        "          WHERE mv.ticker = p.ticker "
        "          ORDER BY mv.id DESC LIMIT 1) AS minutes_to_close, "
        "       (SELECT mv.strike_low FROM market_views mv "
        "          WHERE mv.ticker = p.ticker "
        "          ORDER BY mv.id DESC LIMIT 1) AS floor_strike, "
        "       (SELECT mv.strike_high FROM market_views mv "
        "          WHERE mv.ticker = p.ticker "
        "          ORDER BY mv.id DESC LIMIT 1) AS cap_strike "
        "FROM positions p LEFT JOIN position_marks m ON p.id = m.position_id "
        "WHERE p.status = 'open' ORDER BY p.opened_at DESC")


# ── Kalshi portfolio → cross-bot history ────────────────────────────
# The History tab now sources every closed bet from Kalshi's real
# /portfolio/settlements + /portfolio/fills endpoints (source of
# truth), not from local sim_state / sim.db files. That way the tab
# shows exactly what actually happened on the account — a paper trade
# recorded in a bot's sim_state that never became a Kalshi order
# doesn't appear (matches the user's 2026-07-11 request: "only show a
# bot when at least one contract has been bought"), and the ledger is
# unified across every bot instead of tennis-only.

_KALSHI_HISTORY_CACHE: Dict[str, Any] = {"at": 0.0, "settlements": [],
                                            "fills": []}
_KALSHI_HISTORY_TTL_S = 60.0


def _fetch_all_kalshi_history() -> Tuple[List[dict], List[dict]]:
    """Return ``(settlements, fills)`` across every Kalshi ticker.
    60s cached — settlements never change post-fact and fills only
    append, so a longer TTL would also be safe. Returns cached data
    on any API failure so a transient error doesn't blank the tab."""
    import time
    now = time.time()
    if (now - _KALSHI_HISTORY_CACHE["at"] < _KALSHI_HISTORY_TTL_S):
        return (_KALSHI_HISTORY_CACHE["settlements"],
                _KALSHI_HISTORY_CACHE["fills"])
    try:
        from . import tennis as _tennis  # reuse the tennis-client helper
        client = _tennis._get_kalshi_client()
    except Exception:  # noqa: BLE001
        log.exception("kalshi client init failed")
        return (_KALSHI_HISTORY_CACHE["settlements"],
                _KALSHI_HISTORY_CACHE["fills"])
    if client is None:
        return ([], [])
    try:
        settlements = client.iter_settlements() or []
        fills = client.iter_fills() or []
    except Exception:  # noqa: BLE001
        log.exception("kalshi settlements/fills fetch failed; serving stale")
        return (_KALSHI_HISTORY_CACHE["settlements"],
                _KALSHI_HISTORY_CACHE["fills"])
    _KALSHI_HISTORY_CACHE.update(
        {"at": now, "settlements": settlements, "fills": fills})
    return settlements, fills


def _bot_ticker_prefix_index(bots: List[dict]
                              ) -> List[Tuple[str, dict]]:
    """Build ``[(prefix, bot_dict)]`` sorted longest-prefix-first so a
    KXNBASUMMERGAME ticker matches the NBA bot instead of any bot
    whose config only carries ``KXNBAGAME``."""
    idx: List[Tuple[str, dict]] = []
    for b in bots:
        prefixes = list(b.get("series_prefixes") or [])
        if not prefixes and b.get("series_ticker"):
            prefixes = [str(b["series_ticker"])]
        for p in prefixes:
            p = str(p).strip().rstrip("*")
            if p:
                idx.append((p, b))
    idx.sort(key=lambda x: len(x[0]), reverse=True)
    return idx


def _ticker_to_bot(ticker: str,
                    prefix_index: List[Tuple[str, dict]]
                    ) -> Optional[dict]:
    """Return the bot config whose ticker prefix matches ``ticker``,
    or None if no bot claims this ticker."""
    t = (ticker or "").upper()
    for prefix, bot in prefix_index:
        if t.startswith(prefix.upper()):
            return bot
    return None


def _load_sim_state_enrichment(bots: List[dict]) -> Dict[str, dict]:
    """Index every bot's ``sim_state.json`` ``closed_positions`` +
    ``positions`` arrays by Kalshi ticker so ``build_kalshi_cross_bot
    _history`` can hydrate the History-tab columns the ledger alone
    doesn't carry — Title, Model p, Entry EV — the same way the
    tennis-only history did.

    Sport bots (tennis / WNBA / NBA / darts / TT / MLB / world-cup)
    write records in a shared schema (originally the tennis
    executor's): each entry has ``ticker`` OR ``match_id``, plus
    ``event_title`` / ``title``, ``entry_model_prob`` /
    ``entry_market_prob``, ``player_a`` / ``player_b`` /
    ``side_player``. Standard bots (gas / claims / cpi / natgas)
    persist trades in sim.db rather than sim_state.json — those
    rows land without enrichment (Title falls back to the ticker,
    Model p / Entry EV display "—").

    Returns ``{ticker → enrichment_dict}``. Keys are indexed with
    BOTH the side-specific ticker (``KXATPMATCH-…-DAL``) and the
    event ticker / ``match_id`` (``KXATPMATCH-…``) so a settlement
    row can join on either.
    """
    idx: Dict[str, dict] = {}
    for b in bots:
        sim_path = b.get("sim_state_path")
        if not sim_path:
            continue
        # Index BOTH sibling ledgers (outputs/ paper + outputs-live/
        # executor), not just the current mode's file: the account
        # settlements this joins against are REAL trades recorded by
        # the live executors, so the sim dashboard's paper-side path
        # alone left almost every settlement without a model prob
        # ("untagged" in the By-edge panel — user 2026-07-10 asked for
        # those to be mapped).
        paths = [sim_path]
        if "outputs-live" in sim_path:
            paths.append(sim_path.replace("outputs-live", "outputs"))
        elif "/outputs/" in sim_path:
            paths.append(sim_path.replace("/outputs/", "/outputs-live/"))
        # Archived ledgers too: the 2026-07-08 tennis audit reset the
        # live state file but kept the full pre-reset ledger beside it
        # as *.pre-audit-backup — that's where the May/June real
        # trades' model probs live. Current files are indexed first,
        # so a backup never overrides a live record.
        paths += [f"{_p}.pre-audit-backup" for _p in list(paths)]
        records: list = []
        for _p in paths:
            try:
                with open(_p, "r", encoding="utf-8") as f:
                    st = json.load(f) or {}
            except (OSError, json.JSONDecodeError):
                continue
            records += (st.get("closed_positions") or [])                        + (st.get("positions") or [])
        for c in records:
            ticker = c.get("ticker") or c.get("match_id") or ""
            if not ticker:
                continue
            pa = c.get("player_a") or ""
            pb = c.get("player_b") or ""
            matchup = (f"{pa} vs {pb}" if pa and pb else "")
            side_player = c.get("side_player") or ""
            title = (c.get("event_title") or c.get("title") or ""
                     or (matchup and side_player
                         and f"{matchup} — bet on {side_player}")
                     or matchup or "")
            entry_mp = c.get("entry_model_prob")
            entry_kp = (c.get("entry_market_prob")
                        or c.get("entry_kalshi_prob"))
            payload = {
                "_title": title,
                "_match": matchup,
                "_side_player": side_player,
                "entry_model_prob": entry_mp,
                "entry_market_prob": entry_kp,
            }
            # Primary key: side-specific ticker. Also register the
            # match_id / event_ticker so settlements arriving on the
            # base ticker can still resolve.
            for key in {ticker, c.get("match_id") or "",
                        c.get("event_ticker") or ""}:
                if not key:
                    continue
                # First writer wins (closed_positions are iterated
                # before open positions) — EXCEPT that a record
                # carrying a real model prob upgrades one that
                # doesn't, so a paper-side stub can't block the live
                # executor's fully-tagged record for the same ticker.
                prev = idx.get(key)
                if prev is None or (prev.get("entry_model_prob") is None
                                     and entry_mp is not None):
                    idx[key] = payload
    return idx


def _summarize_fills_by_ticker(fills: List[dict]) -> Dict[str, dict]:
    """Group fills by ticker, tracking open (buy) vs close (sell)
    legs. Yes-price is used uniformly so entry and exit sit on the
    same axis regardless of which side was actually traded."""
    by: Dict[str, dict] = {}
    for f in fills:
        t = f.get("ticker") or f.get("market_ticker") or ""
        if not t:
            continue
        n = float(f.get("count_fp") or 0)
        if n <= 0:
            continue
        action = (f.get("action") or "").lower()
        side = (f.get("side") or "yes").lower()
        yes_p = float(f.get("yes_price_dollars") or 0)
        if yes_p > 1.0:
            yes_p = yes_p / 100.0
        d = by.setdefault(t, {"open_n": 0.0, "open_yn": 0.0,
                                 "close_n": 0.0, "close_yn": 0.0,
                                 "side": side, "first_open_time": ""})
        if action == "sell":
            d["close_n"] += n
            d["close_yn"] += yes_p * n
        else:
            d["open_n"] += n
            d["open_yn"] += yes_p * n
            ct = f.get("created_time") or ""
            if ct and (not d["first_open_time"]
                       or ct < d["first_open_time"]):
                d["first_open_time"] = ct
    out: Dict[str, dict] = {}
    for t, d in by.items():
        open_avg = (d["open_yn"] / d["open_n"]) if d["open_n"] else None
        close_avg = ((d["close_yn"] / d["close_n"])
                      if d["close_n"] else None)
        out[t] = {
            "open_avg_dollars": open_avg,
            "close_avg_dollars": close_avg,
            "contracts": int(round(d["open_n"])),
            "side": d["side"],
            "first_open_time": d["first_open_time"],
        }
    return out


def build_kalshi_cross_bot_history(bots: List[dict]) -> List[dict]:
    """Real Kalshi portfolio → the standard ``global_history`` row
    shape, unified across every bot with settled positions on the
    account. Bot attribution derives from the ticker prefix using each
    bot's ``series_prefixes`` / ``series_ticker`` config.

    P&L math mirrors the tennis history adapter — see
    ``_join_settlement_with_sim_state`` in tennis.py — using Kalshi's
    per-side ``yes_count_fp`` / ``no_count_fp`` counts,
    ``yes_total_cost_dollars`` / ``no_total_cost_dollars`` costs,
    ``fee_cost`` and ``market_result``. Handles the "offset-closed"
    case where the bot opened YES then closed via a NO-side trade —
    Kalshi books both legs to settlement and the naive
    ``revenue - avg_price × count`` formula would double-count.
    """
    settlements, fills = _fetch_all_kalshi_history()
    prefix_idx = _bot_ticker_prefix_index(bots)
    fills_by_ticker = _summarize_fills_by_ticker(fills)
    enrich_idx = _load_sim_state_enrichment(bots)
    out: List[dict] = []
    for s in settlements:
        ticker = s.get("ticker") or s.get("market_ticker") or ""
        if not ticker:
            continue
        yes_n = float(s.get("yes_count_fp") or 0)
        no_n = float(s.get("no_count_fp") or 0)
        f = fills_by_ticker.get(ticker) or {}
        fill_contracts = float(f.get("contracts") or 0)
        fill_side = (f.get("side") or "yes").lower()
        if yes_n > 0 or no_n > 0:
            side_held = "yes" if yes_n > 0 else "no"
            contracts = int(yes_n if side_held == "yes" else no_n)
        elif fill_contracts > 0:
            side_held = fill_side
            contracts = int(fill_contracts)
        else:
            # Phantom settlement — Kalshi surfaces the row but we never
            # held or filled. Skip so the ledger + attribution stays
            # honest.
            continue

        yes_cost = float(s.get("yes_total_cost_dollars") or 0)
        no_cost = float(s.get("no_total_cost_dollars") or 0)
        fee = float(s.get("fee_cost") or 0)
        market_result = (s.get("market_result") or "").lower()
        offset_closed = (yes_n > 0 and no_n > 0)
        if offset_closed:
            total_cost = yes_cost + no_cost
            payout = (yes_n if market_result == "yes" else 0.0) \
                     + (no_n if market_result == "no" else 0.0)
            pnl_dollars = payout - total_cost - fee
        else:
            cost = yes_cost if side_held == "yes" else no_cost
            payout = float(contracts) if market_result == side_held else 0.0
            pnl_dollars = payout - cost - fee
        realized_cents = int(round(pnl_dollars * 100))

        # Weighted-avg fill price on the yes-axis (fills always
        # normalize to yes_price_dollars — see
        # ``_summarize_fills_by_ticker``). Fall back to a derived value
        # from settlement's per-side cost when fills aren't available.
        # The derived fallback normalizes to yes-axis too so downstream
        # EV math stays consistent regardless of which side we bought.
        open_avg = f.get("open_avg_dollars")
        if open_avg is None and contracts > 0:
            if side_held == "yes":
                derived = yes_cost / contracts
            else:
                # no_cost/contracts is NO-axis price paid; flip.
                derived = 1.0 - (no_cost / contracts)
            open_avg = derived if 0 < derived < 1 else None
        # Display entry price = price paid on the side we bought
        # (tennis idiom — NO bet at 40¢ displays "40c", not "60c"),
        # so History rows line up with what the user saw at open.
        if open_avg is None:
            entry_cents = None
        elif side_held == "yes":
            entry_cents = int(round(open_avg * 100))
        else:
            entry_cents = int(round((1 - open_avg) * 100))
        # Exit price on the side we bought: 100¢ if we won, 0¢ if we lost.
        exit_cents = 100 if market_result == side_held else 0

        opened_at = f.get("first_open_time") or ""
        settled_at = (s.get("settled_time")
                      or s.get("settle_time")
                      or s.get("created_time") or "")
        bot = _ticker_to_bot(ticker, prefix_idx)

        # Sim-state enrichment (Title / Model p / Entry EV). Look the
        # settlement up by side-specific ticker first, then event
        # ticker — the sport bots store either shape depending on when
        # the row was written. Missing entries fall through with None,
        # matching the tennis-only history renderer's em-dash idiom.
        enrich = (enrich_idx.get(ticker)
                  or enrich_idx.get(s.get("event_ticker") or "")
                  or {})
        entry_mp = enrich.get("entry_model_prob")
        try:
            entry_mp_f = (float(entry_mp)
                          if entry_mp is not None else None)
        except (TypeError, ValueError):
            entry_mp_f = None
        # Store on the yes-axis so ``_render_bet_history_block`` can
        # side-flip for NO bets (its rule: YES ⇒ p; NO ⇒ 1 − p).
        model_yes = entry_mp_f
        # Entry EV per contract on the side we bought, in decimal $:
        #   YES bet: mp − yes_price
        #   NO bet:  (1 − mp) − (1 − yes_price) = yes_price − mp
        # ``open_avg`` is the yes-axis fill price. Falls to None when
        # either the model prob or the fill wasn't recorded.
        if entry_mp_f is not None and open_avg is not None:
            if side_held == "yes":
                entry_ev = entry_mp_f - float(open_avg)
            else:
                entry_ev = float(open_avg) - entry_mp_f
        else:
            entry_ev = None

        out.append({
            "ticker": ticker,
            "side": side_held.upper(),
            "entry_price_cents": entry_cents,
            "exit_price_cents": exit_cents,
            "contracts": contracts,
            "realized_pnl_cents": realized_cents,
            "opened_at": opened_at,
            "exited_at": settled_at,
            "expected_ev_at_entry": entry_ev,
            "model_yes_prob_at_entry": model_yes,
            "_title": enrich.get("_title") or "",
            "_match": enrich.get("_match") or "",
            "_side_player": enrich.get("_side_player") or "",
            "_bot_key": bot.get("key") if bot else "unknown",
            "_bot_name": bot.get("name") if bot else "Unknown bot",
            "_dashboard_type": (bot.get("dashboard_type")
                                 if bot else "standard") or "standard",
            "_display": (bot.get("display") if bot else {}) or {},
        })
    out.sort(key=lambda r: r.get("exited_at") or "", reverse=True)
    return out


def filter_history_by_period(rows: List[dict],
                              period_days: int | None) -> List[dict]:
    """Trim closed-bet rows to the user-selected period window by
    ``exited_at``. None keeps everything. Shared by the server's
    cross-bot rollup and the History panel so the two ledgers always
    agree on what "this week" means. Rows whose timestamp can't be
    parsed are dropped (same rule both call sites already applied)."""
    if period_days is None:
        return rows
    cutoff_ts = (datetime.now(timezone.utc).timestamp()
                 - period_days * 86400)

    def _within(h: dict) -> bool:
        ex = h.get("exited_at") or ""
        try:
            if "T" in ex:
                t = datetime.fromisoformat(
                    ex.replace("Z", "+00:00")).timestamp()
            else:
                t = datetime.strptime(
                    ex[:19], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc).timestamp()
        except (TypeError, ValueError):
            return False
        return t >= cutoff_ts

    return [h for h in rows if _within(h)]


def fetch_bet_history(db_path: str, limit: int = 100) -> List[dict]:
    """Closed positions only — for the Bet History section.

    Tolerates schema drift across bots. ``gas_price_at_close`` only exists
    on the gas-prices simulator schema; bots without it still appear in
    the cross-bot Summary with an empty Gas-at-close cell. floor_strike
    + cap_strike are pulled via subqueries on market_views so the bet-
    history view can render the human Question text per row.
    """
    if not Path(db_path).exists():
        return []
    base_cols = ("p.id, p.ticker, p.side, p.entry_price_cents, p.exit_price_cents, "
                 "p.contracts, p.realized_pnl_cents, p.opened_at, p.exited_at")
    try:
        with closing(_conn(db_path)) as c:
            # Probe the schema once instead of try/except-ing the full query;
            # cheaper and clearer about *why* the fallback path runs.
            cols = {r["name"] for r in
                    c.execute("PRAGMA table_info(positions)").fetchall()}
            if not cols:
                # No positions table at all (e.g. natural-gas bot writes
                # only model_snapshots + market_views). Empty history.
                return []
            extras = [f"p.{c_}" for c_ in (
                "gas_price_at_close",
                "model_yes_prob_at_entry",
                "kalshi_yes_prob_at_entry",
                "break_even_probability",
                "expected_ev_at_entry",
                "error_type",
                # decision_json is the catch-all entry-time payload
                # every bot writes. Natural-gas (and any older bot
                # schema that doesn't have dedicated model_yes_prob_
                # at_entry / kalshi_yes_prob_at_entry columns)
                # stashes ``model_prob`` + ``kalshi_implied_prob``
                # in here — we parse them out below so the History
                # tab's Model-p cell is populated regardless of which
                # schema the bot ships with.
                "decision_json",
                # Set to N > 1 by the same-ticker dedupe pass when
                # multiple flap-trades on the same (ticker, side) were
                # collapsed into this row. The History renderer surfaces
                # a "×N" badge so the user can tell merged rows apart.
                "merged_trade_count",
                "merged_position_ids",
            ) if c_ in cols]
            select_cols = base_cols + (", " + ", ".join(extras) if extras else "")
            rows = c.execute(
                f"SELECT {select_cols}, "
                f"(SELECT mv.strike_low FROM market_views mv "
                f"   WHERE mv.ticker = p.ticker "
                f"   ORDER BY mv.id DESC LIMIT 1) AS floor_strike, "
                f"(SELECT mv.strike_high FROM market_views mv "
                f"   WHERE mv.ticker = p.ticker "
                f"   ORDER BY mv.id DESC LIMIT 1) AS cap_strike "
                f"FROM positions p WHERE p.status='closed' "
                f"ORDER BY p.exited_at DESC LIMIT ?", (limit,),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    out = [dict(r) for r in rows]
    # Backfill model_yes_prob_at_entry / kalshi_yes_prob_at_entry /
    # expected_ev_at_entry from decision_json for bots that don't have
    # dedicated columns (the natural-gas bot's older schema is the case
    # in production).
    for h in out:
        raw = h.get("decision_json")
        if not raw:
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        mp = payload.get("model_prob")
        kp = payload.get("kalshi_implied_prob")
        if mp is not None and h.get("model_yes_prob_at_entry") is None:
            try:
                h["model_yes_prob_at_entry"] = float(mp)
            except (TypeError, ValueError):
                pass
        if kp is not None and h.get("kalshi_yes_prob_at_entry") is None:
            try:
                h["kalshi_yes_prob_at_entry"] = float(kp)
            except (TypeError, ValueError):
                pass
        # Entry EV: derive from model_prob + side + ask + spread when the
        # bot's schema doesn't store it directly. Same formula as the
        # SDK's ev_for_side: p_side - ask_dollars - half_spread.
        if h.get("expected_ev_at_entry") is None and mp is not None:
            try:
                model_p = float(mp)
                side = (h.get("side") or "").upper()
                ya = payload.get("yes_ask_cents")
                na = payload.get("no_ask_cents")
                spr = payload.get("spread_cents") or 0
                half_spread = (float(spr) / 2.0) / 100.0
                ev = None
                if side == "YES" and ya is not None:
                    ev = model_p - (float(ya) / 100.0) - half_spread
                elif side == "NO" and na is not None:
                    ev = (1.0 - model_p) - (float(na) / 100.0) - half_spread
                if ev is not None:
                    h["expected_ev_at_entry"] = ev
            except (TypeError, ValueError):
                pass
    return out


# Fallback bankroll used when neither display.bankroll_cents nor a
# live Kalshi balance is available. The Kelly sizing column on the
# watchlist multiplies half-Kelly fractions against this — change it
# via display.bankroll_cents in dashboard.yaml if a particular bot
# deserves a bigger or smaller stake size, or set the Kalshi API
# creds on the dashboard host to drive Size off the real balance.
DEFAULT_BANKROLL_CENTS = 100_000  # $1,000 fallback


def bot_regime_status(db_path: str, days: int = 90,
                        min_bets: int = 10) -> dict:
    """Rolling edge-health check used by the Home-tab bot cards.

    Looks at the last ``days`` of closed bets with both
    ``expected_ev_at_entry`` and ``realized_pnl_cents`` recorded, then
    grades how well the realized cents-per-contract is tracking the
    predicted EV. Returns a small dict the renderer turns into a
    status pill.

    Statuses
    --------
    ``"green"``  — realized ≥ predicted × 0.5 (edge largely survived)
    ``"yellow"`` — realized > 0 but well below predicted (eroding)
    ``"red"``    — realized ≤ 0 where predicted was positive (anti-edge)
    ``"gray"``   — not enough data (or schema lacks the EV column)
    """
    empty = {"status": "gray", "label": "no data",
             "reason": "Needs closed bets with a recorded entry EV."}
    if not Path(db_path).exists():
        return empty
    try:
        with closing(_conn(db_path)) as c:
            cols = {r["name"] for r in
                    c.execute("PRAGMA table_info(positions)").fetchall()}
            if "expected_ev_at_entry" not in cols:
                return empty
            rows = c.execute(
                "SELECT expected_ev_at_entry, realized_pnl_cents, contracts "
                "FROM positions WHERE status = 'closed' "
                "  AND expected_ev_at_entry IS NOT NULL "
                "  AND realized_pnl_cents IS NOT NULL "
                "  AND contracts > 0 "
                "  AND date(exited_at) >= date('now', ?)",
                (f"-{int(days)} days",),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return empty
    if len(rows) < min_bets:
        return {
            "status": "gray",
            "label": "warming up",
            "reason": (f"Only {len(rows)} closed bets in the last "
                       f"{days}d — need ≥ {min_bets} for a reading."),
        }
    pred_sum_cents = 0.0
    realized_per_sum = 0.0
    total_pnl_cents = 0
    for r in rows:
        try:
            ev = float(r["expected_ev_at_entry"])
            pnl = int(r["realized_pnl_cents"])
            contracts = int(r["contracts"])
        except (TypeError, ValueError):
            continue
        if contracts <= 0:
            continue
        pred_sum_cents += ev * 100.0
        realized_per_sum += pnl / contracts
        total_pnl_cents += pnl
    n = len(rows)
    pred = pred_sum_cents / n
    realized = realized_per_sum / n
    # Edge-survival rule mirrors the EV-vs-realized table: realized
    # within half of predicted (or within 1¢ at tiny predicted EV)
    # is "the edge held". Negative realized with positive predicted is
    # the anti-edge regime that warrants pausing the bot.
    if pred <= 0:
        status = "green" if realized > 0 else "red"
        label = "edge confirmed" if status == "green" else "anti-edge"
    elif realized >= max(pred * 0.5, pred - 1.0):
        status, label = "green", "edge confirmed"
    elif realized > 0:
        status, label = "yellow", "edge eroding"
    else:
        status, label = "red", "anti-edge"
    reason = (f"{n} bets · predicted {pred:+.1f}¢/contract · "
              f"realized {realized:+.1f}¢/contract · "
              f"net {'+' if total_pnl_cents > 0 else ('−' if total_pnl_cents < 0 else '')}"
              f"${abs(total_pnl_cents)/100:.2f} over {days}d")
    return {"status": status, "label": label, "reason": reason,
            "predicted_cents": pred, "realized_cents": realized,
            "total_pnl_cents": total_pnl_cents, "n_bets": n,
            "days": days}


def fetch_global_summary(bots: List[dict],
                          period_days: int | None = None) -> dict:
    """Cross-bot rollup for the Summary section's headline cards.

    ``active_bets`` is always the live across-bots count regardless of
    the period filter (per user spec — current state, not historical).
    The other rolled-up fields (``period_bets_made``,
    ``period_net_pnl_cents``, ``period_win_pct``) honor ``period_days``.
    """
    rollup = {
        "active_bets": 0,            # always current — never period-scoped
        "active_contracts": 0,       # always current — sum of contracts open
        "active_bots": 0,            # distinct bots with at least one active bet
        "period_bets_made": 0,
        "period_net_pnl_cents": 0,
        "period_wins": 0,
        "period_losses": 0,
        "period_money_spent_cents": 0,
        "period_money_gained_cents": 0,
        "period_contracts_bought": 0,
        "potential_gain_cents": 0,    # always current — fee-subtracted
        "active_money_spent_cents": 0,  # entry cost of currently-open bets only
        "this_week_pnl_cents": 0,     # always lifetime-of-last-7-days
        # Lifetime fields kept for callers that still want them.
        "total_bets": 0,
        "net_pnl_cents": 0,
        "wins": 0,
        "losses": 0,
        "best_bot_name": "—",
        "best_bot_pnl_cents": 0,
        "per_bot": [],
    }
    for b in bots:
        if not b.get("available"):
            continue
        # Tennis bot keeps its paper-trade ledger in a JSON file rather
        # than a sim.db; we map it onto the same dict shape via
        # ``tennis.summary_for_rollup`` so the cross-bot summary cards
        # at the top of the home page DO include tennis volume + P&L.
        if b.get("dashboard_type") == "sport":
            from . import tennis as _tennis
            s = _tennis.summary_for_rollup(b.get("sim_state_path"))
        elif b.get("dashboard_type") == "survivor":
            from . import survivor as _survivor
            s = _survivor.summary_for_rollup(b.get("sim_state_path"))
        elif b.get("dashboard_type") in ("billboard", "reality"):
            # Billboard + reality-leaks write a real sim.db (standard
            # schema) so the same summary reader the gas / claims
            # bots use works here too. The legacy
            # `_billboard.summary_for_rollup` always returned zeros
            # (advisory-only era).
            s = fetch_summary(b["db_path"], period_days=period_days)
        elif b.get("dashboard_type") and b["dashboard_type"] != "standard":
            continue
        else:
            s = fetch_summary(b["db_path"], period_days=period_days)
        rollup["active_bets"] += s.get("open_count", 0)
        rollup["active_contracts"] += s.get("active_contracts", 0)
        if s.get("open_count", 0) > 0:
            rollup["active_bots"] += 1
        rollup["period_bets_made"] += s.get("period_bets_made", 0)
        rollup["period_net_pnl_cents"] += s.get("period_net_pnl_cents", 0)
        rollup["period_wins"] += s.get("period_wins", 0)
        rollup["period_losses"] += s.get("period_losses", 0)
        rollup["period_money_spent_cents"] += s.get("period_money_spent_cents", 0)
        rollup["period_money_gained_cents"] += s.get("period_money_gained_cents", 0)
        rollup["period_contracts_bought"] += s.get("period_contracts_bought", 0)
        rollup["potential_gain_cents"] += s.get("potential_gain_cents", 0)
        rollup["active_money_spent_cents"] += s.get("active_money_spent_cents", 0)
        rollup["this_week_pnl_cents"] += s.get("this_week_pnl_cents", 0)
        rollup["total_bets"] += s.get("total_bets", 0)
        rollup["net_pnl_cents"] += s.get("realized_pnl_cents", 0)
        rollup["wins"] += s.get("wins_lifetime", 0)
        rollup["losses"] += s.get("losses_lifetime", 0)
        rollup["per_bot"].append((b["name"], s))
        better = (
            s.get("realized_pnl_cents", 0) > rollup["best_bot_pnl_cents"]
            or (s.get("realized_pnl_cents", 0) == rollup["best_bot_pnl_cents"]
                and s.get("total_bets", 0) > 0
                and rollup["best_bot_name"] == "—")
        )
        if better:
            rollup["best_bot_name"] = b["name"]
            rollup["best_bot_pnl_cents"] = s.get("realized_pnl_cents", 0)
    period_closed = rollup["period_wins"] + rollup["period_losses"]
    rollup["period_win_pct"] = (
        rollup["period_wins"] / period_closed if period_closed else 0.0
    )
    return rollup


def _compute_active_bets_totals(active_bets: List[dict],
                                  hide_settled: bool = True) -> dict:
    """Compute the Active-bets headline values directly from the same
    list that feeds ``_render_active_bets_table`` — guarantees the
    Home-tab cards match the table column totals cell-for-cell
    regardless of how each bot's data is sourced.

    Returns dict with: ``open_count``, ``active_contracts``,
    ``active_bots`` (distinct bots in the table), ``active_money_spent_cents``
    (sum of Entry cost column, = entry × contracts + Kalshi fee per row),
    ``potential_gain_cents`` (sum of Potential gain column, = (100 −
    entry) × contracts − Kalshi fee per row).
    """
    if hide_settled:
        active_bets = [
            b for b in active_bets
            if (
                (b.get("minutes_to_close")
                 if b.get("minutes_to_close") is not None
                 else minutes_to_close_from_ticker(b.get("ticker"))) or 0
            ) >= -60
        ]
    contracts_total = 0
    money_spent_cents = 0
    potential_gain_cents = 0
    bots_in_table = set()
    for b in active_bets:
        entry = b.get("entry_price_cents") or 0
        ctr = b.get("contracts") or 0
        fee_c = kalshi_fee_cents(entry, ctr)
        contracts_total += ctr
        money_spent_cents += entry * ctr + fee_c
        potential_gain_cents += (100 - entry) * ctr - fee_c
        bk = b.get("_bot_key") or b.get("_bot_name") or ""
        if bk:
            bots_in_table.add(bk)
    return {
        "open_count": len(active_bets),
        # ``active_bets`` mirrors ``open_count`` so callers that
        # ``summary.update(...)`` this dict also overwrite the rollup's
        # ``active_bets`` field — the card value the Home tab renders
        # for "Active bets". Without this, the card kept the un-filtered
        # per-bot sum (which counts zombie positions whose ticker close
        # time is already past) and disagreed with the table beneath it.
        "active_bets": len(active_bets),
        "active_contracts": int(contracts_total),
        "active_bots": len(bots_in_table),
        "active_money_spent_cents": int(money_spent_cents),
        "potential_gain_cents": int(potential_gain_cents),
    }


def _live_kalshi_held_tickers() -> Optional[set]:
    """Set of tickers currently held on Kalshi's portfolio, or ``None``
    when the fetch fails / no creds. Used by the LIVE dashboard to keep
    the home-page Active bets table honest against actual portfolio
    state — the bot sim_state files can drift when the executor's
    reconciliation lags, and the last thing the user wants is the
    Home tab claiming positions they don't own.
    """
    from .kalshi_client import get_open_positions
    pos, _err = get_open_positions()
    if pos is None:
        return None
    return {p.get("ticker") for p in pos if p.get("ticker")}


def _build_global_active_bets(bots: List[dict],
                                mode: str | None = None) -> List[dict]:
    """Cross-bot list of active-bet dicts in the shape the active-bets
    table renderer (and ``_compute_active_bets_totals``) expects.
    Tagged with ``_bot_key`` so the distinct-bots count works. Skips
    bots whose data source isn't available, matching the page-render
    bot iteration.

    ``mode="live"`` cross-references sport / billboard bots' rows
    against Kalshi's live /portfolio/positions so the Home tab's
    Active bets card shows the real held-portfolio count, not
    whatever the bot's local sim_state has drifted to. Same graceful
    degradation as the per-bot Active bets: when the Kalshi fetch
    fails we leave the rows in place rather than blanking them.
    """
    held: Optional[set] = None
    if mode == "live":
        held = _live_kalshi_held_tickers()
    out: List[dict] = []
    for b in bots:
        if not b.get("available"):
            continue
        dt = b.get("dashboard_type") or "standard"
        if dt == "sport":
            from . import tennis as _tennis
            rows = _tennis.active_bets_for_rollup(
                b.get("sim_state_path"),
                watchlist_path=b.get("watchlist_json_path"))
        elif dt == "survivor":
            continue  # advisory bot — no positions
        elif dt == "billboard":
            # Live trader writes the standard sim.db, so the shared
            # active-bets reader works.
            rows = fetch_active_bets_with_marks(b["db_path"])
        elif dt != "standard":
            continue
        else:
            rows = fetch_active_bets_with_marks(b["db_path"])
        if held is not None:
            rows = [r for r in rows if r.get("ticker") in held]
        for ab in rows:
            ab["_bot_key"] = b["key"]
            ab["_bot_name"] = b["name"]
            out.append(ab)
    return out


def fetch_watchlist(db_path: str) -> List[dict]:
    """Latest market_view per ticker, filtered to markets with at least
    one of (Kalshi YES ask, Kalshi NO ask) populated AND still open.

    The user wants to see tickers when there is a probability to compare
    against — not when both sides are empty. Loose OR means "show if
    Kalshi has at least one quoted side".

    Markets that have closed are dropped: we never delete old
    market_views rows, so the latest-per-ticker cache contains entries
    for tickers from previous resolution weeks. We project the recorded
    minutes_to_close forward by however long ago the row was captured,
    and require the projection to still be positive.
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            rows = c.execute(
                "WITH latest AS ("
                "  SELECT ticker, MAX(id) AS id FROM market_views GROUP BY ticker"
                ") "
                "SELECT mv.* FROM market_views mv "
                "JOIN latest l ON mv.id = l.id "
                "WHERE mv.model_prob_yes IS NOT NULL "
                "  AND (mv.yes_ask_cents IS NOT NULL OR mv.no_ask_cents IS NOT NULL) "
                # Drop rows whose market has already closed in real time.
                # Project minutes_to_close forward by the elapsed time
                # since this row was recorded; positive => still open.
                # julianday() handles ISO8601 with timezone offsets.
                "  AND mv.minutes_to_close IS NOT NULL "
                "  AND mv.minutes_to_close - "
                "      (julianday('now') - julianday(mv.captured_at)) * 1440 > 0 "
                # Safety net: drop anything captured > 24 hours ago even
                # if its mtc projection is somehow still positive (e.g.
                # bot was offline; data is stale, not actionable).
                "  AND mv.captured_at >= datetime('now', '-1 day') "
                "ORDER BY "
                "  CASE mv.bot_verdict "
                "    WHEN 'BUY_YES' THEN 0 WHEN 'BUY_NO' THEN 0 "
                "    WHEN 'WATCH' THEN 1 ELSE 2 END, "
                "  mv.minutes_to_close ASC"
            ).fetchall()
        return [dict(r) for r in rows]
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []


def _watchlist_from_kalshi(markets: List[dict]) -> List[dict]:
    """Translate Kalshi /markets payloads into the same row shape that
    the local-DB watchlist uses, so the render path is uniform.

    Bot-specific fields (`model_prob_yes`, `bot_verdict`, `edge_yes`,
    `rejection_reason`, …) are left None — those only exist when the
    bot's running and writing market_views. Market-state fields
    (strikes, prices, volume, OI, mtc) come straight from Kalshi.

    Used when the bot service isn't running locally, so the user can
    still see the strike ladder for whichever event in the series is
    currently open. Auto-refreshes via the 60s Kalshi cache.
    """
    out: List[dict] = []
    now = datetime.now(timezone.utc)
    for m in markets:
        ticker = m.get("ticker", "")
        if not ticker:
            continue
        # Strike: floor_strike/cap_strike are dollars-as-floats. Some
        # markets are "between" range; we mirror gas-prices' convention
        # of using strike_low for "above $X".
        floor = m.get("floor_strike")
        cap = m.get("cap_strike")
        try:
            strike_low = float(floor) if floor is not None else None
        except (TypeError, ValueError):
            strike_low = None
        try:
            strike_high = float(cap) if cap is not None else None
        except (TypeError, ValueError):
            strike_high = None
        # Direction: prefer Kalshi's strike_type field — it's
        # authoritative ("greater" → above, "less" → below, "between"
        # → between). For above-style markets Kalshi sets BOTH
        # floor_strike and cap_strike to the same value (a convention,
        # not a range), so the old "both set ⇒ between" heuristic
        # mis-renders them as a zero-width band like "0.90pp – 0.90pp".
        # Fall back to that heuristic only when strike_type is absent.
        strike_type = (m.get("strike_type") or "").lower()
        if strike_type in ("greater", "above"):
            direction = "above"
            strike_high = None
        elif strike_type in ("less", "below"):
            direction = "below"
            strike_high = None
        elif strike_type == "between":
            direction = "between"
        elif strike_low is not None and strike_high is not None and strike_low != strike_high:
            direction = "between"
        else:
            direction = "above"
            strike_high = None
        # Prices come as decimal-dollar strings ("0.7600"). The local
        # schema uses int cents.
        def _cents(v: Any) -> int | None:
            if v is None or v == "":
                return None
            try:
                return int(round(float(v) * 100))
            except (TypeError, ValueError):
                return None
        ya = _cents(m.get("yes_ask_dollars"))
        na = _cents(m.get("no_ask_dollars"))
        yb = _cents(m.get("yes_bid_dollars"))
        spread = None
        if ya is not None and yb is not None:
            spread = max(0, ya - yb)
        # Time-to-close in minutes from close_time.
        close_time_str = m.get("close_time") or ""
        mtc_min: float | None = None
        if close_time_str:
            try:
                ct = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                mtc_min = max(0.0, (ct - now).total_seconds() / 60.0)
            except ValueError:
                mtc_min = None
        # Volume / open-interest come as float-precision strings.
        def _intv(v: Any) -> int | None:
            if v is None or v == "":
                return None
            try:
                return int(round(float(v)))
            except (TypeError, ValueError):
                return None
        out.append({
            "ticker": ticker,
            "title": m.get("title") or "",
            "direction": direction,
            "strike_low": strike_low,
            "strike_high": strike_high,
            "minutes_to_close": mtc_min,
            "model_prob_yes": None,
            "raw_model_prob_yes": None,
            "yes_ask_cents": ya,
            "no_ask_cents": na,
            "yes_bid_cents": yb,
            "spread_cents": spread,
            "edge_yes": None,
            "edge_no": None,
            "bot_verdict": None,
            "rejection_reason": None,
            "rules_primary": m.get("rules_primary") or "",
            "rules_secondary": m.get("rules_secondary") or "",
            "event_title": m.get("event_ticker") or "",
            "event_sub_title": m.get("yes_sub_title") or "",
            "volume": _intv(m.get("volume_fp")),
            "open_interest": _intv(m.get("open_interest_fp")),
            "yes_ask_depth": _intv(m.get("yes_ask_size_fp")),
            "captured_at": now.isoformat(),
        })
    # Sort by strike ascending — matches the local fetch_watchlist order.
    out.sort(key=lambda r: (r.get("strike_low")
                            if r.get("strike_low") is not None else 9_999.0,
                            r.get("ticker") or ""))
    return out


def _merge_kalshi_with_local(kalshi_markets: List[dict],
                              local_rows: List[dict]) -> List[dict]:
    """Build the watchlist from Kalshi (the spine), then for each row
    look up the matching ticker in the local DB and copy over the
    bot-computed fields (model_prob_yes, bot_verdict, edge_*, etc.).

    Why Kalshi-spine: the local sim.db can be empty or stale (different
    event's tickers from a prior week), but Kalshi always knows the
    current event's full strike ladder. Layering local data on top
    means the user sees the right ladder AND the bot's view per row,
    when the bot has one.
    """
    base = _watchlist_from_kalshi(kalshi_markets)
    by_ticker = {r["ticker"]: r for r in local_rows if r.get("ticker")}
    bot_fields = (
        "model_prob_yes", "raw_model_prob_yes",
        "edge_yes", "edge_no",
        "bot_verdict", "rejection_reason",
        "rules_primary", "rules_secondary",
    )
    for row in base:
        local = by_ticker.get(row["ticker"])
        if not local:
            continue
        for f in bot_fields:
            if local.get(f) is not None:
                row[f] = local[f]
    return base


def fetch_underlying_history(db_path: str, hours: int = 72,
                              max_points: int = 400) -> List[dict]:
    """Time-series of the bot's underlying value (model_snapshots.current_gas_price)
    over the last N hours. Used to draw the watchlist hero chart.

    Each row carries both ``captured_at`` (ISO string for compatibility
    with older code paths) and ``ts`` (unix seconds, what
    ``svg_kalshi_chart`` reads). Empty list when the bot's DB doesn't
    exist or has no snapshots in the window.
    """
    if not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            rows = c.execute(
                "SELECT captured_at, current_gas_price AS value "
                "FROM model_snapshots "
                "WHERE current_gas_price IS NOT NULL "
                "  AND captured_at >= datetime('now', ?) "
                "ORDER BY captured_at ASC LIMIT ?",
                (f"-{int(hours)} hours", max_points),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    out: List[dict] = []
    for r in rows:
        d = dict(r)
        ts = _iso_to_unix(d.get("captured_at"))
        if ts is not None:
            d["ts"] = ts
        out.append(d)
    return out


def pick_recent_market_view_ticker(db_path: str) -> str | None:
    """Most-recently-updated ticker in ``market_views`` with a non-null
    yes_ask_cents. Used as a robust chart-ticker fallback when neither
    the active bet nor Kalshi's ATM market produces a ticker that has
    fresh data locally (e.g. a bot mid-event-rollover).
    """
    if not Path(db_path).exists():
        return None
    try:
        with closing(_conn(db_path)) as c:
            row = c.execute(
                "SELECT ticker FROM market_views "
                "WHERE yes_ask_cents IS NOT NULL "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return None
    return (dict(row).get("ticker") if row else None)


def fetch_ticker_yes_prob_history(db_path: str, ticker: str | None,
                                    hours: int = 168) -> List[dict]:
    """Time-series of YES probability (in cents, 0-100) for one ticker.

    Reads ``market_views`` rows for the ticker over the last N hours.
    Each row is a snapshot the bot took at scoring time, so the chart
    point density mirrors how often the bot polls.

    Returns ``[{"ts": unix_seconds, "value": yes_ask_cents}, …]``.
    Empty when the DB / ticker has no data.
    """
    if not ticker or not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            rows = c.execute(
                "SELECT captured_at, yes_ask_cents AS value "
                "FROM market_views "
                "WHERE ticker = ? AND yes_ask_cents IS NOT NULL "
                "  AND captured_at >= datetime('now', ?) "
                "ORDER BY captured_at ASC",
                (ticker, f"-{int(hours)} hours"),
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    out: List[dict] = []
    for r in rows:
        d = dict(r)
        ts = _iso_to_unix(d.get("captured_at"))
        if ts is not None:
            d["ts"] = ts
            out.append(d)
    return out


def _iso_to_unix(s: str | None) -> float | None:
    """Parse SQLite's `YYYY-MM-DD HH:MM:SS` (UTC) strings into unix
    seconds. Returns None on malformed input — callers fall back to the
    captured_at string.
    """
    if not s:
        return None
    try:
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError):
        return None


def _kalshi_held_for_snapshot() -> List[dict]:
    """Real Kalshi holdings as [{ticker, side}], with parent-event
    aliases — the ONLY source the client-side row highlighter keys on
    (user 2026-07-10: only rows actually bought get highlighted).
    Empty on missing creds / API failure so a transient error clears
    highlights rather than inventing them."""
    held: List[dict] = []
    try:
        from .kalshi_client import get_open_positions as _gop
        _kp, _err = _gop()
        for _p in (_kp or []):
            _tk = _p.get("ticker")
            if not _tk:
                continue
            try:
                _fp = float(_p.get("position_fp")
                             or _p.get("position") or 0)
            except (TypeError, ValueError):
                _fp = 0.0
            if not _fp:
                continue
            _sd = "YES" if _fp > 0 else "NO"
            held.append({"ticker": _tk, "side": _sd})
            if "-" in _tk:
                held.append({"ticker": _tk.rsplit("-", 1)[0],
                             "side": _sd})
    except Exception:  # noqa: BLE001
        log.exception("snapshot kalshi_held lookup failed")
    return held


def build_snapshot(db_path: str, bots: List[dict],
                    edge_cfg: dict,
                    period_days: int | None = None,
                    *, mode: str | None = None) -> dict:
    """Compact JSON snapshot for live page updates.

    The browser polls this every few seconds and patches DOM cells in
    place — no full page reload. Format is a flat dict where each
    top-level key corresponds to a stable element id in the rendered
    HTML, so the JS can do `document.getElementById(...)` lookups
    directly without needing to reparse layout.

    Includes:
      • Cross-bot summary card values (Total bets / P&L / win % / etc.)
      • The current bot's watchlist rows (one entry per ticker)
      • Active bet mark, entry, current EV, P&L
      • A monotonically-increasing tick counter for the JS to detect
        skipped updates.
    """
    summary = fetch_global_summary(bots, period_days=period_days)
    # Override the active-bets headline fields with values computed
    # directly from the global active-bets list — guarantees the cards
    # equal the table column totals cell-for-cell, even when a bot's
    # ``summary_for_rollup`` shape drifts.
    summary.update(_compute_active_bets_totals(
        _build_global_active_bets(bots, mode=mode)))
    watchlist = fetch_watchlist(db_path)
    active_bets = fetch_active_bets_with_marks(db_path)

    def _ev_yes(p, ya_c, spread_c):
        if p is None or ya_c is None:
            return None
        fee_d = kalshi_fee_cents(ya_c, 1) / 100.0
        return (float(p) - (ya_c / 100.0)
                - ((spread_c or 0) / 200.0) - fee_d)
    def _ev_no(p, na_c, spread_c):
        if p is None or na_c is None:
            return None
        fee_d = kalshi_fee_cents(na_c, 1) / 100.0
        return ((1.0 - float(p)) - (na_c / 100.0)
                - ((spread_c or 0) / 200.0) - fee_d)

    rows = []
    min_ev = edge_cfg.get("min_ev_per_contract", 0.03)
    for v in watchlist:
        ya = v.get("yes_ask_cents"); na = v.get("no_ask_cents")
        sp = v.get("spread_cents")
        p = v.get("model_prob_yes")
        ey = _ev_yes(p, ya, sp); en = _ev_no(p, na, sp)
        rows.append({
            "ticker": v.get("ticker"),
            "kalshi_yes": ya, "kalshi_no": na,
            "spread": sp,
            "volume": v.get("volume"), "open_interest": v.get("open_interest"),
            "minutes_to_close": v.get("minutes_to_close"),
            "model_prob_yes": p,
            # Raw (un-blended) model prob — null on bots that don't
            # surface it. Used by the dashboard to disambiguate "is
            # the displayed edge real raw disagreement, or shrinkage
            # blend noise?"
            "raw_model_prob_yes": v.get("raw_model_prob_yes"),
            "ev_yes": ey, "ev_no": en,
            "bot_verdict": v.get("bot_verdict"),
            "rejection_reason": v.get("rejection_reason"),
        })

    actives = []
    for ab in active_bets:
        actives.append({
            "id": ab.get("id"),
            "ticker": ab.get("ticker"),
            "side": ab.get("side"),
            "entry": ab.get("entry_price_cents"),
            "contracts": ab.get("contracts"),
            "mark_yes_ask": ab.get("mark_yes_ask"),
            "mark_no_ask": ab.get("mark_no_ask"),
            "mark_mid": ab.get("mark_mid"),
            "unreal_pnl_cents": unrealized_pnl_cents(ab),
        })

    period_closed = (summary.get("period_wins", 0)
                     + summary.get("period_losses", 0))
    return {
        "summary": {
            "active_bets": summary.get("active_bets"),
            "active_contracts": summary.get("active_contracts"),
            "active_bots": summary.get("active_bots"),
            "period_closed_bets": period_closed,
            "period_money_spent_cents": summary.get("period_money_spent_cents"),
            "period_money_gained_cents": summary.get("period_money_gained_cents"),
            "active_money_spent_cents": summary.get("active_money_spent_cents"),
            "potential_gain_cents": summary.get("potential_gain_cents"),
            "period_net_pnl_cents": summary.get("period_net_pnl_cents"),
            "period_win_pct": summary.get("period_win_pct"),
            "period_has_closed": period_closed > 0,
            "this_week_pnl_cents": summary.get("this_week_pnl_cents"),
            "net_pnl_cents": summary.get("net_pnl_cents"),
        },
        "watchlist": rows,
        "active_bets": actives,
        "kalshi_held": _kalshi_held_for_snapshot(),
        "min_ev": min_ev,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _tennis_like_snapshot(
    watchlist_rows: List[dict], active_bets: List[dict],
    bots: List[dict], *, edge_cfg: dict,
    period_days: int | None,
    mode: str | None = None,
) -> dict:
    """Shape a snapshot for tennis-shape (JSON-source) bots.

    The watchlist + active bets come pre-adapted via
    ``tennis.build_standard_watchlist_rows`` and
    ``tennis.active_bets_for_rollup`` — both already produce the
    standard row schema. Cross-bot summary fields come from the same
    rollup the sim.db bots use so the Home tab cards stay live.
    """
    summary = fetch_global_summary(bots, period_days=period_days)
    # Same override as the standard snapshot — see ``build_snapshot``.
    summary.update(_compute_active_bets_totals(
        _build_global_active_bets(bots, mode=mode)))
    rows = []
    for v in watchlist_rows:
        ya = v.get("yes_ask_cents")
        na = v.get("no_ask_cents")
        sp = v.get("spread_cents") or 0
        # Reference prob for EV — Pinnacle when the sport bot ships it,
        # else the bot's own model. Mirrors the server-side render at
        # ``_render_watchlist`` so the poll-refresh values agree with
        # the initial page load.
        pinn_p = v.get("pinnacle_prob_yes")
        p = v.get("model_prob_yes")
        ref_p = pinn_p if pinn_p is not None else p
        ev_yes = None
        ev_no = None
        if ref_p is not None and ya is not None:
            fee_yes_d = kalshi_fee_cents(ya, 1) / 100.0
            ev_yes = (float(ref_p) - (ya / 100.0)
                      - (sp / 200.0) - fee_yes_d)
        if ref_p is not None and na is not None:
            fee_no_d = kalshi_fee_cents(na, 1) / 100.0
            ev_no = ((1.0 - float(ref_p)) - (na / 100.0)
                     - (sp / 200.0) - fee_no_d)
        rows.append({
            "ticker": v.get("ticker"),
            "kalshi_yes": ya, "kalshi_no": na,
            "spread": v.get("spread_cents"),
            "volume": v.get("volume"),
            "open_interest": v.get("open_interest"),
            "minutes_to_close": v.get("minutes_to_close"),
            "model_prob_yes": p,
            "raw_model_prob_yes": v.get("raw_model_prob_yes"),
            "pinnacle_prob_yes": pinn_p,
            "ev_yes": ev_yes, "ev_no": ev_no,
            "bot_verdict": v.get("bot_verdict"),
            "rejection_reason": v.get("rejection_reason"),
        })
    actives = []
    for ab in active_bets:
        actives.append({
            "id": ab.get("id"),
            "ticker": ab.get("ticker"),
            "side": ab.get("side"),
            "entry": ab.get("entry_price_cents"),
            "contracts": ab.get("contracts"),
            "mark_yes_ask": ab.get("mark_yes_ask"),
            "mark_no_ask": ab.get("mark_no_ask"),
            "mark_mid": ab.get("mark_mid"),
            "unreal_pnl_cents": unrealized_pnl_cents(ab) if
                ab.get("entry_price_cents") is not None
                and ab.get("contracts") is not None
                and ab.get("side") else None,
        })
    period_closed = (summary.get("period_wins", 0)
                     + summary.get("period_losses", 0))
    return {
        "summary": {
            "active_bets": summary.get("active_bets"),
            "active_contracts": summary.get("active_contracts"),
            "active_bots": summary.get("active_bots"),
            "period_closed_bets": period_closed,
            "period_money_spent_cents": summary.get("period_money_spent_cents"),
            "period_money_gained_cents": summary.get("period_money_gained_cents"),
            "active_money_spent_cents": summary.get("active_money_spent_cents"),
            "potential_gain_cents": summary.get("potential_gain_cents"),
            "period_net_pnl_cents": summary.get("period_net_pnl_cents"),
            "period_win_pct": summary.get("period_win_pct"),
            "period_has_closed": period_closed > 0,
            "this_week_pnl_cents": summary.get("this_week_pnl_cents"),
            "net_pnl_cents": summary.get("net_pnl_cents"),
        },
        "watchlist": rows,
        "active_bets": actives,
        "kalshi_held": _kalshi_held_for_snapshot(),
        "min_ev": edge_cfg.get("min_ev_per_contract", 0.03),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
