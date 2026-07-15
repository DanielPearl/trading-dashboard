"""Tennis-forecast (Baseline Break) dashboard view.

Different shape than the gas-bot-style page:

  - Source is JSON (the watchlist file written by the tennis-forecast
    project's ``src/dashboard/export_watchlist.py``), not SQLite.
  - There are no Kalshi tickers / strikes / hedges. The "watchlist"
    is one row per upcoming-or-live tennis match, with the model's
    pre-match probability, the live-adjusted probability, the
    market-implied probability, and the resulting edge / EV / signal.

Reuses the standard dashboard's CSS + page chrome (title, tab bar,
.section/.body blocks) so the tennis page is visually indistinguishable
from the Kalshi-bot pages — it just renders tennis-shaped data.

Tab structure mirrors the standard renderer's three-tab bar:

  Home      → ``/``        (the cross-bot home — true website home)
  Watchlist → tennis page  (the only tennis-specific tab)
  History   → ``/?tab=history`` (cross-bot history)

So clicking Home or History on the tennis page takes the user out of
the tennis context and into the cross-bot dashboard. The tennis page
itself only renders the watchlist content.
"""
from __future__ import annotations

import html
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger("dashboard.tennis")


def _kalshi_fee_cents(price_cents: int | None, contracts: int | None) -> int:
    """Inline copy of dashboard.kalshi_fee_cents to avoid a circular
    import. Kalshi's published fee = ceil(0.07 × contracts × p × (1−p))
    where p is the price in dollars; returns the equivalent cents.
    Zero on inputs that wouldn't be charged (settled / missing).
    """
    if price_cents is None or contracts is None:
        return 0
    try:
        p = int(price_cents)
        n = int(contracts)
    except (TypeError, ValueError):
        return 0
    if n <= 0 or p <= 0 or p >= 100:
        return 0
    return int(math.ceil(0.07 * n * p * (100 - p) / 100.0))

_LABEL_COLORS = {
    "STRONG_EDGE":         "#3fb950",
    "SMALL_EDGE":          "#56d364",
    "MARKET_OVERREACTION": "#e3b341",
    "WATCH":               "#58a6ff",
    "AVOID_VOLATILE":      "#d29922",
    "INJURY_RISK":         "#f85149",
    "NO_TRADE":            "#8b949e",
}


# --------------------------------------------------------------------------- #
# Data loaders                                                                #
# --------------------------------------------------------------------------- #

def load_watchlist(path: str | None) -> Dict[str, Any]:
    if not path:
        return {"generated_at": None, "rows": []}
    p = Path(path)
    if not p.exists():
        return {"generated_at": None, "rows": []}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"generated_at": None, "rows": []}


def load_metrics(metrics_path: str | None) -> Dict[str, Any]:
    if not metrics_path:
        return {}
    p = Path(metrics_path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_coefficients(coefs_path: str | None) -> Dict[str, Any]:
    if not coefs_path:
        return {}
    p = Path(coefs_path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_sim_state(sim_state_path: str | None) -> Dict[str, Any]:
    """Read the paper-trade simulator's state file. Tolerates missing
    file (returns an empty stub so renderers show 'no positions yet')."""
    empty = {
        "open_positions": [], "closed_positions": [],
        "stats": {"open_count": 0, "total_closed": 0, "wins": 0, "losses": 0,
                   "total_realized_pnl": 0.0, "total_unrealized_pnl": 0.0,
                   "total_staked": 0.0, "win_rate": None, "roi": None},
    }
    if not sim_state_path:
        return empty
    p = Path(sim_state_path)
    if not p.exists():
        return empty
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return empty
    for k, v in empty.items():
        data.setdefault(k, v)
    for k, v in empty["stats"].items():
        data["stats"].setdefault(k, v)
    return data


def model_summary_for_card(metrics_path: str | None,
                            sim_state_path: str | None = None) -> Dict[str, Any]:
    """Return a dict shaped like ``fetch_latest_model``'s output so the
    cross-bot card grid renders the tennis bot with the same eight
    cells as every other card (Accuracy / F1 / Precision / ROC AUC /
    Recall / Features / Actual win % / Gain / loss).
    """
    metrics = load_metrics(metrics_path)
    if not metrics:
        return {}
    blended = metrics.get("blended") or metrics.get("ensemble") or {}
    sim = load_sim_state(sim_state_path) if sim_state_path else {}
    stats = (sim or {}).get("stats") or {}
    # Real feature count from the trainer's importance file so this
    # doesn't drift when the panel changes — falls back to the length
    # of the trainer's per-model output only if the file is absent.
    feature_count: int | None = None
    try:
        from pathlib import Path
        if metrics_path:
            fi_path = Path(metrics_path).parent / "feature_importance.csv"
            if fi_path.exists():
                with fi_path.open("r", encoding="utf-8") as _f:
                    # subtract the header line
                    feature_count = sum(1 for _ in _f) - 1
    except (OSError, ValueError):
        feature_count = None
    return {
        "classifier_accuracy": blended.get("accuracy"),
        "training_brier": blended.get("brier"),
        "training_log_loss": blended.get("log_loss"),
        "training_f1": blended.get("f1"),
        "training_precision": blended.get("precision"),
        "training_recall": blended.get("recall"),
        "training_roc_auc": blended.get("roc_auc"),
        "feature_count": feature_count,
        # Training- and held-out test-set sizes. Surface on the Home
        # tab's bot card as "Train rows" / "Test rows".
        "rows_train": metrics.get("rows_train"),
        "rows_test": metrics.get("rows_test"),
        "actual_wins": int(stats.get("wins", 0) or 0),
        "actual_losses": int(stats.get("losses", 0) or 0),
    }


def build_standard_watchlist_rows(
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Translate tennis watchlist.json rows into the row shape that the
    standard sport-bot ``_render_watchlist`` consumes. One row per match,
    with player A mapped to the YES side and player B to NO.

    Kalshi's ``yes_ask_cents_a`` and ``yes_ask_cents_b`` are the per-side
    market prices — the "NO" cents column is just the other player's
    YES contract price (the pair sums to ~100¢ minus the spread).

    Rows with zero open interest are dropped so the standard renderer
    surfaces only tradeable matches, matching the old tennis-specific
    table's filter.
    """
    raw_rows = payload.get("rows") or []
    out: List[Dict[str, Any]] = []
    for r in raw_rows:
        match_id = str(r.get("match_id") or "")
        if not match_id:
            continue
        oi = r.get("open_interest")
        try:
            if ((oi is None or float(oi) <= 0)
                    and not r.get("_skip_oi_filter")):
                continue
        except (TypeError, ValueError):
            if not r.get("_skip_oi_filter"):
                continue
        # Prefer the live (in-play adjusted) probability since that's what
        # the bot actually trades on; fall back to the pre-match prior.
        p_a = r.get("live_prob_a")
        if p_a is None:
            p_a = r.get("pre_match_prob_a")
        # Decide which player goes on the "YES" / top side of the row.
        # The standard sport-bot renderer treats YES as the favoured /
        # action side — top of every stacked cell (My %, Kalshi %,
        # Edge, EV) and the bold player name in the Side cell. For
        # tennis, both players have their own YES market (separate
        # Kalshi tickers), and player A is just alphabetical first —
        # not necessarily the favoured side. If we leave the mapping
        # unconditional, rows where player B has the edge render with
        # player A as the bold "Side" name but the bot is actually
        # betting on B (Title cell + Verdict reflect this), inverting
        # the visual. Flip every side-paired field so YES always tracks
        # the favoured side.
        pinn_a = r.get("pinnacle_prob_a")
        b_favoured = (p_a is not None and float(p_a) < 0.5)
        if b_favoured:
            top_ask = r.get("yes_ask_cents_b")
            bot_ask = r.get("yes_ask_cents_a")
            top_prob = (1.0 - float(p_a)) if p_a is not None else None
            top_raw = ((1.0 - float(r.get("pre_match_prob_a")))
                       if r.get("pre_match_prob_a") is not None else None)
            top_pinn = ((1.0 - float(pinn_a))
                         if pinn_a is not None else None)
            top_label = r.get("player_b")
            bot_label = r.get("player_a")
            top_title = r.get("title_b") or r.get("title") or ""
        else:
            top_ask = r.get("yes_ask_cents_a")
            bot_ask = r.get("yes_ask_cents_b")
            top_prob = p_a
            top_raw = r.get("pre_match_prob_a")
            top_pinn = pinn_a
            top_label = r.get("player_a")
            bot_label = r.get("player_b")
            top_title = r.get("title_a") or r.get("title") or ""
        # The dashboard's Title cell should match what Kalshi shows on
        # the event page the user lands on when they click the ticker
        # (e.g. "Choinski vs Herbert", not "Will Herbert win the
        # Choinski vs Herbert: Qual R2 match?"). Prefer the bot's
        # stored event_title when present; fall back to the per-side
        # market title for older rows that haven't been re-exported.
        display_title = r.get("event_title") or top_title
        buy_eligible = bool(r.get("buy_eligible"))
        buy_side = (r.get("buy_side") or "").upper()
        # ``BUY_YES`` = act on the favoured (top-of-row) side. Whether
        # that's PLAYER_A or PLAYER_B in the underlying row is now
        # encoded by the flip above.
        if buy_eligible and ((buy_side == "A" and not b_favoured)
                              or (buy_side == "B" and b_favoured)):
            verdict = "BUY_YES"
        elif buy_eligible:
            verdict = "BUY_NO"
        else:
            verdict = "SKIP"
        blockers = r.get("buy_blockers") or []
        if blockers:
            rej_reason = ", ".join(str(b) for b in blockers)
        else:
            rej_reason = str(r.get("reason_for_signal") or "")
        out.append({
            "ticker": match_id,
            "direction": "yes",
            "strike_low": None,
            "strike_high": None,
            "yes_ask_cents": top_ask,
            "no_ask_cents": bot_ask,
            "spread_cents": r.get("spread_cents"),
            "volume": r.get("volume"),
            "open_interest": oi,
            "model_prob_yes": top_prob,
            "raw_model_prob_yes": top_raw,
            # Pinnacle sportsbook devigged probability for the top side.
            # None when the match isn't in Pinnacle's book (e.g. between
            # tournaments, ITF events, or the API is down).
            "pinnacle_prob_yes": top_pinn,
            "_skip_oi_filter": bool(r.get("_skip_oi_filter")),
            "bot_verdict": verdict,
            "rejection_reason": rej_reason,
            "title": display_title,
            "minutes_to_close": None,
            "_yes_label": top_label,
            "_no_label": bot_label,
            # Kalshi side-tickers for the top (YES) / bottom (NO) sides
            # of THIS row after the favored-side flip. The dashboard
            # renderer joins these against ``kalshi_held_by_ticker`` to
            # decide whether a held position sits on the row's YES side
            # (no re-orient needed) or its NO side (flip Model % /
            # Kalshi % / Entry % onto the held axis so all three
            # columns agree). Without these, held rows where our side
            # is the underdog per our own model render three
            # inconsistent axes across the stacked cells.
            "_yes_ticker": r.get("ticker_b") if b_favoured else r.get("ticker_a"),
            "_no_ticker": r.get("ticker_a") if b_favoured else r.get("ticker_b"),
            # Kalshi ``rules_primary`` — the resolution paragraph the
            # trading dashboard's "Kalshi rules" section renders. The
            # tennis-forecast exporter writes it onto the raw
            # watchlist row; pass it through unchanged.
            "rules_primary": r.get("rules_primary"),
            # Competition label ("WNBA", "NBA", tennis tournament...).
            # The Event column falls back to this when the rules text
            # doesn't match any known event template.
            "tournament": r.get("tournament"),
        })
    return out


def _current_model_prob_yes(pos: dict,
                             fresh_pinn: Dict[str, dict]) -> float | None:
    """Return today's Pinnacle prob on the SIDE this position holds,
    from the fresh watchlist ``fresh_pinn`` lookup — or None when the
    match isn't on today's board. Sport positions store the held side
    as ``PLAYER_A`` / ``PLAYER_B`` and the specific ticker as
    ``ticker`` (side-specific) with ``match_id`` as the event ticker;
    the lookup is keyed under both forms."""
    # Try side-specific ticker first, then event ticker.
    for key in (pos.get("ticker") or "", pos.get("match_id") or ""):
        pair = fresh_pinn.get(key)
        if pair is None:
            continue
        s = str(pos.get("side", "")).upper()
        if s == "PLAYER_A":
            return pair["a"]
        if s == "PLAYER_B":
            return pair["b"]
        # side_player fallback — match against player name
        return None
    return None


def active_bets_for_rollup(sim_state_path: str | None,
                             watchlist_path: str | None = None
                             ) -> List[Dict[str, Any]]:
    """Return tennis open paper positions in the dict shape the
    standard ``_render_active_bets_table`` expects.

    Mapping from sim_state.json position record → standard schema:
      ticker          ← match_id (= the real Kalshi event_ticker)
      _match          ← "{player_a} vs {player_b}"
      _side_player    ← side_player (the player we're betting on)
      side            ← "YES" (we always buy the favoured side)
      contracts       ← stake (= 1.0 default; expressed as $1 = 1 contract)
      entry_price_cents ← entry_market_prob * 100
      mark_mid        ← current_market_prob * 100
      opened_at       ← opened_at
      minutes_to_close ← derived from the matching watchlist row's
                         ``expected_expiration_time`` so the standard
                         "Closes in" cell renders the time to match
                         resolution rather than dashing out.
      _bot_name       ← caller fills in

    Tennis stake is in dollars rather than Kalshi contracts; we use a
    1-contract / dollar mapping so the existing dollar columns
    (Entry cost / Potential gain) render in the same units as Kalshi
    bets without special-casing the renderer.
    """
    s = load_sim_state(sim_state_path)
    # Build a per-match_id → expected_expiration_time map from the
    # canonical live-state file so the standard "Closes in" cell can
    # render a real countdown. When the watchlist path isn't in a
    # standard layout, the lookup falls through and Closes-in dashes.
    exp_by_id: Dict[str, str] = {}
    if watchlist_path:
        try:
            wl_path = Path(watchlist_path).parent.parent / "raw" / "live_state.json"
            if wl_path.exists():
                with wl_path.open("r", encoding="utf-8") as f:
                    for rec in json.load(f) or []:
                        mid = rec.get("match_id")
                        exp = rec.get("expected_expiration_time")
                        if mid and exp:
                            exp_by_id[str(mid)] = str(exp)
        except (OSError, json.JSONDecodeError):
            pass
    # Current-Pinnacle lookup for each ticker in the fresh watchlist.
    # Home page's Active bets renderer preferred entry-time model
    # (``entry_model_prob``) which drifts away from today's Pinnacle
    # line as the market moves — the Chiba/Seibu case 2026-07-15
    # showed 68% on Home (entry-time) vs 56% on the baseball
    # watchlist page (fresh Pinnacle) for the same match. Attach the
    # current per-side value here so both pages read the same
    # source of truth.
    fresh_pinn: Dict[str, dict] = {}
    if watchlist_path:
        try:
            with open(watchlist_path, "r", encoding="utf-8") as f:
                _wl = json.load(f) or {}
            for r in _wl.get("rows") or []:
                pa_prob = r.get("pinnacle_prob_a")
                if pa_prob is None:
                    continue
                pa_prob = float(pa_prob)
                # Register under BOTH ticker forms — the position's
                # ``ticker`` is the side-specific one (KX..-A/B),
                # while ``match_id`` on the position may be the event
                # ticker; the caller keys off either.
                mid = r.get("match_id") or ""
                if mid:
                    fresh_pinn[mid] = {"a": pa_prob, "b": 1.0 - pa_prob}
                for tk_field, side in (("ticker_a", "a"), ("ticker_b", "b")):
                    tk = r.get(tk_field)
                    if tk:
                        fresh_pinn[tk] = {"a": pa_prob, "b": 1.0 - pa_prob}
        except (OSError, json.JSONDecodeError):
            pass
    now = datetime.now(timezone.utc)
    out: List[Dict[str, Any]] = []
    for p in s.get("open_positions") or []:
        entry = p.get("entry_market_prob")
        if entry is None:
            # No real Kalshi quote at open — shouldn't happen given the
            # simulator's filter, but if it slips through we drop the
            # row from the table rather than show a fabricated 50%.
            continue
        # Skip unenriched orphan stubs (executor's ``_adopt_orphans``
        # writes a placeholder record with empty player names, side_
        # player="?", and 0.5/0.5 probs for Kalshi positions it didn't
        # place itself). The Kalshi-side path below (via
        # ``kalshi_positions_to_active_bets`` in the caller) will
        # surface those tickers with the current watchlist enrichment
        # applied — proper player names, matchup, model prob — so we
        # let it own the row instead of showing a "? vs — Model 50%"
        # stub. Detected via the "recovered-" order-id prefix the
        # orphan adopter stamps, backed by an empty player_a as
        # belt-and-braces for pre-2026-07-11 stubs that lack the
        # order_id.
        _oid = str(p.get("order_id") or "")
        if _oid.startswith("recovered-") and not p.get("player_a"):
            continue
        entry = float(entry)
        mark = float(p.get("current_market_prob") or entry)
        mid = p.get("match_id", "")
        mtc: float | None = None
        exp = exp_by_id.get(mid)
        if exp:
            try:
                ts = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                mtc = max(0.0, (ts - now).total_seconds() / 60.0)
            except (TypeError, ValueError):
                mtc = None
        out.append({
            "ticker": mid,
            "_match": f"{p.get('player_a','')} vs {p.get('player_b','')}",
            "_side_player": p.get("side_player", ""),
            # Prefer the Kalshi event-page heading so the active-bets
            # Title column matches what the user sees on click-through.
            # Falls back to the per-side market question for pre-fix
            # rows that don't have event_title stored.
            "_title": p.get("event_title") or p.get("title") or "",
            "title": p.get("event_title") or p.get("title") or "",
            "_tournament": p.get("tournament", ""),
            "_surface": p.get("surface", ""),
            "side": "YES",  # tennis always buys the favoured side
            # 1 contract per paper bet — same convention as Kalshi
            # (1 contract = $1 face value at settlement). The
            # standard renderer multiplies entry_price_cents × contracts
            # / 100 for Entry cost; with ``contracts=1`` and entry =
            # real cents from Kalshi's yes_ask, the dollar columns
            # match what the user would pay on the actual exchange.
            "contracts": 1,
            "entry_price_cents": int(round(entry * 100)),
            "mark_mid": mark * 100,
            "opened_at": p.get("opened_at", ""),
            "minutes_to_close": mtc,
            "label_at_open": p.get("label_at_open", ""),
            "reason_at_open": p.get("reason_at_open", ""),
            # Required by the renderer's "why was this bet chosen" hook.
            "model_yes_prob_at_entry": float(p.get("entry_model_prob") or entry),
            # Current-Pinnacle prob on the SIDE WE HELD, from today's
            # fresh watchlist row. When available, renderers should
            # prefer this over ``model_yes_prob_at_entry`` for the
            # Model % display cell so the Home page's Active bets
            # doesn't drift from the per-bot Watchlist page's number
            # once the Pinnacle line moves after we opened. None when
            # Pinnacle isn't quoting the match today.
            "current_model_prob_yes": _current_model_prob_yes(
                p, fresh_pinn),
            "kalshi_yes_prob_at_entry": entry,
            # Reconstruct the net-of-fee EV the bot saw at open from
            # (entry_model_prob, entry_market_prob). Same formula the
            # closed-position rollup uses, so the buy-criteria popup
            # on the cross-bot active-bets row shows a real EV figure
            # for sport bots instead of a missing value.
            "expected_ev_at_entry": (
                float(p.get("entry_model_prob") or entry) - entry
                - (_kalshi_fee_cents(int(round(entry * 100)), 1) / 100.0)
            ),
        })
    return out


def _ticker_matches_series(ticker: str, series_prefixes) -> bool:
    """True when a Kalshi market ticker belongs to one of the bot's
    series. Entries ending in ``*`` are family prefixes (``KXDARTS*``
    matches KXDARTSMATCH / KXPDCDARTS...); everything else must equal
    the ticker's first dash-segment exactly, so ``KXNBAGAME`` can't
    accidentally claim the unrelated KXNBAGAMES series."""
    seg = (ticker or "").split("-", 1)[0]
    for p in series_prefixes or []:
        p = (p or "").strip()
        if not p:
            continue
        if p.endswith("*"):
            if seg.startswith(p[:-1]):
                return True
        elif seg == p:
            return True
    return False


def kalshi_positions_to_active_bets(kalshi_positions: List[Dict[str, Any]],
                                    watchlist_payload: Dict[str, Any] | None,
                                    series_prefixes,
                                    exclude_tickers=(),
                                    ) -> List[Dict[str, Any]]:
    """Project REAL Kalshi portfolio positions in a bot's series into
    the same active-bets row shape ``active_bets_for_rollup`` emits,
    so manually-placed (or externally-placed) orders show up on the
    bot's Active bets table with the standard tennis columns.

    Only positions whose ticker matches ``series_prefixes`` and isn't
    already covered by ``exclude_tickers`` (the executor's own rows —
    either the exact market ticker or its parent event ticker) are
    returned. Rows are enriched from the bot's watchlist payload when
    the market is on today's board (matchup title, model prob, current
    mark); positions on markets that already left the watchlist render
    from the portfolio data alone.
    """
    rows = list((watchlist_payload or {}).get("rows") or [])
    by_ticker: Dict[str, tuple] = {}
    for r in rows:
        if r.get("ticker_a"):
            by_ticker[r["ticker_a"]] = (r, "a")
        if r.get("ticker_b"):
            by_ticker[r["ticker_b"]] = (r, "b")

    excluded = {t for t in (exclude_tickers or ()) if t}
    out: List[Dict[str, Any]] = []
    for p in kalshi_positions or []:
        ticker = str(p.get("ticker") or "")
        if not ticker or not _ticker_matches_series(ticker, series_prefixes):
            continue
        event_ticker = ticker.rsplit("-", 1)[0]
        if ticker in excluded or event_ticker in excluded:
            continue
        try:
            fp = float(p.get("position_fp") or 0)
        except (TypeError, ValueError):
            continue
        if not fp:
            continue
        contracts = max(1, int(round(abs(fp))))
        traded = None
        # Exposure first: it is the open position's cost basis;
        # total_traded is gross volume and misprices any position
        # with a partial exit (2026-07-14 FARWAW 102% entry bug).
        for k in ("market_exposure_dollars", "total_traded_dollars"):
            try:
                v = float(p.get(k))
                if v > 0:
                    traded = v
                    break
            except (TypeError, ValueError):
                continue
        entry = (traded / contracts) if traded else None
        if entry is None:
            continue  # can't price the row honestly — skip
        entry_cents = int(round(entry * 100))

        row_entry = by_ticker.get(ticker)
        match_label = event_ticker
        side_player = ticker.rsplit("-", 1)[-1]
        title = ""
        tournament = surface = ""
        mark = entry
        model_p = entry
        mtc = None
        if row_entry:
            row, side_key = row_entry
            match_label = (f"{row.get('player_a', '')} vs "
                           f"{row.get('player_b', '')}").strip()
            side_player = (row.get("player_a") if side_key == "a"
                           else row.get("player_b")) or side_player
            title = row.get("event_title") or row.get("title") or ""
            tournament = row.get("tournament") or ""
            surface = row.get("surface") or ""
            m = (row.get("market_prob_a") if side_key == "a"
                 else row.get("market_prob_b"))
            if m is not None:
                mark = float(m)
            lp = (row.get("live_prob_a") if side_key == "a"
                  else row.get("live_prob_b"))
            if lp is not None:
                model_p = float(lp)
            exp = (row.get("expected_expiration_time")
                   or row.get("kickoff"))
            if exp:
                try:
                    ts = datetime.fromisoformat(
                        str(exp).replace("Z", "+00:00"))
                    mtc = max(0.0, (ts - datetime.now(timezone.utc)
                                    ).total_seconds() / 60.0)
                except (TypeError, ValueError):
                    mtc = None

        out.append({
            "ticker": ticker,
            "_match": match_label,
            "_side_player": side_player,
            "_title": title,
            "title": title,
            "_tournament": tournament,
            "_surface": surface,
            "side": "YES" if fp > 0 else "NO",
            "contracts": contracts,
            "entry_price_cents": entry_cents,
            "mark_mid": mark * 100,
            "opened_at": p.get("last_updated_ts", ""),
            "minutes_to_close": mtc,
            "label_at_open": "KALSHI",
            "reason_at_open": ("held on Kalshi (manual or external "
                               "order — not opened by this bot's "
                               "executor)"),
            "model_yes_prob_at_entry": model_p,
            "kalshi_yes_prob_at_entry": entry,
            "expected_ev_at_entry": (
                model_p - entry
                - (_kalshi_fee_cents(entry_cents, contracts)
                   / 100.0 / contracts)
            ),
        })
    return out


def _is_real_fill(c: Dict[str, Any]) -> bool:
    """True when a closed sim_state record traces back to a REAL
    Kalshi order. Real fills carry the exchange's UUID order_id (or
    the orphan-adoption 'recovered-…' marker); dry-run records carry
    'DRY-RUN-…' ids / 'dry_run_simulated' status; paper-simulator
    closes have no order_id at all."""
    status = str(c.get("order_status") or "").lower()
    if status == "dry_run_simulated":
        return False
    oid = str(c.get("order_id") or "")
    if oid.startswith("DRY-RUN"):
        return False
    return bool(oid)


def closed_positions_for_rollup(sim_state_path: str | None,
                                  limit: int = 100,
                                  real_only: bool = False,
                                  ) -> List[Dict[str, Any]]:
    """Project tennis ``closed_positions`` into the shape the standard
    ``_render_bet_history_block`` expects.

    Mapping:
      ticker             ← match_id (= real Kalshi event_ticker)
      side               ← "YES" (tennis always buys YES on the
                            favoured side; the dashboard's outcome
                            badge keys off realized_pnl sign anyway)
      entry_price_cents  ← entry_market_prob × 100
      exit_price_cents   ← exit_market_prob × 100 (hedge exit) or
                            settle_market_prob × 100 (natural settle)
      contracts          ← 1 (tennis uses 1-contract = $1 face value)
      realized_pnl_cents ← realized_pnl × 100
      opened_at / exited_at ← as recorded
      _title             ← the Kalshi-published YES question
      error_type         ← exit_reason from the hedge engine
                            (hedge_pl / hedge_sl) — surfaces on the
                            Outcome column tooltip when present.
    """
    s = load_sim_state(sim_state_path)
    closed = list(s.get("closed_positions") or [])
    # LIVE dashboard's History must only show bets that actually
    # traded on Kalshi (user 2026-07-10) — the live executors' state
    # files accumulate dry-run evaluations alongside real fills, and
    # mixing them makes the real-money ledger unreadable.
    if real_only:
        closed = [c for c in closed if _is_real_fill(c)]
    # Most recently closed first; honour the caller's limit so the
    # cross-bot history loop doesn't pull thousands of rows from a
    # long-running paper-trade ledger.
    closed.sort(key=lambda c: c.get("closed_at", ""), reverse=True)
    out: List[Dict[str, Any]] = []
    for c in closed[:limit]:
        entry = c.get("entry_market_prob")
        exit_p = (c.get("exit_market_prob")
                    or c.get("settle_market_prob"))
        try:
            entry_cents = (int(round(float(entry) * 100))
                            if entry is not None else None)
        except (TypeError, ValueError):
            entry_cents = None
        try:
            exit_cents = (int(round(float(exit_p) * 100))
                           if exit_p is not None else None)
        except (TypeError, ValueError):
            exit_cents = None
        try:
            realized_cents = int(round(float(c.get("realized_pnl", 0)) * 100))
        except (TypeError, ValueError):
            realized_cents = 0
        # Recover Entry EV from what the bot recorded at open. Tennis-
        # style sim state doesn't persist the dashboard's per-row EV,
        # but it does record (entry_model_prob, entry_market_prob), so
        # we can reconstruct the same net-of-fee figure the watchlist
        # column shows for open bets:
        #     EV = entry_model_prob − entry_market_prob − fee_at_entry
        # (no half-spread term — sport bots don't store the bid-ask
        # spread on the closed-position record.)
        entry_model_p = c.get("entry_model_prob")
        try:
            entry_model_p = (float(entry_model_p)
                             if entry_model_p is not None else None)
            entry_market_p = float(entry) if entry is not None else None
        except (TypeError, ValueError):
            entry_model_p = entry_market_p = None
        if (entry_model_p is not None and entry_market_p is not None
                and entry_cents is not None):
            fee_d = _kalshi_fee_cents(entry_cents, 1) / 100.0
            expected_ev = entry_model_p - entry_market_p - fee_d
        else:
            expected_ev = None
        out.append({
            "ticker": c.get("match_id"),
            # Prefer the Kalshi event-page heading so the History tab
            # title matches what the user sees on click-through. Falls
            # back to the per-side market question for pre-fix rows.
            "_title": c.get("event_title") or c.get("title", ""),
            "side": "YES",
            "entry_price_cents": entry_cents,
            "exit_price_cents": exit_cents,
            "contracts": 1,
            "realized_pnl_cents": realized_cents,
            "opened_at": c.get("opened_at", ""),
            "exited_at": c.get("closed_at", ""),
            "error_type": c.get("exit_reason"),
            "model_yes_prob_at_entry": c.get("entry_model_prob"),
            "kalshi_yes_prob_at_entry": entry,
            "expected_ev_at_entry": expected_ev,
            "break_even_probability": entry,
            # > 1 when multiple flap-trades on this match were
            # collapsed into this row by the dedupe pass.
            "merged_trade_count": int(c.get("merged_trade_count", 1) or 1),
            "merged_position_ids": c.get("merged_position_ids"),
        })
    return out


def summary_for_rollup(sim_state_path: str | None) -> Dict[str, Any]:
    """Tennis summary in the shape the cross-bot rollup expects.
    Cents conversion: tennis stake is dollars (1.0 = $1) → ×100 for cents.

    Tennis convention: each paper bet is 1 contract face-value
    (``active_bets_for_rollup`` returns contracts=1, entry in cents
    from the market prob × 100). The active-bets totals mirror that
    so the Home-tab summary cards agree with the rendered table.
    """
    from .dashboard import kalshi_fee_cents
    s = load_sim_state(sim_state_path)
    stats = s.get("stats") or {}
    open_positions = s.get("open_positions") or []
    closed = s.get("closed_positions") or []
    money_spent_cents = int(round(sum(
        float(c.get("stake", 0)) * 100.0 for c in closed
    )))
    money_gained_cents = 0
    for c in closed:
        stake = float(c.get("stake", 0))
        pnl = float(c.get("realized_pnl", 0))
        money_gained_cents += int(round((stake + pnl) * 100.0))
    realized_pnl_cents = money_gained_cents - money_spent_cents
    active_contracts = 0
    active_money_spent_cents = 0
    potential_gain_cents = 0
    for p in open_positions:
        entry = p.get("entry_market_prob")
        if entry is None:
            continue
        entry_c = int(round(float(entry) * 100))
        ctr = 1  # tennis paper bet face-value matches the table renderer
        fee_c = kalshi_fee_cents(entry_c, ctr)
        active_contracts += ctr
        active_money_spent_cents += entry_c * ctr + fee_c
        potential_gain_cents += (100 - entry_c) * ctr - fee_c
    return {
        "open_count": len(open_positions),
        "active_contracts": active_contracts,
        "period_bets_made": int(stats.get("total_closed", 0)) + len(open_positions),
        "period_net_pnl_cents": realized_pnl_cents,
        "period_wins": int(stats.get("wins", 0)),
        "period_losses": int(stats.get("losses", 0)),
        "period_money_spent_cents": money_spent_cents,
        "period_money_gained_cents": money_gained_cents,
        "potential_gain_cents": potential_gain_cents,
        "active_money_spent_cents": active_money_spent_cents,
        "total_bets": int(stats.get("total_closed", 0)) + len(open_positions),
        "realized_pnl_cents": realized_pnl_cents,
        "wins_lifetime": int(stats.get("wins", 0)),
        "losses_lifetime": int(stats.get("losses", 0)),
    }


# --------------------------------------------------------------------------- #
# Formatters                                                                  #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Sections                                                                    #
# --------------------------------------------------------------------------- #


# JS: vanilla, ~80 lines. Reads the payload from the inline JSON tag,
# wires up row clicks on the ticker table, redraws an SVG. No D3 / no
# Chart.js — keeps the dashboard's stdlib-only footprint. The chart
# layout is a horizontal bar showing the three probability points on
# a 0-100% axis, with a confidence band shaded behind the live point.
_FORECAST_GRAPH_JS = """
<script>
(function() {
  const dataEl = document.getElementById('tfg-data');
  if (!dataEl) return;
  const payload = JSON.parse(dataEl.textContent || '{}');
  const svg = document.getElementById('tfg-svg');
  const titleEl = document.getElementById('tfg-title');
  const subEl = document.getElementById('tfg-sub');
  const legendEl = document.getElementById('tfg-legend');
  const W = 700, H = 220, PAD_L = 50, PAD_R = 30, PAD_T = 30, PAD_B = 40;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const xOf = (p) => PAD_L + p * innerW;

  function el(tag, attrs, children) {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    (children || []).forEach(c => e.appendChild(c));
    return e;
  }
  function txt(t) { return document.createTextNode(String(t)); }
  function tspan(content) { return el('text', {}, [txt(content)]); }

  function draw(mid) {
    const d = payload[mid];
    svg.innerHTML = '';
    legendEl.innerHTML = '';
    if (!d) {
      titleEl.textContent = 'No forecast available';
      subEl.textContent = '';
      return;
    }
    titleEl.textContent = d.player_a + ' vs ' + d.player_b;
    const labelTxt = (d.label || '').replace('_', ' ');
    const edgeStr = (d.edge !== null && d.edge !== undefined)
      ? (d.edge >= 0 ? '+' : '') + (d.edge * 100).toFixed(1) + 'pp'
      : '—';
    subEl.textContent = d.tournament + ' · ' + d.surface
      + ' · score ' + (d.score || '0-0')
      + ' · edge ' + edgeStr
      + ' · ' + labelTxt;

    // Axis: probability bar 0..1 (player_a's perspective).
    const axisY = PAD_T + innerH - 18;
    // Background track.
    svg.appendChild(el('rect', {
      x: PAD_L, y: axisY - 8,
      width: innerW, height: 16,
      fill: '#1d232c', stroke: '#30363d', 'stroke-width': '1', rx: 4,
    }));
    // 50% reference line.
    svg.appendChild(el('line', {
      x1: xOf(0.5), x2: xOf(0.5),
      y1: PAD_T + 8, y2: PAD_T + innerH + 4,
      stroke: '#30363d', 'stroke-dasharray': '3,4',
    }));
    // Confidence band around live.
    if (d.ci_low !== undefined && d.ci_high !== undefined) {
      svg.appendChild(el('rect', {
        x: xOf(d.ci_low), y: axisY - 14,
        width: Math.max(2, xOf(d.ci_high) - xOf(d.ci_low)),
        height: 28, fill: '#58a6ff22', stroke: '#58a6ff55',
        'stroke-width': '1', rx: 3,
      }));
    }

    // Plot points: pre / live / market.
    const points = [
      { v: d.pre, color: '#8b949e', label: 'Pre-match' },
      { v: d.live, color: '#58a6ff', label: 'Live model' },
      { v: d.market, color: '#e3b341', label: 'Market' },
    ];
    points.forEach((p) => {
      if (p.v === null || p.v === undefined) return;
      const x = xOf(p.v);
      svg.appendChild(el('circle', {
        cx: x, cy: axisY,
        r: 7, fill: p.color, stroke: '#0d1117', 'stroke-width': '2',
      }));
      const lbl = el('text', {
        x: x, y: axisY - 16, fill: p.color,
        'text-anchor': 'middle', 'font-size': '11', 'font-weight': '600',
      });
      lbl.appendChild(txt((p.v * 100).toFixed(0) + '%'));
      svg.appendChild(lbl);
    });

    // X-axis ticks: 0%, 25%, 50%, 75%, 100% — gives the user a
    // visual sense of where the dot sits without staring at numbers.
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      const x = xOf(t);
      svg.appendChild(el('line', {
        x1: x, x2: x, y1: axisY + 10, y2: axisY + 14,
        stroke: '#30363d',
      }));
      const lbl = el('text', {
        x: x, y: axisY + 28, fill: '#8b949e',
        'text-anchor': 'middle', 'font-size': '10',
      });
      lbl.appendChild(txt((t * 100).toFixed(0) + '%'));
      svg.appendChild(lbl);
    });
    // Y-axis label — implicit 'P(' + player_a + ' wins)'.
    const yAxisLbl = el('text', {
      x: PAD_L, y: PAD_T + 14, fill: '#8b949e', 'font-size': '11',
    });
    yAxisLbl.appendChild(txt('P(' + d.player_a + ' wins)'));
    svg.appendChild(yAxisLbl);

    // Legend.
    points.forEach((p) => {
      if (p.v === null || p.v === undefined) return;
      const item = document.createElement('span');
      item.style.display = 'inline-flex';
      item.style.alignItems = 'center';
      item.style.gap = '6px';
      const dot = document.createElement('span');
      dot.style.cssText = 'display:inline-block;width:10px;height:10px;'
        + 'border-radius:50%;background:' + p.color + ';';
      item.appendChild(dot);
      item.appendChild(txt(p.label + ' ' + (p.v * 100).toFixed(0) + '%'));
      legendEl.appendChild(item);
    });
    if (d.ci_low !== undefined && d.ci_high !== undefined) {
      const ci = document.createElement('span');
      ci.style.color = '#8b949e';
      ci.appendChild(txt('Live 95% CI ' + (d.ci_low * 100).toFixed(0)
        + '% – ' + (d.ci_high * 100).toFixed(0) + '%'));
      legendEl.appendChild(ci);
    }
  }

  // Highlight the currently-selected row.
  function setSelected(mid) {
    document.querySelectorAll('tr.tennis-row').forEach((tr) => {
      tr.classList.toggle('tennis-row-selected', tr.dataset.mid === mid);
    });
  }

  const container = document.getElementById('tennis-forecast-graph');
  const defaultMid = container ? container.dataset.defaultMid : '';
  if (defaultMid) { draw(defaultMid); setSelected(defaultMid); }

  // Wire row clicks → redraw.
  document.addEventListener('click', function (ev) {
    const tr = ev.target.closest('tr.tennis-row');
    if (!tr) return;
    const mid = tr.dataset.mid;
    if (!mid) return;
    draw(mid);
    setSelected(mid);
    // Smooth-scroll the graph into view if it's offscreen.
    const c = document.getElementById('tennis-forecast-graph');
    if (c) c.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
})();
</script>
<style>
tr.tennis-row { cursor: pointer; }
tr.tennis-row-selected td { background: #1f2630 !important; }
tr.tennis-row:hover td { background: #1c222b; }
</style>
"""


# --------------------------------------------------------------------------- #
# History page — sourced directly from Kalshi's settlements API                #
# --------------------------------------------------------------------------- #

# Cached Kalshi client + settlements/fills lists. Settlements and
# fills are immutable once written, so a 60-second cache is just
# politeness to the portfolio endpoint, not a freshness compromise.
_SETTLEMENTS_CACHE: dict[str, Any] = {"at": 0.0, "rows": []}
_FILLS_CACHE: dict[str, Any] = {"at": 0.0, "rows": []}
_SETTLEMENTS_TTL_S = 60.0
_KALSHI_CLIENT: Any = None


def _get_kalshi_client():
    """Lazy Kalshi client init — same env-var auth the live executor uses."""
    global _KALSHI_CLIENT
    if _KALSHI_CLIENT is not None:
        return _KALSHI_CLIENT
    import os
    try:
        from kalshi_sdk import KalshiClient
    except ImportError:
        log.warning("kalshi_sdk unavailable; tennis history will be empty")
        return None
    api = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    pkey = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not api or not pkey:
        log.warning("KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH unset; "
                     "tennis history will be empty")
        return None
    try:
        _KALSHI_CLIENT = KalshiClient(api_key_id=api, private_key_path=pkey)
    except Exception:  # noqa: BLE001
        log.exception("KalshiClient init failed; tennis history will be empty")
        return None
    return _KALSHI_CLIENT


# Player → country (IOC 3-letter) lookup. Populated lazily from
# tennis-forecast's matches_clean.csv on first History-tab render and
# cached for the life of the process — the file only changes on
# retrain (~once a day) and re-reading it is cheap.
_PLAYER_IOC_CACHE: dict[str, dict[str, str]] = {"by_path": {}}


# IOC (Olympic) codes → ISO-3166-alpha-2. The two diverge on a few
# dozen countries — the ones below cover every IOC code that appears
# in the Sackmann ATP/WTA panel since 2015. Unmapped codes render
# with no flag (defensive: better empty than wrong).
_IOC_TO_ISO2: dict[str, str] = {
    # Common ATP/WTA countries — alphabetical for grep-ability.
    "ALG": "DZ", "ANG": "AO", "ARG": "AR", "ARM": "AM", "AUS": "AU",
    "AUT": "AT", "AZE": "AZ", "BAH": "BS", "BAN": "BD", "BAR": "BB",
    "BEL": "BE", "BIH": "BA", "BLR": "BY", "BOL": "BO", "BOT": "BW",
    "BRA": "BR", "BRN": "BH", "BUL": "BG", "BUR": "BF", "CAM": "KH",
    "CAN": "CA", "CHI": "CL", "CHN": "CN", "CIV": "CI", "CMR": "CM",
    "COD": "CD", "COL": "CO", "CRC": "CR", "CRO": "HR", "CUB": "CU",
    "CYP": "CY", "CZE": "CZ", "DEN": "DK", "DMA": "DM", "DOM": "DO",
    "ECU": "EC", "EGY": "EG", "ESA": "SV", "ESP": "ES", "EST": "EE",
    "ETH": "ET", "FIJ": "FJ", "FIN": "FI", "FRA": "FR", "GBR": "GB",
    "GEO": "GE", "GER": "DE", "GHA": "GH", "GRE": "GR", "GRN": "GD",
    "GUA": "GT", "HAI": "HT", "HKG": "HK", "HON": "HN", "HUN": "HU",
    "INA": "ID", "IND": "IN", "IRI": "IR", "IRL": "IE", "ISL": "IS",
    "ISR": "IL", "ITA": "IT", "JAM": "JM", "JOR": "JO", "JPN": "JP",
    "KAZ": "KZ", "KEN": "KE", "KGZ": "KG", "KOR": "KR", "KSA": "SA",
    "KUW": "KW", "LAT": "LV", "LBA": "LY", "LBN": "LB", "LCA": "LC",
    "LIB": "LB", "LIE": "LI", "LTU": "LT", "LUX": "LU", "MAD": "MG",
    "MAR": "MA", "MAS": "MY", "MDA": "MD", "MEX": "MX", "MGL": "MN",
    "MKD": "MK", "MLT": "MT", "MNE": "ME", "MON": "MC", "NCA": "NI",
    "NED": "NL", "NEP": "NP", "NGR": "NG", "NOR": "NO", "NZL": "NZ",
    "OMA": "OM", "PAK": "PK", "PAN": "PA", "PAR": "PY", "PER": "PE",
    "PHI": "PH", "POL": "PL", "POR": "PT", "PRK": "KP", "PUR": "PR",
    "QAT": "QA", "ROU": "RO", "RSA": "ZA", "RUS": "RU", "SEN": "SN",
    "SEY": "SC", "SIN": "SG", "SLO": "SI", "SMR": "SM", "SRB": "RS",
    "SRI": "LK", "SUI": "CH", "SVK": "SK", "SWE": "SE", "SYR": "SY",
    "TAH": "PF", "TAN": "TZ", "THA": "TH", "TJK": "TJ", "TKM": "TM",
    "TOG": "TG", "TPE": "TW", "TRI": "TT", "TUN": "TN", "TUR": "TR",
    "UAE": "AE", "UGA": "UG", "UKR": "UA", "URU": "UY", "USA": "US",
    "UZB": "UZ", "VEN": "VE", "VIE": "VN", "ZAM": "ZM", "ZIM": "ZW",
}


# --------------------------------------------------------------------------- #
# Models tab                                                                  #
# --------------------------------------------------------------------------- #


def _render_tennis_models_page(metrics: dict, coefficients: dict,
                                sim_state: dict,
                                metrics_path: str | None = None) -> str:
    """Tennis Models tab — unified two-section layout used across every
    bot: (1) a table of every model the trainer produced with the same
    stats surfaced on the home-page model cards plus Brier, and
    (2) the readable-features panel with source colouring and
    permutation-importance bars. Everything else on this page has been
    stripped so the layout is identical across sports.
    """
    from pathlib import Path
    from .dashboard import (  # type: ignore
        _read_feature_importance, _render_feature_source_table,
        _render_models_run_table,
    )
    out: List[str] = []

    artifacts_dir = (Path(metrics_path).parent if metrics_path else None)

    # Feature-importance CSV — powers both the readable table and the
    # feature count that shows in the shared models table.
    feats: List[dict] = []
    if artifacts_dir:
        fi_path = artifacts_dir / "feature_importance.csv"
        if fi_path.exists():
            feats = _read_feature_importance(str(fi_path))

    # Bundle mtime → "Last trained" for the table.
    last_retrain = "—"
    if artifacts_dir:
        bundle_path = artifacts_dir / "prematch_model.joblib"
        if bundle_path.exists():
            try:
                import datetime as _dt
                mt = _dt.datetime.fromtimestamp(
                    bundle_path.stat().st_mtime, tz=_dt.timezone.utc)
                last_retrain = mt.strftime("%Y-%m-%d")
            except (OSError, OverflowError):
                pass

    # 1) Table of models run.
    out.append(_render_models_run_table(
        metrics or {},
        feature_count=len(feats) if feats else None,
        last_trained=last_retrain,
    ))

    # 2) Features with definitions and bars.
    if feats:
        out.append(_render_feature_source_table(feats))
    else:
        out.append("<div class='empty' style='margin-top:12px;'>"
                    "Feature importance not yet written for this bot — "
                    "the file lands after the next retrain.</div>")

    return "".join(out)


# --------------------------------------------------------------------------- #
# Page renderer                                                                #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Training Data tab                                                           #
# --------------------------------------------------------------------------- #

# Default location of the tennis training DB on the droplet — matches the
# trainer's ``upsert_training_panel`` path. ``Path.exists`` decides whether
# the dashboard shows a populated table or a "not initialised yet" stub.
_TRAINING_DB_PATH = Path("/root/tennis-forecast/data/training_history.db")

# All training-data table columns and their definitions. Each entry is
# (sql-column, short-label, full-definition). The short label goes in
# the table header; clicking it pops up a definition modal. ``None``
# label hides the column from rendering (e.g. internal IDs).
#
# The order here is the rendered column order.
_TRAINING_COLUMNS: list[tuple[str, str, str]] = [
    # ── Match identity ───────────────────────────────────────────────
    # Tournament is deliberately the leftmost column — it's the most
    # recognisable per-row identifier, and users scanning the table
    # anchor on tournament name before date.
    ("tourney_name", "Tournament",
     "Tournament name from the official ATP/WTA tour calendar."),
    ("tourney_date", "Date",
     "Date of the match's tournament round, in YYYY-MM-DD."),
    ("tour", "Tour",
     "ATP (men's) or WTA (women's). Determined by which Sackmann "
     "match file the row came from."),
    ("surface", "Surface",
     "Hard / Clay / Grass / Carpet — the playing surface."),
    ("level", "Level",
     "Tournament tier raw code: G = Grand Slam, M = Masters 1000, "
     "A = ATP 500/250 or WTA tier-equivalent, F = Tour Finals, "
     "D = Davis Cup, C = Challenger, S = ITF Futures."),
    ("round", "Round",
     "Round of the match: R128, R64, R32, R16, QF, SF, F (final), "
     "RR = round-robin, BR = bronze."),
    ("draw_size", "Draw",
     "Total number of players in the main draw. 128 for a Slam, 64 "
     "for most Masters, 32 for most 250s."),
    ("best_of", "BO",
     "Best-of-3 or best-of-5 sets. Slams + Davis Cup men's are "
     "best-of-5; everything else is best-of-3."),
    # ── Outcome ─────────────────────────────────────────────────────
    ("player_a", "Player A",
     "First player in the matchup. Each historical match is stored "
     "twice in the training panel — once with Player A = the actual "
     "winner and once with Player A = the actual loser — so the "
     "feature differences cancel and the trained model isn't biased "
     "toward putting the winner on either side."),
    ("player_b", "Player B",
     "Second player in the matchup. See Player A for the orientation "
     "note."),
    ("winner", "Winner",
     "Name of the player who actually won this match."),
    # ── Player A raw attributes ─────────────────────────────────────
    ("a_age", "A age",
     "Player A's age in years at match start."),
    ("a_height_cm", "A height",
     "Player A's listed height in centimetres."),
    ("a_hand", "A hand",
     "Player A's playing hand: R = right, L = left, U = unknown / "
     "ambidextrous."),
    ("a_country", "A country",
     "Player A's nationality (IOC 3-letter code)."),
    ("a_rank", "A rank",
     "Player A's ATP/WTA singles ranking at the time of the match. "
     "Lower number = better."),
    ("a_rank_points", "A pts",
     "Player A's ranking points entering the match."),
    ("a_seed", "A seed",
     "Player A's seeding in this draw, if seeded."),
    ("a_entry", "A entry",
     "How Player A entered the draw: Q = qualifier, WC = wild card, "
     "LL = lucky loser, SE = special exempt, ALT = alternate, "
     "PR = protected ranking."),
    # ── Player B raw attributes ─────────────────────────────────────
    ("b_age", "B age", "Player B's age in years at match start."),
    ("b_height_cm", "B height", "Player B's listed height in cm."),
    ("b_hand", "B hand",
     "Player B's playing hand: R = right, L = left, U = unknown."),
    ("b_country", "B country",
     "Player B's nationality (IOC 3-letter code)."),
    ("b_rank", "B rank",
     "Player B's ATP/WTA singles ranking at match time."),
    ("b_rank_points", "B pts",
     "Player B's ranking points entering the match."),
    ("b_seed", "B seed",
     "Player B's seeding in this draw, if seeded."),
    ("b_entry", "B entry",
     "Player B's entry route into the draw (Q, WC, LL, SE, ALT, PR)."),
    # ── Engineered features the model actually trains on ─────────────
    ("diff_elo_pre", "Elo Δ",
     "Player A's pre-match overall Elo minus Player B's. Computed "
     "rolling through the historical match panel. Larger positive = "
     "Player A is the all-surface favourite. ⚙ MODEL FEATURE."),
    ("diff_surface_elo_pre", "Surface Elo Δ",
     "Surface-specific Elo difference (A − B) for this match's "
     "surface. Captures the fact that some players are clay "
     "specialists, others grass specialists, etc. ⚙ MODEL FEATURE."),
    ("diff_form_last5", "Form 5 Δ",
     "Win-rate over the last 5 matches (A − B). Captures short-term "
     "momentum / cold streaks. ⚙ MODEL FEATURE."),
    ("diff_form_last10", "Form 10 Δ",
     "Win-rate over the last 10 matches (A − B). Smoother form "
     "signal that's less reactive to a single bad day. ⚙ MODEL "
     "FEATURE."),
    ("diff_avg_serve_pts_won_10", "Serve % Δ",
     "Average serve points won % over the last 10 matches (A − B). "
     "A direct measure of who's holding serve better recently. "
     "⚙ MODEL FEATURE."),
    ("diff_avg_return_pts_won_10", "Return % Δ",
     "Average return points won % over the last 10 matches (A − B). "
     "Captures who's been breaking serve / pressuring the opponent's "
     "delivery. ⚙ MODEL FEATURE."),
    ("diff_avg_bp_saved_10", "BP saved Δ",
     "Average break-points saved % over the last 10 matches (A − B). "
     "Clutch-on-serve indicator. ⚙ MODEL FEATURE."),
    ("diff_days_rest", "Days rest Δ",
     "Days since each player's last match (A − B). Positive = "
     "Player A had more rest. Top permutation-importance feature in "
     "the current model. ⚙ MODEL FEATURE."),
    ("h2h_a_wins_minus_b_wins", "H2H Δ",
     "Career head-to-head record up to (not including) this match: "
     "A's wins minus B's wins. ⚙ MODEL FEATURE."),
    ("rank_diff", "Rank Δ",
     "B's ranking minus A's ranking (so positive = A is higher-"
     "ranked / better). ⚙ MODEL FEATURE."),
    ("level_rank", "Level rank",
     "Tournament tier as a numeric code: Grand Slam = 4, Masters / "
     "WTA 1000 = 3, ATP 500 / WTA 500 / ATP 250 / WTA 250 = 2, "
     "Davis Cup / Challenger / other = 1. ⚙ MODEL FEATURE."),
    ("round_rank", "Round rank",
     "Round depth as a numeric code: R128 = 1, R64 = 2, R32 = 3, "
     "R16 = 4, QF = 5, SF = 6, F = 8. ⚙ MODEL FEATURE."),
    # ── Derived / candidate features (NOT currently selected) ───────
    ("age_diff", "Age Δ",
     "Player A's age minus B's. Negative = Player A is younger. "
     "Computed but not currently in the selected feature list — "
     "tracked here in case it surfaces signal in a future search."),
    ("height_diff_cm", "Height Δ",
     "Player A's height minus B's, in cm. Taller players historically "
     "have an edge on fast surfaces. Candidate feature, not yet "
     "selected."),
    ("rank_points_diff", "Rank pts Δ",
     "Player A's ranking points minus B's. A continuous version of "
     "the rank diff that's more sensitive to the gap between top-5 "
     "and top-20. Candidate feature."),
    ("seed_diff", "Seed Δ",
     "Player B's seed minus A's. Positive = A is higher-seeded. "
     "Candidate feature."),
    ("hand_match", "Same hand?",
     "1 if both players are right-handed or both left-handed, "
     "0 if one is left and one is right (the lefty advantage case), "
     "blank when at least one is unknown. Candidate feature."),
    ("same_country", "Same flag?",
     "1 if both players share the same IOC country code, else 0. "
     "Captures the rare same-country matchup. Candidate feature."),
]

# Backwards-compat alias used in older render code paths. Maps the
# columns the rendered table previously knew about to the new
# (label, definition) pair the modal reads.
_FEATURE_LABELS: Dict[str, Tuple[str, str]] = {
    sql: (label, definition)
    for sql, label, definition in _TRAINING_COLUMNS
}


def _open_training_db():
    """Try to import the trainer-side training_db module to reuse its
    pagination helpers. Returns ``None`` if tennis-forecast's ``src/``
    isn't on sys.path (which happens at dashboard startup before the
    bots' upstream packages are loaded — the alias registered by
    bots/tennis.py adds ``src.data`` to sys.modules)."""
    try:
        # The tennis bot loads its upstream package under the alias
        # ``tennis_src`` (see bots/_base.load_upstream_as_alias) and
        # registers ``src`` -> ``tennis_src`` aliases for joblib unpickle.
        # That alias chain also makes ``src.data.training_db`` reachable.
        import importlib
        return importlib.import_module("src.data.training_db")
    except ImportError:
        try:
            return importlib.import_module("tennis_src.data.training_db")
        except Exception:  # noqa: BLE001
            return None


def render_training_data_panel(*, current_bot: str | None,
                                  page: int = 1, page_size: int = 50,
                                  tour_filter: str | None = None,
                                  split_filter: str | None = None,
                                  current_tab: str = "training",
                                  period_key: str = "all") -> str:
    """Render the Training Data tab. Tennis-only — the panel reads the
    rows the trainer wrote to ``training_history.db`` and paginates
    over them. Shows the 12 engineered features the model trains on,
    the binary winner label, and which train/val/test split slice the
    row belongs to.

    Other bots see a brief explanation: this DB is tennis-specific
    until their trainers also adopt the pattern.
    """
    if current_bot and current_bot != "tennis":
        return (
            "<section class='card'><div class='body'>"
            "<h2>Training Data</h2>"
            "<p class='small gray'>The Training Data tab is currently "
            "tennis-only. Each row shown reflects what the model trained "
            "on — engineered features and the binary winner label. "
            "Other bots haven't been wired into the training database "
            "yet.</p></div></section>"
        )
    db_mod = _open_training_db()
    if db_mod is None or not _TRAINING_DB_PATH.exists():
        return (
            "<section class='card'><div class='body'>"
            "<h2>Training Data — tennis</h2>"
            "<p class='small gray'>The training database hasn't been "
            "populated yet. It's written by the daily tennis-forecast "
            "retrain (see <code>src/models/train_prematch_model.py</code>'s "
            "<code>upsert_training_panel</code> call). Run the trainer "
            "once and the rows appear here.</p></div></section>"
        )

    # Whitelist filter values so a malformed query string can't be
    # forwarded into the SQL.
    tour_filter = tour_filter if tour_filter in ("ATP", "WTA") else None
    split_filter = split_filter if split_filter in ("train", "val",
                                                       "test") else None

    # Only surface fully-populated historical rows. Kalshi-only bookings
    # (matches the bot recorded on Kalshi before Sackmann caught up)
    # would render with every engineered feature blank — they don't
    # represent training data the model ever saw and the mostly-empty
    # rows just add noise here. They're still tracked in
    # ``kalshi_outcomes`` for the P&L / bookings views elsewhere.
    n_historical = db_mod.count_training_matches(
        _TRAINING_DB_PATH, tour=tour_filter, split=split_filter,
    )
    total = n_historical
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)

    kalshi_only_rows: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    if n_historical > 0:
        rows = db_mod.fetch_training_matches(
            _TRAINING_DB_PATH,
            page=page, page_size=page_size,
            tour=tour_filter, split=split_filter,
        )

    # Derive Winner from label for every training row (label=1 means
    # player_a won; label=0 means player_b won). Done here rather
    # than in SQL because the column is computed only at render time.
    for r in rows:
        label_v = r.get("label")
        if label_v is None:
            r["winner"] = None
        else:
            r["winner"] = (r.get("player_a") if int(label_v) == 1
                            else r.get("player_b"))


    out: List[str] = []
    out.append("<section class='card'><div class='body'>")
    out.append("<h2>Training Data — tennis</h2>")
    out.append(
        f"<p class='small gray'>Every fully-populated match the model "
        f"actually trained on — <b>{total:,}</b> rows from the trainer's "
        f"most recent fit. Each row is a single match with the winner "
        f"and every candidate feature. Sorted newest first; columns "
        f"flagged ⚙ MODEL FEATURE in their definition are the ones the "
        f"model actually trains on.</p>"
    )

    # Tour / split filter pills. Hand-rolled query-string preservation
    # so the pagination links below also keep the active filter.
    def _filter_link(key: str, value: str | None, label: str,
                      active: bool) -> str:
        params = [("tab", current_tab)]
        if current_bot:
            params.append(("bot", current_bot))
        if period_key and period_key != "all":
            params.append(("period", period_key))
        # Preserve the OTHER filter dim
        if key != "tour" and tour_filter:
            params.append(("tour", tour_filter))
        if key != "split" and split_filter:
            params.append(("split", split_filter))
        if value is not None:
            params.append((key, value))
        qs = "&".join(f"{k}={v}" for k, v in params)
        cls = "tab-pill" + (" tab-pill-active" if active else "")
        return f"<a class='{cls}' href='?{qs}'>{html.escape(label)}</a>"

    out.append("<div class='tab-bar' style='margin-top:8px;'>")
    out.append("<span class='small gray' style='margin-right:8px;'>Tour:</span>")
    out.append(_filter_link("tour", None, "All", tour_filter is None))
    out.append(_filter_link("tour", "ATP", "ATP", tour_filter == "ATP"))
    out.append(_filter_link("tour", "WTA", "WTA", tour_filter == "WTA"))
    out.append("</div>")
    out.append(
        "<p class='small gray' style='margin-top:8px;'>"
        "Click any column header for its definition. Columns marked "
        "<b>⚙ MODEL FEATURE</b> in the definition are the ones the "
        "current model actually trains on; everything else is a "
        "candidate feature carried for review.</p>"
    )

    # Table — every column is a clickable header that opens a modal
    # with the column's definition (see the inline script below).
    out.append("<div style='overflow-x:auto;margin-top:12px;'>")
    out.append("<table class='training-data-table'><thead><tr>")
    for sql, label, _ in _TRAINING_COLUMNS:
        # Numeric columns get .num for right-alignment; player names /
        # categorical attrs stay left-aligned for readability.
        is_num = sql not in {
            "tourney_date", "tourney_name", "tour", "surface", "level",
            "round", "player_a", "player_b", "a_hand", "a_country",
            "a_entry", "b_hand", "b_country", "b_entry",
        }
        cls = " class='num'" if is_num else ""
        out.append(
            f"<th{cls}><button type='button' class='col-def-btn' "
            f"data-col='{html.escape(sql)}'>"
            f"{html.escape(label)}</button></th>"
        )
    out.append("</tr></thead><tbody>")
    if not rows and not kalshi_only_rows:
        out.append(
            f"<tr><td colspan='{len(_TRAINING_COLUMNS)}' "
            f"class='empty'>No rows for the selected filter.</td></tr>"
        )

    def _fmt_cell(sql: str, v: Any, row: dict | None = None) -> str:
        if v is None or v == "":
            return "—"
        if sql == "tourney_name":
            # Prefix the tour ("ATP" / "WTA") to the tournament name so
            # "Wimbledon" reads as "ATP · Wimbledon" — a single-column
            # answer to "which tour and event was this?" without making
            # the user cross-reference the Tour column.
            tour = ((row or {}).get("tour") or "").strip().upper()
            name = html.escape(str(v))
            return f"{tour} · {name}" if tour else name
        if sql in {"a_hand", "b_hand"}:
            return html.escape(str(v))
        if sql in {"hand_match", "same_country"}:
            return "Yes" if int(v) == 1 else "No"
        if isinstance(v, float):
            # Diffs render with a sign; raw stats render plain.
            if sql.endswith("_diff") or sql.startswith("diff_") or \
                    sql == "h2h_a_wins_minus_b_wins":
                return f"{v:+.3f}"
            return f"{v:.3f}" if abs(v) < 1000 else f"{int(v):,}"
        if isinstance(v, int):
            return f"{v:,}" if abs(v) >= 1000 else str(v)
        return html.escape(str(v))

    # Render Kalshi-only rows first (most recent, page 1 only), then
    # the paginated historical rows below.
    for r in kalshi_only_rows + rows:
        out.append("<tr>")
        for sql, _, _ in _TRAINING_COLUMNS:
            v = r.get(sql)
            is_num = sql not in {
                "tourney_date", "tourney_name", "tour", "surface", "level",
                "round", "player_a", "player_b", "a_hand", "a_country",
                "a_entry", "b_hand", "b_country", "b_entry",
            }
            cls = " class='num'" if is_num else ""
            out.append(f"<td{cls}>{_fmt_cell(sql, v, r)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")

    # Pagination — Prev | page N of M | Next + jump-to dropdown
    def _page_link(p: int) -> str:
        params = [("tab", current_tab)]
        if current_bot:
            params.append(("bot", current_bot))
        if period_key and period_key != "all":
            params.append(("period", period_key))
        if tour_filter:
            params.append(("tour", tour_filter))
        if split_filter:
            params.append(("split", split_filter))
        params.append(("page", str(p)))
        return "?" + "&".join(f"{k}={v}" for k, v in params)

    out.append("<div class='small' style='margin-top:14px;display:flex;"
                "align-items:center;gap:12px;'>")
    if page > 1:
        out.append(f"<a class='tab-pill' href='{_page_link(page - 1)}'>← Prev</a>")
    else:
        out.append("<span class='tab-pill tab-pill-disabled'>← Prev</span>")
    out.append(
        f"<span>Page <b>{page:,}</b> of <b>{total_pages:,}</b> "
        f"<span class='gray'>({total:,} rows)</span></span>"
    )
    # Jump-to dropdown — covers up to 1000 pages cheaply; beyond that
    # rendering options gets heavy, so fall back to a small text input.
    if total_pages <= 1000:
        out.append("<form method='get' style='display:inline;'>")
        # Preserve filters as hidden fields so the dropdown's submit
        # navigates to the right URL.
        out.append(f"<input type='hidden' name='tab' value='{html.escape(current_tab)}'>")
        if current_bot:
            out.append(f"<input type='hidden' name='bot' value='{html.escape(current_bot)}'>")
        if period_key and period_key != "all":
            out.append(f"<input type='hidden' name='period' value='{html.escape(period_key)}'>")
        if tour_filter:
            out.append(f"<input type='hidden' name='tour' value='{html.escape(tour_filter)}'>")
        if split_filter:
            out.append(f"<input type='hidden' name='split' value='{html.escape(split_filter)}'>")
        out.append("<label class='gray' style='margin-right:6px;'>Jump:</label>")
        out.append("<select name='page' onchange='this.form.submit()'>")
        for p in range(1, total_pages + 1):
            sel = " selected" if p == page else ""
            out.append(f"<option value='{p}'{sel}>{p}</option>")
        out.append("</select></form>")
    else:
        out.append(
            "<form method='get' style='display:inline;'>"
            f"<input type='hidden' name='tab' value='{html.escape(current_tab)}'>"
            f"<input type='hidden' name='bot' value='{html.escape(current_bot or '')}'>"
            "<label class='gray' style='margin-right:6px;'>Jump to:</label>"
            f"<input type='number' name='page' min='1' max='{total_pages}' "
            f"value='{page}' style='width:80px;'><button type='submit'>Go</button>"
            "</form>"
        )
    if page < total_pages:
        out.append(f"<a class='tab-pill' href='{_page_link(page + 1)}'>Next →</a>")
    else:
        out.append("<span class='tab-pill tab-pill-disabled'>Next →</span>")
    out.append("</div>")

    # ── Column-definition popover + inline JS ────────────────────────
    # Lightweight tooltip-style panel that anchors below the clicked
    # column header — no backdrop, no page dim, dismissed by clicking
    # away or pressing Escape. Re-uses the same defs map keyed by
    # SQL column name; positioned at render time via
    # ``getBoundingClientRect`` of the clicked button.
    import json as _json
    defs = {sql: {"label": label, "def": definition}
             for sql, label, definition in _TRAINING_COLUMNS}
    out.append(
        "<div id='col-def-pop' class='col-def-pop' hidden>"
        "<div class='col-def-pop-title'></div>"
        "<div class='col-def-pop-body'></div>"
        "</div>"
    )
    out.append(
        "<style>"
        ".col-def-btn { background:none; border:0; color:inherit; "
        "font:inherit; cursor:pointer; padding:0; "
        "text-decoration:underline dotted; }"
        ".col-def-btn:hover { color:#79c0ff; }"
        ".col-def-pop { position:absolute; z-index:1000; max-width:320px; "
        "padding:10px 12px; background:#0d1117; color:#c9d1d9; "
        "border:1px solid #30363d; border-radius:6px; "
        "box-shadow:0 6px 20px rgba(0,0,0,.45); font-size:12px; "
        "line-height:1.45; pointer-events:auto; }"
        ".col-def-pop[hidden] { display:none; }"
        ".col-def-pop-title { font-weight:700; margin-bottom:4px; "
        "color:#f0f6fc; }"
        ".training-data-table { font-size:12px; }"
        ".training-data-table th { white-space:nowrap; }"
        ".training-data-table td { white-space:nowrap; }"
        "</style>"
    )
    out.append(
        "<script>(function(){"
        "var defs = " + _json.dumps(defs) + ";"
        "var pop = document.getElementById('col-def-pop');"
        "if (!pop) return;"
        "var titleEl = pop.querySelector('.col-def-pop-title');"
        "var bodyEl = pop.querySelector('.col-def-pop-body');"
        "var currentBtn = null;"
        "function hide(){ pop.hidden = true; currentBtn = null; }"
        "function show(btn){"
        "  var d = defs[btn.dataset.col];"
        "  if (!d) return;"
        "  titleEl.textContent = d.label;"
        "  bodyEl.textContent = d['def'];"
        "  pop.hidden = false;"
        "  var rect = btn.getBoundingClientRect();"
        "  pop.style.visibility = 'hidden';"
        "  pop.style.left = '0px';"
        "  pop.style.top = '0px';"
        "  var popRect = pop.getBoundingClientRect();"
        "  var pageX = window.scrollX, pageY = window.scrollY;"
        "  var left = rect.left + pageX;"
        "  var top = rect.bottom + pageY + 6;"
        "  var maxLeft = pageX + window.innerWidth - popRect.width - 8;"
        "  if (left > maxLeft) left = Math.max(pageX + 8, maxLeft);"
        "  pop.style.left = left + 'px';"
        "  pop.style.top = top + 'px';"
        "  pop.style.visibility = 'visible';"
        "  currentBtn = btn;"
        "}"
        "document.querySelectorAll('.col-def-btn').forEach(function(b){"
        "  b.addEventListener('click', function(e){"
        "    e.stopPropagation();"
        "    if (currentBtn === b) { hide(); return; }"
        "    show(b);"
        "  });"
        "});"
        "document.addEventListener('click', function(e){"
        "  if (pop.hidden) return;"
        "  if (!pop.contains(e.target)) hide();"
        "});"
        "document.addEventListener('keydown', function(e){"
        "  if (e.key === 'Escape') hide();"
        "});"
        "window.addEventListener('scroll', hide, true);"
        "window.addEventListener('resize', hide);"
        "})();</script>"
    )

    out.append("</section>")
    return "".join(out)


# Path to the live executor's sim_state — used to enrich Kalshi-only
# rows with full player names + tournament metadata recorded at order-
# placement time (Kalshi's /portfolio/settlements doesn't carry those).
_LIVE_SIM_STATE_PATH = Path(
    "/root/tennis-forecast/data/outputs-live/sim_state.json"
)

