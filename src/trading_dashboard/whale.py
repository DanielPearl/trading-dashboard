"""Whale-watcher dashboard view.

Different shape than the gas-bot-style dashboard:

  - Source is JSONL, not SQLite. Whale-watcher writes
    `data/signal_tracking.jsonl` (one row per detected whale event,
    accepted or rejected) and `data/orders.jsonl` (one row per entry
    the bot placed).
  - There are no "users" — Kalshi's public API does not expose trader
    identity on trades. The "user" abstraction here is the *whale event*
    itself. As a substitute for "profitable user", we score each signal
    against a *cohort win rate*: signals matching the same ticker
    family + size bucket + direction whose +30m checkpoint moved in
    the bet's favor. High cohort win-rate + high z-score + tight book
    + meaningful notional ≈ "this looks like someone who knows
    something" — surfaced as the per-signal "insider probability".

Reads are cheap (small JSONL files) and best-effort — missing files
render the empty state instead of raising.

Three tabs (Home / Watchlist / History) mirror the main dashboard's
shape so users can hop between bot-equity and whale-signal context
without learning a second navigation idiom.
"""
from __future__ import annotations

import html
import json
import logging
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("dashboard.whale")


# --------------------------------------------------------------------------- #
# Data loaders                                                                #
# --------------------------------------------------------------------------- #

def load_events(path: str | None, limit: int = 1000) -> List[dict]:
    """Read up to `limit` most-recent rows from signal_tracking.jsonl.

    Each row already has all checkpoints captured (signal_tracker only
    flushes a row once every checkpoint resolves). So we can trust
    `checkpoints[-1].favorable_cents` as the +30m signal-quality value.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: List[dict] = []
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    if limit and len(out) > limit:
        out = out[-limit:]
    return out


def load_orders(path: str | None, limit: int = 500) -> List[dict]:
    """Read recent entries from orders.jsonl. Unused after the tab
    restructure (entries are surfaced as simulated buys in the
    Home tab) but kept around so external callers / tests can still
    poke at the orders file.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: List[dict] = []
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    if limit and len(out) > limit:
        out = out[-limit:]
    return out


# --------------------------------------------------------------------------- #
# Derived metrics                                                             #
# --------------------------------------------------------------------------- #

def _last_favorable(event: dict) -> Optional[float]:
    """+30m (last) favorable_cents, or None if not captured."""
    ckpts = event.get("checkpoints") or []
    if not ckpts:
        return None
    fav = ckpts[-1].get("favorable_cents")
    return None if fav is None else float(fav)


def _favorable_at(event: dict, idx: int) -> Optional[float]:
    ckpts = event.get("checkpoints") or []
    if idx >= len(ckpts):
        return None
    fav = ckpts[idx].get("favorable_cents")
    return None if fav is None else float(fav)


def _ticker_prefix(ticker: str) -> str:
    """Kalshi tickers look like 'KXNFLGAME-26APR18SEAWAS-SEA'. We bucket
    by the part before the first '-' so events on the same series cluster
    together."""
    if "-" not in ticker:
        return ticker
    return ticker.split("-", 1)[0]


def _size_bucket(notional_cents: int) -> str:
    """Log-bucket bet size into human labels."""
    dollars = notional_cents / 100.0
    if dollars < 5:
        return "<$5"
    if dollars < 20:
        return "$5-$20"
    if dollars < 100:
        return "$20-$100"
    if dollars < 500:
        return "$100-$500"
    return "$500+"


# --------------------------------------------------------------------------- #
# Cohort win rates — pseudo-identity stand-in                                 #
# --------------------------------------------------------------------------- #

def compute_cohort_winrates(events: List[dict],
                              min_n: int = 3
                              ) -> Dict[Tuple[str, str, str], dict]:
    """For each (ticker_prefix, size_bucket, direction) cohort, compute
    historical win rate from completed signals (those with at least one
    captured checkpoint with favorable_cents > 0 → win).

    Cohorts with fewer than ``min_n`` completed samples are kept but
    flagged — score callers can choose to weight them lower or skip.

    Returns ``{ (prefix, size, direction): {"win_rate": float, "n": int,
    "mean_fav_30m": float} }``.
    """
    buckets: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for e in events:
        ticker = e.get("ticker") or ""
        notional = e.get("whale_notional_cents") or 0
        direction = e.get("direction") or "?"
        key = (_ticker_prefix(ticker), _size_bucket(int(notional)), direction)
        buckets[key].append(e)

    out: Dict[Tuple[str, str, str], dict] = {}
    for key, members in buckets.items():
        favs = [f for f in (_last_favorable(e) for e in members) if f is not None]
        if not favs:
            continue
        wins = sum(1 for f in favs if f > 0)
        out[key] = {
            "win_rate": wins / len(favs),
            "n": len(favs),
            "mean_fav_30m": sum(favs) / len(favs),
        }
    return out


def cohort_winrate_for(event: dict,
                         cohort_winrates: Dict[Tuple[str, str, str], dict],
                         min_n: int = 3,
                         ) -> Optional[float]:
    """Win rate for the cohort this event belongs to. Returns None when
    the cohort is too thin (n < min_n) to be trustworthy.
    """
    ticker = event.get("ticker") or ""
    notional = event.get("whale_notional_cents") or 0
    direction = event.get("direction") or "?"
    key = (_ticker_prefix(ticker), _size_bucket(int(notional)), direction)
    row = cohort_winrates.get(key)
    if row is None:
        return None
    if row["n"] < min_n:
        return None
    return row["win_rate"]


# --------------------------------------------------------------------------- #
# Insider-probability model                                                   #
# --------------------------------------------------------------------------- #
#
# Heuristic (no labels yet — this is a transparent linear scorer):
#
#   z_norm   — z-score of the trade's contract count vs the market's
#              own rolling lookback. Capped at 10, normalised to [0, 1].
#   size     — log-cents notional, normalised: 0¢ → 0, $100+ → 1.
#   dir_conf — direction_confidence (already 0..1).
#   cohort   — cohort_winrate_for(event); if unknown, default 0.5
#              (no prior either way).
#   tight    — book quality at signal time: tighter spread + deeper
#              book → 1, wide/empty → 0. Insider-y trades happen on
#              books they didn't have to pay much spread to enter.
#
# Weighted sum, then clipped to [0, 1].
# Weights chosen so each signal contributes ~equally; tunable.

# Feature weights — sum to 1.0 so the linear combination stays in
# [0, 1]. Tuned to give cohort + contrarian price the biggest say
# (those are the strongest "this trader knows something" tells) and
# size + book-quality less (those just say "the trade is real").
_W_Z          = 0.18   # size unusualness vs market history
_W_SIZE       = 0.12   # absolute notional (log-scaled)
_W_DIRCONF    = 0.10   # taker_side reliability
_W_COHORT     = 0.25   # historical win rate of similar past signals
_W_TIGHT      = 0.05   # book quality at trade time (spread + depth)
_W_TTC        = 0.15   # time-to-close — insiders cluster near resolution
_W_CONTRARIAN = 0.15   # taking the otherwise-unlikely side


def insider_score(event: dict,
                    cohort_winrates: Dict[Tuple[str, str, str], dict]
                    ) -> float:
    """Return [0, 1] probability that this whale signal looks like
    "someone who knows something". Transparent linear combination of
    seven features — easy to override / tune.

    See module-level _W_* constants for weights. Cohort contribution
    is scaled by sample-size confidence so a 100%-WR cohort built on
    n=2 doesn't look as strong as one built on n=30. Contrarian
    price + time-to-close are the new "informed flow" features.
    """
    z_raw = float(event.get("zscore") or 0.0)
    z_norm = max(0.0, min(1.0, z_raw / 10.0))

    notional = float(event.get("whale_notional_cents") or 0)
    if notional <= 0:
        size = 0.0
    else:
        # log10(cents) ≈ 0..5 (1¢ → 0, $1000 → 5). Normalise to [0, 1]
        # by dividing by 4 (so $100 = 1).
        size = max(0.0, min(1.0, math.log10(notional) / 4.0))

    dir_conf = float(event.get("direction_confidence") or 0.0)

    # Cohort contribution: shrunk toward 0.5 prior when n is small.
    # confidence = sqrt(n / 20) capped at 1, so n=20 reads at full
    # weight, n=5 at ~50%, n=1 at ~22%.
    coh_row = _cohort_row_for(event, cohort_winrates)
    if coh_row is None:
        cohort = 0.5
    else:
        n = max(1, int(coh_row.get("n") or 1))
        wr = float(coh_row.get("win_rate") or 0.5)
        confidence = min(1.0, math.sqrt(n / 20.0))
        cohort = 0.5 + (wr - 0.5) * confidence

    spread = event.get("entry_spread_cents")
    depth = event.get("entry_depth_within_3c") or 0
    if spread is None:
        tight = 0.5
    else:
        sp = max(0, min(8, int(spread)))
        s_norm = 1.0 - (sp / 8.0)
        d_norm = max(0.0, min(1.0, depth / 200.0))
        tight = 0.5 * s_norm + 0.5 * d_norm

    # Time-to-close: highest informational value when the bet is
    # 1h–24h before resolution (insider edge has limited shelf-life
    # and resolution-source data tends to be pinned down by then).
    # Outside that window we shrink toward 0.5 (no info).
    minutes_to_close = event.get("minutes_to_close")
    if minutes_to_close is None:
        ttc = 0.5
    else:
        m = float(minutes_to_close)
        if 60 <= m <= 60 * 24:
            ttc = 1.0
        elif m < 60:
            # Last-hour bets are still informational but resolution
            # may already be priced in. Linear ramp 30m→1h: 0.6→1.0.
            ttc = max(0.0, 0.6 + (m - 30) / 30 * 0.4) if m >= 30 else 0.4
        else:
            # 1d+: decay slowly. 24h→0.7, 7d→0.3.
            d = m / 1440.0
            ttc = max(0.3, 1.0 - 0.4 * (d - 1) / 6)

    # Contrarian-price: taking the side the market thinks is unlikely.
    # entry_mid_cents is the YES side mid (0..100). Implied prob of
    # the side bet on = mid for YES, 100-mid for NO.
    mid = event.get("entry_mid_cents")
    direction = (event.get("direction") or "").lower()
    if mid is None or direction not in {"yes", "no"}:
        contrarian = 0.5
    else:
        implied = float(mid) if direction == "yes" else (100.0 - float(mid))
        # Contrarian zone is implied < 25c — paying for a side the
        # market thinks is a long-shot. The further from 50c (in the
        # underdog direction) the more contrarian. Symmetrically the
        # market-favourite zone (>75c) is "agreement" — not insider.
        if implied < 25:
            contrarian = 1.0 - (implied / 25.0) * 0.4   # 25c → 0.6, 0c → 1.0
        elif implied > 75:
            contrarian = 0.3   # paying through to lock in a sure thing
        else:
            contrarian = 0.5

    raw = (
        _W_Z          * z_norm     +
        _W_SIZE       * size       +
        _W_DIRCONF    * dir_conf   +
        _W_COHORT     * cohort     +
        _W_TIGHT      * tight      +
        _W_TTC        * ttc        +
        _W_CONTRARIAN * contrarian
    )
    return max(0.0, min(1.0, raw))


def _cohort_row_for(event: dict,
                      cohort_winrates: Dict[Tuple[str, str, str], dict],
                      ) -> Optional[dict]:
    """Internal: return the full cohort dict (win_rate + n) for
    confidence-weighted scoring. The public cohort_winrate_for()
    keeps returning a bare float for callers that just need the
    headline number.
    """
    ticker = event.get("ticker") or ""
    notional = event.get("whale_notional_cents") or 0
    direction = event.get("direction") or "?"
    key = (_ticker_prefix(ticker), _size_bucket(int(notional)), direction)
    return cohort_winrates.get(key)


# --------------------------------------------------------------------------- #
# Validators — mirror the bot's pre-trade checks                              #
# --------------------------------------------------------------------------- #

# Tunables. Keep these in sync with the bot's validator config when
# possible; the dashboard re-runs the same logic locally so simulated
# buys reflect what the bot would actually have entered.
VALIDATOR_MIN_ZSCORE         = 3.0
VALIDATOR_MIN_NOTIONAL_CENTS = 5_000   # $50
VALIDATOR_MIN_DIR_CONF       = 0.5
VALIDATOR_MAX_SPREAD_CENTS   = 8
VALIDATOR_MIN_BOOK_DEPTH     = 25      # contracts within 3¢ of best
VALIDATOR_PROB_LO_CENTS      = 15
VALIDATOR_PROB_HI_CENTS      = 85
VALIDATOR_MIN_COHORT_WIN     = 0.50    # need ≥ 50% historical win rate
VALIDATOR_MIN_COHORT_N       = 5       # … on at least 5 prior signals
VALIDATOR_INSIDER_THRESHOLD  = 0.65    # score ≥ this → eligible to "buy"

# "Looks like manipulation, not informed flow" filters. Hidden from
# the visible table (the user wanted a leaner row) but contribute to
# all_pass / Verdict so coordinated-bet patterns vs lone-pump
# patterns are differentiated.
#
# Cluster-size validator   — informed traders rarely act alone; one
#                             big bet with zero supporting flow on the
#                             same ticker in the lookback window is
#                             more likely manipulation than insider
#                             knowledge.
VALIDATOR_MIN_CLUSTER_SIZE   = 2          # this trade + at least 1 other big
VALIDATOR_CLUSTER_MIN_NOTIONAL = 25_000   # $250+ each to count as "supporting"
# Round-number validator   — retail traders bet in 100/250/500/1000-
#                             contract round lots. Insiders sized to
#                             specific dollar amounts rarely land on
#                             a round count exactly. Fail if the
#                             count is at one of these neat values.
VALIDATOR_ROUND_NUMBERS      = (100, 200, 250, 500, 1000, 2500, 5000)
# Price-stability validator — if the post-trade mid has drifted far
#                             from the bet price, the trade either
#                             pushed the market itself (manipulation
#                             attempt) OR a real news event hit. Both
#                             are reasons to be cautious.
VALIDATOR_MAX_PRICE_DRIFT    = 10         # cents

# Live-big-bets pull thresholds. Kalshi exposes /markets/trades; we
# fetch recent trades across the bots' configured series and surface
# every trade above this notional as a candidate. Smaller than the
# whale_detector min_notional_cents (5000c = $50) used by the bot —
# the dashboard wants to *show* every big bet, not just the
# z-score-anomalous ones.
# Default minimum notional for a trade to surface as a "big bet". The
# whale page is for genuinely large flow only — anything below $1000
# is retail noise on this dashboard's series. When Kalshi is quiet
# the table will be empty; that's the honest signal that no whales
# are active right now. Users can dial the floor down via the
# ?min=<dollars> query parameter (see render_page) if they want to
# inspect smaller flow.
LIVE_MIN_NOTIONAL_CENTS = 100_000        # $1000+ trades surface by default
LIVE_LOOKBACK_HOURS     = 24             # recent activity window


def validate_whale(event: dict,
                     cohort_winrates: Dict[Tuple[str, str, str], dict],
                     ) -> List[Tuple[str, bool, str]]:
    """Run every validator and return a list of (name, ok, detail).

    The dashboard surfaces the per-validator pass/fail status next to
    each candidate so users see exactly which gate a signal cleared
    and which it didn't — same idiom as the per-bet "criteria met"
    panel on the main dashboard.
    """
    out: List[Tuple[str, bool, str]] = []

    # 1. Z-score (size unusualness)
    z = float(event.get("zscore") or 0.0)
    out.append((
        "Z-score",
        z >= VALIDATOR_MIN_ZSCORE,
        f"{z:.2f} (≥ {VALIDATOR_MIN_ZSCORE:.1f} required)",
    ))

    # 2. Notional floor
    n_cents = int(event.get("whale_notional_cents") or 0)
    out.append((
        "Notional ≥ $50",
        n_cents >= VALIDATOR_MIN_NOTIONAL_CENTS,
        f"${n_cents/100:.2f}",
    ))

    # 3. Direction confidence
    dc = float(event.get("direction_confidence") or 0.0)
    out.append((
        "Direction confidence",
        dc >= VALIDATOR_MIN_DIR_CONF,
        f"{dc:.2f} (≥ {VALIDATOR_MIN_DIR_CONF:.2f})",
    ))

    # 4. Spread at entry
    sp = event.get("entry_spread_cents")
    if sp is None:
        out.append(("Spread ≤ 8¢", False, "no orderbook"))
    else:
        out.append((
            "Spread ≤ 8¢",
            int(sp) <= VALIDATOR_MAX_SPREAD_CENTS,
            f"{int(sp)}¢",
        ))

    # 5. Book depth at entry
    depth = int(event.get("entry_depth_within_3c") or 0)
    out.append((
        "Book depth ≥ 25",
        depth >= VALIDATOR_MIN_BOOK_DEPTH,
        f"{depth} contracts",
    ))

    # 6. Probability bounds
    mid = event.get("entry_mid_cents")
    if mid is None:
        out.append(("Probability in [15¢, 85¢]", False, "no mid"))
    else:
        direction = event.get("direction") or "?"
        implied = float(mid) if direction == "yes" else (100.0 - float(mid))
        ok = VALIDATOR_PROB_LO_CENTS <= implied <= VALIDATOR_PROB_HI_CENTS
        out.append((
            "Probability in [15¢, 85¢]",
            ok,
            f"{implied:.0f}¢",
        ))

    # 7. Cohort win rate (need ≥ N prior signals AND ≥ 50% win rate)
    ticker = event.get("ticker") or ""
    notional = event.get("whale_notional_cents") or 0
    direction = event.get("direction") or "?"
    key = (_ticker_prefix(ticker), _size_bucket(int(notional)), direction)
    row = cohort_winrates.get(key)
    if row is None:
        out.append(("Cohort track record", False,
                     "no matching cohort yet"))
    elif row["n"] < VALIDATOR_MIN_COHORT_N:
        out.append((
            "Cohort track record", False,
            f"only {row['n']} prior signals (need {VALIDATOR_MIN_COHORT_N})",
        ))
    else:
        ok = row["win_rate"] >= VALIDATOR_MIN_COHORT_WIN
        out.append((
            "Cohort track record",
            ok,
            f"{row['win_rate']*100:.0f}% win rate over {row['n']} prior signals",
        ))

    # 8. Cluster size — informed flow tends to come in waves, not as
    #    a single isolated bet. Lone outliers are more likely manipulation.
    cluster = int(event.get("cluster_size") or 0)
    out.append((
        "Cluster of large trades",
        cluster >= VALIDATOR_MIN_CLUSTER_SIZE,
        f"{cluster} large trades on this ticker in the lookback "
        f"(≥ {VALIDATOR_MIN_CLUSTER_SIZE} required)",
    ))

    # 9. Not a round-lot count — retail / wash trades cluster on
    #    100 / 250 / 500 / 1000-contract round numbers; insider
    #    sizing is dollar-budgeted and rarely lands exactly there.
    count = int(event.get("whale_count") or 0)
    is_round = count in VALIDATOR_ROUND_NUMBERS
    out.append((
        "Non-round-lot size",
        not is_round,
        f"{count} contracts" + (" (looks retail-round)" if is_round else ""),
    ))

    # 10. Price stability — if the post-trade mid has drifted far
    #     from the bet price, this trade pushed the market itself
    #     (manipulation tell) OR a news event hit alongside.
    bet_mid = event.get("entry_mid_cents")
    cur_mid = event.get("current_mid_cents")
    if bet_mid is None or cur_mid is None:
        out.append(("Price stable since trade", True, "no current mid"))
    else:
        drift = abs(int(cur_mid) - int(bet_mid))
        out.append((
            "Price stable since trade",
            drift <= VALIDATOR_MAX_PRICE_DRIFT,
            f"{drift}¢ drift from {int(bet_mid)}¢ → {int(cur_mid)}¢",
        ))

    return out


# --------------------------------------------------------------------------- #
# Live big-bet ingestion                                                      #
# --------------------------------------------------------------------------- #

def _market_question(market: Optional[dict],
                      event_title: Optional[str] = None) -> str:
    """Compact human-readable label for a Kalshi market.

    Tries each candidate string in order and returns the first one
    that *looks like a question* — i.e. it isn't the comma-dump that
    Kalshi's multi-option markets put in the title field
    ("yes Over 2.5 runs, yes Over 3.5 runs, …"). A real question
    has at most ~4 commas and at most one "yes "/"no " prefix.
    """
    if not market and not event_title:
        return "—"

    def _looks_like_question(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        # Comma-dump: too many commas suggests a list of sub-options.
        if s.count(",") > 4:
            return False
        # Multi "yes …" / "no …" prefixes also indicate an option dump.
        if s.lower().count("yes ") + s.lower().count("no ") > 2:
            return False
        return True

    market = market or {}
    candidates: List[str] = []
    # 1. Market title — the actual question Kalshi shows on the page.
    candidates.append(market.get("title") or "")
    # 2. Event title — passed in by caller when available; useful when
    #    the market title is a multi-option dump but the event has a
    #    sensible parent question.
    if event_title:
        candidates.append(event_title)
    # 3. First sentence of rules_primary. Split on ". " (period +
    #    space) so decimal numbers like "2.75" don't truncate.
    rules = (market.get("rules_primary") or "").strip()
    if rules:
        candidates.append(rules.split(". ", 1)[0].rstrip("."))
    # 4. Subtitle as a last resort — only useful when it's short.
    candidates.append(market.get("yes_sub_title") or
                       market.get("subtitle") or "")

    for c in candidates:
        c = (c or "").strip()
        if _looks_like_question(c):
            return c
    return "—"


def fetch_live_big_bets(series_tickers: List[str],
                          min_notional_cents: int = LIVE_MIN_NOTIONAL_CENTS,
                          lookback_hours: int = LIVE_LOOKBACK_HOURS,
                          *,
                          scan_all_markets: bool = False,
                          ) -> List[dict]:
    """Pull recent trades from Kalshi, filter to big bets, and return
    event-shaped dicts ready for the existing scorer + validator
    pipeline.

    Two modes:
      * Per-series (default): walk each series in ``series_tickers``,
        list its open markets, fetch trades for each market.
      * Global (``scan_all_markets=True``): pull the global trades
        feed once. Surfaces big bets across the whole exchange — not
        just the series this dashboard's bots care about. Looks up
        market metadata lazily for each ticker that produced a
        qualifying trade.

    No checkpoint data (these are live trades, not yet aged out), so
    `last_favorable_cents` will be None on every row. The "Outcome"
    column on the History tab simply won't include them until the
    bot's signal_tracker has a chance to capture +30m checkpoints.
    """
    from . import kalshi_client
    client = kalshi_client.default_client()
    if not client.available:
        return []

    seen_trade_ids: set = set()
    out: List[dict] = []
    # Inline parsers shared across both modes.
    def _trade_count(t):
        c = t.get("count_fp")
        if c is None:
            c = t.get("count")
        try:
            return int(round(float(c))) if c is not None else 0
        except (TypeError, ValueError):
            return 0

    def _trade_price_cents(t, side: str):
        key = f"{side}_price_dollars"
        v = t.get(key)
        if v is None:
            v = t.get(f"{side}_price")  # legacy field name
        if v is None:
            return 0
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0
        return int(round(f * 100)) if f < 5 else int(f)

    if scan_all_markets:
        # Global path — one trades call, then per-ticker market lookup
        # for the survivors. Pulls the most-recent N trades across all
        # markets; the limit caps how far back into history we see.
        try:
            trades = client.fetch_trades(
                market_ticker=None,
                lookback_hours=lookback_hours,
                limit=1000,
            )
        except Exception:  # noqa: BLE001
            log.exception("whale: global fetch_trades failed")
            return []

        # Pre-filter trades by notional so we only do per-ticker
        # market lookups on tickers that actually have big bets.
        big_trades: List[dict] = []
        for t in trades:
            count = _trade_count(t)
            yes_p = _trade_price_cents(t, "yes")
            no_p = _trade_price_cents(t, "no")
            price = yes_p if yes_p > 0 else no_p
            if count * price >= min_notional_cents:
                big_trades.append(t)

        # Group by ticker so we can compute per-ticker stats (cluster
        # size, current mid, z-score) over each ticker's trade batch.
        from collections import defaultdict
        by_ticker: Dict[str, List[dict]] = defaultdict(list)
        for t in big_trades:
            tk = t.get("ticker")
            if tk:
                by_ticker[tk].append(t)
        # We also want non-big trades on these same tickers for the
        # cluster + z-score calculations. Pull those out of the
        # original trades list.
        per_ticker_all: Dict[str, List[dict]] = defaultdict(list)
        for t in trades:
            tk = t.get("ticker")
            if tk in by_ticker:
                per_ticker_all[tk].append(t)

        for ticker, ticker_trades in per_ticker_all.items():
            # Look up market metadata once per ticker (cached).
            try:
                market = client.get_market(ticker)
            except Exception:  # noqa: BLE001
                log.warning("whale: get_market failed for %s", ticker)
                market = None
            close_time = (market or {}).get("close_time")
            mtc: Optional[float] = None
            if close_time:
                try:
                    ct = datetime.fromisoformat(
                        close_time.replace("Z", "+00:00")
                    ).timestamp()
                    mtc = max(0.0, (ct - time.time()) / 60.0)
                except (TypeError, ValueError):
                    mtc = None
            # Some Kalshi markets put a comma-dump of every option in
            # the `title` field; the parent event's title is then the
            # only place to find the real question. Fetch lazily —
            # only when the market itself doesn't yield a clean
            # question. get_event is cached so repeats are cheap.
            event_title: Optional[str] = None
            preview = _market_question(market)
            if preview == "—" and market and market.get("event_ticker"):
                try:
                    event = client.get_event(market["event_ticker"])
                    event_title = (event or {}).get("title") or None
                except Exception:  # noqa: BLE001
                    event_title = None
            question = _market_question(market, event_title=event_title)
            # Compute per-ticker stats off this ticker's trades.
            counts = [_trade_count(t) for t in ticker_trades]
            if len(counts) >= 5:
                mean_c = sum(counts) / len(counts)
                var = sum((c - mean_c) ** 2 for c in counts) / len(counts)
                std_c = math.sqrt(var) if var > 0 else 0.0
            else:
                mean_c = std_c = 0.0
            cluster_size = 0
            for t in ticker_trades:
                tc = _trade_count(t)
                tp = _trade_price_cents(t, "yes")
                if tp == 0:
                    tp = _trade_price_cents(t, "no")
                if tc * tp >= VALIDATOR_CLUSTER_MIN_NOTIONAL:
                    cluster_size += 1
            current_mid_cents: Optional[int] = None
            if ticker_trades:
                latest = ticker_trades[-1]
                lp = _trade_price_cents(latest, "yes")
                if lp == 0:
                    lp = 100 - _trade_price_cents(latest, "no")
                current_mid_cents = lp if lp > 0 else None

            # Build event dicts for the qualifying (big) trades only.
            for t in by_ticker[ticker]:
                _emit_event(
                    out, seen_trade_ids, t, ticker,
                    _trade_count, _trade_price_cents,
                    mean_c=mean_c, std_c=std_c,
                    cluster_size=cluster_size,
                    current_mid_cents=current_mid_cents,
                    mtc=mtc, question=question,
                    min_notional_cents=min_notional_cents,
                )
        return out

    # Per-series path (legacy).
    for series in series_tickers:
        try:
            markets = client.list_markets(series)
        except Exception:  # noqa: BLE001
            log.exception("whale: list_markets failed for %s", series)
            continue
        for m in markets:
            ticker = m.get("ticker")
            if not ticker:
                continue
            close_time = m.get("close_time")
            mtc = None
            if close_time:
                try:
                    ct = datetime.fromisoformat(
                        close_time.replace("Z", "+00:00")
                    ).timestamp()
                    mtc = max(0.0, (ct - time.time()) / 60.0)
                except (TypeError, ValueError):
                    mtc = None
            question = _market_question(m)
            try:
                trades = client.fetch_trades(
                    market_ticker=ticker,
                    lookback_hours=lookback_hours,
                    limit=200,
                )
            except Exception:  # noqa: BLE001
                log.exception("whale: fetch_trades failed for %s", ticker)
                continue
            # Per-ticker stats (z-score baseline, cluster size,
            # current mid). Same logic the global-scan path uses,
            # factored into _per_ticker_stats below.
            mean_c, std_c, cluster_size, current_mid_cents = (
                _per_ticker_stats(trades, _trade_count, _trade_price_cents)
            )
            for t in trades:
                _emit_event(
                    out, seen_trade_ids, t, ticker,
                    _trade_count, _trade_price_cents,
                    mean_c=mean_c, std_c=std_c,
                    cluster_size=cluster_size,
                    current_mid_cents=current_mid_cents,
                    mtc=mtc, question=question,
                    min_notional_cents=min_notional_cents,
                )
    return out


def _per_ticker_stats(trades: List[dict], _trade_count, _trade_price_cents,
                       ) -> Tuple[float, float, int, Optional[int]]:
    """Per-ticker z-score baseline, cluster size, current mid. Shared
    by the per-series and global-scan paths so the two trees compute
    identical signal context for each event.
    """
    counts = [_trade_count(t) for t in trades]
    if len(counts) >= 5:
        mean_c = sum(counts) / len(counts)
        var = sum((c - mean_c) ** 2 for c in counts) / len(counts)
        std_c = math.sqrt(var) if var > 0 else 0.0
    else:
        mean_c = std_c = 0.0
    cluster_size = 0
    for t in trades:
        tc = _trade_count(t)
        tp = _trade_price_cents(t, "yes")
        if tp == 0:
            tp = _trade_price_cents(t, "no")
        if tc * tp >= VALIDATOR_CLUSTER_MIN_NOTIONAL:
            cluster_size += 1
    current_mid_cents: Optional[int] = None
    if trades:
        latest = trades[-1]
        lp = _trade_price_cents(latest, "yes")
        if lp == 0:
            lp = 100 - _trade_price_cents(latest, "no")
        current_mid_cents = lp if lp > 0 else None
    return mean_c, std_c, cluster_size, current_mid_cents


def _emit_event(out: List[dict], seen_trade_ids: set, t: dict, ticker: str,
                  _trade_count, _trade_price_cents, *,
                  mean_c: float, std_c: float, cluster_size: int,
                  current_mid_cents: Optional[int],
                  mtc: Optional[float], question: str,
                  min_notional_cents: int) -> None:
    """Build one event dict from a raw Kalshi trade and append to ``out``
    if it clears the notional floor. Shared by both ingestion paths so
    the event shape stays uniform.
    """
    trade_id = t.get("trade_id") or (
        f"{ticker}:{t.get('created_time')}:{_trade_count(t)}"
    )
    if trade_id in seen_trade_ids:
        return
    seen_trade_ids.add(trade_id)
    count = _trade_count(t)
    yes_price = _trade_price_cents(t, "yes")
    no_price = _trade_price_cents(t, "no")
    price = yes_price if yes_price > 0 else no_price
    notional = count * price
    if notional < min_notional_cents:
        return
    taker_side = (t.get("taker_side") or "").lower()
    if taker_side in {"yes", "no"}:
        direction = taker_side
        dir_conf = 0.8
    else:
        direction = "yes" if yes_price >= no_price else "no"
        dir_conf = 0.4
    ts_str = t.get("created_time") or ""
    try:
        signal_ts = datetime.fromisoformat(
            ts_str.replace("Z", "+00:00")
        ).timestamp()
    except (TypeError, ValueError):
        signal_ts = time.time()
    mid = yes_price if yes_price > 0 else (100 - no_price)
    zscore = (count - mean_c) / std_c if std_c > 0 else 0.0
    out.append({
        "ticker": ticker,
        "question": question,
        "signal_ts": signal_ts,
        "zscore": zscore,
        "minutes_to_close": mtc,
        "cluster_size": cluster_size,
        "current_mid_cents": current_mid_cents,
        "direction": direction,
        "direction_confidence": dir_conf,
        "whale_count": count,
        "whale_notional_cents": notional,
        "taker_side": taker_side,
        "entry_mid_cents": mid,
        "entry_spread_cents": None,
        "entry_depth_within_3c": 0,
        "entered": False,
        "rejection_reason": None,
        "checkpoints": [],
        "_source": "live",
    })


# --------------------------------------------------------------------------- #
# Candidate construction                                                      #
# --------------------------------------------------------------------------- #

def compute_action(insider_score_v: float,
                     all_pass: bool,
                     n_failed: int,
                     ) -> Tuple[str, str]:
    """Recommend what the bot should *do* with a candidate signal.

    Three states, mirroring the regular bots' verdict idiom:
      • BUY    — every validator passed AND insider P clears the
                 threshold. The bot would simulate an entry on this.
      • WATCH  — high insider P (≥ 0.5) but at least one validator
                 failed. Worth keeping an eye on; don't enter yet.
      • SKIP   — low insider P or many validator failures. Noise.

    Returns (verdict, reason) where reason is a short explanation
    suitable for a tooltip / row label.
    """
    if all_pass and insider_score_v >= VALIDATOR_INSIDER_THRESHOLD:
        return ("BUY", f"All gates passed; insider P "
                       f"{insider_score_v*100:.0f}% ≥ "
                       f"{VALIDATOR_INSIDER_THRESHOLD*100:.0f}%")
    if insider_score_v >= 0.50:
        if n_failed == 1:
            return ("WATCH", "1 validator failed — close to actionable")
        return ("WATCH",
                f"Promising (P {insider_score_v*100:.0f}%) but "
                f"{n_failed} validators failed")
    return ("SKIP",
            f"Insider P {insider_score_v*100:.0f}% — below threshold")


def compute_candidates(events: List[dict],
                         cohort_winrates: Dict[Tuple[str, str, str], dict],
                         ) -> List[dict]:
    """Per-signal feature record for the Watchlist tab + simulated-buy
    detection. Returns one dict per event with:
      ticker, signal_ts, direction, notional_cents, zscore,
      entry_mid_cents, entry_spread_cents, entry_depth_within_3c,
      cohort_winrate (None if unknown), insider_score,
      validators (list of (name, ok, detail) tuples),
      all_pass (bool — every validator passed),
      simulated_buy (bool — all_pass AND insider_score ≥ threshold),
      last_favorable_cents (None if not yet captured).
    """
    out: List[dict] = []
    for e in events:
        validators = validate_whale(e, cohort_winrates)
        all_pass = all(ok for _name, ok, _detail in validators)
        n_failed = sum(1 for _name, ok, _detail in validators if not ok)
        score = insider_score(e, cohort_winrates)
        sim_buy = all_pass and score >= VALIDATOR_INSIDER_THRESHOLD
        action, action_reason = compute_action(score, all_pass, n_failed)
        last_fav = _last_favorable(e)
        coh = cohort_winrate_for(e, cohort_winrates)
        # Bet vs current probability for the side this trader took.
        # entry_mid_cents is the YES side mid; flip for NO bets.
        direction = (e.get("direction") or "?").lower()
        bet_mid = e.get("entry_mid_cents")
        cur_mid = e.get("current_mid_cents")
        bet_prob = (None if bet_mid is None
                     else (float(bet_mid) if direction == "yes"
                            else 100.0 - float(bet_mid)))
        cur_prob = (None if cur_mid is None
                     else (float(cur_mid) if direction == "yes"
                            else 100.0 - float(cur_mid)))
        out.append({
            "ticker": e.get("ticker") or "",
            "question": e.get("question") or "",
            "signal_ts": e.get("signal_ts"),
            "direction": e.get("direction") or "?",
            "bet_prob_pct": bet_prob,
            "current_prob_pct": cur_prob,
            "notional_cents": int(e.get("whale_notional_cents") or 0),
            "zscore": (None if e.get("zscore") is None
                        else float(e.get("zscore") or 0.0)),
            "whale_count": int(e.get("whale_count") or 0),
            "entry_mid_cents": e.get("entry_mid_cents"),
            "entry_spread_cents": e.get("entry_spread_cents"),
            "entry_depth_within_3c": int(e.get("entry_depth_within_3c") or 0),
            "direction_confidence": float(
                e.get("direction_confidence") or 0.0),
            "cohort_winrate": coh,
            "insider_score": score,
            "validators": validators,
            "all_pass": all_pass,
            "simulated_buy": sim_buy,
            "action": action,
            "action_reason": action_reason,
            "last_favorable_cents": last_fav,
            "rejection_reason": e.get("rejection_reason"),
            "entered": bool(e.get("entered")),
            "source": e.get("_source", "tracked"),
        })
    return out


# --------------------------------------------------------------------------- #
# Top-line summary stats                                                      #
# --------------------------------------------------------------------------- #

def summarize(events: List[dict],
              candidates: Optional[List[dict]] = None,
              ) -> dict:
    """Top-line aggregate stats for the Home tab cards."""
    n = len(events)
    if n == 0:
        return {
            "n_signals": 0, "n_simulated_buys": 0,
            "n_passed_all_validators": 0,
            "win_rate_30m": None, "mean_fav_30m": None,
            "first_ts": None, "last_ts": None, "verdict": "no signals yet",
        }
    favs: List[float] = [
        f for f in (_last_favorable(e) for e in events) if f is not None
    ]
    n_with_fav = len(favs)
    mean_fav = (sum(favs) / n_with_fav) if favs else None
    n_pos = sum(1 for f in favs if f > 0)
    win_rate = (n_pos / n_with_fav) if n_with_fav else None
    timestamps = [e.get("signal_ts") for e in events
                  if e.get("signal_ts") is not None]
    first_ts = min(timestamps) if timestamps else None
    last_ts = max(timestamps) if timestamps else None

    n_passed = 0
    n_sim_buys = 0
    if candidates:
        n_passed = sum(1 for c in candidates if c["all_pass"])
        n_sim_buys = sum(1 for c in candidates if c["simulated_buy"])

    if n < 50:
        verdict = "too few signals to judge"
    elif mean_fav is None:
        verdict = "no checkpoint data"
    elif mean_fav > 1.5 and win_rate and win_rate > 0.55:
        verdict = "looks like an edge"
    elif mean_fav < -1.0 or (win_rate and win_rate < 0.45):
        verdict = "fading edge — review"
    else:
        verdict = "noise — keep collecting"

    return {
        "n_signals": n,
        "n_simulated_buys": n_sim_buys,
        "n_passed_all_validators": n_passed,
        "win_rate_30m": win_rate,
        "mean_fav_30m": mean_fav,
        "first_ts": first_ts, "last_ts": last_ts, "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Formatters                                                                  #
# --------------------------------------------------------------------------- #

def _fmt_dollars(cents: int | float | None) -> str:
    if cents is None:
        return "—"
    return f"${cents/100:.2f}"


def _fmt_signed_cents(cents: float | None) -> str:
    if cents is None:
        return "—"
    sign = "+" if cents >= 0 else "−"
    return f"{sign}{abs(cents):.1f}¢"


def _fmt_ts(ts: float | str | None) -> str:
    if ts is None:
        return "—"
    if isinstance(ts, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%SZ")
        except (TypeError, ValueError, OSError):
            return "—"
    return str(ts)[:19].replace("T", " ")


def _fmt_pct(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p*100:.0f}%"


def _fmt_age(signal_ts: float | None) -> str:
    """Compact age string — '3m', '47m', '4.2h', '2.1d'."""
    if signal_ts is None:
        return "—"
    try:
        secs = max(0, time.time() - float(signal_ts))
    except (TypeError, ValueError):
        return "—"
    if secs < 60:
        return f"{int(secs)}s"
    if secs < 3600:
        return f"{int(secs/60)}m"
    if secs < 86400:
        return f"{secs/3600:.1f}h"
    return f"{secs/86400:.1f}d"


# --------------------------------------------------------------------------- #
# Render — tab-based                                                          #
# --------------------------------------------------------------------------- #

WHALE_TABS = [
    ("home", "Home"),
    ("watchlist", "Watchlist"),
    ("history", "History"),
]


def render_page(
    *,
    events: List[dict],
    orders: List[dict],
    available_bots: List[dict],
    current_bot_key: str,
    sort_by: str = "recent",
    tab_key: str = "watchlist",  # ignored, kept for backwards-compat
    min_notional_dollars: int | None = None,
) -> str:
    """Whole HTML page for the whale-watcher view.

    Layout matches the main dashboard's Watchlist tab so the user
    has one navigation idiom across all bots:

        [Home] [Watchlist] [History]   ← top-level dashboard tabs
        Bot: [Whale Watcher ▾]         ← bot dropdown
        [stats cards row]              ← summary stats in boxes
        [insider candidates table]     ← the main content

    The top-level tabs link to /?tab=home and /?tab=history (the
    main dashboard pages) — clicking Watchlist stays here. So
    selecting whale-watcher in the bot dropdown swaps the watchlist
    body for the insider-candidates view; selecting any other bot
    swaps to the strike-ladder + chart view.
    """
    # Imported lazily to avoid a circular import at module load time.
    from .dashboard import CSS, _favicon_link, _render_bot_filter

    cohorts = compute_cohort_winrates(events)

    # Pull live big bets across the configured bots' series so the
    # whale page surfaces real Kalshi activity even when the
    # whale-watcher bot's signal_tracking.jsonl is sparse / empty.
    # ?min=<dollars> from the URL overrides the default $1000 floor.
    if min_notional_dollars is not None and min_notional_dollars >= 0:
        effective_min_cents = min_notional_dollars * 100
        floor_was_explicit = True
    else:
        effective_min_cents = LIVE_MIN_NOTIONAL_CENTS
        floor_was_explicit = False
    # Whale watcher scans the WHOLE Kalshi exchange for big bets, not
    # just the series this dashboard's bots track. Reads global trades
    # once, then looks up market metadata only for tickers that
    # produced a qualifying trade.
    live_events = fetch_live_big_bets(
        [],  # ignored in scan_all_markets mode
        min_notional_cents=effective_min_cents,
        scan_all_markets=True,
    )
    # Auto-bump: if too many bets cleared the default floor, widen
    # the bar so the user sees only the truly notable ones. Skipped
    # when the user explicitly set ?min= — they asked for that bar.
    AUTO_BUMP_THRESHOLD = 30
    AUTO_BUMP_FLOOR_CENTS = 500_000  # $5000
    auto_bumped = False
    if (not floor_was_explicit
            and effective_min_cents < AUTO_BUMP_FLOOR_CENTS
            and len(live_events) > AUTO_BUMP_THRESHOLD):
        live_events = fetch_live_big_bets(
            [], min_notional_cents=AUTO_BUMP_FLOOR_CENTS,
            scan_all_markets=True,
        )
        effective_min_cents = AUTO_BUMP_FLOOR_CENTS
        auto_bumped = True

    # Combine: tracked signals (from JSONL, with checkpoints) + live
    # big bets (from Kalshi API, no checkpoints yet). The scorer
    # handles both shapes — live trades just lack a z-score so the
    # corresponding insider feature defaults to 0 for them.
    #
    # Enforce the same floor on the JSONL side too — the bot's own
    # whale_detector ingests at $50, but the dashboard's "whale"
    # bar is $1000+ so smaller bets shouldn't leak into the cards
    # or the unusual-whales table just because the bot tracked them.
    def _clears_floor(e: dict) -> bool:
        try:
            return int(e.get("whale_notional_cents") or 0) >= effective_min_cents
        except (TypeError, ValueError):
            return False
    filtered_events = [e for e in events if _clears_floor(e)]
    combined_events = filtered_events + live_events
    candidates = compute_candidates(combined_events, cohorts)
    summary = summarize(combined_events, candidates)

    out: List[str] = []
    out.append("<!doctype html><html><head>")
    out.append("<meta charset='utf-8'>")
    out.append("<meta http-equiv='refresh' content='30'>")
    out.append("<title>Whale Watcher — Kalshi simulation dashboard</title>")
    out.append(_favicon_link())
    out.append(f"<style>{CSS}</style>")
    out.append("<style>"
               ".side-yes { color:#3fb950; font-weight:600; }"
               ".side-no  { color:#f85149; font-weight:600; }"
               ".pos { color:#3fb950; }"
               ".neg { color:#f85149; }"
               ".whale-score { display:inline-block; padding:2px 8px; "
                  "border-radius:10px; font-size:11px; font-weight:600; "
                  "font-variant-numeric: tabular-nums; }"
               ".whale-score.high { background: rgba(63,185,80,0.18); "
                  "color:#3fb950; border:1px solid rgba(63,185,80,0.35); }"
               ".whale-score.med  { background: rgba(212,153,0,0.18); "
                  "color:#d49900; border:1px solid rgba(212,153,0,0.35); }"
               ".whale-score.low  { background: rgba(139,148,158,0.15); "
                  "color:#8b949e; border:1px solid rgba(139,148,158,0.30); }"
               ".valid-pill { display:inline-block; padding:1px 6px; "
                  "border-radius:4px; font-size:10px; font-weight:600; "
                  "text-transform:uppercase; letter-spacing:0.04em; "
                  "margin-right:4px; line-height:1.5; }"
               ".valid-pill.pass { background:rgba(63,185,80,0.15); "
                  "color:#3fb950; border:1px solid rgba(63,185,80,0.30); }"
               ".valid-pill.fail { background:rgba(248,81,73,0.15); "
                  "color:#f85149; border:1px solid rgba(248,81,73,0.30); }"
               # Question cell — clip long titles so the column
               # doesn't push the rest of the table off-screen. Full
               # text stays accessible via the tooltip on hover.
               ".whale-question { max-width: 320px; overflow: hidden; "
                  "text-overflow: ellipsis; white-space: nowrap; "
                  "color: #c9d1d9; }"
               "</style>")
    out.append("</head><body>")
    out.append("<h1>Kalshi simulation dashboard</h1>")
    out.append("<div class='meta'>"
               f"Loaded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
               " · live updates every 30s · DRY-RUN mode (no real orders)</div>")

    # ── Top-level dashboard tabs (above the bot filter, link to the
    # main dashboard's pages so the navigation is one idiom across
    # the whole app). Watchlist is the active tab — that's what this
    # whale view is, a Watchlist variant.
    main_tabs = [
        ("home",      "Home",      "?tab=home"),
        ("watchlist", "Watchlist", f"?tab=watchlist&bot={html.escape(current_bot_key)}"),
        ("history",   "History",   "?tab=history"),
    ]
    out.append("<div class='tab-bar'>")
    for key, label, href in main_tabs:
        cls = "tab-pill" + (" tab-pill-active" if key == "watchlist" else "")
        out.append(f"<a class='{cls}' href='{href}'>{html.escape(label)}</a>")
    out.append("</div>")

    # Bot filter sits BELOW the tabs (inside the section body, like the
    # main dashboard's Watchlist tab does it).
    out.append("<div class='section'><h2>"
               "Watchlist — model vs market</h2>"
               "<div class='body'>")
    _render_bot_filter(out, available_bots, current_bot_key)

    if summary["n_signals"] == 0:
        _render_empty_state(
            out,
            min_dollars=effective_min_cents // 100,
            lookback_hours=LIVE_LOOKBACK_HOURS,
        )
        out.append("</div></div>")
        out.append(_BOT_SELECT_NAVIGATE_JS)
        out.append("</body></html>")
        return "".join(out)

    # Stats cards row — mirrors the prediction-cards rhythm on the
    # regular Watchlist tab. Same `<div class='row compact'>` shape.
    _render_summary_cards(out, summary)

    # Simulated buys ("current bets" equivalent — sits where Active
    # bet sits on the regular Watchlist tab).
    _render_simulated_buys(out, candidates)

    # Main table — "huge bets that could be insiders". This is the
    # whale-watcher analog of the strike-ladder ticker table on the
    # regular Watchlist tab. Each row is one suspicious whale trade
    # ranked by insider probability.
    _render_unusual_whales(out, candidates)

    # Past whales with realised outcomes — same role as Kalshi rules
    # at the bottom of the regular Watchlist tab (reference content
    # that lives below the actionable view).
    _render_signal_history(out, candidates, cohorts)

    out.append("</div></div>")  # /body /section

    # Bot dropdown onchange handler. The main dashboard wires this up
    # inside _live_update_script, but the whale page doesn't load
    # that script (no live-poll cells to patch). Inline the same
    # snippet so switching the bot from this page navigates to the
    # chosen URL — `<option value=...>` carries the destination.
    out.append(_BOT_SELECT_NAVIGATE_JS)
    out.append("</body></html>")
    return "".join(out)


# Module-level constant so the empty-state and full-render paths
# share one source of truth.
_BOT_SELECT_NAVIGATE_JS = """<script>
(function () {
  const sel = document.getElementById("bot-select");
  if (!sel) return;
  sel.addEventListener("change", function () {
    if (sel.value) window.location.href = sel.value;
  });
})();
</script>"""


# --------------------------------------------------------------------------- #
# Per-tab render helpers                                                      #
# --------------------------------------------------------------------------- #

def _render_summary_cards(out: List[str], summary: dict) -> None:
    """Stats cards row — same `<div class='row compact'>` shape as the
    main dashboard's Watchlist prediction-cards so they line up
    visually across bots.
    """
    win = summary.get("win_rate_30m")
    win_cls = ("green" if win is not None and win > 0.55
               else ("red" if win is not None and win < 0.45 else "gray"))
    win_str = _fmt_pct(win) if win is not None else "—"
    out.append("<div class='row compact'>")
    out.append(f"<div class='card'><div class='label'>Whale signals</div>"
               f"<div class='value'>{summary['n_signals']}</div></div>")
    out.append(f"<div class='card'><div class='label'>Passed validators</div>"
               f"<div class='value'>{summary['n_passed_all_validators']}</div></div>")
    out.append(f"<div class='card'><div class='label'>Simulated buys</div>"
               f"<div class='value'>{summary['n_simulated_buys']}</div></div>")
    out.append(f"<div class='card'><div class='label'>+30m win rate</div>"
               f"<div class='value {win_cls}'>{win_str}</div></div>")
    out.append("</div>")


def _render_simulated_buys(out: List[str], candidates: List[dict]) -> None:
    """Current-bets table — only signals that passed every validator
    AND have insider_score ≥ threshold. Sort by recency, highest
    insider score first.
    """
    sims = [c for c in candidates if c["simulated_buy"]]
    sims.sort(key=lambda c: (c.get("signal_ts") or 0, c["insider_score"]),
                reverse=True)

    out.append("<h3 class='subhead'>Current bets — simulated</h3>")
    if not sims:
        out.append("<div class='empty'>No signals have cleared every "
                   "validator yet. The Watchlist tab shows everything "
                   "that triggered.</div>")
        return
    out.append("<div class='small gray' style='margin-bottom:8px;'>"
               "Whale signals that passed every gate AND scored "
               f"≥ {VALIDATOR_INSIDER_THRESHOLD:.2f} on insider probability. "
               "These are the trades the bot would have placed.</div>")
    out.append("<table><thead><tr>"
               "<th>Signal time</th><th>Ticker</th><th>Side</th>"
               "<th class='num'>Whale size</th><th class='num'>Z-score</th>"
               "<th class='num'>Entry mid</th><th class='num'>Insider P</th>"
               "<th class='num'>+30m</th>"
               "</tr></thead><tbody>")
    for c in sims[:50]:
        side_cls = "side-yes" if c["direction"] == "yes" else "side-no"
        score_cls = _score_class(c["insider_score"])
        fav = c.get("last_favorable_cents")
        fav_cls = ("pos" if fav is not None and fav > 0
                    else ("neg" if fav is not None and fav < 0 else "gray"))
        out.append(
            f"<tr><td>{html.escape(_fmt_ts(c['signal_ts']))}</td>"
            f"<td class='mono'>{html.escape(c['ticker'])}</td>"
            f"<td><span class='{side_cls}'>{c['direction'].upper()}</span></td>"
            f"<td class='num'>{_fmt_dollars(c['notional_cents'])}</td>"
            f"<td class='num'>{c['zscore']:.2f}</td>"
            f"<td class='num'>"
            f"{c['entry_mid_cents']:.0f}¢"
            if c.get('entry_mid_cents') is not None else "<td class='num'>—"
            f"</td>"
            f"<td class='num'><span class='whale-score {score_cls}'>"
            f"{c['insider_score']*100:.0f}%</span></td>"
            f"<td class='num {fav_cls}'>{_fmt_signed_cents(fav)}</td>"
            f"</tr>"
        )
    out.append("</tbody></table>")


def _render_unusual_whales(out: List[str], candidates: List[dict]) -> None:
    """The main table on the whale-watcher Watchlist tab — every
    candidate signal with score + validator pass/fail. Sort by
    insider score (highest first). Replaces the strike-ladder table
    that the regular bots show on this tab.
    """
    out.append("<h3 class='subhead'>"
               "Huge bets that could be insiders</h3>")
    if not candidates:
        out.append("<div class='empty'>No signals captured yet.</div>")
        return
    out.append("<div class='small gray' style='margin-bottom:8px;'>"
               "Every flagged whale trade, ranked by insider probability. "
               "Insider P ≥ "
               f"{VALIDATOR_INSIDER_THRESHOLD*100:.0f}% AND every validator "
               "green = the bot simulates a buy (shown in Current bets above).</div>")
    pool = sorted(candidates, key=lambda c: c["insider_score"], reverse=True)
    out.append("<table><thead><tr>"
               "<th>Time</th><th>Ticker</th>"
               "<th title='Human-readable market question — what "
               "the bet pays out on. Pulled from Kalshi.'>Question</th>"
               "<th class='num'>Bet size</th><th>Side</th>"
               "<th class='num' title='Implied probability of the bet "
               "side at the moment the trade hit.'>Bet prob</th>"
               "<th class='num' title='Current implied probability for "
               "the same side — compare to Bet prob to see how the "
               "market has moved since.'>Current prob</th>"
               "<th class='num'>Z-score</th>"
               "<th class='num'>Cohort win %</th>"
               "<th class='num'>Insider P</th>"
               "<th title='Recommended action: BUY (simulate entry), "
               "WATCH (promising but not actionable), or SKIP. "
               "Hover for the reason — includes hidden manipulation-"
               "pattern checks (cluster size, round-lot test, price "
               "stability).'>Verdict</th>"
               "</tr></thead><tbody>")
    for c in pool[:200]:
        # Side rendered as the same badge-yes/badge-no idiom the
        # regular dashboard's active-bets / watchlist tables use, so
        # the visual idiom is consistent across every page.
        side = (c["direction"] or "?").upper()
        side_badge_cls = "badge-yes" if c["direction"] == "yes" else "badge-no"
        score_cls = _score_class(c["insider_score"])
        coh = c["cohort_winrate"]
        coh_str = _fmt_pct(coh) if coh is not None else "—"
        action = c["action"]
        action_cls = {
            "BUY":   "badge-yes",
            "WATCH": "badge-hedge",
            "SKIP":  "badge-skip",
        }.get(action, "badge-skip")
        # Verdict tooltip lists every failed validator inline so the
        # user can hover to see why something was downgraded — the
        # manipulation-pattern checks (cluster size, round-lot,
        # price stability) live here even though they don't have
        # their own visible columns.
        failed = [f"{n}: {d}" for n, ok, d in c["validators"] if not ok]
        verdict_tt = c.get("action_reason", "")
        if failed:
            verdict_tt += " · Failed checks — " + "; ".join(failed)
        action_badge = (
            f"<span class='badge {action_cls}' "
            f"title='{html.escape(verdict_tt)}'>"
            f"{action}</span>"
        )
        z_str = (f"{c['zscore']:.2f}" if c.get('zscore') is not None
                  else "—")
        # Ticker → clickable link to the Kalshi market page (same
        # convention as the regular watchlist table). Series prefix is
        # everything before the first "-" lowercased.
        tt = c["ticker"]
        series_lower = (tt.split("-", 1)[0] if tt else "").lower()
        ticker_url = (f"https://kalshi.com/markets/{series_lower}"
                      if series_lower else "")
        ticker_cell = (
            f"<a href='{html.escape(ticker_url)}' target='_blank' "
            f"rel='noopener noreferrer' class='ticker-link'>"
            f"{html.escape(tt)}</a>"
            if ticker_url else html.escape(tt)
        )
        bet_prob_str = (f"{c['bet_prob_pct']:.0f}%"
                         if c.get("bet_prob_pct") is not None else "—")
        cur_prob_str = (f"{c['current_prob_pct']:.0f}%"
                         if c.get("current_prob_pct") is not None else "—")
        # Color the current-prob cell green when it has moved in our
        # favor (current > bet), red when against. Same convention as
        # the active-bets table on the regular dashboard.
        if (c.get("bet_prob_pct") is not None
                and c.get("current_prob_pct") is not None):
            delta = c["current_prob_pct"] - c["bet_prob_pct"]
            cur_cls = ("green" if delta > 0.5
                        else ("red" if delta < -0.5 else ""))
        else:
            cur_cls = ""
        # Question cell — the market's human-readable label. Long
        # values are clipped via CSS but the full text stays in a
        # title tooltip so the user can hover for the rest.
        question = (c.get("question") or "—").strip() or "—"
        question_cell = (
            f"<td class='whale-question' "
            f"title='{html.escape(question)}'>"
            f"{html.escape(question)}</td>"
        )
        out.append(
            f"<tr><td>{html.escape(_fmt_ts(c['signal_ts']))}</td>"
            f"<td class='mono'>{ticker_cell}</td>"
            f"{question_cell}"
            f"<td class='num'>{_fmt_dollars(c['notional_cents'])}</td>"
            f"<td><span class='badge {side_badge_cls}'>{side}</span></td>"
            f"<td class='num'>{bet_prob_str}</td>"
            f"<td class='num {cur_cls}'>{cur_prob_str}</td>"
            f"<td class='num'>{z_str}</td>"
            f"<td class='num'>{coh_str}</td>"
            f"<td class='num'><span class='whale-score {score_cls}'>"
            f"{c['insider_score']*100:.0f}%</span></td>"
            f"<td>{action_badge}</td>"
            f"</tr>"
        )
    out.append("</tbody></table>")


def _render_signal_history(out: List[str], candidates: List[dict],
                             cohorts: Dict[Tuple[str, str, str], dict],
                             ) -> None:
    """History tab — completed signals (those with a captured +30m
    favorable_cents) ordered by time, plus a cohort breakdown table.
    """
    completed = [c for c in candidates
                 if c["last_favorable_cents"] is not None]
    completed.sort(key=lambda c: c.get("signal_ts") or 0, reverse=True)

    out.append("<h3 class='subhead'>Completed whale signals (with +30m outcome)</h3>")
    if not completed:
        out.append("<div class='empty'>No signals have completed their "
                   "+30m checkpoint yet.</div>")
    else:
        out.append("<table><thead><tr>"
                   "<th>Signal time</th><th>Ticker</th><th>Side</th>"
                   "<th class='num'>Bet size</th>"
                   "<th class='num'>Z-score</th>"
                   "<th class='num'>Insider P</th>"
                   "<th class='num'>+30m favorable</th>"
                   "<th>Outcome</th>"
                   "</tr></thead><tbody>")
        for c in completed[:200]:
            side_cls = "side-yes" if c["direction"] == "yes" else "side-no"
            score_cls = _score_class(c["insider_score"])
            fav = c["last_favorable_cents"]
            fav_cls = ("pos" if fav is not None and fav > 0
                        else ("neg" if fav is not None and fav < 0 else "gray"))
            outcome = ("WIN" if fav is not None and fav > 0
                        else ("LOSS" if fav is not None and fav < 0
                              else "FLAT"))
            out.append(
                f"<tr><td>{html.escape(_fmt_ts(c['signal_ts']))}</td>"
                f"<td class='mono'>{html.escape(c['ticker'])}</td>"
                f"<td><span class='{side_cls}'>{c['direction'].upper()}</span></td>"
                f"<td class='num'>{_fmt_dollars(c['notional_cents'])}</td>"
                f"<td class='num'>{c['zscore']:.2f}</td>"
                f"<td class='num'><span class='whale-score {score_cls}'>"
                f"{c['insider_score']*100:.0f}%</span></td>"
                f"<td class='num {fav_cls}'>{_fmt_signed_cents(fav)}</td>"
                f"<td class='{fav_cls}'>{outcome}</td>"
                f"</tr>"
            )
        out.append("</tbody></table>")

    # Cohort breakdown
    rows = sorted(cohorts.items(),
                    key=lambda kv: kv[1]["win_rate"], reverse=True)
    if rows:
        out.append("<h3 class='subhead' style='margin-top:24px;'>"
                   "Cohort track records</h3>")
        out.append("<div class='small gray' style='margin-bottom:8px;'>"
                   "Pseudo-identity buckets (ticker family × bet size × "
                   "side). Cohort win rate is the score input for the "
                   "Insider P column above.</div>")
        out.append("<table><thead><tr>"
                   "<th>Cohort</th>"
                   "<th class='num'>n signals</th>"
                   "<th class='num'>Win rate</th>"
                   "<th class='num'>Mean +30m</th>"
                   "</tr></thead><tbody>")
        for (prefix, size, direction), row in rows[:50]:
            wr = row["win_rate"]
            wr_cls = ("pos" if wr > 0.55 else ("neg" if wr < 0.45 else "gray"))
            mf = row["mean_fav_30m"]
            mf_cls = ("pos" if mf > 0 else ("neg" if mf < 0 else "gray"))
            out.append(
                f"<tr><td><span class='mono'>{html.escape(prefix)}</span> · "
                f"{html.escape(size)} · "
                f"<span class='side-{direction}'>{direction.upper()}</span></td>"
                f"<td class='num'>{row['n']}</td>"
                f"<td class='num {wr_cls}'>{wr*100:.0f}%</td>"
                f"<td class='num {mf_cls}'>{_fmt_signed_cents(mf)}</td>"
                f"</tr>"
            )
        out.append("</tbody></table>")


def _render_empty_state(out: List[str], min_dollars: int = 0,
                          lookback_hours: int = 0) -> None:
    """Empty-state copy for the whale page.

    Tells the user *why* the table is empty — usually because no
    trades cleared the dollar floor in the lookback window. Without
    this hint the page reads "broken" when in fact Kalshi flow is
    just quiet.
    """
    if min_dollars > 0 and lookback_hours > 0:
        msg = (
            f"No bets ≥ ${min_dollars:,} in the last "
            f"{lookback_hours}h. Try "
            f"<a href='?bot=whale-watcher&min=10'>$10</a> · "
            f"<a href='?bot=whale-watcher&min=50'>$50</a> · "
            f"<a href='?bot=whale-watcher&min=100'>$100</a> · "
            f"<a href='?bot=whale-watcher&min=1000'>$1000</a>."
        )
    else:
        msg = "No whale signals captured yet."
    out.append(f"<div class='empty'>{msg}</div>")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _score_class(score: float) -> str:
    if score >= VALIDATOR_INSIDER_THRESHOLD:
        return "high"
    if score >= 0.5:
        return "med"
    return "low"


def _validator_tooltip(name: str, ok: bool, detail: str) -> str:
    return f"{name}: {'PASS' if ok else 'FAIL'} — {detail}"


# --------------------------------------------------------------------------- #
# Backwards-compat surface — older imports                                    #
# --------------------------------------------------------------------------- #
#
# The legacy single-page render imported `compute_cohorts` etc. Keep
# thin shims that delegate to the new helpers so any external caller
# (tests, scripts) continues to work.

def compute_cohorts(events: List[dict], min_n: int = 3) -> List[dict]:
    """Legacy-shape cohort list (unchanged shape from prior callers).
    Computed off the same buckets as compute_cohort_winrates but with
    extra fields the old UI consumed.
    """
    buckets: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for e in events:
        ticker = e.get("ticker") or ""
        notional = e.get("whale_notional_cents") or 0
        direction = e.get("direction") or "?"
        key = (_ticker_prefix(ticker), _size_bucket(int(notional)), direction)
        buckets[key].append(e)
    out: List[dict] = []
    for (prefix, size, direction), members in buckets.items():
        if len(members) < min_n:
            continue
        favs = [f for f in (_last_favorable(e) for e in members) if f is not None]
        if not favs:
            continue
        mean_fav = sum(favs) / len(favs)
        wins = sum(1 for f in favs if f > 0)
        win_rate = wins / len(favs)
        sizes = sorted(int(e.get("whale_notional_cents") or 0) / 100.0
                       for e in members)
        median = sizes[len(sizes) // 2] if sizes else 0.0
        out.append({
            "key": f"{prefix} · {size} · {direction.upper()}",
            "ticker_prefix": prefix,
            "size_bucket": size,
            "direction": direction,
            "n": len(members),
            "n_entered": sum(1 for e in members if e.get("entered")),
            "mean_fav_30m": mean_fav,
            "win_rate": win_rate,
            "median_size_dollars": median,
        })
    out.sort(key=lambda r: r["mean_fav_30m"], reverse=True)
    return out


def standouts(events: List[dict], sort_by: str = "recent",
              limit: int = 50) -> List[dict]:
    """Legacy sorter — kept for any test that still imports it."""
    pool = list(events)
    if sort_by == "size":
        pool.sort(key=lambda e: e.get("whale_notional_cents") or 0, reverse=True)
    elif sort_by == "zscore":
        pool.sort(key=lambda e: e.get("zscore") or 0, reverse=True)
    elif sort_by == "favorable":
        pool.sort(key=lambda e: _last_favorable(e) if _last_favorable(e) is not None else -1e9,
                  reverse=True)
    else:  # recent / unknown
        pool.sort(key=lambda e: e.get("signal_ts") or 0, reverse=True)
    return pool[:limit]


def rejection_histogram(events: List[dict]) -> List[Tuple[str, int]]:
    """Legacy — count rejection reasons across all events."""
    cnt: Counter = Counter()
    for e in events:
        if e.get("entered"):
            continue
        reason = (e.get("rejection_reason") or "unknown").strip()
        cnt[reason] += 1
    return cnt.most_common()
