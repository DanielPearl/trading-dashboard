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

_W_Z       = 0.25
_W_SIZE    = 0.20
_W_DIRCONF = 0.15
_W_COHORT  = 0.30
_W_TIGHT   = 0.10


def insider_score(event: dict,
                    cohort_winrates: Dict[Tuple[str, str, str], dict]
                    ) -> float:
    """Return [0, 1] probability that this whale signal looks like
    "someone who knows something". Transparent linear combination of
    the features above — easy to override / tune.
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

    coh = cohort_winrate_for(event, cohort_winrates)
    cohort = 0.5 if coh is None else float(coh)

    spread = event.get("entry_spread_cents")
    depth = event.get("entry_depth_within_3c") or 0
    if spread is None:
        tight = 0.5
    else:
        # Tight = 0¢ spread; loose = 8¢+. Linear in between.
        sp = max(0, min(8, int(spread)))
        s_norm = 1.0 - (sp / 8.0)
        # Depth: 0..200+ contracts within 3¢ of best.
        d_norm = max(0.0, min(1.0, depth / 200.0))
        tight = 0.5 * s_norm + 0.5 * d_norm

    raw = (
        _W_Z       * z_norm   +
        _W_SIZE    * size     +
        _W_DIRCONF * dir_conf +
        _W_COHORT  * cohort   +
        _W_TIGHT   * tight
    )
    return max(0.0, min(1.0, raw))


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

# Live-big-bets pull thresholds. Kalshi exposes /markets/trades; we
# fetch recent trades across the bots' configured series and surface
# every trade above this notional as a candidate. Smaller than the
# whale_detector min_notional_cents (5000c = $50) used by the bot —
# the dashboard wants to *show* every big bet, not just the
# z-score-anomalous ones.
LIVE_MIN_NOTIONAL_CENTS = 500            # $5+ trades surface (these
                                         # markets are thin so the bar
                                         # is set low to populate the
                                         # page; bump for sportsbook-
                                         # scale series)
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

    return out


# --------------------------------------------------------------------------- #
# Live big-bet ingestion                                                      #
# --------------------------------------------------------------------------- #

def fetch_live_big_bets(series_tickers: List[str],
                          min_notional_cents: int = LIVE_MIN_NOTIONAL_CENTS,
                          lookback_hours: int = LIVE_LOOKBACK_HOURS,
                          ) -> List[dict]:
    """Pull recent trades from Kalshi for each series, filter to big
    bets, and return event-shaped dicts ready for the existing
    scorer + validator pipeline.

    No checkpoint data (these are live trades, not yet aged out), so
    `last_favorable_cents` will be None on every row. The "Outcome"
    column on the History tab simply won't include them until the
    bot's signal_tracker has a chance to capture +30m checkpoints.

    Each event-shaped dict has keys matching what compute_candidates /
    insider_score / validate_whale already consume:
        ticker, signal_ts, zscore (None — no per-market history yet),
        direction, direction_confidence, whale_count,
        whale_notional_cents, taker_side, entry_mid_cents,
        entry_spread_cents (None — no orderbook fetched per trade),
        entry_depth_within_3c (0), entered (False), checkpoints ([]).
    """
    from . import kalshi_client
    client = kalshi_client.default_client()
    if not client.available:
        return []

    seen_trade_ids: set = set()
    out: List[dict] = []
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
            try:
                trades = client.fetch_trades(
                    market_ticker=ticker,
                    lookback_hours=lookback_hours,
                    limit=200,
                )
            except Exception:  # noqa: BLE001
                log.exception("whale: fetch_trades failed for %s", ticker)
                continue
            # Kalshi /markets/trades returns count as `count_fp` (a
            # fixed-point integer count) and prices as
            # `yes_price_dollars` / `no_price_dollars` (dollar
            # strings like "0.6500"). Normalise once.
            def _trade_count(t):
                # Kalshi sends count_fp as a float (e.g. 46.85). Round
                # to whole contracts — Kalshi's UI displays the
                # rounded integer too.
                c = t.get("count_fp")
                if c is None:
                    c = t.get("count")
                try:
                    return int(round(float(c))) if c is not None else 0
                except (TypeError, ValueError):
                    return 0

            def _trade_price_cents(t, side: str):
                # side = "yes" or "no"
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
                # Heuristic: if value is < 5 it's dollars; otherwise
                # already cents. Kalshi's _dollars suffix is reliable
                # so we expect dollars in the range [0, 1].
                return int(round(f * 100)) if f < 5 else int(f)

            # Per-ticker z-score: how unusual is each trade's count
            # relative to the other trades we're seeing on the same
            # market in the lookback window? Cheap stand-in for the
            # rolling-stats approach the bot uses on its live tape.
            counts = [_trade_count(t) for t in trades]
            if len(counts) >= 5:
                mean_c = sum(counts) / len(counts)
                var = sum((c - mean_c) ** 2 for c in counts) / len(counts)
                std_c = math.sqrt(var) if var > 0 else 0.0
            else:
                mean_c = std_c = 0.0
            for t in trades:
                trade_id = t.get("trade_id") or (
                    f"{ticker}:{t.get('created_time')}:{_trade_count(t)}"
                )
                if trade_id in seen_trade_ids:
                    continue
                seen_trade_ids.add(trade_id)
                count = _trade_count(t)
                yes_price = _trade_price_cents(t, "yes")
                no_price = _trade_price_cents(t, "no")
                # Notional = whichever side was actually paid.
                price = yes_price if yes_price > 0 else no_price
                notional = count * price
                if notional < min_notional_cents:
                    continue
                taker_side = (t.get("taker_side") or "").lower()
                # Direction inference — same heuristic as the bot's
                # whale_detector when taker_side isn't reliable.
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
                # Mid for probability bounds — use the trade's own
                # yes_price as the implied mid since we don't have
                # the orderbook at trade time.
                mid = yes_price if yes_price > 0 else (100 - no_price)
                # Per-ticker z-score (computed above off the trades
                # batch). Falls back to 0 when the sample is too
                # thin — the validator will then fail z-score.
                zscore = (
                    (count - mean_c) / std_c
                    if std_c > 0 else 0.0
                )
                out.append({
                    "ticker": ticker,
                    "signal_ts": signal_ts,
                    "zscore": zscore,
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
    return out


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
        out.append({
            "ticker": e.get("ticker") or "",
            "signal_ts": e.get("signal_ts"),
            "direction": e.get("direction") or "?",
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
    series_to_scan: List[str] = []
    for b in available_bots:
        s = b.get("series_ticker")
        if s and s not in series_to_scan:
            series_to_scan.append(s)
    live_events = fetch_live_big_bets(series_to_scan) if series_to_scan else []

    # Combine: tracked signals (from JSONL, with checkpoints) + live
    # big bets (from Kalshi API, no checkpoints yet). The scorer
    # handles both shapes — live trades just lack a z-score so the
    # corresponding insider feature defaults to 0 for them.
    combined_events = list(events) + live_events
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
        _render_empty_state(out)
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
               "<th>Age</th><th>Ticker</th><th>Side</th>"
               "<th class='num'>Bet size</th>"
               "<th class='num'>Z-score</th>"
               "<th class='num'>Cohort win %</th>"
               "<th class='num'>Insider P</th>"
               "<th>Validators</th>"
               "<th title='Recommended action: BUY (simulate entry), "
               "WATCH (promising but not actionable), or SKIP.'>"
               "Action</th>"
               "</tr></thead><tbody>")
    for c in pool[:200]:
        side_cls = "side-yes" if c["direction"] == "yes" else "side-no"
        score_cls = _score_class(c["insider_score"])
        coh = c["cohort_winrate"]
        coh_str = _fmt_pct(coh) if coh is not None else "—"
        # Validators summary as inline pills.
        pills = []
        for name, ok, _detail in c["validators"]:
            cls = "valid-pill " + ("pass" if ok else "fail")
            short = name.split(" ")[0][:8]  # compact
            pills.append(f"<span class='{cls}' title='{html.escape(_validator_tooltip(name, ok, _detail))}'>{html.escape(short)}</span>")
        action = c["action"]
        action_cls = {
            "BUY":   "badge-yes",
            "WATCH": "badge-hedge",
            "SKIP":  "badge-skip",
        }.get(action, "badge-skip")
        action_badge = (
            f"<span class='badge {action_cls}' "
            f"title='{html.escape(c.get('action_reason', ''))}'>"
            f"{action}</span>"
        )
        z_str = (f"{c['zscore']:.2f}" if c.get('zscore') is not None
                  else "—")
        out.append(
            f"<tr><td>{html.escape(_fmt_age(c['signal_ts']))}</td>"
            f"<td class='mono'>{html.escape(c['ticker'])}</td>"
            f"<td><span class='{side_cls}'>{c['direction'].upper()}</span></td>"
            f"<td class='num'>{_fmt_dollars(c['notional_cents'])}</td>"
            f"<td class='num'>{z_str}</td>"
            f"<td class='num'>{coh_str}</td>"
            f"<td class='num'><span class='whale-score {score_cls}'>"
            f"{c['insider_score']*100:.0f}%</span></td>"
            f"<td>{''.join(pills)}</td>"
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


def _render_empty_state(out: List[str]) -> None:
    out.append("<div class='empty'>No whale signals captured yet.</div>")


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
