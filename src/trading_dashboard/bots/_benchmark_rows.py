"""Benchmark row pass — make the sharp-book line the model.

Shared by the sport bots whose probability source is the devigged
Pinnacle benchmark itself (darts, table-tennis; MLB and NBA do the
same thing inside their upstream exporters). Takes the tennis-shape
watchlist rows an upstream exporter produced with its own model and
rewrites the probability / edge / gate fields so that:

  * pre_match_prob / live_prob / pinnacle_prob all carry the devigged
    benchmark probability (the benchmark IS the model),
  * edge / EV / buy_* are recomputed from that benchmark against the
    Kalshi asks with the same uniform gates the MLB exporter uses,
  * rows without a benchmark line are never buy-eligible (WATCH) —
    no sharp reference, no trade. This includes tournament-winner
    rows (player_b == "Field"), which have no two-way moneyline.

The upstream's simulator and the live executor both read
buy_eligible / buy_side / buy_score / live_prob off the row, so this
single pass flips the entire trading pipeline to benchmark-driven
without touching the upstream repo.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

log = logging.getLogger("dashboard.benchmark-rows")

# Uniform gates — keep in lockstep with the MLB exporter's DEFAULTS
# (Baseball Forecast src/dashboard/export_watchlist.py).
DEFAULTS = {
    "slippage": 0.02,
    "min_edge": 0.09,   # raised 0.05 → 0.09 (2026-07-13 real-money audit: −15% ROI; 5pp gross barely cleared spread+fee)
    "strong_edge": 0.10,
    "max_edge": 0.15,   # suspect-benchmark ceiling — see edge_sane gate
    "min_entry_price": 0.30,  # 30–60¢ band (audit: 20-29¢ ran −43%, 60-69¢ −41%)
    "max_entry_price": 0.60,
    "max_spread_cents": 6,
    "min_open_interest": 1,
}

TRADEABLE_LABELS = {"STRONG_EDGE", "SMALL_EDGE"}


def match_pair_probs(lookup: Dict[frozenset, Dict[str, float]],
                     name_a: str, name_b: str):
    """(prob_a, prob_b) from a benchmark lookup, tolerant of casing and
    alias drift between Kalshi titles and Pinnacle participant names
    (e.g. "L. Littler" vs "Luke Littler")."""
    if not (name_a and name_b):
        return None, None
    probs = lookup.get(frozenset({name_a, name_b}))
    if probs is None:
        want_a, want_b = name_a.lower(), name_b.lower()
        for names, p in lookup.items():
            lowered = {n.lower() for n in names if not n.startswith("_")}
            if all(any(w in n or n in w for n in lowered)
                   for w in (want_a, want_b)):
                probs = p
                break
    if not probs:
        return None, None

    def _match(name: str):
        t = name.lower()
        for k, v in probs.items():
            if k.startswith("_"):
                continue  # meta keys (_source / _start), not names
            kl = k.lower()
            if kl == t or t in kl or kl in t:
                return v
        return None

    pa, pb = _match(name_a), _match(name_b)
    # BOTH sides must match — the old one-sided complement fallback
    # manufactured degenerate benchmark probs from bad single-name
    # matches (fantasy +25pp edges, 2026-07-13 WNBA audit).
    if pa is None or pb is None:
        return None, None
    return pa, pb


def match_pair_start(lookup: Dict[frozenset, Dict[str, float]],
                     name_a: str, name_b: str):
    """The benchmark feed's scheduled start (ISO string or None) for a
    pair — Pinnacle guest ``startTime`` / Odds-API ``commence_time``.
    Same tolerant pair resolution as ``match_pair_probs``."""
    if not (name_a and name_b):
        return None
    probs = lookup.get(frozenset({name_a, name_b}))
    if probs is None:
        want_a, want_b = name_a.lower(), name_b.lower()
        for names, p in lookup.items():
            lowered = {n.lower() for n in names if not n.startswith("_")}
            if all(any(w in n or n in w for n in lowered)
                   for w in (want_a, want_b)):
                probs = p
                break
    return (probs or {}).get("_start")


def apply_benchmark(rows: List[Dict[str, Any]],
                    records: List[Dict[str, Any]],
                    lookup: Dict[frozenset, Dict[str, float]],
                    cfg: Dict[str, Any] | None = None,
                    win_verb: str = "winning",
                    keep_model_probs: bool = False) -> int:
    """Rewrite ``rows`` in place; returns how many rows got a benchmark.

    ``records`` (the collapse_to_matches output) supplies per-side
    market tickers for rows whose exporter doesn't pass them through —
    the live executor needs ticker_a/ticker_b to place orders.

    ``keep_model_probs``: when True, rows WITHOUT a benchmark keep the
    upstream exporter's own model probabilities in the pre_match /
    live prob fields (display only — the row is still forced to WATCH
    / not-buy-eligible, and edge / EV stay blank so nothing about it
    looks tradeable). Used by table-tennis, whose Model-vs-market
    table would otherwise be permanently blank while Pinnacle isn't
    quoting TT at all. Rows WITH a benchmark are overwritten either
    way — the sharp line is the model wherever it exists.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec_by_id = {r.get("match_id"): r for r in (records or [])
                 if r.get("match_id")}
    matched = 0

    for row in rows:
        rec = rec_by_id.get(row.get("match_id")) or {}
        for k in ("ticker_a", "ticker_b"):
            if not row.get(k) and rec.get(k):
                row[k] = rec[k]

        # Tournament-winner rows (one player vs the Field) have no
        # two-way moneyline to benchmark — never tradeable here.
        is_h2h = bool(row.get("ticker_b")) and (
            (row.get("player_b") or "").lower() != "field")

        p_a = p_b = None
        if is_h2h:
            p_a, p_b = match_pair_probs(
                lookup, row.get("player_a") or "", row.get("player_b") or "")
            # Stamp the benchmark's scheduled start onto rows that
            # don't carry one (darts / TT exporters have no kickoff
            # field) so the live executor's prematch check can block
            # in-play entries against a stale line — the exact failure
            # the NBA executor hit on 2026-07-09.
            if p_a is not None and not row.get("kickoff"):
                start = match_pair_start(
                    lookup, row.get("player_a") or "",
                    row.get("player_b") or "")
                if start:
                    row["kickoff"] = start

        ask_a = row.get("market_prob_a")
        ask_b = row.get("market_prob_b")
        if ask_b is None and row.get("yes_ask_cents_b") is not None:
            ask_b = row["yes_ask_cents_b"] / 100.0
            row["market_prob_b"] = ask_b

        row["pinnacle_prob_a"] = p_a
        row["pinnacle_prob_b"] = p_b
        if p_a is not None or not keep_model_probs:
            row["pre_match_prob_a"] = p_a
            row["pre_match_prob_b"] = p_b
            row["live_prob_a"] = p_a
            row["live_prob_b"] = p_b
        row["confidence_score"] = (max(p_a, p_b)
                                   if p_a is not None else None)

        edge_a = (p_a - ask_a) if (p_a is not None and ask_a) else None
        edge_b = (p_b - ask_b) if (p_b is not None and ask_b) else None
        ev_a = (edge_a - cfg["slippage"]) if edge_a is not None else None
        ev_b = (edge_b - cfg["slippage"]) if edge_b is not None else None
        row["edge_a"], row["edge_b"] = edge_a, edge_b
        row["ev_a"], row["ev_b"] = ev_a, ev_b

        side, side_edge, side_ev, side_price = "A", edge_a, ev_a, ask_a
        spread = row.get("spread_cents_a", row.get("spread_cents"))
        if (edge_b is not None) and (edge_a is None or edge_b > edge_a):
            side, side_edge, side_ev, side_price = "B", edge_b, ev_b, ask_b
            spread = row.get("spread_cents_b", row.get("spread_cents"))
        oi = row.get("open_interest",
                     (row.get("open_interest_a") or 0)
                     + (row.get("open_interest_b") or 0))

        gates: Dict[str, bool] = {}
        if p_a is None:
            label = "WATCH"
            if not is_h2h:
                reason = "no two-way moneyline to benchmark"
            elif keep_model_probs:
                reason = ("no Pinnacle line — model view only, "
                          "not tradeable")
            else:
                reason = "no Pinnacle line for this match yet"
            eligible = False
        else:
            matched += 1
            gates = {
                "edge": (side_edge or 0) >= cfg["min_edge"],
                # Fake-edge guard: beyond max_edge the benchmark is
                # almost certainly stale or mispaired — WATCH, never buy.
                "edge_sane": (side_edge or 0) <= cfg["max_edge"],
                "ev": (side_ev or 0) > 0,
                "price_band": (side_price is not None
                               and cfg["min_entry_price"] <= side_price
                               <= cfg["max_entry_price"]),
                "open_interest": (oi or 0) >= cfg["min_open_interest"],
                "spread": (spread is not None
                           and spread <= cfg["max_spread_cents"]),
            }
            eligible = all(gates.values())
            if (side_edge or 0) >= cfg["strong_edge"]:
                label = "STRONG_EDGE"
            elif (side_edge or 0) >= cfg["min_edge"]:
                label = "SMALL_EDGE"
            else:
                label = "WATCH"
            eligible = eligible and label in TRADEABLE_LABELS
            blocked = [k for k, ok in gates.items() if not ok]
            side_name = (row.get("player_a") if side == "A"
                         else row.get("player_b"))
            if eligible:
                reason = (f"Pinnacle {side_edge * 100:+.1f}pp vs market "
                          f"on {side_name} {win_verb}")
            elif label in TRADEABLE_LABELS and blocked:
                reason = "edge present but blocked: " + ", ".join(blocked)
            else:
                reason = (f"Pinnacle within {cfg['min_edge'] * 100:.0f}pp "
                          f"of market")

        row["recommended_action"] = label
        row["recommendation"] = "Buy" if eligible else (
            "Watch" if label == "WATCH" else "Hold")
        row["reason_for_signal"] = reason
        row["buy_eligible"] = eligible
        row["buy_score"] = float(side_edge or 0.0)
        row["buy_side"] = side
        row["buy_side_edge"] = side_edge
        row["buy_side_ev"] = side_ev
        row["buy_gates"] = gates
        row["buy_blockers"] = [k for k, ok in gates.items() if not ok]
        row["last_updated"] = now

    return matched
