"""CLV + calibration analytics — the professional-practice layer.

Two responsibilities, one daemon thread (live dashboard only):

1. **Line logger** (every ``LOG_INTERVAL_S``): snapshot each sport
   bot's benchmark line per SIDE ticker into
   ``data/analytics/line_log.db``. This is the raw material for
   closing-line value: the *closing line* for a position is the last
   pre-kickoff benchmark snapshot for its ticker. Read-only against
   the watchlist JSONs the bots already write — this daemon can slow
   nothing down and break nothing upstream.

2. **Analytics computer** (daily + once shortly after startup):
   walk every bot's closed positions and produce
   ``data/analytics/analytics.json`` with, per bot:

     * calibration bins — what the model said vs how often it won,
     * Brier score of the model vs Brier of the MARKET at entry
       (the market's price is the benchmark to beat: a model with a
       worse Brier than the price it trades against is noise, per
       the 2026-09-10 tennis audit),
     * CLV — entry price vs the closing benchmark line (sports only;
       positions opened before logging began have no closing line
       and are excluded rather than faked),
     * a sizing verdict: ``sizing_eligible`` turns True only after
       ``SIZING_MIN_CLV_N`` CLV-measured trades with positive average
       CLV. The Kelly path in the executors stays at 1 contract until
       then (and additionally until the operator sets a bankroll).

Why CLV: realized P&L needs ~1000 trades to separate skill from
variance; CLV converges in ~100 because every trade is scored against
the sharp close, not the coin-flip outcome. It is the standard yard-
stick professional betting operations use to judge a strategy while
the P&L sample is still meaningless.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("dashboard.analytics")

ANALYTICS_DIR = Path(__file__).resolve().parents[2] / "data" / "analytics"
LINE_LOG_DB = ANALYTICS_DIR / "line_log.db"
ARTIFACT_PATH = ANALYTICS_DIR / "analytics.json"

LOG_INTERVAL_S = 300          # benchmark snapshot cadence
HEARTBEAT_S = 1800            # re-log an unchanged line at least this often
MIN_DELTA = 0.005             # log when the line moved at least this much
PRUNE_DAYS = 60               # line-log retention
DAILY_S = 24 * 3600           # analytics recompute cadence
STARTUP_DELAY_S = 300         # let the bots' first ticks land first

SIZING_MIN_CLV_N = 100        # trades with measured CLV before sizing up
SIZING_MIN_AVG_CLV = 0.0      # and the average must be positive

# Sports whose exporter puts the benchmark in live_prob_a (the
# benchmark IS the model there); everywhere else the benchmark is
# pinnacle_prob_a and live_prob_a is the bot's own internal model.
_BENCHMARK_IS_MODEL = {"basketball", "mlb", "world-cup"}

# Closed-position exit reasons that reflect a real outcome. Voids /
# scalar refunds carry no information about the model.
_SCORED_EXITS = {"settled_match", "profit_lock", "external_close"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(t: Any) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


# ────────────────────────────── line logger ──────────────────────────


def _line_db() -> sqlite3.Connection:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(LINE_LOG_DB)
    c.execute(
        "CREATE TABLE IF NOT EXISTS line_snapshots ("
        " bot TEXT NOT NULL,"
        " ticker TEXT NOT NULL,"
        " bench_prob REAL NOT NULL,"
        " kalshi_yes_ask INTEGER,"
        " kickoff TEXT,"
        " captured_at TEXT NOT NULL)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_line_ticker "
        "ON line_snapshots (ticker, captured_at)"
    )
    return c


# In-memory last-logged value per ticker so the common no-move case
# costs zero writes. {ticker: (bench_prob, monotonic_ts)}
_last_logged: Dict[str, tuple] = {}


def _rows_for_bot(bot: Dict[str, Any]) -> List[tuple]:
    """(ticker, bench_prob, yes_ask_cents, kickoff) per SIDE for one
    sport bot's current watchlist. Empty list on any read problem —
    a mid-write JSON or missing file is normal, not an error."""
    wl_path = bot.get("watchlist_json_path")
    if not wl_path:
        return []
    try:
        with open(wl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    rows = data.get("rows") if isinstance(data, dict) else data
    out: List[tuple] = []
    for r in rows or []:
        bench_a = r.get("pinnacle_prob_a")
        if bench_a is None and bot.get("key") in _BENCHMARK_IS_MODEL:
            bench_a = r.get("live_prob_a")
        if bench_a is None:
            continue
        try:
            bench_a = float(bench_a)
        except (TypeError, ValueError):
            continue
        kickoff = r.get("kickoff")
        t_a, t_b = r.get("ticker_a"), r.get("ticker_b")
        if t_a:
            out.append((t_a, bench_a, r.get("yes_ask_cents_a"), kickoff))
        if t_b:
            out.append((t_b, 1.0 - bench_a, r.get("yes_ask_cents_b"),
                        kickoff))
    return out


def log_lines_tick(bots: List[Dict[str, Any]]) -> int:
    """One snapshot pass. Returns how many rows were appended.

    Kickoff is normalised to a UTC isoformat string at write time so
    the closing-line query can compare timestamps lexically — raw
    exporter kickoffs mix "Z", "+00:00" and "-04:00" offsets, and
    comparing those as strings mis-orders events across offsets
    (basketball's ET-stamped tips would land on the wrong side of a
    UTC captured_at).
    """
    now_iso = _now().isoformat(timespec="seconds")
    mono = time.monotonic()
    appended = 0
    with closing(_line_db()) as db:
        for b in bots:
            if b.get("dashboard_type") != "sport":
                continue
            for ticker, prob, ask, kickoff in _rows_for_bot(b):
                last = _last_logged.get(ticker)
                if (last is not None
                        and abs(last[0] - prob) < MIN_DELTA
                        and mono - last[1] < HEARTBEAT_S):
                    continue
                kdt = _parse_ts(kickoff)
                kick_utc = (kdt.astimezone(timezone.utc)
                            .isoformat(timespec="seconds")
                            if kdt else None)
                db.execute(
                    "INSERT INTO line_snapshots "
                    "(bot, ticker, bench_prob, kalshi_yes_ask, kickoff,"
                    " captured_at) VALUES (?,?,?,?,?,?)",
                    (b.get("key"), ticker, prob, ask, kick_utc, now_iso))
                _last_logged[ticker] = (prob, mono)
                appended += 1
        db.commit()
    if len(_last_logged) > 20000:   # tickers cycle out; cap the map
        _last_logged.clear()
    return appended


def prune_line_log() -> None:
    cutoff = datetime.fromtimestamp(
        _now().timestamp() - PRUNE_DAYS * 86400,
        tz=timezone.utc).isoformat(timespec="seconds")
    with closing(_line_db()) as db:
        db.execute("DELETE FROM line_snapshots WHERE captured_at < ?",
                   (cutoff,))
        db.commit()


def closing_line(ticker: str) -> Optional[float]:
    """Last pre-kickoff benchmark snapshot for a ticker, or None.

    Uses the snapshot's own kickoff stamp; a ticker whose snapshots
    never carried a kickoff has no honest closing line and returns
    None — CLV is only reported where it is real.
    """
    try:
        with closing(_line_db()) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT bench_prob FROM line_snapshots "
                "WHERE ticker = ? AND kickoff IS NOT NULL "
                "  AND captured_at <= kickoff "
                "ORDER BY captured_at DESC LIMIT 1",
                (ticker,)).fetchone()
        return float(row["bench_prob"]) if row else None
    except sqlite3.Error:
        return None


# ─────────────────────────── analytics computer ──────────────────────


def _bins(pairs: List[tuple]) -> List[Dict[str, Any]]:
    """10pp calibration bins over (predicted, won) pairs."""
    out = []
    for lo10 in range(0, 10):
        lo, hi = lo10 / 10.0, lo10 / 10.0 + 0.1
        sel = [(p, w) for p, w in pairs if lo <= p < hi]
        if not sel:
            continue
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": len(sel),
            "predicted": round(sum(p for p, _ in sel) / len(sel), 3),
            "actual": round(sum(1 for _, w in sel if w) / len(sel), 3),
        })
    return out


def _brier(pairs: List[tuple]) -> Optional[float]:
    if not pairs:
        return None
    return round(sum((p - (1.0 if w else 0.0)) ** 2
                     for p, w in pairs) / len(pairs), 4)


def _sport_bot_stats(bot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calibration + CLV for one tennis-shape sports bot from its
    live sim_state.json closed positions."""
    sp = bot.get("sim_state_path")
    if not sp or not Path(sp).exists():
        return None
    try:
        with open(sp, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    closed = state.get("closed_positions") or []
    model_pairs: List[tuple] = []      # (model prob of side, won)
    market_pairs: List[tuple] = []     # (price of side, won)
    clvs: List[float] = []
    pnl = 0.0
    for p in closed:
        if p.get("exit_reason") not in _SCORED_EXITS:
            continue
        won = bool(p.get("won"))
        pnl += float(p.get("realized_pnl") or 0.0)
        mp = p.get("entry_model_prob")
        kp = p.get("entry_market_prob")
        if mp is not None:
            model_pairs.append((float(mp), won))
        if kp is not None:
            market_pairs.append((float(kp), won))
        # CLV: closing benchmark prob for the SIDE ticker minus the
        # price paid. Positive = we bought better than the close.
        tick = p.get("ticker")
        if tick and kp is not None:
            close = closing_line(tick)
            if close is not None:
                clvs.append(round(close - float(kp), 4))
    if not model_pairs:
        return None
    n_clv = len(clvs)
    avg_clv = (sum(clvs) / n_clv) if n_clv else None
    return {
        "kind": "sport",
        "n": len(model_pairs),
        "realized_pnl": round(pnl, 2),
        "win_rate": round(sum(1 for _, w in model_pairs if w)
                          / len(model_pairs), 3),
        "avg_model_prob": round(sum(p for p, _ in model_pairs)
                                / len(model_pairs), 3),
        "brier_model": _brier(model_pairs),
        "brier_market": _brier(market_pairs),
        "bins": _bins(model_pairs),
        "clv": {
            "n": n_clv,
            "avg": round(avg_clv, 4) if avg_clv is not None else None,
            "pos_rate": (round(sum(1 for c in clvs if c > 0) / n_clv, 3)
                         if n_clv else None),
        },
        "sizing_eligible": bool(n_clv >= SIZING_MIN_CLV_N
                                and avg_clv is not None
                                and avg_clv > SIZING_MIN_AVG_CLV),
    }


def _model_prob_from_decision(dj: str, side: str) -> Optional[float]:
    """Held-side model probability out of a stored decision_json."""
    try:
        d = json.loads(dj or "{}")
    except json.JSONDecodeError:
        return None
    p = None
    for k in ("model_prob_yes", "model_prob", "prob_yes"):
        if d.get(k) is not None:
            try:
                p = float(d[k])
            except (TypeError, ValueError):
                p = None
            break
    if p is None or not (0.0 <= p <= 1.0) or math.isnan(p):
        return None
    return p if (side or "").upper() == "YES" else 1.0 - p


def _db_bot_stats(bot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calibration for a sqlite-ledger bot (macro / weather). No CLV —
    these bots' reference IS the model, so 'closing line value' would
    be self-referential; calibration + realized P&L is the honest
    read. Curves render once n is meaningful."""
    dbp = bot.get("db_path") or ""
    if not dbp.endswith(".db") or not Path(dbp).exists():
        return None
    pairs: List[tuple] = []
    market_pairs: List[tuple] = []
    pnl_cents = 0
    try:
        with closing(sqlite3.connect(dbp)) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT ticker, side, entry_price_cents, "
                "realized_pnl_cents, decision_json FROM positions "
                "WHERE status = 'closed' "
                "AND realized_pnl_cents IS NOT NULL").fetchall()
    except sqlite3.Error:
        return None
    # Bots that share one ledger (rain + temp both live in the
    # weather sim.db) are told apart by their card's series prefixes
    # — without this both cards would report the union of the file
    # (2026-09-10: rain and temp rendered identical stats).
    prefixes = tuple(bot.get("series_prefixes") or ())
    if prefixes:
        rows = [r for r in rows
                if str(r["ticker"] or "").startswith(prefixes)]
    for r in rows:
        won = (r["realized_pnl_cents"] or 0) > 0
        pnl_cents += int(r["realized_pnl_cents"] or 0)
        p = _model_prob_from_decision(r["decision_json"], r["side"])
        if p is not None:
            pairs.append((p, won))
        if r["entry_price_cents"] is not None:
            market_pairs.append((r["entry_price_cents"] / 100.0, won))
    if not pairs:
        return None
    return {
        "kind": "ledger",
        "n": len(pairs),
        "realized_pnl": round(pnl_cents / 100.0, 2),
        "win_rate": round(sum(1 for _, w in pairs if w) / len(pairs), 3),
        "avg_model_prob": round(sum(p for p, _ in pairs) / len(pairs), 3),
        "brier_model": _brier(pairs),
        "brier_market": _brier(market_pairs),
        "bins": _bins(pairs) if len(pairs) >= 20 else [],
        "clv": {"n": 0, "avg": None, "pos_rate": None},
        "sizing_eligible": False,
    }


def compute_artifact(bots: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_bot: Dict[str, Any] = {}
    for b in bots:
        key = b.get("key")
        if not key:
            continue
        try:
            if b.get("dashboard_type") == "sport":
                stats = _sport_bot_stats(b)
            else:
                stats = _db_bot_stats(b)
        except Exception:  # noqa: BLE001 — one bot never blanks the rest
            log.exception("analytics failed for bot=%s", key)
            stats = None
        if stats:
            stats["name"] = b.get("name", key)
            per_bot[key] = stats
    return {
        "generated_at": _now().isoformat(timespec="seconds"),
        "sizing_rule": {
            "min_clv_n": SIZING_MIN_CLV_N,
            "min_avg_clv": SIZING_MIN_AVG_CLV,
        },
        "bots": per_bot,
    }


def write_artifact(bots: List[Dict[str, Any]]) -> Optional[Path]:
    try:
        artifact = compute_artifact(bots)
        ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = ARTIFACT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(artifact, indent=1))
        tmp.replace(ARTIFACT_PATH)
        log.info("analytics artifact written (%d bots)",
                 len(artifact["bots"]))
        return ARTIFACT_PATH
    except Exception:  # noqa: BLE001
        log.exception("analytics artifact write failed")
        return None


def read_artifact() -> Optional[Dict[str, Any]]:
    """For the Home-tab panel and the sizing gate. None when absent."""
    try:
        with ARTIFACT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def sizing_eligibility(bot_key: str) -> bool:
    """Kelly gate: True only when the nightly artifact says this bot
    has enough measured, positive CLV. Missing artifact → False —
    sizing fails closed to 1 contract."""
    art = read_artifact()
    if not art:
        return False
    return bool((art.get("bots") or {}).get(bot_key, {})
                .get("sizing_eligible"))


# ────────────────────────────── daemon ───────────────────────────────


def start_daemon(bots: List[Dict[str, Any]], mode: str) -> Optional[threading.Thread]:
    """Live-mode only: the sim dashboard reading the same repo would
    double-write the line log and score paper fills as if they were
    executions. One writer, real fills only."""
    if mode != "live":
        log.info("analytics daemon not started (mode=%s — live only)",
                 mode)
        return None

    def _loop() -> None:
        log.info("analytics daemon started (log every %ds, "
                 "recompute daily)", LOG_INTERVAL_S)
        time.sleep(STARTUP_DELAY_S)
        last_daily = 0.0
        while True:
            try:
                n = log_lines_tick(bots)
                if n:
                    log.info("line log +%d snapshots", n)
            except Exception:  # noqa: BLE001
                log.exception("line-log tick failed")
            if time.monotonic() - last_daily > DAILY_S or last_daily == 0.0:
                try:
                    prune_line_log()
                    write_artifact(bots)
                    last_daily = time.monotonic()
                except Exception:  # noqa: BLE001
                    log.exception("daily analytics pass failed")
            time.sleep(LOG_INTERVAL_S)

    t = threading.Thread(target=_loop, daemon=True, name="analytics")
    t.start()
    return t
