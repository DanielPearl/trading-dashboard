"""Per-bet probability chart for the Active bets section.

User 2026-09-16: "for every active live bot, put the line graph back
showing the probability changes between the model % and the kalshi
market % from when it was bought to when the contract ends. green if
YES bought, red if NO bought. show update every time it's updated.
default to the most recent bet; clicking an active bet switches the
graph to that bet."

Both series plot the probability THE BET WINS (NO bets flip the
yes-axis), so line-up always reads as winning regardless of side.

Sources per bet:
- Kalshi %: the market's candlestick history from open time on
  (works for any full market ticker). Sport paper bets key on the
  EVENT ticker (no candles) — they fall back to a two-point
  entry → current line, which still advances every refresh.
- Model %: the bot's own market_views history for the ticker
  (ladder/macro bots record model_prob_yes every tick); bots
  without that table fall back to entry → current model points.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("dashboard.bet_chart")

MAX_BETS = 12
MAX_POINTS = 240


def _ts(iso: Any) -> Optional[float]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(
            str(iso).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _side_pct(yes_prob01: Optional[float], side: str) -> Optional[float]:
    """Yes-axis prob (0..1) → the bet side's win chance in %."""
    if yes_prob01 is None:
        return None
    p = max(0.0, min(1.0, float(yes_prob01)))
    return round((p if side == "YES" else 1.0 - p) * 100.0, 1)


def _downsample(pts: List[list]) -> List[list]:
    if len(pts) <= MAX_POINTS:
        return pts
    step = len(pts) / float(MAX_POINTS)
    keep = [pts[int(i * step)] for i in range(MAX_POINTS)]
    if keep[-1] != pts[-1]:
        keep.append(pts[-1])
    return keep


def _candle_series(ticker: str, open_ts: float,
                   side: str) -> List[list]:
    """[[ts, side_pct], ...] from Kalshi candlesticks, open→now."""
    try:
        from . import kalshi_client as kc
        client = kc.get_client()
        now = datetime.now(timezone.utc).timestamp()
        lookback_h = max(2, int((now - open_ts) / 3600) + 2)
        # 1-minute candles for young bets, hourly beyond two days —
        # "every update" at a resolution the chart can actually show.
        period = 1 if lookback_h <= 48 else 60
        cdls = kc._candlesticks(client, ticker.split("-")[0], ticker,
                                period, min(lookback_h, 45 * 24))
        out = []
        for c in cdls or []:
            ts = _num(c.get("end_period_ts") or c.get("ts"))
            if ts is None or ts < open_ts:
                continue
            p = kc._candle_yes_prob(c)
            sp = _side_pct(p, side)
            if sp is not None:
                out.append([int(ts), sp])
        return _downsample(out)
    except Exception:  # noqa: BLE001
        log.exception("candle series failed for %s", ticker)
        return []


def _model_series(db_path: str, ticker: str, open_ts: float,
                  side: str) -> List[list]:
    """[[ts, side_pct], ...] from the bot's own market_views ticks."""
    if not db_path or not db_path.endswith(".db") \
            or not Path(db_path).exists():
        return []
    out: List[list] = []
    try:
        with closing(sqlite3.connect(db_path)) as c:
            cols = {r[1] for r in
                    c.execute("PRAGMA table_info(market_views)")}
            if "model_prob_yes" not in cols:
                return []
            for cap, p in c.execute(
                    "SELECT captured_at, model_prob_yes FROM market_views"
                    " WHERE ticker = ? AND model_prob_yes IS NOT NULL"
                    " ORDER BY id", (ticker,)):
                ts = _ts(cap)
                if ts is None or ts < open_ts:
                    continue
                sp = _side_pct(_num(p), side)
                if sp is not None:
                    out.append([int(ts), sp])
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    return _downsample(out)


def _contract_title(db_path: str, ticker: str) -> Optional[str]:
    """The Kalshi contract's own title + question phrasing from the
    bot's latest market_views row — "Resident Evil Rotten Tomatoes
    score? · above 92" — so the chart caption reads like the listing
    instead of a ticker."""
    if not db_path or not db_path.endswith(".db") \
            or not Path(db_path).exists():
        return None
    try:
        with closing(sqlite3.connect(db_path)) as c:
            cols = {r[1] for r in
                    c.execute("PRAGMA table_info(market_views)")}
            if "title" not in cols:
                return None
            row = c.execute(
                "SELECT title, direction, strike_low FROM market_views"
                " WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (ticker,)).fetchone()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return None
    if not row or not row[0]:
        return None
    title, direction, strike = row
    q = ""
    if strike is not None:
        n = _num(strike)
        n_txt = (str(int(n)) if n is not None and n == int(n)
                 else str(strike))
        q = f" · {direction or 'above'} {n_txt}"
    return f"{title}{q}"


def build_payload(bot: dict, bets: List[dict]) -> Optional[dict]:
    """Chart payload for the pane: newest-first, capped, one entry
    per bet with both series. None when there is nothing to draw."""
    rows: List[dict] = []
    now = datetime.now(timezone.utc).timestamp()
    usable = [b for b in (bets or []) if b.get("ticker")
              and _ts(b.get("opened_at"))]
    usable.sort(key=lambda b: b.get("opened_at") or "", reverse=True)
    for b in usable[:MAX_BETS]:
        ticker = str(b.get("ticker"))
        side = "NO" if str(b.get("side") or "YES").upper() == "NO" \
            else "YES"
        open_ts = _ts(b.get("opened_at"))
        entry_c = _num(b.get("entry_price_cents"))
        # Entry point on the bet's own side (entry price ≈ the market
        # prob paid for that side).
        entry_pct = round(entry_c, 1) if entry_c is not None else None

        kal = _candle_series(ticker, open_ts, side)
        if not kal:
            kal = []
            if entry_pct is not None:
                kal.append([int(open_ts), entry_pct])
            cur = (_num(b.get("mark_mid"))
                   or _num(b.get("current_market_prob")))
            if cur is not None:
                cur01 = cur / 100.0 if cur > 1.5 else cur
                # sport rows already carry the bet side's prob; sim
                # rows carry yes-axis marks — mark_mid is yes-axis.
                sp = _side_pct(cur01, side) \
                    if b.get("mark_mid") is not None \
                    else round(max(0.0, min(1.0, cur01)) * 100.0, 1)
                kal.append([int(now), sp])
        elif entry_pct is not None and (not kal
                                        or kal[0][0] > open_ts + 60):
            kal.insert(0, [int(open_ts), entry_pct])

        mdl = _model_series(bot.get("db_path") or "", ticker,
                            open_ts, side)
        if not mdl:
            emp = _num(b.get("model_yes_prob_at_entry"))
            if emp is not None:
                sp = _side_pct(emp if emp <= 1.5 else emp / 100.0, side)
                mdl.append([int(open_ts), sp])
            cmp_ = (_num(b.get("current_model_prob_yes"))
                    if b.get("current_model_prob_yes") is not None
                    else _num(b.get("current_model_prob")))
            if cmp_ is not None:
                sp = _side_pct(cmp_ if cmp_ <= 1.5 else cmp_ / 100.0,
                               side)
                if sp is not None:
                    mdl.append([int(now), sp])

        mtc = _num(b.get("minutes_to_close"))
        close_ts = int(now + mtc * 60) if mtc and mtc > 0 else None
        label = (b.get("_match")
                 or _contract_title(bot.get("db_path") or "", ticker)
                 or b.get("title") or ticker)
        rows.append({
            "ticker": ticker,
            "side": side,
            "label": str(label)[:80],
            "open_ts": int(open_ts),
            "close_ts": close_ts,
            "kalshi": kal,
            "model": mdl,
        })
    if not rows:
        return None
    return {"bets": rows}


def _series_cache_path(ticker: str, side: str) -> Path:
    d = Path(__file__).resolve().parents[2] / "data" / "series_cache"
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                   for ch in f"{ticker}_{side}")
    return d / f"{safe}.json"


def history_series(bot: dict, ticker: str, side: str,
                   open_ts: float, close_ts: float) -> dict:
    """Both series for a SETTLED contract over the market's FULL
    life — from Kalshi listing to settlement, not just from the
    purchase (user 2026-09-17: "show the full kalshi history of the
    contract and then show where i bought it"). ``market_open`` in
    the response lets the chart scale its axis to the whole
    lifetime; the caller draws the buy marker from the bet row."""
    side = "NO" if str(side).upper() == "NO" else "YES"
    # A settled contract's history is immutable — cache it forever so
    # every view after the first is a snapshot (user 2026-09-17:
    # "this should just be a snapshot").
    cache = _series_cache_path(ticker, side)
    if cache.exists():
        try:
            cached = json.loads(cache.read_text())
            if cached.get("kalshi"):
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    grace = 900
    kal: List[list] = []
    market_open = open_ts
    try:
        from . import kalshi_client as kc
        client = kc.get_client()
        mk = client.get_market(ticker) or {}
        mk = mk.get("market") or mk
        mo = _ts(mk.get("open_time"))
        if mo and mo < open_ts:
            market_open = mo
        span_h = (close_ts - market_open) / 3600.0
        period = 1 if span_h <= 48 else 60
        resp = client.get_market_candlesticks(
            ticker=ticker, series_ticker=ticker.split("-")[0],
            start_ts=int(market_open - 3600),
            end_ts=int(close_ts + grace),
            period_interval=period)
        for c in (resp or {}).get("candlesticks") or []:
            ts = _num(c.get("end_period_ts") or c.get("ts"))
            if ts is None or ts < market_open or ts > close_ts + grace:
                continue
            sp = _side_pct(kc._candle_yes_prob(c), side)
            if sp is not None:
                kal.append([int(ts), sp])
    except Exception:  # noqa: BLE001
        log.exception("history candles failed for %s", ticker)
    mdl = [p for p in _model_series(bot.get("db_path") or "", ticker,
                                    market_open, side)
           if p[0] <= close_ts + grace]
    if not mdl:
        mdl = [p for p in _benchmark_history_series(
                   bot, ticker, market_open, side)
               if p[0] <= close_ts + grace]
    out = {"kalshi": _downsample(kal), "model": mdl,
           "market_open": int(market_open)}
    if kal:
        try:
            cache.write_text(json.dumps(out))
        except OSError:
            pass
    return out


def _bench_db_path(bot: dict) -> Optional[Path]:
    """benchmark_history.db lives beside the sport bot's sim_state —
    written by the dashboard's benchmark recorder (2026-09-17) so
    future sport contracts get a model line; nothing exists for bets
    that settled before the recorder started."""
    sp = bot.get("sim_state_path") or ""
    if not sp:
        return None
    return Path(sp).parent / "benchmark_history.db"


def _benchmark_history_series(bot: dict, ticker: str,
                              open_ts: float, side: str) -> List[list]:
    p = _bench_db_path(bot)
    if not p or not p.exists():
        return []
    out: List[list] = []
    try:
        with closing(sqlite3.connect(p)) as c:
            for ts, prob in c.execute(
                    "SELECT ts, prob_yes FROM bench WHERE ticker = ?"
                    " ORDER BY ts", (ticker,)):
                if ts is None or prob is None or ts < open_ts:
                    continue
                sp = _side_pct(float(prob), side)
                if sp is not None:
                    out.append([int(ts), sp])
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    return _downsample(out)


def record_sport_benchmarks(bots: List[dict]) -> int:
    """Append every sport bot's current benchmark probs to its
    benchmark_history.db — the movement record the History drill-down
    reads for sport contracts. Called on an interval by the server."""
    import time as _time
    n = 0
    now = int(_time.time())
    for bot in bots or []:
        if (bot.get("dashboard_type") or "") != "sport":
            continue
        wl = bot.get("watchlist_json_path")
        p = _bench_db_path(bot)
        if not wl or not p or not Path(wl).exists():
            continue
        try:
            rows = json.loads(Path(wl).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        try:
            with closing(sqlite3.connect(p)) as c:
                c.execute("CREATE TABLE IF NOT EXISTS bench ("
                          " ts INTEGER, ticker TEXT, prob_yes REAL,"
                          " PRIMARY KEY (ts, ticker))")
                for r in rows or []:
                    for sfx in ("a", "b"):
                        tk = r.get(f"ticker_{sfx}")
                        pr = (r.get(f"pinnacle_prob_{sfx}")
                              if r.get(f"pinnacle_prob_{sfx}") is not None
                              else r.get(f"live_prob_{sfx}"))
                        if tk and pr is not None:
                            c.execute("INSERT OR IGNORE INTO bench"
                                      " VALUES (?,?,?)",
                                      (now, str(tk), float(pr)))
                            n += 1
                c.commit()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            log.exception("benchmark recorder failed for %s",
                          bot.get("key"))
    return n
