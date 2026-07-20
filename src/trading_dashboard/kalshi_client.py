"""Re-export shim for the Kalshi REST client.

The dashboard's previous 26KB ``kalshi_client.py`` was an
almost-identical re-implementation of the bots' clients with TTL
caching baked on top. The canonical implementation now lives in
``kalshi_sdk.client`` (TTL caching is part of it via ``cache_ttl=``);
this shim preserves the dashboard's import sites.

Usage in dashboard code (unchanged):
    from .kalshi_client import KalshiClient
    kc = KalshiClient()
    kc.list_markets(series_ticker="KX...")  # cached 5min

The shim builds a single process-wide instance lazily on first use, so
the dashboard renderer can keep calling ``KalshiClient()`` without
re-loading the private key on every request.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from kalshi_sdk import KalshiClient as _SDKClient
from kalshi_sdk.exceptions import KalshiError

log = logging.getLogger("dashboard.kalshi")


_lock = threading.Lock()
_cached: Optional["KalshiClient"] = None


class KalshiClient:
    """Lazy-init Kalshi client wrapper.

    Reads creds from env on first method call; if either KALSHI_API_KEY_ID
    or KALSHI_PRIVATE_KEY_PATH is missing, every method short-circuits
    to ``None`` / ``[]`` rather than raising — the dashboard treats that
    as "no live data" and renders the empty-state placeholder.
    """

    def __init__(self) -> None:
        self._sdk: Optional[_SDKClient] = None
        self._init_attempted = False
        self._available = False

    def _init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        key_id = os.getenv("KALSHI_API_KEY_ID")
        priv = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not key_id or not priv:
            log.info("kalshi_client: KALSHI_API_KEY_ID / "
                     "KALSHI_PRIVATE_KEY_PATH not set; "
                     "live Kalshi data disabled")
            return
        try:
            self._sdk = _SDKClient(
                api_key_id=key_id,
                private_key_path=priv,
                cache_ttl=300.0,    # 5-minute cache, same as before
            )
        except Exception as e:  # noqa: BLE001
            log.warning("kalshi_client: failed to init SDK: %s", e)
            return
        self._available = True
        log.info("kalshi_client: ready (key id %s…)", key_id[:8])

    @property
    def available(self) -> bool:
        self._init()
        return self._available

    # ------------------------------------------------------------------ #
    # Endpoints the dashboard cares about. Each returns None / [] on
    # failure rather than raising, to match the previous shim's contract.
    # ------------------------------------------------------------------ #

    def list_markets(self, series_ticker: Optional[str] = None,
                     event_ticker: Optional[str] = None,
                     status: Optional[str] = "open",
                     limit: int = 200) -> List[dict]:
        self._init()
        if not self._available:
            return []
        try:
            assert self._sdk is not None
            out: List[dict] = []
            cursor: Optional[str] = None
            while True:
                resp = self._sdk.get_markets(
                    limit=limit, cursor=cursor,
                    status=None if event_ticker else status,
                    series_ticker=None if event_ticker else series_ticker,
                    event_ticker=event_ticker,
                )
                out.extend(resp.get("markets") or [])
                cursor = resp.get("cursor") or None
                if not cursor:
                    break
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("list_markets failed: %s", e)
            return []

    def get_market(self, ticker: str) -> Optional[dict]:
        self._init()
        if not self._available:
            return None
        try:
            assert self._sdk is not None
            resp = self._sdk.get_market(ticker)
            return resp.get("market") or resp
        except Exception as e:  # noqa: BLE001
            log.warning("get_market(%s) failed: %s", ticker, e)
            return None

    def get_event(self, event_ticker: str) -> Optional[dict]:
        self._init()
        if not self._available:
            return None
        try:
            assert self._sdk is not None
            resp = self._sdk.get_event(event_ticker)
            return resp.get("event") or resp
        except Exception as e:  # noqa: BLE001
            log.warning("get_event(%s) failed: %s", event_ticker, e)
            return None

    def fetch_trades(self, market_ticker: Optional[str] = None,
                     lookback_hours: float = 1.0,
                     limit: int = 1000) -> List[dict]:
        """Recent trades, paginated. Used by whale.py to detect
        large-notional bets; whale reads ``count_fp`` / ``yes_price_dollars``
        / ``no_price_dollars`` / ``ticker`` directly from the dict shape,
        so this method preserves the raw payload (the SDK's
        ``iter_trades`` returns ``Trade`` dataclasses which would lose
        those fields).

        Args:
            market_ticker: optional ticker filter. None = all markets
                in the most-recent ``lookback_hours``.
            lookback_hours: window before now to scan, translated to
                Kalshi's ``min_ts`` query param (epoch seconds).
            limit: hard cap across all paginated pages so a chatty
                series doesn't blow up the dashboard render.

        Returns: list of raw trade dicts (empty on failure / no creds).
        """
        self._init()
        if not self._available:
            return []
        import time as _time
        min_ts = int(_time.time() - lookback_hours * 3600)
        out: List[dict] = []
        cursor: Optional[str] = None
        try:
            assert self._sdk is not None
            while len(out) < limit:
                page_limit = min(1000, limit - len(out))
                resp = self._sdk.get_trades(
                    ticker=market_ticker,
                    limit=page_limit,
                    min_ts=min_ts,
                    cursor=cursor,
                )
                trades = resp.get("trades") or []
                out.extend(trades)
                cursor = resp.get("cursor") or None
                if not cursor or not trades:
                    break
        except Exception as e:  # noqa: BLE001
            log.warning("fetch_trades(%s, lookback=%sh) failed: %s",
                        market_ticker, lookback_hours, e)
            return out  # return whatever we got before the error
        return out

    def get_market_candlesticks(
        self,
        ticker: str,
        series_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> Optional[dict]:
        self._init()
        if not self._available:
            return None
        try:
            assert self._sdk is not None
            return self._sdk.get_market_candlesticks(
                ticker=ticker,
                series_ticker=series_ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=period_interval,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("candlesticks(%s) failed: %s", ticker, e)
            return None

    def get_balance(self) -> Optional[dict]:
        """Authenticated /portfolio/balance pull. Returns the raw
        Kalshi response dict (typically ``{"balance": <cents>,
        "payout": <cents>}``), or None if the SDK isn't available or
        the call errored. Errors are logged at warning-level — the
        renderer treats None as "no live balance" and falls back to
        the configured default.
        """
        self._init()
        if not self._available:
            return None
        try:
            assert self._sdk is not None
            return self._sdk.get_balance()
        except Exception as e:  # noqa: BLE001
            log.warning("get_balance() failed: %s", e)
            return None


_MARK_CACHE: dict = {}
_OVERVIEW_CACHE: list = [0.0, None]
_SNAPSHOT_PATH = os.environ.get(
    "KALSHI_PORTFOLIO_SNAPSHOTS",
    str(Path(__file__).resolve().parents[2] / "data"
        / "portfolio_snapshots.jsonl"))


def _position_mark_cents(ticker: str) -> Optional[int]:
    """Current value of one contract: last trade, else yes bid.
    Cached 60s per ticker — the overview recomputes on every page
    render and a dozen live GET /markets calls per render would be
    wasteful."""
    now = time.time()
    hit = _MARK_CACHE.get(ticker)
    if hit and now - hit[0] < 60:
        return hit[1]
    m = get_client().get_market(ticker) or {}
    mark = market_cents(m, "last_price")
    if mark is None:
        mark = market_cents(m, "yes_bid")
    _MARK_CACHE[ticker] = (now, mark)
    return mark


def get_portfolio_overview() -> dict:
    """Account headline numbers for the Home/History summary cards:

      cash_cents             — Kalshi cash balance
      positions_value_cents  — open YES/NO positions at last price
      portfolio_cents        — cash + positions value ("Predictions"
                               total, matching the Kalshi app)
      unrealized_cents       — positions value − their cost basis
      change_24h_cents       — portfolio vs the snapshot nearest 24h
                               ago (None until a day of snapshots
                               exists)

    Appends a throttled snapshot line to data/portfolio_snapshots.jsonl
    so the 24h delta has a baseline. Cached 60s.
    """
    import json as _json
    now = time.time()
    if _OVERVIEW_CACHE[1] is not None and now - _OVERVIEW_CACHE[0] < 60:
        return _OVERVIEW_CACHE[1]
    cash, _balance_err = get_balance_cents()
    value = cost = 0
    positions, _err = get_open_positions()
    for pos in positions or []:
        t = pos.get("ticker") or ""
        try:
            fp = float(pos.get("position_fp") or 0)
        except (TypeError, ValueError):
            continue
        if not t or not fp:
            continue
        mark = _position_mark_cents(t)
        if mark is None:
            continue
        n = abs(fp)
        # NO positions (fp < 0) are worth (100 − yes mark).
        per = mark if fp > 0 else (100 - mark)
        value += int(round(n * per))
        try:
            cost += int(round(
                float(pos.get("market_exposure_dollars") or 0) * 100))
        except (TypeError, ValueError):
            pass
    out = {
        "cash_cents": cash,
        "positions_value_cents": value,
        "portfolio_cents": (cash + value) if cash is not None else None,
        "unrealized_cents": value - cost,
        "change_24h_cents": None,
    }
    # Snapshot + 24h baseline (best-effort; file I/O never breaks a render)
    try:
        snap_p = Path(_SNAPSHOT_PATH)
        snaps = []
        if snap_p.exists():
            for line in snap_p.read_text().splitlines()[-2000:]:
                try:
                    snaps.append(_json.loads(line))
                except ValueError:
                    continue
        if out["portfolio_cents"] is not None and (
                not snaps or now - snaps[-1].get("ts", 0) >= 600):
            snap_p.parent.mkdir(parents=True, exist_ok=True)
            with open(snap_p, "a") as fh:
                fh.write(_json.dumps(
                    {"ts": int(now), "cash": cash,
                     "portfolio": out["portfolio_cents"]}) + "\n")
        target = now - 86400
        older = [sn for sn in snaps if sn.get("ts", 0) <= now - 72000]
        if older and out["portfolio_cents"] is not None:
            base = min(older, key=lambda sn: abs(sn.get("ts", 0) - target))
            out["change_24h_cents"] = (out["portfolio_cents"]
                                        - base.get("portfolio", 0))
    except OSError:
        pass
    _OVERVIEW_CACHE[0] = now
    _OVERVIEW_CACHE[1] = out
    return out


def market_cents(m: dict, name: str) -> Optional[int]:
    """Price field in cents, tolerant of the API's migration from int
    cents (``yes_ask``) to dollar strings (``yes_ask_dollars``)."""
    v = m.get(name)
    if v not in (None, 0):
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
    d = m.get(f"{name}_dollars")
    if d not in (None, "", "0", "0.0000"):
        try:
            return int(round(float(d) * 100))
        except (TypeError, ValueError):
            pass
    return None


def market_count(m: dict, name: str) -> Optional[float]:
    """Count field (open_interest / volume), tolerant of the ``_fp``
    string variants the API migrated to."""
    v = m.get(name)
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    fp = m.get(f"{name}_fp")
    if fp is not None:
        try:
            return float(fp)
        except (TypeError, ValueError):
            pass
    return None


def get_client() -> KalshiClient:
    """Process-wide singleton — keeps the loaded RSA key in memory."""
    global _cached
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is None:
            _cached = KalshiClient()
        return _cached


# ---------------------------------------------------------------------------
# Live account balance — drives the Size column's Kelly bankroll.
# ---------------------------------------------------------------------------
#
# Kalshi's /portfolio/balance is cheap (~80ms over the wire) but every
# dashboard page render would issue one call per panel without caching.
# We cache the last successful pull for 60 seconds — long enough to
# absorb a tab full of opened-at-once snapshot polls, short enough that
# a real deposit / withdrawal shows up on the next refresh after.
#
# Returns the *settled* balance in cents (Kalshi's "balance" field).
# We deliberately ignore the "payout" field (max possible win across
# open positions) — for Kelly sizing you want the cash you can deploy
# right now, not the optimistic upper bound on existing positions.

_BALANCE_CACHE: dict[str, Any] = {"ts": 0.0, "cents": None,
                                     "error": None}
_BALANCE_TTL_SECONDS = 60.0
_balance_lock = threading.Lock()


def get_balance_cents(force_refresh: bool = False
                       ) -> tuple[Optional[int], Optional[str]]:
    """Return (balance_cents, error_or_none).

    On success: (int cents, None).
    On no-creds / API failure: (None, "human-readable reason").

    Cached for 60 seconds. Pass ``force_refresh=True`` to bypass the
    cache (e.g. for a "Refresh balance" button in the future).
    """
    import time as _time
    now = _time.monotonic()
    with _balance_lock:
        cached = _BALANCE_CACHE.get("cents")
        ts = _BALANCE_CACHE.get("ts") or 0.0
        if (not force_refresh
                and cached is not None
                and now - ts < _BALANCE_TTL_SECONDS):
            return int(cached), None

    client = get_client()
    if not client.available:
        with _balance_lock:
            _BALANCE_CACHE["error"] = (
                "Kalshi API key not configured — Size uses default bankroll."
            )
        return None, _BALANCE_CACHE["error"]

    resp = client.get_balance()
    if not resp or "balance" not in resp:
        err = "Kalshi /portfolio/balance returned no balance field."
        with _balance_lock:
            _BALANCE_CACHE["error"] = err
        return None, err

    try:
        cents = int(resp["balance"])
    except (TypeError, ValueError) as e:
        err = f"Could not parse Kalshi balance ({e})."
        with _balance_lock:
            _BALANCE_CACHE["error"] = err
        return None, err

    with _balance_lock:
        _BALANCE_CACHE["ts"] = now
        _BALANCE_CACHE["cents"] = cents
        _BALANCE_CACHE["error"] = None
    return cents, None


# ---------------------------------------------------------------------------
# Open Kalshi positions — drives the "Active bets" section on live
# dashboards so held-position rows reflect actual Kalshi portfolio state
# instead of the bot's local sim_state (which can drift when the
# executor's reconciliation is slow or partial). Same cache pattern as
# the balance helper above — 30-second TTL so a fresh fill / close
# shows up on the next refresh without hammering the endpoint on every
# panel poll during a page load.
# ---------------------------------------------------------------------------

_POSITIONS_CACHE: dict[str, Any] = {"ts": 0.0, "positions": None,
                                       "error": None}
_POSITIONS_TTL_SECONDS = 30.0
_positions_lock = threading.Lock()


def get_open_positions(force_refresh: bool = False
                        ) -> tuple[Optional[List[Dict[str, Any]]],
                                    Optional[str]]:
    """Return (open_positions, error_or_none).

    ``open_positions`` is a list of ``/portfolio/positions`` entries
    filtered to non-zero ``position_fp``. Each entry keeps the raw
    Kalshi shape (``ticker``, ``position_fp``, ``market_exposure``,
    ``realized_pnl``, etc.). On no-creds / API failure returns
    ``(None, "reason")``.

    Cached for 30 seconds — long enough to absorb a tab full of panel
    polls, short enough that a fresh fill / close shows up on the next
    refresh.
    """
    import time as _time
    now = _time.monotonic()
    with _positions_lock:
        cached = _POSITIONS_CACHE.get("positions")
        ts = _POSITIONS_CACHE.get("ts") or 0.0
        if (not force_refresh
                and cached is not None
                and now - ts < _POSITIONS_TTL_SECONDS):
            return list(cached), None

    client = get_client()
    if not client.available:
        with _positions_lock:
            _POSITIONS_CACHE["error"] = (
                "Kalshi API key not configured — Active bets falls back "
                "to bot sim_state."
            )
        return None, _POSITIONS_CACHE["error"]

    try:
        resp = client._sdk._request(
            "GET", "/portfolio/positions",
            params={"limit": 500},
        )
    except Exception as exc:  # noqa: BLE001
        err = f"/portfolio/positions call failed ({exc})."
        with _positions_lock:
            _POSITIONS_CACHE["error"] = err
        return None, err

    raw = ((resp or {}).get("market_positions")
           or (resp or {}).get("positions") or [])
    open_only: List[Dict[str, Any]] = []
    for p in raw:
        try:
            v = float(p.get("position_fp") or p.get("position") or 0)
        except (TypeError, ValueError):
            v = 0.0
        if abs(v) > 0.0 and p.get("ticker"):
            open_only.append(p)

    with _positions_lock:
        _POSITIONS_CACHE["ts"] = now
        _POSITIONS_CACHE["positions"] = open_only
        _POSITIONS_CACHE["error"] = None
    return list(open_only), None


# ---------------------------------------------------------------------------
# Live "implied spot" for above-$X bot series — drives the Home grid's
# stale-forecast badge.
# ---------------------------------------------------------------------------
#
# For an "above $X" market priced at YES = 50¢, the market is saying
# "50/50 the underlying is above X" — i.e. underlying ≈ X. We find the
# strike where YES crosses 50¢ on the most-imminent event of a given
# series and return it as the market-implied current value of the
# underlying.
#
# Used by the natural-gas card to flag the bot's stored forecast when
# it has drifted away from the live Kalshi-implied price (typically a
# sign of EIA-feed lag or a missed retrain).
#
# Cached for 60s — the same cadence as the balance helper above. Long
# enough to absorb a tab full of card grids hitting it simultaneously,
# short enough that a real intra-day move shows up on the next refresh.

_IMPLIED_SPOT_CACHE: dict = {}
_IMPLIED_SPOT_TTL_SECONDS = 60.0
_implied_spot_lock = threading.Lock()


def _parse_threshold(ticker: str) -> Optional[float]:
    """Parse the numeric strike out of an above-$X ticker — accepts
    both the -T2.750 (top) and -B2.750 (bottom) conventions."""
    import re
    m = re.search(r"-[BT](\d+(?:\.\d+)?)$", ticker or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _yes_ask_cents(market: dict) -> Optional[int]:
    """Read yes_ask from either legacy int (yes_ask) or new float
    (yes_ask_dollars) — same convention the bots use."""
    legacy = market.get("yes_ask")
    if legacy not in (None, ""):
        try:
            return int(legacy)
        except (TypeError, ValueError):
            pass
    dollars = market.get("yes_ask_dollars")
    if dollars not in (None, ""):
        try:
            return int(round(float(dollars) * 100))
        except (TypeError, ValueError):
            pass
    return None


def get_implied_spot(series_ticker: str,
                       force_refresh: bool = False
                       ) -> tuple[Optional[float], Optional[str]]:
    """Return (implied_spot_usd, error_or_none) for a series.

    Finds the 50¢-crossover strike on the most-imminent open event of
    ``series_ticker``. Per-series cached for 60s so the home grid's
    bot-card pass doesn't issue duplicate Kalshi calls.

    Returns (None, "reason") when the SDK is unavailable, no markets
    are returned, or the YES-curve doesn't cross 50¢ in the visible
    strike range.
    """
    import time as _time
    now = _time.monotonic()
    with _implied_spot_lock:
        cached = _IMPLIED_SPOT_CACHE.get(series_ticker)
        if (not force_refresh and cached
                and now - cached.get("ts", 0.0) < _IMPLIED_SPOT_TTL_SECONDS):
            return cached.get("spot"), cached.get("error")

    client = get_client()
    if not client.available:
        return None, "Kalshi API key not configured."

    markets = client.list_markets(series_ticker=series_ticker, limit=400) or []
    if not markets:
        return None, f"No open {series_ticker} markets."

    # Group by event so we can pick the soonest-closing one — Kalshi
    # may return strikes across multiple events when a series has
    # overlap (e.g. daily natgas with both today + tomorrow open).
    by_event: dict[str, list] = {}
    for m in markets:
        ev = m.get("event_ticker") or ""
        by_event.setdefault(ev, []).append(m)

    def _event_close_ts(ms: list) -> float:
        for m in ms:
            ts = m.get("close_time") or m.get("expected_expiration_time")
            if ts:
                try:
                    from datetime import datetime as _dt
                    return _dt.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                except (TypeError, ValueError):
                    continue
        return float("inf")

    soonest = min(by_event.values(), key=_event_close_ts)

    # Build sorted (strike, yes_ask) pairs and find the 50¢ crossover.
    pairs: list[tuple[float, int]] = []
    for m in soonest:
        thr = _parse_threshold(m.get("ticker", ""))
        ya = _yes_ask_cents(m)
        if thr is not None and ya is not None:
            pairs.append((thr, ya))
    pairs.sort(key=lambda x: x[0])
    if len(pairs) < 2:
        return None, "Not enough priced strikes to interpolate spot."

    spot: Optional[float] = None
    for (lo_s, lo_y), (hi_s, hi_y) in zip(pairs, pairs[1:]):
        if lo_y >= 50 >= hi_y:
            span = lo_y - hi_y
            spot = (lo_s if span <= 0
                    else lo_s + ((lo_y - 50) / span) * (hi_s - lo_s))
            break
    if spot is None:
        # 50¢ outside the visible strike range — anchor to the
        # closer endpoint so the badge still has SOMETHING to
        # compare against rather than dashing out.
        lo_s, lo_y = pairs[0]
        hi_s, hi_y = pairs[-1]
        spot = float(lo_s) if lo_y < 50 else float(hi_s) if hi_y > 50 else None

    with _implied_spot_lock:
        _IMPLIED_SPOT_CACHE[series_ticker] = {
            "ts": now, "spot": spot,
            "error": None if spot is not None else "no crossover",
        }
    return spot, (None if spot is not None else "no 50¢ crossover")


# ============================================================================ #
# Module-level helpers used by the dashboard's render path.
#
# Pre-SDK-shim, these lived in the bespoke 26 KB kalshi_client.py module.
# The shim refactor moved the singleton + REST methods onto KalshiClient
# but didn't re-export the higher-level functions the dashboard's
# render_page() calls — `fetch_underlying_history` and friends. The
# dashboard's try/except around the call swallowed the AttributeError
# silently, leaving every bot's Watchlist tab empty / showing only stale
# local-DB data.
#
# The implementations below are restored ~as-was, with one change: the
# candlestick fetch goes through the shim's get_market_candlesticks
# (which forwards to kalshi_sdk) instead of the legacy `client.candlesticks`
# method that no longer exists.
# ============================================================================ #

import time as _time
from datetime import datetime as _dt

# In-process candlestick cache — 60s TTL is plenty given Kalshi
# updates each candle at the period boundary. Keeps simultaneous
# dashboard tabs from each issuing the same underlying-history fetch.
_CDL_CACHE: dict[tuple, tuple[float, list[dict]]] = {}
_CDL_TTL_SECONDS = 60.0


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_iso(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        return _dt.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def pick_atm_market(markets: List[dict]) -> Optional[dict]:
    """Return the market whose YES probability is closest to 50¢.

    The ATM strike is where chart movement carries the most info — a
    30→70¢ swing on a 50% market matters more than 1→2¢ on a tail.
    Falls back to highest-OI when no market has a price. Reads both
    legacy (yes_bid/yes_ask integer cents) and the new
    yes_bid_dollars/yes_ask_dollars float fields.
    """
    if not markets:
        return None

    def _mid(m: dict) -> Optional[float]:
        # Prefer dollars (new schema), fall back to cents (legacy).
        ya = _to_float(m.get("yes_ask_dollars"))
        yb = _to_float(m.get("yes_bid_dollars"))
        if ya is None and yb is None:
            ya_c = _to_float(m.get("yes_ask"))
            yb_c = _to_float(m.get("yes_bid"))
            if ya_c is not None:
                ya = ya_c / 100.0
            if yb_c is not None:
                yb = yb_c / 100.0
        if ya is None and yb is None:
            return None
        if ya is not None and yb is not None:
            return (ya + yb) / 2.0
        return ya if ya is not None else yb

    scored: List[Tuple[float, dict]] = []
    for m in markets:
        mid = _mid(m)
        if mid is not None:
            scored.append((abs(mid - 0.5), m))
    if not scored:
        for m in markets:
            oi = (_to_float(m.get("open_interest_fp"))
                  or _to_float(m.get("open_interest"))
                  or 0.0)
            scored.append((-oi, m))
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _candle_yes_prob(cdl: dict) -> Optional[float]:
    """Pull the cleanest YES probability (0..1) from one candlestick."""
    price = cdl.get("price") or {}
    yp = (_to_float(price.get("mean_dollars"))
          or _to_float(price.get("close_dollars")))
    if yp is not None:
        return yp
    ya = cdl.get("yes_ask") or {}
    yb = cdl.get("yes_bid") or {}
    ya_c = _to_float(ya.get("close_dollars"))
    yb_c = _to_float(yb.get("close_dollars"))
    if ya_c is not None and yb_c is not None:
        return (ya_c + yb_c) / 2.0
    return ya_c if ya_c is not None else yb_c


def _candlesticks(client: KalshiClient, series_ticker: str, ticker: str,
                   period_minutes: int, lookback_hours: int
                   ) -> List[dict]:
    """Fetch + cache a single market's candlesticks.

    Adapts the dashboard's old (period_minutes, lookback_hours) shape
    to the SDK's (start_ts, end_ts, period_interval). period_interval
    is constrained to {1, 60, 1440} by the API; pick the closest
    supported value.
    """
    if period_minutes >= 1440:
        api_period = 1440
    elif period_minutes >= 60:
        api_period = 60
    else:
        api_period = 1
    end_ts = int(_time.time())
    start_ts = end_ts - max(1, lookback_hours) * 3600
    cache_key = (ticker, series_ticker, api_period, start_ts // 60,
                 end_ts // 60)
    now = _time.time()
    cached = _CDL_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CDL_TTL_SECONDS:
        return cached[1]
    resp = client.get_market_candlesticks(
        ticker=ticker, series_ticker=series_ticker,
        start_ts=start_ts, end_ts=end_ts, period_interval=api_period,
    )
    if not resp:
        return []
    cdls = resp.get("candlesticks") or resp.get("candles") or []
    if isinstance(cdls, list):
        _CDL_CACHE[cache_key] = (now, cdls)
        return cdls
    return []


def _interpolate_event_history(
    client: KalshiClient,
    series_ticker: str,
    event_markets: List[dict],
    anchor_value: Optional[float],
    period_minutes: int,
    lookback_hours: int,
    max_strikes: int = 6,
) -> List[dict]:
    """Strike-ladder → implied-underlying interpolation per timestamp.

    For each candle ts at which ≥2 strikes reported a YES probability,
    find the strike where YES crosses 50% (linear interpolation between
    the bracketing strikes). That value is the implied underlying for
    the chart line.
    """
    scored: List[Tuple[float, dict]] = []
    for m in event_markets:
        floor = m.get("floor_strike")
        if floor is None:
            continue
        try:
            strike = float(floor)
        except (TypeError, ValueError):
            continue
        if anchor_value is not None:
            scored.append((abs(strike - anchor_value), m))
            continue
        ya = _to_float(m.get("yes_ask_dollars"))
        yb = _to_float(m.get("yes_bid_dollars"))
        if ya is None and yb is None:
            continue
        if ya is not None and yb is not None:
            mid = (ya + yb) / 2.0
        else:
            mid = ya if ya is not None else yb
        scored.append((abs(mid - 0.5), m))
    scored.sort(key=lambda x: x[0])
    picked = [m for _, m in scored[:max_strikes]]
    if len(picked) < 2:
        return []

    bucket_seconds = max(1, period_minutes) * 60
    market_data: List[Tuple[float, dict]] = []
    for m in picked:
        try:
            strike = float(m["floor_strike"])
        except (TypeError, ValueError):
            continue
        cdls = _candlesticks(client, series_ticker, m["ticker"],
                              period_minutes=period_minutes,
                              lookback_hours=lookback_hours)
        ts_to_yes: dict[float, float] = {}
        if period_minutes >= 60 or period_minutes <= 1:
            for cdl in cdls:
                ts = cdl.get("end_period_ts")
                yp = _candle_yes_prob(cdl)
                if ts is None or yp is None:
                    continue
                ts_to_yes[float(ts)] = yp
        else:
            buckets: dict[int, Tuple[float, float]] = {}
            for cdl in cdls:
                ts = cdl.get("end_period_ts")
                yp = _candle_yes_prob(cdl)
                if ts is None or yp is None:
                    continue
                ts_f = float(ts)
                bucket = int(ts_f // bucket_seconds)
                cur = buckets.get(bucket)
                if cur is None or ts_f > cur[0]:
                    buckets[bucket] = (ts_f, float(yp))
            for bucket, (ts_f, yp) in buckets.items():
                snapped = bucket * bucket_seconds + bucket_seconds
                ts_to_yes[float(snapped)] = yp
        market_data.append((strike, ts_to_yes))

    if not market_data:
        return []
    all_ts = sorted({t for _, ts_map in market_data
                     for t in ts_map.keys()})
    history: List[dict] = []
    for ts in all_ts:
        pairs: List[Tuple[float, float]] = []
        for strike, ts_map in market_data:
            yp = ts_map.get(ts)
            if yp is None:
                continue
            pairs.append((strike, yp))
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda x: x[0])
        implied: Optional[float] = None
        for i in range(len(pairs) - 1):
            s1, p1 = pairs[i]
            s2, p2 = pairs[i + 1]
            if (p1 >= 0.5 >= p2) or (p2 >= 0.5 >= p1):
                if p1 == p2:
                    implied = (s1 + s2) / 2
                else:
                    t = (p1 - 0.5) / (p1 - p2)
                    implied = s1 + t * (s2 - s1)
                break
        if implied is None:
            if pairs[-1][1] >= 0.5:
                implied = pairs[-1][0]
            elif pairs[0][1] < 0.5:
                implied = pairs[0][0]
        if implied is not None:
            history.append({"ts": float(ts), "value": float(implied)})
    return history


def _pick_current_event_markets(
    markets: List[dict],
    extra_tickers: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Filter ``markets`` down to the single "current" event in the series.

    Kalshi often has overlapping events open simultaneously — for CPI
    a typical day might have April / May / June markets all listed
    open, because Kalshi pre-publishes the upcoming months. The
    Watchlist table is meant for ONE contract at a time: the one most
    imminently going to settle. When that one closes, the next month
    naturally becomes the earliest-future event and rolls forward.

    Selection rule:
      1. Group by ``event_ticker``.
      2. Among events with at least one market whose ``close_time`` is
         still in the future, pick the one with the earliest
         ``close_time`` (most-imminent-but-not-yet-closed).
      3. If no future close_times are available (e.g., test fixture),
         fall back to the event with the highest aggregate open
         interest. This is a safety net — production Kalshi data
         always has close_time set.

    Additionally, any event that contains a market whose ticker is in
    ``extra_tickers`` is always included. This is used by the Watchlist
    so that markets we hold open positions on stay visible even if
    they sit on a non-imminent event (e.g. NBA games scheduled later
    in the week, where the most-imminent event is tonight's game).

    Returns markets belonging to the chosen event(s), in the same order
    they appeared in the input. Empty list maps to empty output.
    """
    if not markets:
        return []
    by_event: dict[str, List[dict]] = {}
    for m in markets:
        et = m.get("event_ticker")
        if not et:
            continue
        by_event.setdefault(et, []).append(m)
    if not by_event:
        # No event_ticker on any market — surface them all rather than
        # hiding everything; the upstream fetch already scopes to one
        # series so this is conservative.
        return list(markets)

    now_ts = _time.time()

    def _earliest_future_close(group: List[dict]) -> Optional[float]:
        ts: List[float] = []
        for m in group:
            t = _parse_iso(m.get("close_time"))
            if t is not None and t > now_ts:
                ts.append(t)
        return min(ts) if ts else None

    def _total_oi(group: List[dict]) -> float:
        oi = 0.0
        for m in group:
            v = (_to_float(m.get("open_interest_fp"))
                 or _to_float(m.get("open_interest")) or 0.0)
            oi += v
        return oi

    scored: List[Tuple[float, str]] = []
    for et, group in by_event.items():
        close = _earliest_future_close(group)
        if close is not None:
            # Earliest future close wins; use the timestamp itself as
            # the sort key (smaller = more imminent).
            scored.append((close, et))
    if scored:
        scored.sort(key=lambda x: x[0])
        chosen = scored[0][1]
    else:
        # Fallback: pick by aggregate OI.
        chosen = max(by_event.keys(), key=lambda e: _total_oi(by_event[e]))

    chosen_events: set[str] = {chosen}
    extras = set(extra_tickers or ())
    if extras:
        for et, group in by_event.items():
            if any((m.get("ticker") in extras) for m in group):
                chosen_events.add(et)

    out: List[dict] = []
    for m in markets:
        if m.get("event_ticker") in chosen_events:
            out.append(m)
    return out


def fetch_underlying_history(
    series_ticker: str,
    period_minutes: int = 60,
    lookback_hours: Optional[int] = None,
    client: Optional[KalshiClient] = None,
    max_strikes: int = 6,
    extra_tickers: Optional[Iterable[str]] = None,
    all_open_events: bool = False,
) -> Tuple[List[dict], Optional[dict], List[dict],
            Optional[float], Optional[float], Optional[str]]:
    """Build the chart + watchlist payload for one series.

    Returns
    -------
    history : list[{ts, value}]
        Implied-underlying timeseries derived from the strike ladder.
    atm_market : dict or None
        The ATM market (used as the chart anchor).
    markets : list[dict]
        Markets feeding the Watchlist table. By default this is just
        the current event (gas / CPI / unemployment-style series have
        only one contract open at a time). When ``all_open_events`` is
        true — NBA and other sport series where many concurrent games
        are open simultaneously — every market with a future
        ``close_time`` is returned.
    contract_open_ts : float or None
        Unix epoch of the current event's open. Chart x-axis start.
    contract_close_ts : float or None
        Unix epoch of the current event's close.
    event_title : str or None
        Human-readable event title for the chart header.
    """
    c = client or get_client()
    if not c.available:
        return [], None, [], None, None, None
    raw_markets = c.list_markets(series_ticker=series_ticker)
    if not raw_markets:
        return [], None, raw_markets, None, None, None
    # The chart always anchors to ONE event (the most-imminent close,
    # with ``extra_tickers`` keeping events containing open positions
    # visible too). For sport series like KXNBAGAME where every game
    # is its own concurrently-open event, the watchlist needs every
    # one of them — narrowing to one event would hide the rest of
    # tonight's slate.
    chart_markets = _pick_current_event_markets(
        raw_markets, extra_tickers=extra_tickers,
    )
    if all_open_events:
        now_ts = _time.time()
        watchlist_markets = [
            m for m in raw_markets
            if (_parse_iso(m.get("close_time")) or 0) > now_ts
        ]
        # If nothing has a future close_time (test fixtures / odd
        # state), fall back to the chart-side picker output so the
        # watchlist still renders something.
        if not watchlist_markets:
            watchlist_markets = chart_markets
    else:
        watchlist_markets = chart_markets
    atm = pick_atm_market(chart_markets)
    contract_open_ts = _parse_iso(atm.get("open_time")) if atm else None
    contract_close_ts = _parse_iso(atm.get("close_time")) if atm else None
    event_title: Optional[str] = None
    if atm and atm.get("event_ticker"):
        event = c.get_event(atm["event_ticker"])
        if event:
            event_title = event.get("title")

    if lookback_hours is None:
        if contract_open_ts is not None:
            elapsed_h = max(1, int((_time.time() - contract_open_ts) / 3600) + 1)
            lookback_hours = min(elapsed_h, 7 * 24)
        else:
            lookback_hours = 24

    # Daily candles need ≥48h elapsed before they yield more than 1-2
    # points; auto-downgrade to hourly so a fresh contract still has a
    # chart line.
    effective_period = period_minutes
    if period_minutes >= 1440 and lookback_hours < 48:
        effective_period = 60

    history = _interpolate_event_history(
        c, series_ticker, chart_markets, anchor_value=None,
        period_minutes=effective_period,
        lookback_hours=lookback_hours,
        max_strikes=max_strikes,
    )
    return (history, atm, watchlist_markets, contract_open_ts,
            contract_close_ts, event_title)


def fetch_event_metadata(
    series_ticker: str,
    client: Optional[KalshiClient] = None,
) -> Tuple[Optional[dict], List[dict],
            Optional[float], Optional[float], Optional[str]]:
    """Slim metadata-only counterpart to fetch_underlying_history.

    Skips the per-strike candlestick fetches; useful when the chart is
    sourced from the bot's recorded snapshots and the dashboard only
    needs the event framing (ATM, open/close, title) + market list.
    """
    c = client or get_client()
    if not c.available:
        return None, [], None, None, None
    raw_markets = c.list_markets(series_ticker=series_ticker)
    if not raw_markets:
        return None, [], None, None, None
    markets = _pick_current_event_markets(raw_markets)
    atm = pick_atm_market(markets)
    contract_open_ts = _parse_iso(atm.get("open_time")) if atm else None
    contract_close_ts = _parse_iso(atm.get("close_time")) if atm else None
    event_title: Optional[str] = None
    if atm and atm.get("event_ticker"):
        event = c.get_event(atm["event_ticker"])
        if event:
            event_title = event.get("title")
    return atm, markets, contract_open_ts, contract_close_ts, event_title


__all__ = ["KalshiClient", "KalshiError", "get_client",
           "fetch_underlying_history", "fetch_event_metadata",
           "pick_atm_market"]
