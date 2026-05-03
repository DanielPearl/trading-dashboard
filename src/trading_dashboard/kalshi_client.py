"""Slim signed-REST client for Kalshi's public market API.

The dashboard uses this to fetch candlestick history for each bot's
at-the-money market and to discover the active markets in each series.
We only ever GET — never write — and only ever read public market data
that the dashboard would render, never account-scoped data.

Auth: same RSA-PSS scheme the bots use. The signature is over
`{millis}{method}{path}` where `path` is the bare endpoint path WITHOUT
the `/trade-api/v2` prefix and WITHOUT any query string.

Caching: tiny in-process TTL cache. Each candlesticks fetch costs ~1s
of round-trip + Kalshi-side query, so caching for 60s keeps page loads
fast and stays well under any plausible rate limit.
"""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

log = logging.getLogger("dashboard.kalshi")

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class _CacheEntry:
    fetched_at: float
    value: Any


class _TTLCache:
    """Process-wide TTL cache; thread-safe lookups."""
    def __init__(self, ttl_seconds: float = 60.0):
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._store: Dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if now - entry.fetched_at > self.ttl:
                self._store.pop(key, None)
                return None
            return entry.value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = _CacheEntry(fetched_at=time.time(), value=value)


# 60-min candles only refresh once per hour, so a 5-minute cache hits
# ~12× fewer API calls than the 60s default and stays well under
# Kalshi's per-IP rate limit. Markets list is also cached here.
_CACHE = _TTLCache(ttl_seconds=300.0)


class KalshiClient:
    """Minimal signed Kalshi REST client. Loads creds from env vars on
    first use and never re-reads — the systemd unit owns the values.

        KALSHI_API_KEY_ID         — public key id (uuid-ish string)
        KALSHI_PRIVATE_KEY_PATH   — absolute path to the .pem private key

    If either is missing or the cryptography lib isn't installed, every
    method short-circuits and returns an empty result. The dashboard's
    chart renderer treats that as "no data" and shows the empty-state
    placeholder, so the page never hard-fails on missing creds.
    """

    def __init__(self) -> None:
        self._priv: Optional[Any] = None
        self._key_id: Optional[str] = None
        self._init_attempted = False
        self._available = False

    def _init(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        if requests is None or not HAS_CRYPTO:
            log.info("kalshi_client: requests/cryptography not installed; "
                     "live Kalshi data disabled")
            return
        key_id = os.getenv("KALSHI_API_KEY_ID")
        priv_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not key_id or not priv_path:
            log.info("kalshi_client: KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH "
                     "not set; live Kalshi data disabled")
            return
        try:
            key_bytes = Path(priv_path).expanduser().read_bytes()
            self._priv = serialization.load_pem_private_key(
                key_bytes, password=None,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("kalshi_client: failed to load private key: %s", e)
            return
        self._key_id = key_id
        self._available = True
        log.info("kalshi_client: ready (key id %s…)", key_id[:8])

    @property
    def available(self) -> bool:
        self._init()
        return self._available

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method}{path}".encode()
        sig = base64.b64encode(self._priv.sign(  # type: ignore[union-attr]
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                         salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )).decode()
        return {
            "KALSHI-ACCESS-KEY": self._key_id or "",
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "Accept": "application/json",
        }

    def _get(self, path: str, params: Optional[dict] = None,
             timeout: float = 8.0) -> Optional[dict]:
        self._init()
        if not self._available:
            return None
        try:
            r = requests.get(
                BASE_URL + path,
                headers=self._headers("GET", path),
                params=params or {},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("kalshi_client GET %s failed: %s", path, e)
            return None

    # ---------------------------------------------------------------- #
    # Endpoints we care about                                          #
    # ---------------------------------------------------------------- #

    def list_markets(self, series_ticker: Optional[str] = None,
                     event_ticker: Optional[str] = None,
                     status: Optional[str] = "open",
                     limit: int = 200) -> List[dict]:
        """Markets in a series (or one event). Cached 5min.

        If ``event_ticker`` is given, scopes to that one event regardless
        of status. Otherwise filters by ``series_ticker`` + ``status``.

        Each market dict has the canonical Kalshi fields:
            ticker, event_ticker, floor_strike, cap_strike,
            yes_ask_dollars, yes_bid_dollars, no_ask_dollars,
            volume_fp, open_interest_fp, last_price_dollars,
            expiration_value, expected_expiration_time, ...
        """
        if event_ticker:
            cache_key = f"markets-evt:{event_ticker}:{limit}"
        else:
            cache_key = f"markets:{series_ticker}:{status}:{limit}"
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached
        out: List[dict] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"limit": limit}
            if event_ticker:
                params["event_ticker"] = event_ticker
            else:
                if series_ticker:
                    params["series_ticker"] = series_ticker
                if status:
                    params["status"] = status
            if cursor:
                params["cursor"] = cursor
            resp = self._get("/markets", params=params)
            if not resp:
                break
            out.extend(resp.get("markets") or [])
            cursor = resp.get("cursor") or None
            if not cursor:
                break
        _CACHE.put(cache_key, out)
        return out

    def get_event(self, event_ticker: str) -> Optional[dict]:
        """Fetch a single event's metadata. Cached.

        We need the event's `title` ("Initial jobless claims for the
        week ending May 2, 2026?", "Natural gas price on May 04, 2026
        at 5:00 PM EDT?", etc.) to label the chart. The market's own
        `title` is more verbose ("Will the natural gas close price be
        above 2.750 USD/MMBtu on …?") so we prefer the event title.
        """
        cache_key = f"event:{event_ticker}"
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached
        resp = self._get(f"/events/{event_ticker}")
        if not resp:
            return None
        event = resp.get("event") or resp
        _CACHE.put(cache_key, event)
        return event

    def list_events(self, series_ticker: str,
                    statuses: Tuple[str, ...] = ("open", "settled"),
                    limit: int = 50) -> List[dict]:
        """Events in a series across the given status filter, most-
        recent first. Cached 5min. Used to assemble multi-event history
        windows (e.g. 5-day rolling charts on daily-cycle bots).
        """
        cache_key = f"events:{series_ticker}:{','.join(statuses)}:{limit}"
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached
        out: List[dict] = []
        for status in statuses:
            cursor: Optional[str] = None
            while True:
                params: Dict[str, Any] = {
                    "series_ticker": series_ticker,
                    "status": status,
                    "limit": limit,
                }
                if cursor:
                    params["cursor"] = cursor
                resp = self._get("/events", params=params)
                if not resp:
                    break
                out.extend(resp.get("events") or [])
                cursor = resp.get("cursor") or None
                if not cursor:
                    break
        _CACHE.put(cache_key, out)
        return out

    def candlesticks(self, series_ticker: str, market_ticker: str,
                     period_minutes: int = 60,
                     lookback_hours: int = 72) -> List[dict]:
        """Historical candlesticks for one market. Cached 60s per (market,
        period). Each candle has end_period_ts (unix seconds) and price /
        yes_ask / yes_bid sub-dicts with close_dollars / mean_dollars.
        """
        cache_key = f"candles:{market_ticker}:{period_minutes}:{lookback_hours}"
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached
        end_ts = int(time.time())
        start_ts = end_ts - lookback_hours * 3600
        path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        params = {
            "period_interval": period_minutes,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }
        resp = self._get(path, params=params)
        if not resp:
            return []
        candles = resp.get("candlesticks") or []
        _CACHE.put(cache_key, candles)
        return candles


_DEFAULT_CLIENT: Optional[KalshiClient] = None
_CLIENT_LOCK = threading.Lock()


def default_client() -> KalshiClient:
    """Singleton accessor — used by the chart fetcher."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        with _CLIENT_LOCK:
            if _DEFAULT_CLIENT is None:
                _DEFAULT_CLIENT = KalshiClient()
    return _DEFAULT_CLIENT


# --------------------------------------------------------------------------- #
# High-level helpers used by the dashboard's chart render
# --------------------------------------------------------------------------- #

def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pick_atm_market(markets: List[dict]) -> Optional[dict]:
    """Pick the at-the-money market: the one whose YES probability is
    closest to 50¢. That's the strike whose chart movement carries the
    most information about the underlying — moving 30¢→70¢ on a 50%
    market matters more than 1¢→2¢ on a 1% tail.
    """
    if not markets:
        return None
    scored: List[Tuple[float, dict]] = []
    for m in markets:
        ya = _to_float(m.get("yes_ask_dollars"))
        yb = _to_float(m.get("yes_bid_dollars"))
        if ya is None and yb is None:
            continue
        # Mid of bid/ask if both, else single side.
        if ya is not None and yb is not None:
            mid = (ya + yb) / 2.0
        else:
            mid = ya if ya is not None else yb  # type: ignore[assignment]
        scored.append((abs(mid - 0.5), m))
    if not scored:
        # Fallback: highest open interest.
        for m in markets:
            oi = _to_float(m.get("open_interest_fp")) or 0.0
            scored.append((-oi, m))
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _candle_yes_prob(cdl: dict) -> Optional[float]:
    """Pull the cleanest YES probability (0..1) from one candlestick.
    Prefers the mean trade price (smoothest), falls back to close price,
    then to bid/ask midpoint for thin bars without trades.
    """
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


def _interpolate_event_history(client: "KalshiClient",
                                series_ticker: str,
                                event_markets: List[dict],
                                anchor_value: Optional[float],
                                period_minutes: int,
                                lookback_hours: int,
                                max_strikes: int = 6,
                                ) -> List[dict]:
    """Strike-ladder interpolation for one event, producing one (ts,
    value) point per candle timestamp where at least 2 markets reported
    a YES probability.

    `anchor_value`: for OPEN events pass None — markets are picked by
    current YES≈50% (the ATM heuristic). For SETTLED events pass the
    expiration_value so we pick markets whose strikes bracketed the
    eventual settlement (those carry the most informative intraday
    price action — they're where the market thought 50% was).
    """
    scored: List[Tuple[float, dict]] = []
    for m in event_markets:
        if m.get("floor_strike") is None:
            continue
        try:
            strike = float(m["floor_strike"])
        except (TypeError, ValueError):
            continue
        if anchor_value is not None:
            # Settled event — score by distance from settlement value.
            scored.append((abs(strike - anchor_value), m))
            continue
        ya = _to_float(m.get("yes_ask_dollars"))
        yb = _to_float(m.get("yes_bid_dollars"))
        mid = None
        if ya is not None and yb is not None:
            mid = (ya + yb) / 2.0
        elif ya is not None:
            mid = ya
        elif yb is not None:
            mid = yb
        if mid is None:
            continue
        scored.append((abs(mid - 0.5), m))
    scored.sort(key=lambda x: x[0])
    picked = [m for _, m in scored[:max_strikes]]
    if len(picked) < 2:
        return []

    # Kalshi's candlesticks API only supports period_interval ∈ {1,60,
    # 1440}. Three resolution paths:
    #   * >=1440 → daily bars from Kalshi (1:1, one point per day)
    #   * >=60   → hourly bars from Kalshi (1:1)
    #   * <60    → fetch 1-min bars and bucket down to the target window
    if period_minutes >= 1440:
        api_period = 1440
    elif period_minutes >= 60:
        api_period = 60
    else:
        api_period = 1
    bucket_seconds = max(1, period_minutes) * 60

    market_data: List[Tuple[float, Dict[float, float]]] = []
    for m in picked:
        strike = float(m["floor_strike"])
        cache_key = f"candles:{m['ticker']}:{api_period}:{lookback_hours}"
        was_cached = _CACHE.get(cache_key) is not None
        candles = client.candlesticks(
            series_ticker, m["ticker"],
            period_minutes=api_period, lookback_hours=lookback_hours,
        )
        if not was_cached:
            time.sleep(0.1)
        ts_to_yes: Dict[float, float] = {}
        if api_period in (60, 1440) or period_minutes <= 1:
            # 1:1 mapping — every candle is one data point.
            for cdl in candles:
                ts = cdl.get("end_period_ts")
                yp = _candle_yes_prob(cdl)
                if ts is None or yp is None:
                    continue
                ts_to_yes[float(ts)] = yp
        else:
            # Bucket 1-min candles into N-min windows. Take the LAST
            # candle in each bucket as the bucket's representative
            # value (matches how Kalshi displays close-of-period).
            buckets: Dict[int, Tuple[float, float]] = {}
            for cdl in candles:
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
                # Snap to bucket boundary so different markets'
                # bucketed timestamps line up exactly during
                # cross-strike interpolation.
                snapped = bucket * bucket_seconds + bucket_seconds
                ts_to_yes[float(snapped)] = yp
        market_data.append((strike, ts_to_yes))

    if not market_data:
        return []

    all_ts = sorted({t for _, ts_map in market_data for t in ts_map.keys()})
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
            # Tail of the distribution — clamp to nearest strike.
            if pairs[-1][1] >= 0.5:
                implied = pairs[-1][0]
            elif pairs[0][1] < 0.5:
                implied = pairs[0][0]
        if implied is not None:
            history.append({"ts": float(ts), "value": float(implied)})
    return history



def _parse_iso(ts: str | None) -> Optional[float]:
    if not ts:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def fetch_underlying_history(series_ticker: str,
                              period_minutes: int = 60,
                              lookback_hours: Optional[int] = None,
                              client: Optional[KalshiClient] = None,
                              max_strikes: int = 6,
                              ) -> Tuple[List[dict], Optional[dict], List[dict], Optional[float], Optional[float], Optional[str]]:
    """Derive the implied underlying price for one ticker group from the
    full strike ladder.

    For each candle timestamp we have (strike, YES probability) pairs
    across every market in the series. The implied underlying is the
    strike where YES probability crosses 50% (linear interpolation
    between the bracketing strikes). That mirrors the chart Kalshi
    displays at the top of every market page — same data source, just
    derived rather than fetched directly.

    Returns ``(history, atm_market, markets, contract_open_ts, contract_close_ts, event_title)``:
      * history — list of ``{"ts": float, "value": float}`` points, oldest
        first. Y-values are in the underlying's native units (USD/MMBtu
        for natgas, USD/gal for retail gas, raw count for claims).
      * atm_market — the market closest to the money right now (used by
        the chart title / watchlist "Chance" highlight).
      * markets — the full open-market list for this series, so the
        caller can render a Kalshi-derived watchlist when the local
        DB is empty.
      * contract_open_ts — unix timestamp of when the current event
        opened. The chart uses this as the x-axis start so the line
        spans the entire contract life, not just the lookback window.

    Limited to the ``max_strikes`` closest-to-money markets to keep API
    load bounded. Each market's candlesticks call is cached for 60s.
    """
    c = client or default_client()
    if not c.available:
        return [], None, [], None, None, None
    markets = c.list_markets(series_ticker)
    if not markets:
        return [], None, markets, None, None, None
    atm = pick_atm_market(markets)
    # Contract open + close times — same for any market in the event.
    # Chart x-axis spans [open, close] so the user can see the full
    # contract duration with the line filling the elapsed portion.
    contract_open_ts = _parse_iso(atm.get("open_time")) if atm else None
    contract_close_ts = _parse_iso(atm.get("close_time")) if atm else None
    # Event title — used as the chart title (matches Kalshi's market
    # page header, e.g. "Initial jobless claims for the week ending
    # May 2, 2026?").
    event_title: Optional[str] = None
    if atm and atm.get("event_ticker"):
        event = c.get_event(atm["event_ticker"])
        if event:
            event_title = event.get("title")
    # Auto-size lookback to span the current event's life. Cap at 7
    # days so Kalshi doesn't reject an absurdly long range with a fine
    # period_interval.
    if lookback_hours is None:
        if contract_open_ts is not None:
            elapsed_h = max(1, int((time.time() - contract_open_ts) / 3600) + 1)
            lookback_hours = min(elapsed_h, 7 * 24)
        else:
            lookback_hours = 24

    # Daily candles need ≥48h of elapsed history to yield more than
    # 1-2 useful points. Below that, auto-downgrade to hourly so the
    # chart isn't empty for fresh contracts. Weekly contracts that
    # outlive 48h naturally re-promote to daily on the next render.
    effective_period = period_minutes
    if period_minutes >= 1440 and lookback_hours < 48:
        effective_period = 60

    history = _interpolate_event_history(
        c, series_ticker, markets, anchor_value=None,
        period_minutes=effective_period, lookback_hours=lookback_hours,
        max_strikes=max_strikes,
    )
    return history, atm, markets, contract_open_ts, contract_close_ts, event_title


def fetch_event_metadata(series_ticker: str,
                          client: Optional[KalshiClient] = None,
                          ) -> Tuple[Optional[dict], List[dict], Optional[float], Optional[float], Optional[str]]:
    """Slim metadata-only fetch for the currently-open event in a series.

    Same payload as ``fetch_underlying_history`` minus the candle history
    — useful when the chart is sourced from the bot's own recorded
    snapshots rather than from the Kalshi strike ladder. Skips the
    expensive per-strike candlestick fetches.

    Returns ``(atm_market, markets, contract_open_ts, contract_close_ts,
    event_title)`` — the same metadata fields the dashboard needs to
    frame the chart (x-axis span, contract title, ATM strike).
    """
    c = client or default_client()
    if not c.available:
        return None, [], None, None, None
    markets = c.list_markets(series_ticker)
    if not markets:
        return None, [], None, None, None
    atm = pick_atm_market(markets)
    contract_open_ts = _parse_iso(atm.get("open_time")) if atm else None
    contract_close_ts = _parse_iso(atm.get("close_time")) if atm else None
    event_title: Optional[str] = None
    if atm and atm.get("event_ticker"):
        event = c.get_event(atm["event_ticker"])
        if event:
            event_title = event.get("title")
    return atm, markets, contract_open_ts, contract_close_ts, event_title
