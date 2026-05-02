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

    def list_markets(self, series_ticker: str, status: str = "open",
                     limit: int = 200) -> List[dict]:
        """Returns the current open markets in a series. Cached 60s.

        Each market dict has the canonical Kalshi fields:
            ticker, event_ticker, floor_strike, cap_strike,
            yes_ask_dollars, yes_bid_dollars, no_ask_dollars,
            volume_fp, open_interest_fp, last_price_dollars, ...
        """
        cache_key = f"markets:{series_ticker}:{status}:{limit}"
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached
        out: List[dict] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": limit,
            }
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
                              ) -> Tuple[List[dict], Optional[dict], List[dict], Optional[float]]:
    """Derive the implied underlying price for one ticker group from the
    full strike ladder.

    For each candle timestamp we have (strike, YES probability) pairs
    across every market in the series. The implied underlying is the
    strike where YES probability crosses 50% (linear interpolation
    between the bracketing strikes). That mirrors the chart Kalshi
    displays at the top of every market page — same data source, just
    derived rather than fetched directly.

    Returns ``(history, atm_market, markets, contract_open_ts)``:
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
        return [], None, [], None
    markets = c.list_markets(series_ticker)
    if not markets:
        return [], None, markets, None
    atm = pick_atm_market(markets)
    # Contract open time — ATM is in the current event so any market in
    # that event reports the same open_time. Chart x-axis starts here.
    contract_open_ts = _parse_iso(atm.get("open_time")) if atm else None
    # Auto-size lookback to cover the contract's full life so far. Cap
    # at 7 days; Kalshi rejects very long ranges with very fine periods.
    if lookback_hours is None:
        if contract_open_ts is not None:
            elapsed_h = max(1, int((time.time() - contract_open_ts) / 3600) + 1)
            lookback_hours = min(elapsed_h, 7 * 24)
        else:
            lookback_hours = 24
    # Pick the top-N markets nearest the ATM by current YES probability.
    # Anything far from 50% has no useful information for interpolation
    # — the strike where YES≈50% is what determines the implied price.
    scored: List[Tuple[float, dict]] = []
    for m in markets:
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
        if m.get("floor_strike") is None:
            continue
        scored.append((abs(mid - 0.5), m))
    scored.sort(key=lambda x: x[0])
    picked = [m for _, m in scored[:max_strikes]]
    if len(picked) < 2:
        return [], atm, markets, contract_open_ts

    # Pull candles for each picked market. Cached 5 minutes per market
    # so the same lookup costs ~0 on subsequent renders. Cache misses
    # space the API calls 100ms apart to stay under Kalshi's per-IP
    # rate limit (we've seen 429s when bursting 30+ calls in a second).
    market_data: List[Tuple[float, Dict[float, float]]] = []
    for m in picked:
        try:
            strike = float(m["floor_strike"])
        except (TypeError, ValueError):
            continue
        cache_key = (f"candles:{m['ticker']}:{period_minutes}:{lookback_hours}")
        was_cached = _CACHE.get(cache_key) is not None
        candles = c.candlesticks(
            series_ticker, m["ticker"],
            period_minutes=period_minutes, lookback_hours=lookback_hours,
        )
        if not was_cached:
            time.sleep(0.1)
        ts_to_yes: Dict[float, float] = {}
        for cdl in candles:
            ts = cdl.get("end_period_ts")
            yp = _candle_yes_prob(cdl)
            if ts is None or yp is None:
                continue
            ts_to_yes[float(ts)] = yp
        market_data.append((strike, ts_to_yes))

    if not market_data:
        return [], atm, markets, contract_open_ts

    # Union of timestamps observed across all markets.
    all_ts = sorted({t for _, ts_map in market_data for t in ts_map.keys()})
    history: List[dict] = []
    for ts in all_ts:
        # Build (strike, yes_prob) pairs at this ts, sorted by strike asc.
        pairs: List[Tuple[float, float]] = []
        for strike, ts_map in market_data:
            yp = ts_map.get(ts)
            if yp is None:
                continue
            pairs.append((strike, yp))
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda x: x[0])
        # Higher strike → lower YES probability. Walk pairs and find the
        # bracket where probability crosses 0.5.
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
            # All probs > 0.5 → underlying is above the highest sampled
            # strike. All probs < 0.5 → below the lowest. Clamp so the
            # chart still has data instead of going blank in extreme
            # tail-of-distribution moments.
            if pairs[-1][1] >= 0.5:
                implied = pairs[-1][0]
            elif pairs[0][1] < 0.5:
                implied = pairs[0][0]
        if implied is not None:
            history.append({"ts": float(ts), "value": float(implied)})

    return history, atm, markets, contract_open_ts
