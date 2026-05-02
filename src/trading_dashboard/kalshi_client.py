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


_CACHE = _TTLCache(ttl_seconds=60.0)


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


def fetch_chart_series(series_ticker: str,
                        period_minutes: int = 60,
                        lookback_hours: int = 72,
                        client: Optional[KalshiClient] = None
                        ) -> Tuple[List[dict], Optional[dict]]:
    """Fetch the chart data for one ticker group (= one series).

    Returns (history, atm_market). `history` is a list of
    ``{"ts": float, "yes_pct": float, "no_pct": float, "volume": float}``
    points, oldest first. `atm_market` is the picked market dict (or None
    if no markets are open / Kalshi is unavailable). Empty history is a
    valid result — the chart renderer shows the empty-state in that case.
    """
    c = client or default_client()
    if not c.available:
        return [], None
    markets = c.list_markets(series_ticker)
    atm = pick_atm_market(markets)
    if atm is None:
        return [], None
    candles = c.candlesticks(
        series_ticker, atm["ticker"],
        period_minutes=period_minutes, lookback_hours=lookback_hours,
    )
    history: List[dict] = []
    for cdl in candles:
        ts = cdl.get("end_period_ts")
        if ts is None:
            continue
        # Prefer mean_dollars if populated; fall back to close_dollars.
        # Both are in 0..1 (dollar fraction).
        price = cdl.get("price") or {}
        ya = cdl.get("yes_ask") or {}
        yb = cdl.get("yes_bid") or {}
        yes_close = (_to_float(price.get("mean_dollars"))
                     or _to_float(price.get("close_dollars")))
        if yes_close is None:
            # Some thin bars only have ask/bid, no trades. Fall back to
            # the bid/ask midpoint so the line stays continuous.
            ya_close = _to_float(ya.get("close_dollars"))
            yb_close = _to_float(yb.get("close_dollars"))
            if ya_close is not None and yb_close is not None:
                yes_close = (ya_close + yb_close) / 2.0
            elif ya_close is not None:
                yes_close = ya_close
            elif yb_close is not None:
                yes_close = yb_close
        if yes_close is None:
            continue
        history.append({
            "ts": float(ts),
            "yes_pct": yes_close * 100.0,
            "no_pct": (1.0 - yes_close) * 100.0,
            "volume": _to_float(cdl.get("volume_fp")) or 0.0,
        })
    return history, atm
