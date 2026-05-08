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


def get_client() -> KalshiClient:
    """Process-wide singleton — keeps the loaded RSA key in memory."""
    global _cached
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is None:
            _cached = KalshiClient()
        return _cached


__all__ = ["KalshiClient", "KalshiError", "get_client"]
