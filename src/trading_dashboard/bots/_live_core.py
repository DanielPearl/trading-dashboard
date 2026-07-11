"""Shared live-trading core for the in-process bot executors.

One home for the machinery every real-money executor needs, so the
buy/sell mechanics stay identical across bots (world-cup today; the
tennis executor predates this module and migrates once the world-cup
port has proven parity — it is live real money mid-season and gets no
casual rewrites):

  DailyOrderCounter      rolling-UTC-day order budget, thread-safe
  atomic_write_json      crash-safe state persistence
  coid                   idempotent client_order_id derivation
  KalshiSession          lazy client + order submission + market reads
  build_closed_record    uniform close/settle record schema
  compute_stats          the sim_state.json stats block

Design rules carried over from the tennis executor:
  * dry_run defaults to True everywhere; real orders require an
    explicit config opt-in by the operator.
  * Orders are IOC limit at the observed price — fill now or cancel,
    no resting orders.
  * client_order_id is stable per (bot, ticker, price, day) so a retry
    de-dupes server-side instead of doubling up.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("dashboard.bots.live-core")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DailyOrderCounter:
    """Rolling-day counter for total orders placed. Resets at UTC
    midnight; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._date = self._today()
        self._count = 0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def increment(self) -> int:
        with self._lock:
            t = self._today()
            if t != self._date:
                self._date, self._count = t, 0
            self._count += 1
            return self._count

    def current(self) -> int:
        with self._lock:
            return self._count if self._today() == self._date else 0


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=str(path.parent),
                               text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def coid(bot: str, kind: str, ticker: str, cents: int) -> str:
    """Idempotency key: stable per (bot, kind, ticker, price, UTC day).
    A retried submit de-dupes server-side; a price move mints a new key
    (the price-deviation gate rejects chasing anyway)."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    seed = f"{bot}-{kind}|{ticker}|{cents}|{today}"
    return f"{kind}-" + hashlib.sha256(seed.encode()).hexdigest()[:28]


class KalshiSession:
    """Lazy Kalshi client with the order/read wrappers all executors
    share. Never constructed in dry-run paths that don't need it."""

    def __init__(self, bot_key: str) -> None:
        self.bot_key = bot_key
        self._client = None

    def client(self):
        if self._client is not None:
            return self._client
        try:
            from kalshi_sdk import KalshiClient
        except ImportError:
            log.exception("kalshi_sdk not importable")
            return None
        key = os.environ.get("KALSHI_API_KEY_ID", "").strip()
        pem = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if not key or not pem:
            log.error("Kalshi creds missing in env")
            return None
        try:
            self._client = KalshiClient(api_key_id=key,
                                        private_key_path=pem)
        except Exception:  # noqa: BLE001
            log.exception("KalshiClient init failed")
            return None
        return self._client

    # ── orders ──────────────────────────────────────────────────────

    def submit_ioc(self, *, ticker: str, action: str, count: int,
                   yes_price_cents: int,
                   kind: str) -> tuple[Optional[str], str]:
        """IOC limit order on the YES side of ``ticker``. ``action`` is
        "buy" or "sell". Returns (order_id, status); (None, reason) on
        failure. Kalshi wants lowercase snake_case time_in_force."""
        client = self.client()
        if client is None:
            return None, "no_client"
        # Kalshi rejects prices outside 1–99¢ ("invalid_price" 400).
        # Profit-lock sells hit this when a decided market's bid pins
        # at 100¢ — clamp so the sell fills at 99¢ instead of
        # error-looping until settlement.
        yes_price_cents = max(1, min(99, int(yes_price_cents)))
        try:
            resp = client.place_order(
                ticker=ticker, side="yes", action=action, count=count,
                order_type="limit", yes_price=yes_price_cents,
                time_in_force="immediate_or_cancel",
                client_order_id=coid(self.bot_key, kind, ticker,
                                     yes_price_cents))
        except Exception as exc:  # noqa: BLE001
            log.exception("%s place_order(%s %s) failed for %s: %s",
                          self.bot_key, action, kind, ticker, exc)
            return None, f"error: {exc!r}"
        order = (resp or {}).get("order") or {}
        return order.get("order_id"), order.get("status") or "submitted"

    # ── reads ───────────────────────────────────────────────────────

    def market(self, ticker: str) -> dict:
        client = self.client()
        if client is None:
            return {}
        try:
            return (client.get_market(ticker) or {}).get("market") or {}
        except Exception:  # noqa: BLE001
            log.exception("get_market failed for %s", ticker)
            return {}

    @staticmethod
    def _cents(mkt: dict, base: str) -> Optional[int]:
        for key, scale in ((base, 1), (f"{base}_dollars", 100)):
            v = mkt.get(key)
            if v is not None:
                try:
                    return int(round(float(v) * scale))
                except (TypeError, ValueError):
                    continue
        return None

    def yes_ask_cents(self, ticker: str,
                      fallback: Optional[int] = None) -> Optional[int]:
        v = self._cents(self.market(ticker), "yes_ask")
        return v if v is not None else fallback

    def yes_bid_cents(self, ticker: str) -> Optional[int]:
        return self._cents(self.market(ticker), "yes_bid")

    def balance_cents(self) -> Optional[int]:
        """Best-effort; None means 'skip the gate' (prefer under-
        trading on an API flake to over-trading on a false zero)."""
        client = self.client()
        if client is None:
            return None
        try:
            resp = client.get_balance() or {}
            for key in ("buying_power", "balance"):
                if resp.get(key) is not None:
                    return int(resp[key])
        except Exception:  # noqa: BLE001
            log.exception("get_balance failed (gate skipped)")
        return None

    def position_count(self, ticker: str) -> Optional[int]:
        """Contracts we actually hold on ``ticker`` per Kalshi's
        portfolio (positive = long YES-side). None on API failure —
        callers should defer rather than assume. Guards sells against
        inventory another process already disposed of."""
        client = self.client()
        if client is None:
            return None
        try:
            resp = client._request("GET", "/portfolio/positions",
                                   params={"ticker": ticker, "limit": 10})
        except Exception:  # noqa: BLE001
            log.exception("position_count failed for %s", ticker)
            return None
        for p in ((resp or {}).get("market_positions")
                  or (resp or {}).get("positions") or []):
            if p.get("ticker") == ticker:
                v = p.get("position_fp") or p.get("position") or 0
                try:
                    return int(float(v))
                except (TypeError, ValueError):
                    return None
        return 0

    def list_positions(self, limit: int = 200) -> Optional[list[dict]]:
        """Full-portfolio scan — every non-zero position on the account.
        Returns None on API failure so callers can defer. Used by the
        orphan-adoption path to reconstruct sim_state entries after a
        crash or manual click."""
        client = self.client()
        if client is None:
            return None
        try:
            resp = client._request("GET", "/portfolio/positions",
                                   params={"limit": limit})
        except Exception:  # noqa: BLE001
            log.exception("list_positions failed")
            return None
        raw = ((resp or {}).get("market_positions")
               or (resp or {}).get("positions") or [])
        out: list[dict] = []
        for p in raw:
            try:
                v = float(p.get("position_fp") or p.get("position") or 0)
            except (TypeError, ValueError):
                v = 0.0
            if abs(v) > 0:
                out.append(p)
        return out

    def market_result(self, ticker: str,
                      snapshot: dict | None = None) -> Optional[str]:
        """'yes' / 'no' once the market has finalized, else None. Uses
        the caller's market snapshot when provided to save a fetch."""
        mkt = snapshot or self.market(ticker)
        status = (mkt.get("status") or "").lower()
        result = (mkt.get("result") or "").lower()
        if not status and snapshot is not None:
            mkt = self.market(ticker)
            status = (mkt.get("status") or "").lower()
            result = (mkt.get("result") or "").lower()
        if status in ("finalized", "settled") and result in ("yes", "no"):
            return result
        return None


def _derive_exit_reason(reason: str) -> str:
    """Map the free-form close reason to the enum the analytics + Home-tab
    breakdowns group by. Copied from the old tennis executor so live
    closes stamp the same tag paper closes always used — no divergent
    row shape across bots.
    """
    r = (reason or "").lower()
    if r.startswith("profit_lock") or r.startswith("hedge_pl"):
        return "profit_lock"
    if r.startswith("auto_close"):
        return "auto_close"
    if r.startswith("stop_loss") or r.startswith("hedge_sl"):
        return "stop_loss"
    if r.startswith("void") or r.startswith("cancel"):
        return "voided"
    if r.startswith("settled") or r == "settlement":
        return "settled_match"
    if r.startswith("closed_externally") or r.startswith("external"):
        return "external_close"
    return r or "unknown"


def build_closed_record(pos: dict, *, settle_prob: float, reason: str,
                        result: str, **extra: Any) -> dict:
    """Uniform close record: realized P&L from settle price vs entry,
    winner_side derived from the held side. Stamps both the free-form
    ``close_reason`` and the ``exit_reason`` enum so analytics can
    group live + paper closes by the same categories."""
    entry = float(pos.get("entry_market_prob") or 0.0)
    contracts = int(pos.get("contracts") or 1)
    realized = (settle_prob - entry) * contracts
    won = realized > 0
    side = pos.get("side") or "PLAYER_A"
    closed = dict(pos)
    closed.update({
        "closed_at": now_iso(),
        "settle_market_prob": settle_prob,
        "realized_pnl": round(realized, 4),
        "won": won,
        "close_reason": reason,
        "exit_reason": _derive_exit_reason(reason),
        "result": result,
        "winner_side": side if won else (
            "PLAYER_B" if side == "PLAYER_A" else "PLAYER_A"),
        **extra,
    })
    return closed


def compute_stats(state: dict) -> dict:
    open_p = state.get("open_positions", [])
    closed = state.get("closed_positions", [])
    graded = [c for c in closed if c.get("result") != "VOID_DRY_RUN"]
    wins = sum(1 for c in graded if c.get("won"))
    losses = sum(1 for c in graded if c.get("won") is False)
    realized = sum(float(c.get("realized_pnl") or 0) for c in graded)
    unreal = sum(float(p.get("unrealized_pnl") or 0) for p in open_p)
    staked = sum(float(p.get("stake") or 0) for p in open_p + graded)
    return {
        "total_opened": len(open_p) + len(graded),
        "total_closed": len(graded),
        "open_count": len(open_p),
        "wins": wins, "losses": losses,
        "win_rate": (wins / len(graded)) if graded else None,
        "total_realized_pnl": round(realized, 4),
        "total_unrealized_pnl": round(unreal, 4),
        "total_staked": round(staked, 4),
        "roi": round(realized / staked, 4) if staked else None,
    }


def void_dry_run_positions(state: dict) -> int:
    """When an executor starts in REAL mode, paper positions left over
    from dry-run testing are voided (moved to closed as VOID_DRY_RUN,
    excluded from stats and from re-entry cooldowns) so real orders
    aren't blocked by paper bookkeeping."""
    open_p = state.get("open_positions", [])
    dry = [p for p in open_p
           if str(p.get("order_id") or "").startswith("DRY-RUN")]
    if not dry:
        return 0
    state["open_positions"] = [p for p in open_p if p not in dry]
    for p in dry:
        closed = build_closed_record(
            p, settle_prob=float(p.get("entry_market_prob") or 0.0),
            reason="void_dry_run", result="VOID_DRY_RUN")
        closed["realized_pnl"] = 0.0
        closed["won"] = None
        state.setdefault("closed_positions", []).append(closed)
        # Free the match for a real entry.
        state.get("last_settled_at_by_match_id", {}).pop(
            p.get("match_id", ""), None)
    return len(dry)
