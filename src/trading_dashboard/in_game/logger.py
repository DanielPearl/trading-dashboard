"""Audit log for in-game model predictions.

Every confident prediction the model issues for an open sport
position gets appended to ``data/in_game_predictions.jsonl``. The
log is the foundation for two things:

  1. A live "recent predictions" panel on each sport bot's Models
     view — shows what the model said, when, and (joined against
     the closed-bet ledger) how the position ultimately resolved.
  2. Future training. Once we have months of (prediction, outcome)
     pairs, a real classifier can replace the heuristic and the
     same audit infrastructure becomes its calibration source.

Write policy
------------
We don't log on every hedge tick — that would be noisy and the
log would explode. Instead we log only **transitions**: when the
model first issues a confident action for a (bot_key, ticker)
pair, and whenever the action changes thereafter. Tracked in
memory via ``_last_action_per_ticker``; resets on dashboard
restart (the log itself is append-only and durable).
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("dashboard.in_game.logger")

# Log file lives in the project's data dir alongside bot_states.json
# and regime_notifications.jsonl. Three sources, same root, easy
# to back up together.
LOG_PATH = (Path(__file__).resolve().parents[3]
            / "data" / "in_game_predictions.jsonl")


# In-memory dedupe — only log when (action) flips for a ticker.
# Keyed by ``"{bot_key}:{ticker}"`` so two bots on the same ticker
# (shouldn't happen in practice but cheap to guard) don't collide.
_last_action_per_ticker: Dict[str, str] = {}
_dedupe_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def maybe_log(bot: Dict[str, Any], position: Dict[str, Any],
                prediction: "LivePrediction",  # noqa: F821 — fwd ref
                ) -> bool:
    """Append a prediction record when it represents a *transition*
    from the last-known action for this ticker. Returns True iff a
    new line was written.

    No-op when:
      - prediction.confidence < 0.5  (advisory threshold)
      - action == "neutral"          (nothing to record)
      - action matches the previous logged action for this ticker
    """
    if prediction is None:
        return False
    if (prediction.confidence or 0) < 0.5:
        return False
    action = (prediction.recommended_action or "").lower()
    if not action or action == "neutral":
        return False
    bot_key = bot.get("key") or ""
    ticker = (position.get("ticker") or position.get("match_id") or "")
    if not bot_key or not ticker:
        return False
    cache_key = f"{bot_key}:{ticker}"
    with _dedupe_lock:
        prev = _last_action_per_ticker.get(cache_key)
        if prev == action:
            return False
        _last_action_per_ticker[cache_key] = action

    entry: Dict[str, Any] = {
        "ts": _now_iso(),
        "bot_key": bot_key,
        "ticker": ticker,
        "side": (position.get("side") or "").upper() or None,
        "entry_price_cents": position.get("entry_price_cents"),
        "live_prob_yes": prediction.live_prob_yes,
        "confidence": prediction.confidence,
        "action": action,
        "reason": prediction.reason,
        # Truncate features to JSON-safe floats; the log shouldn't
        # carry exotic objects.
        "features": {k: (float(v) if isinstance(v, (int, float)) else None)
                      for k, v in (prediction.features or {}).items()},
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except OSError as exc:
        log.warning("in-game prediction log write failed: %s", exc)
        return False
    return True


def read_tail(limit: int = 50) -> List[Dict[str, Any]]:
    """Newest-first tail of the predictions log. Empty when the
    file is missing or unreadable.
    """
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def read_for_bot(bot_key: str, limit: int = 50
                   ) -> List[Dict[str, Any]]:
    """Newest-first tail filtered to one bot. Convenience for the
    Models > In-game "Recent predictions" panel.
    """
    if not bot_key:
        return []
    out: List[Dict[str, Any]] = []
    if not LOG_PATH.exists():
        return out
    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return out
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("bot_key") != bot_key:
            continue
        out.append(entry)
        if len(out) >= limit:
            break
    return out
