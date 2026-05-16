"""Dense per-tick feature snapshot logger.

The transition log in ``in_game/logger.py`` records only when the
model's *recommended action* changes — which is great for an audit
trail but sparse for training. A real classifier needs dense
``(features, eventual_outcome)`` pairs sampled regularly while a
position is open.

This module writes one JSON line per (bot, ticker, hedge-tick) pair
to ``data/in_game_features.jsonl`` — append-only, one row per
prediction the model makes. Each row carries:

  * ``ts``         : ISO 8601 capture time
  * ``bot_key``    : which bot this is for
  * ``ticker``     : the position's ticker / match_id
  * ``side``       : YES / NO
  * ``entry_cents``: position's entry price
  * ``live_prob``  : the in-game model's live_prob_yes for the bet's side
  * ``features``   : full features dict the model produced

The training pipeline at training time joins these rows on
``(bot_key, ticker)`` against the bot's closed-position ledger to
assign each row the eventual ``won`` (1 if realized P&L > 0) label.

Volume control
--------------
Snapshotting every 30s for a bot with 10 open positions across a
3-hour NBA game writes ~3,600 lines/game. That's fine — the file
is JSONL, line-grepped easily, and a season's worth fits inside a
few hundred MB. We do not dedupe by features (unlike the audit
log) because dense sampling IS the point.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

log = logging.getLogger("dashboard.in_game.feature_log")


LOG_PATH = (Path(__file__).resolve().parents[3]
            / "data" / "in_game_features.jsonl")


_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_snapshot(bot: Dict[str, Any], position: Dict[str, Any],
                   prediction: "LivePrediction",  # noqa: F821
                   ) -> bool:
    """Append one feature-snapshot line. Returns True iff written.

    Logs *every* prediction that has features (regardless of
    confidence or action) — we want dense training samples even
    from low-confidence ticks. Errors are swallowed so a write
    failure can never affect the hedge tick.
    """
    if prediction is None:
        return False
    bot_key = bot.get("key") or ""
    ticker = (position.get("ticker") or position.get("match_id") or "")
    if not bot_key or not ticker:
        return False
    entry_c = position.get("entry_price_cents")
    side = (position.get("side") or "").upper() or None
    features = prediction.features or {}
    entry: Dict[str, Any] = {
        "ts": _now_iso(),
        "bot_key": bot_key,
        "ticker": ticker,
        "side": side,
        "entry_cents": entry_c if entry_c is None else int(entry_c),
        "live_prob_yes": float(prediction.live_prob_yes or 0.0),
        "confidence": float(prediction.confidence or 0.0),
        "action": (prediction.recommended_action or "").lower(),
        "features": {k: (float(v) if isinstance(v, (int, float)) else None)
                       for k, v in features.items()},
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except OSError as exc:
        log.warning("feature_log write failed: %s", exc)
        return False
    return True


def iter_snapshots(bot_key: Optional[str] = None
                     ) -> Iterator[Dict[str, Any]]:
    """Stream snapshots oldest-first, optionally filtered to one bot.
    Used by the training script.
    """
    if not LOG_PATH.exists():
        return
    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if bot_key and entry.get("bot_key") != bot_key:
                    continue
                yield entry
    except OSError:
        return
