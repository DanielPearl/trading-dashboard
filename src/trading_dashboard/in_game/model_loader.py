"""Load + apply trained classifier output, if a model exists.

After ``train_classifier.py`` runs and writes
``data/in_game_models/<bot>_logreg.json``, the in-game scorer can
call ``trained_probability(bot_key, features)`` to get the
classifier's predicted win probability and blend it with the
heuristic baseline.

When no model file exists yet (the typical state for the first
few weeks of running), ``trained_probability`` returns ``None``
and the in-game scorer keeps using its heuristic. There's no
fallback drama — the system degrades gracefully to its previous
state.

The loaded model is cached for the dashboard's lifetime. To pick
up a freshly retrained model, restart the dashboard service.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("dashboard.in_game.model_loader")

MODELS_DIR = (Path(__file__).resolve().parents[3]
              / "data" / "in_game_models")


_MODEL_CACHE: Dict[str, Optional[dict]] = {}


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _load(bot_key: str) -> Optional[dict]:
    if bot_key in _MODEL_CACHE:
        return _MODEL_CACHE[bot_key]
    path = MODELS_DIR / f"{bot_key}_logreg.json"
    if not path.exists():
        _MODEL_CACHE[bot_key] = None
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("model load failed for %s: %s", bot_key, exc)
        _MODEL_CACHE[bot_key] = None
        return None
    _MODEL_CACHE[bot_key] = payload
    log.info(
        "loaded trained in-game model for %s "
        "(holdout AUC=%.3f, n=%d)",
        bot_key,
        (payload.get("holdout_metrics") or {}).get("roc_auc", 0.0),
        int(payload.get("n_train") or 0),
    )
    return payload


def trained_probability(bot_key: str,
                          features: Dict[str, float],
                          ) -> Optional[float]:
    """Apply the trained classifier (if any) and return the
    predicted win probability for this position. ``None`` when no
    model is loaded — caller falls back to heuristic.
    """
    model = _load(bot_key)
    if not model:
        return None
    keys = model.get("feature_keys") or []
    weights = model.get("weights") or []
    bias = model.get("bias") or 0.0
    means = model.get("means") or []
    stds = model.get("stds") or []
    if not (len(keys) == len(weights) == len(means) == len(stds)):
        log.debug("model shape mismatch for %s", bot_key)
        return None
    z = float(bias)
    for j, k in enumerate(keys):
        v = features.get(k)
        if v is None:
            v = 0.0
        try:
            xv = (float(v) - means[j]) / max(1e-6, stds[j])
        except (TypeError, ValueError):
            xv = 0.0
        z += weights[j] * xv
    return _sigmoid(z)
