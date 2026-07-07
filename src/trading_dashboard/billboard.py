"""Billboard Hot 100 dashboard adapter.

The bot writes watchlist.json + metrics.json + model_coefficients.json +
holdout_predictions.csv + feature_importance.csv on every tick. This
module is the seam between those JSON-source artifacts and the
standard ``_render_watchlist`` / ``_render_models_panel`` renderers
in dashboard.py.

Routing pattern matches tennis: the GET handler synthesises a
standard ``watchlist`` row list from the bot's JSON output, sets
``model = None``, and falls through to the shared ``render_page`` —
so the Billboard pages are visually indistinguishable from the
retail-gas-prices pages they're modelled on.

For the Models tab, ``render_models_panel`` reproduces the same
sections the standard sim.db-backed renderer produces (headline
metrics card row → top features → ROC curve → calibration curve →
strike-band / EV / hedge empty states for advisory-only bots) using
metrics.json + coefficients.json + holdout_predictions.csv as the
data sources.
"""
from __future__ import annotations

import csv
import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger("dashboard.billboard")


# --------------------------------------------------------------------------- #
# JSON loaders                                                                #
# --------------------------------------------------------------------------- #

def load_watchlist(path: str | None) -> Dict[str, Any]:
    if not path:
        return {"generated_at": None, "rows": []}
    p = Path(path)
    if not p.exists():
        return {"generated_at": None, "rows": []}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"generated_at": None, "rows": []}


def load_metrics(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_coefficients(path: str | None) -> Dict[str, Any]:
    return load_metrics(path)


def load_sim_state(path: str | None) -> Dict[str, Any]:
    """Billboard is advisory-only — no sim_state.json. Returns {} so the
    callers downstream that expect a dict don't crash."""
    return {}


# --------------------------------------------------------------------------- #
# Cross-bot rollup adapters — match the survivor adapter's signatures so      #
# the home grid + models page can iterate every bot uniformly.                #
# --------------------------------------------------------------------------- #

def is_available(metrics_path: str | None) -> bool:
    return bool(metrics_path and Path(metrics_path).exists())


def model_summary_for_card(metrics_path: str | None,
                            sim_state_path: str | None = None
                            ) -> Dict[str, Any]:
    metrics = load_metrics(metrics_path)
    if not metrics:
        return {}
    blended = metrics.get("blended") or {}
    return {
        "classifier_accuracy": blended.get("accuracy"),
        "training_brier": blended.get("brier"),
        "training_log_loss": blended.get("log_loss"),
        "training_f1": blended.get("f1"),
        "training_precision": blended.get("precision"),
        "training_recall": blended.get("recall"),
        "training_roc_auc": blended.get("roc_auc"),
        "threshold": blended.get("threshold"),
        "feature_count": int(metrics.get("feature_count", 0) or 0),
        # Training- and held-out test-set sizes. Surface on the Home
        # bot card as "Train rows" / "Test rows".
        "rows_train": metrics.get("rows_train"),
        "rows_test": metrics.get("rows_test"),
        "actual_wins": 0,
        "actual_losses": 0,
    }


def summary_for_rollup(sim_state_path: str | None) -> Dict[str, Any]:
    return {
        "open_count": 0, "active_contracts": 0,
        "period_bets_made": 0, "period_net_pnl_cents": 0,
        "period_wins": 0, "period_losses": 0, "period_money_spent_cents": 0,
        "period_money_gained_cents": 0, "potential_gain_cents": 0,
        "active_money_spent_cents": 0,
        "total_bets": 0, "realized_pnl_cents": 0,
        "wins_lifetime": 0, "losses_lifetime": 0,
    }


def active_bets_for_rollup(sim_state_path: str | None,
                             watchlist_path: str | None = None
                             ) -> List[Dict[str, Any]]:
    """Billboard is advisory-only — no positions ever — so return []."""
    return []


def closed_positions_for_rollup(sim_state_path: str | None,
                                  limit: int = 100) -> List[Dict[str, Any]]:
    """Billboard is advisory-only — no settled positions ever."""
    return []


# --------------------------------------------------------------------------- #
# Watchlist row adapter — converts billboard rows into the standard schema    #
# the shared ``_render_watchlist`` consumes.                                  #
# --------------------------------------------------------------------------- #

def build_standard_watchlist_rows(payload: Dict[str, Any]
                                    ) -> List[Dict[str, Any]]:
    """Translate billboard watchlist rows into the schema the shared
    ``_render_watchlist`` expects.

    One row per active Kalshi Billboard market. Mapping notes:

    - ``direction`` carries the song title so the standard renderer's
      ``question_str`` returns "<song>" in the Question column.
      ``strike_low`` / ``strike_high`` stay None so question_str
      doesn't append a strike clause.
    - ``model_prob_yes`` = billboard's P(song is in Hot 100 top 10).
    - ``bot_verdict`` maps the billboard vocabulary to the standard
      one (BUY YES → BUY_YES, BUY NO → BUY_NO, WATCH / SKIP → SKIP).
    - ``_skip_oi_filter`` = True so the renderer's
      ``open_interest > 0`` filter passes for billboard rows even
      when Kalshi returns null OI (illiquid early-week markets).
    """
    raw = payload.get("rows") or []
    out: List[Dict[str, Any]] = []
    for r in raw:
        bb_verdict = (r.get("verdict") or "").upper()
        if bb_verdict == "BUY YES":
            bot_verdict = "BUY_YES"
        elif bb_verdict == "BUY NO":
            bot_verdict = "BUY_NO"
        else:
            bot_verdict = "SKIP"
        blockers = r.get("buy_blockers") or []
        out.append({
            "ticker": r.get("ticker") or "",
            "title": r.get("title") or "",
            "direction": r.get("song") or r.get("album") or "",
            # Watchlist renderer reads these to fill the billboard-only
            # Artist + Song columns that replace the generic Question
            # column. ``direction`` is left set so the question_str
            # fallback (used elsewhere) still works.
            "_artist": r.get("artist") or "",
            "_song": r.get("song") or "",
            "strike_low": None,
            "strike_high": None,
            "yes_ask_cents": r.get("yes_ask_cents"),
            "no_ask_cents": r.get("no_ask_cents"),
            "spread_cents": r.get("spread_cents"),
            "volume": r.get("volume"),
            "open_interest": r.get("open_interest"),
            "model_prob_yes": r.get("model_prob"),
            "raw_model_prob_yes": r.get("model_prob"),
            "bot_verdict": bot_verdict,
            "rejection_reason": (", ".join(str(b) for b in blockers)
                                  if blockers else ""),
            "minutes_to_close": r.get("minutes_to_close"),
        })
    return out


# --------------------------------------------------------------------------- #
# Models tab — produces the same sections the standard sim.db renderer        #
# produces, but from billboard's metrics.json / coefficients.json /           #
# holdout_predictions.csv artifacts.                                          #
# --------------------------------------------------------------------------- #

def _read_billboard_holdout(csv_path: str) -> List[Tuple[float, int]]:
    """Read holdout_predictions.csv. Trainer writes columns:
    chart_date, title, artist, rank, is_song_in_top_10_hot_100,
    predicted_prob_no1.
    (``predicted_prob_no1`` is a legacy column name kept across the
    Billboard-200-#1 → Hot-100-top-10 retarget; the value is now
    P(song hits Hot 100 top 10), not P(album hits #1).)
    Returns [(prob, label)] pairs the standard ROC / calibration
    helpers consume. Also reads the legacy
    ``is_billboard_200_number_1`` column for old CSVs so a stale
    artifact on disk doesn't break the page until the next retrain.
    """
    p = Path(csv_path)
    if not p.exists():
        return []
    out: List[Tuple[float, int]] = []
    try:
        with p.open("r") as f:
            rd = csv.DictReader(f)
            for row in rd:
                try:
                    prob = float(row.get("predicted_prob_no1") or 0.0)
                    raw_label = (row.get("is_song_in_top_10_hot_100")
                                 or row.get("is_billboard_200_number_1")
                                 or 0)
                    label = int(float(raw_label))
                except (TypeError, ValueError):
                    continue
                out.append((prob, 1 if label else 0))
    except (OSError, csv.Error):
        return []
    return out


def _logistic_coefs_as_features(coefficients: Dict[str, Any]
                                  ) -> List[Dict[str, Any]]:
    """Convert the logistic block of model_coefficients.json into the
    feats list shape the standard ``_svg_feature_importance_vertical``
    / ``_render_feature_source_table`` consume:
        [{feature, mean_importance, positive_folds, selected}, …]

    Logistic coefficients are signed — the standard helpers want a
    magnitude — so we use |coef|. ``selected`` is True for every
    feature (logistic uses all of them).
    """
    log_block = (coefficients or {}).get("logistic") or {}
    feats = log_block.get("features") or []
    coefs = log_block.get("coefficients") or []
    out: List[Dict[str, Any]] = []
    for n, c in zip(feats, coefs):
        try:
            mag = abs(float(c))
        except (TypeError, ValueError):
            mag = 0.0
        out.append({
            "feature": str(n),
            "mean_importance": mag,
            "positive_folds": 1,   # logistic is single-fit, not k-fold
            "selected": True,
        })
    return out


def render_models_panel(out: List[str], bot: Dict[str, Any]) -> None:
    """Billboard Models tab — two-section layout shared with every
    other bot: (1) a table of every model the trainer produced with
    the stats surfaced on the home-page model cards plus Brier, and
    (2) the readable-features panel with source colouring and
    importance bars. All other historical sections have been removed
    to keep the layout identical across sports.
    """
    from . import dashboard as _d  # peer module — late import avoids cycle

    metrics = load_metrics(bot.get("metrics_path"))
    coefficients = load_coefficients(bot.get("coefficients_path"))

    # Features (from the logistic coefficients, since Billboard's
    # trainer doesn't dump a feature_importance.csv).
    feats = _logistic_coefs_as_features(coefficients)

    # Feature count + last-trained date from the metrics.json when
    # present. Billboard's metrics don't carry ``captured_at`` for the
    # bundle, so fall back to the mtime of metrics.json itself.
    n_feats = (int(metrics.get("feature_count"))
                if metrics.get("feature_count") is not None
                else (len(feats) if feats else None))
    last_trained = "—"
    metrics_path = bot.get("metrics_path")
    if metrics_path:
        try:
            import datetime as _dt
            mt = _dt.datetime.fromtimestamp(
                Path(metrics_path).stat().st_mtime,
                tz=_dt.timezone.utc)
            last_trained = mt.strftime("%Y-%m-%d")
        except (OSError, OverflowError):
            pass

    # 1) Table of models run.
    out.append(_d._render_models_run_table(
        metrics,
        feature_count=n_feats,
        last_trained=last_trained,
    ))

    # 2) Features with definitions and bars.
    out.append(_d._render_feature_source_table(feats))


def _default_holdout_path(metrics_path: str | None) -> str:
    """Derive holdout_predictions.csv path from metrics.json path —
    they live in the same artifacts/ directory."""
    if not metrics_path:
        return ""
    p = Path(metrics_path).parent / "holdout_predictions.csv"
    return str(p)
