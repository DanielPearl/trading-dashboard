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
            # Lets the renderer's OI filter pass for billboard rows
            # whose Kalshi-side OI is null (illiquid early in the
            # chart week — the markets are still tradeable).
            "_skip_oi_filter": True,
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
    """Drop-in replacement for the sim.db branch of
    ``_render_models_panel`` — same visual sections, populated from
    billboard's JSON / CSV artifacts.

    Mirrors retail-gas-prices' Models tab section-for-section:
      1. Headline metrics card row (Accuracy / F1 / Precision /
         Recall / ROC AUC / Features).
      2. Top features chart + sources legend + features table.
      3. ROC curve + held-out confidence card.
      4. Calibration curve.
      5. Empty-state stubs for the closed-bet-driven sections
         (strike-band accuracy, predicted vs realized EV, hedge
         audit) — Billboard is advisory-only so these are always
         empty, matching the empty-state retail gas would show on
         a brand-new bot with no closed bets.
    """
    # Late imports — the dashboard module is a peer of this one. Importing
    # at module load time would create a circular dep at startup.
    from . import dashboard as _d

    metrics = load_metrics(bot.get("metrics_path"))
    coefficients = load_coefficients(bot.get("coefficients_path"))
    blended = metrics.get("blended") or {}

    # ── 1. Headline metrics card row ─────────────────────────────────
    def _pct(v, decimals: int = 0) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v) * 100:.{decimals}f}%"
        except (TypeError, ValueError):
            return "—"

    out.append("<div class='row compact'>")
    if not metrics:
        out.append("<div class='card'><div class='label'>Model</div>"
                    "<div class='value'>No snapshot yet</div></div>")
    else:
        metric_cards = [
            ("Accuracy",
             "Held-out classifier accuracy on the trainer's test set.",
             _pct(blended.get("accuracy"), 1)),
            ("F1",
             "Harmonic mean of precision and recall on the test set.",
             _pct(blended.get("f1"))),
            ("Precision",
             "Fraction of predicted positives that were correct.",
             _pct(blended.get("precision"))),
            ("Recall",
             "Fraction of actual positives the model caught.",
             _pct(blended.get("recall"))),
            ("ROC AUC",
             "Area under the ROC curve on the test set. "
             "0.5 = random, 1.0 = perfect.",
             _pct(blended.get("roc_auc"))),
            ("Features",
             "Number of input features the model uses.",
             (str(int(metrics.get("feature_count")))
              if metrics.get("feature_count") is not None else "—")),
        ]
        for label, title, value in metric_cards:
            out.append(
                f"<div class='card'><div class='label' "
                f"title='{html.escape(title)}'>"
                f"{html.escape(label)}</div>"
                f"<div class='value'>{html.escape(value)}</div></div>"
            )
    out.append("</div>")

    # ── 2. Top features — bars + readable table in one aligned panel ─
    feats = _logistic_coefs_as_features(coefficients)
    out.append(_d._render_feature_source_table(feats))

    # ── 3. ROC curve + confidence card (held-out predictions) ────────
    holdout_csv = bot.get("holdout_predictions_path") \
        or _default_holdout_path(bot.get("metrics_path"))
    pairs = _read_billboard_holdout(str(holdout_csv)) if holdout_csv else []
    conf = _d._holdout_confidence(pairs)
    auc_scalar = blended.get("roc_auc")
    roc_points = _d.roc_from_holdout(pairs)
    n_pairs = len(pairs)
    holdout_blurb = (
        "Sourced from the trainer's held-out historical test set — "
        "predictions the model never saw during training, so the "
        "numbers below are its honest evaluation against past reality, "
        "not a re-run of the live closed-bet ledger. Numbers reflect "
        "the post-leakage-fix feature builder (see Limitations below) "
        "— test F1 dropped from 0.93 → 0.09 on synthetic data after "
        "the audit, because the previous version was scoring labels "
        "via shortcut paths rather than learning real signal."
    )
    out.append(f"<p class='small gray' style='margin:0 0 6px 0;'>"
                f"{html.escape(holdout_blurb)}</p>")
    _d._render_confidence_card(out, conf)
    out.append("<h3 class='subhead' style='margin-top:0;'>"
                "ROC curve <span class='small gray'>(historical "
                f"held-out test set, {n_pairs:,} predictions)</span></h3>")
    out.append(_d._svg_roc_curve(roc_points, auc_scalar=auc_scalar))

    # ── 4. Calibration curve ─────────────────────────────────────────
    out.append("<h3 class='subhead'>Calibration "
                "<span class='small gray'>(historical held-out test "
                "set, predicted prob vs observed positive rate)"
                "</span></h3>")
    cal_bins = _d.calibration_from_holdout(pairs, n_bins=10)
    out.append(_d._svg_calibration(cal_bins, live_bins=None))

    # ── 5. Empty-state stubs for closed-bet sections ─────────────────
    # Billboard is advisory-only — no closed bets ever — so these
    # sections render the same empty state retail gas would show on
    # a brand-new bot. Kept here so the page structure matches
    # retail gas one-to-one.
    out.append("<h3 class='subhead'>Accuracy by strike band "
                "<span class='small gray'>(closed bets only)</span></h3>")
    out.append("<div class='empty'>No closed bets to break down by "
                "strike yet.</div>")

    out.append("<h3 class='subhead'>Predicted vs realized EV "
                "<span class='small gray'>(closed bets only)</span></h3>")
    out.append("<div class='empty'>No closed bets to grade predicted "
                "edge against yet.</div>")

    out.append("<h3 class='subhead'>Hedge effectiveness audit "
                "<span class='small gray'>(closed bets only)</span></h3>")
    out.append("<div class='empty'>No hedge events fired yet.</div>")

    # ── 6. Limitations & audit history (Billboard-specific) ──────────
    # Sits at the bottom of the Models tab so a user can scan the
    # headline metrics first, then read the context behind them.
    out.append("<h3 class='subhead'>Limitations &amp; audit history</h3>")
    out.append(
        "<ul class='small gray' style='margin:0 0 12px 18px; "
        "line-height:1.6;'>"
        "<li><strong>Data source:</strong> Real Billboard Hot 100 "
        "weekly chart history pulled from the "
        "<code>utdata/rwd-billboard-data</code> public mirror "
        "(refreshed daily). Synthetic-data fallback was removed — "
        "if the source is unreachable AND no local cache exists, "
        "the trainer raises rather than fabricating data.</li>"
        "<li><strong>Features:</strong> Pure chart-dynamics — "
        "artist top-10 history, song's prior peak / weeks-on-chart "
        "/ momentum, debut context, and weekly competition. No "
        "Spotify / Reddit / Google Trends / YouTube features. "
        "Those external signals can't be learned without "
        "point-in-time historical snapshots (using today's "
        "popularity to predict 2017 charts would be leakage), and "
        "the prior implementation zero-filled them at train time "
        "so the model assigned them zero weight anyway. Removed to "
        "keep the feature set honest.</li>"
        "<li><strong>Leakage audit (post-v1):</strong> Four label-"
        "leakage paths were found and fixed in the feature builder "
        "(<code>peak_position_so_far</code> debut-week fillna, "
        "<code>debut_rank</code> on debut rows, <code>best_3wk_rank</code> "
        "unshifted rolling min, <code>competition_count</code> "
        "self-inclusion).</li>"
        "<li><strong>No historical backtest yet.</strong> The model's "
        "held-out metrics above (accuracy, F1, ROC AUC, calibration) "
        "come from real Hot 100 outcomes on the trainer's holdout "
        "weeks — those numbers are honest. There is no P&amp;L "
        "backtest because Kalshi price-history snapshots aren't "
        "cached yet, and we won't fabricate a market probability to "
        "compute one. The Watchlist tab's edge column uses live "
        "current Kalshi prices.</li>"
        "</ul>"
    )


def _default_holdout_path(metrics_path: str | None) -> str:
    """Derive holdout_predictions.csv path from metrics.json path —
    they live in the same artifacts/ directory."""
    if not metrics_path:
        return ""
    p = Path(metrics_path).parent / "holdout_predictions.csv"
    return str(p)
