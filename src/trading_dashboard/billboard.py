"""Billboard Hot 100 dashboard adapter.

The bot (Billboard Charts repo) targets the KXTOPSONG series — the
weekly "#1 on the Billboard Hot 100" market — and trains two models
on the same (song × chart-week) popular-pool panel:

    in_hot_100      P(song is on the Hot 100 that week) — the
                    primary/interpretable target
    is_number_one   P(song is the #1) — prices the KXTOPSONG YES
                    contract; watchlist ``model_prob`` carries this

It writes watchlist.json + metrics.json + model_coefficients.json +
training_data.db (the full training panel) on every train/tick, plus
a standard sim.db from the PAPER trader (dry_run — never places real
orders). This module is the seam between those artifacts and the
shared renderers.

Routing pattern matches tennis: the GET handler synthesises a
standard ``watchlist`` row list from the bot's JSON output, sets
``model = None``, and falls through to the shared ``render_page``.
Positions/history come straight from sim.db via the standard
readers (see server.py / data.py billboard branches).
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.billboard")

# Target metadata for the Models + Training Data tabs. Order matters:
# the membership model is the primary/interpretable one, the #1 model
# is what actually prices KXTOPSONG contracts.
_TARGETS = [
    ("in_hot_100",
     "In the Hot 100",
     "P(song is on the Billboard Hot 100 that chart week). Trained on "
     "pool rows only — songs that charted within the trailing 12 weeks "
     "— so every row has a real 0/1 outcome."),
    ("is_number_one",
     "#1 on the Hot 100",
     "P(song is the #1 on the Hot 100 that chart week). Trained on all "
     "panel rows including debuts; this probability prices the "
     "KXTOPSONG YES contract on the Watchlist."),
]


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
    - ``model_prob_yes`` = the bot's P(song is #1 on the Hot 100 for
      the contract's chart week) — the tradeable probability.
    - ``_p_hot100`` = P(song is on the Hot 100 at all) — context
      shown under the Song cell (a #1 candidate near 100% is normal).
    - ``bot_verdict`` maps the billboard vocabulary to the standard
      one (BUY YES → BUY_YES, BUY NO → BUY_NO, WATCH / SKIP → SKIP).
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
            "_p_hot100": r.get("model_prob_hot100"),
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
# Models tab — per-target sections: models-run table + signed logistic        #
# coefficients at the top, feature definitions underneath.                    #
# --------------------------------------------------------------------------- #

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


def _render_coefficients_table(target_label: str,
                                 log_block: Dict[str, Any]) -> str:
    """Signed logistic coefficients for one target, sorted by |coef|
    descending, with green/red signing. These are standardized-input
    coefficients, so magnitudes are directly comparable across
    features."""
    feats = log_block.get("features") or []
    coefs = log_block.get("coefficients") or []
    if not feats or not coefs:
        return ""
    rows = sorted(zip(feats, coefs),
                  key=lambda fc: -abs(float(fc[1] or 0.0)))
    parts: List[str] = []
    parts.append(
        f"<h4 class='subhead' style='margin-top:10px;'>Logistic "
        f"coefficients — {html.escape(target_label)} "
        "<span class='small gray'>(standardized inputs; positive = "
        "raises the probability)</span></h4>"
    )
    parts.append(
        "<div style='overflow-x:auto;margin-bottom:14px;'>"
        "<table style='border-collapse:collapse;font-size:12.5px;'>"
        "<thead><tr><th style='text-align:left;'>Feature</th>"
        "<th class='num'>Coefficient</th></tr></thead><tbody>"
    )
    for n, c in rows:
        try:
            cf = float(c)
        except (TypeError, ValueError):
            cf = 0.0
        color = "#3fb950" if cf > 0 else ("#f85149" if cf < 0 else "#8b949e")
        parts.append(
            "<tr>"
            f"<td style='padding:2px 18px 2px 0;'>{html.escape(str(n))}</td>"
            f"<td class='num' style='color:{color};'>{cf:+.4f}</td>"
            "</tr>"
        )
    try:
        icpt = float(log_block.get("intercept"))
        parts.append(
            "<tr><td style='padding:2px 18px 2px 0;' class='gray'>"
            "(intercept)</td>"
            f"<td class='num gray'>{icpt:+.4f}</td></tr>"
        )
    except (TypeError, ValueError):
        pass
    parts.append("</tbody></table></div>")
    return "".join(parts)


def render_models_panel(out: List[str], bot: Dict[str, Any]) -> None:
    """Billboard Models tab — per-target sections at the top (models
    run + signed logistic coefficients for each of the two targets),
    then the shared readable-features panel underneath. Layout per
    user spec 2026-07-15: "coefficients at the top and the features
    underneath"."""
    from . import dashboard as _d  # peer module — late import avoids cycle

    metrics = load_metrics(bot.get("metrics_path"))
    coefficients = load_coefficients(bot.get("coefficients_path"))
    target_metrics = metrics.get("targets") or {}
    target_coefs = coefficients.get("targets") or {}

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

    rendered_any = False
    for key, label, desc in _TARGETS:
        tm = target_metrics.get(key) or {}
        families = tm.get("families") or {}
        if not families:
            continue
        rendered_any = True
        best = tm.get("best_model") or ""
        out.append(
            f"<h3 class='subhead' style='margin-top:18px;'>Target: "
            f"<code>{html.escape(key)}</code> — {html.escape(label)}"
            "</h3>"
            f"<p class='small gray'>{html.escape(desc)} "
            f"Trained on <b>{tm.get('rows_train', 0):,}</b> rows "
            f"(positive rate {float(tm.get('train_positive_rate') or 0):.2%}), "
            f"tested on <b>{tm.get('rows_test', 0):,}</b>. "
            f"Production model: <b>{html.escape(best.upper())}</b>.</p>"
        )
        # Models-run table for this target: reshape families into the
        # per_model dict the unified renderer expects, plus the
        # winner's held-out block as "Blended (final)".
        per_model = {name: fam.get("test") or {}
                     for name, fam in families.items()}
        table_metrics = {
            "per_model": per_model,
            "blended": (families.get(best) or {}).get("test"),
            "rows_train": tm.get("rows_train"),
            "rows_test": tm.get("rows_test"),
        }
        out.append(_d._render_models_run_table(
            table_metrics,
            feature_count=metrics.get("feature_count"),
            last_trained=last_trained,
        ))
        # Signed coefficients for this target.
        log_block = (target_coefs.get(key) or {}).get("logistic") or {}
        out.append(_render_coefficients_table(label, log_block))

    if not rendered_any:
        # Pre-retrain artifact on disk (no "targets" block) — fall back
        # to the legacy single-target layout so the page never blanks.
        out.append(_d._render_models_run_table(
            metrics,
            feature_count=metrics.get("feature_count"),
            last_trained=last_trained,
        ))
        out.append(_render_coefficients_table(
            "primary target", coefficients.get("logistic") or {}))

    # Feature definitions + importance bars underneath. Importance
    # bars use the membership target's |coef| (primary/interpretable).
    primary_coefs = (target_coefs.get("in_hot_100")
                     or {k: v for k, v in coefficients.items()
                         if k != "targets"})
    feats = _logistic_coefs_as_features(primary_coefs)
    out.append(_d._render_feature_source_table(feats))


# --------------------------------------------------------------------------- #
# Training Data tab — pages the full (song × chart-week) panel from           #
# training_data.db (table ``training_rows``, written by the trainer).         #
# --------------------------------------------------------------------------- #

# (key, label, definition). Order = column order on the page.
_TD_COLUMNS = [
    ("chart_date", "Week",
     "Billboard chart week (Saturday chart date). The row asks: was "
     "this song on the Hot 100 dated this Saturday?"),
    ("title", "Song", "Song title as it appears on the chart."),
    ("artist", "Artist", "Recording artist."),
    ("in_hot_100", "In Hot 100",
     "DEPENDENT VARIABLE — 1 if the song is on this week's Hot 100, "
     "0 if it is not. The in_hot_100 model is trained on pool rows "
     "only (In pool = 1)."),
    ("is_number_one", "Is #1",
     "Second label — 1 if the song is #1 this week. The is_number_one "
     "model (trained on all rows) prices KXTOPSONG contracts."),
    ("rank_this_week", "Rank",
     "The song's actual Hot 100 rank this week (blank when the song "
     "is not on the chart). Not a feature — shown for context."),
    ("in_pool", "In pool",
     "1 = the song charted within the trailing 12 weeks (a 'popular "
     "song' with a real 0/1 membership outcome). 0 = the row only "
     "exists because the song debuted/re-entered this week."),
    ("artist_prior_top10_songweeks", "Artist top-10 song-wks",
     "Lifetime count of this artist's top-10 song-weeks strictly "
     "before this week."),
    ("artist_weeks_since_last_top10", "Artist wks since top-10",
     "Weeks since the artist last had any top-10 song (9999 = never)."),
    ("artist_prior_top10_weeks", "Artist top-10 wks",
     "Number of prior chart weeks in which the artist had at least "
     "one top-10 song."),
    ("artist_prior_no1_weeks", "Artist #1 wks",
     "Number of prior chart weeks in which the artist had the #1 "
     "song."),
    ("weeks_on_chart", "Wks on chart",
     "How many chart weeks this song has appeared, strictly before "
     "this week (0 = debut row)."),
    ("peak_position_so_far", "Peak so far",
     "Best (lowest) rank the song reached before this week (999 = no "
     "prior appearance)."),
    ("debut_rank", "Debut rank",
     "Rank in the song's first chart week (0 on debut rows — the "
     "debut rank isn't knowable before the chart publishes)."),
    ("weeks_since_debut", "Wks since debut",
     "Weeks elapsed since the song's first chart appearance."),
    ("last_seen_rank", "Last rank",
     "The song's rank at its most recent prior appearance (200 "
     "sentinel = never charted)."),
    ("weeks_since_last_on_chart", "Wks since on chart",
     "Gap since the song last appeared (1 = it was on last week's "
     "chart; 99 sentinel = never charted)."),
    ("best_3wk_rank", "Best 3-wk rank",
     "Best rank across the song's last three prior appearances."),
    ("rank_change_last_week", "Rank Δ",
     "Rank change between the song's last two prior appearances "
     "(negative = climbing)."),
    ("weeks_in_top10_so_far", "Wks in top 10",
     "Prior weeks spent in the top 10."),
    ("weeks_in_top40_so_far", "Wks in top 40",
     "Prior weeks spent in the top 40."),
    ("weeks_at_no1_so_far", "Wks at #1",
     "Prior weeks spent at #1."),
    ("debut_month_sin", "Debut month (sin)",
     "Seasonality encoding of the song's debut month (sine "
     "component)."),
    ("debut_month_cos", "Debut month (cos)",
     "Seasonality encoding of the song's debut month (cosine "
     "component)."),
    ("competition_last_week", "Competition",
     "Fresh top-40 debuts on the most recent prior chart week — how "
     "crowded the release window is."),
    ("is_new_to_pool", "New to pool",
     "1 = this row is a debut/re-entry from outside the trailing "
     "pool (its in_hot_100 label is 1 by construction, so the "
     "membership model does not train on it)."),
]
_TD_TEXT_COLS = {"chart_date", "title", "artist"}


def _query_training_db(db_path: str, week: str | None,
                       offset: int, limit: int):
    """(total, rows, weeks) page from training_data.db, newest week
    first, rank ascending inside a week."""
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        weeks = [r[0] for r in con.execute(
            "SELECT DISTINCT chart_date FROM training_rows "
            "ORDER BY chart_date DESC")]
        where = "WHERE chart_date = ?" if week else ""
        args = [week] if week else []
        total = con.execute(
            f"SELECT COUNT(*) FROM training_rows {where}", args
        ).fetchone()[0]
        cur = con.execute(
            f"SELECT * FROM training_rows {where} "
            "ORDER BY chart_date DESC, "
            "CASE WHEN rank_this_week IS NULL THEN 1 ELSE 0 END, "
            "rank_this_week ASC, title ASC LIMIT ? OFFSET ?",
            args + [limit, offset])
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()
    return total, rows, weeks


def _td_fmt(key: str, v: Any) -> str:
    if v is None or v == "":
        return "—"
    if key in _TD_TEXT_COLS:
        return html.escape(str(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if key in ("in_hot_100", "is_number_one", "in_pool", "is_new_to_pool"):
        return "1" if f >= 0.5 else "0"
    if key in ("rank_this_week", "weeks_on_chart", "peak_position_so_far",
               "debut_rank", "last_seen_rank", "best_3wk_rank",
               "weeks_in_top10_so_far", "weeks_in_top40_so_far",
               "weeks_at_no1_so_far", "competition_last_week",
               "artist_prior_top10_songweeks", "artist_prior_top10_weeks",
               "artist_prior_no1_weeks"):
        return f"{int(f)}"
    if key in ("weeks_since_debut", "weeks_since_last_on_chart",
               "artist_weeks_since_last_top10", "rank_change_last_week"):
        return f"{f:.0f}"
    return f"{f:.3f}"


def render_training_data_panel(*, bot: Dict[str, Any],
                                  current_bot: str | None,
                                  page: int = 1, page_size: int = 100,
                                  week: str | None = None,
                                  current_tab: str = "training",
                                  period_key: str = "all") -> str:
    """Billboard Training Data tab — every (song, chart-week) row the
    trainer saw, all weeks, newest first: id columns + the dependent
    variable (in_hot_100) + the second label + every model feature.
    """
    out: List[str] = []
    out.append("<section class='card'><div class='body'>")
    out.append("<h2>Training Data — Billboard Hot 100</h2>")

    db_path = bot.get("training_db_path")
    if not db_path or not Path(db_path).exists():
        out.append(
            "<p class='small gray'>The training panel hasn't been "
            "generated on this host yet. Run "
            "<code>scripts/run_daily_train.py --offline</code> in the "
            "Billboard Charts repo (writes "
            "<code>data/processed/artifacts/training_data.db</code>).</p>"
            "</div></section>")
        return "".join(out)

    page = max(1, page)
    try:
        total, window, weeks = _query_training_db(
            db_path, week, (page - 1) * page_size, page_size)
    except Exception:  # noqa: BLE001
        log.exception("training_data.db query failed")
        out.append("<p class='small gray'>training_data.db exists but "
                   "could not be read — see dashboard log.</p>"
                   "</div></section>")
        return "".join(out)
    if week and week not in weeks:
        week = None
        total, window, weeks = _query_training_db(
            db_path, None, (page - 1) * page_size, page_size)
    total_pages = max(1, (total + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages
        _, window, _ = _query_training_db(
            db_path, week, (page - 1) * page_size, page_size)

    n_weeks = len(weeks)
    out.append(
        f"<p class='small gray'>One row per (song, chart week) — the "
        f"popular pool (songs that charted within the trailing 12 "
        f"weeks) plus this week's debuts. <b>{total:,}</b> rows"
        f"{' in this week' if week else f' across {n_weeks:,} weeks'}"
        f". <b>In Hot 100</b> is the dependent variable (1 = on that "
        f"week's chart, 0 = not); <b>Is #1</b> is the second label "
        f"that prices KXTOPSONG. Every feature uses only data from "
        f"strictly BEFORE the row's chart week. Sorted newest week "
        f"first, chart rank ascending. Click a column header for its "
        f"definition.</p>"
    )

    # Week filter — dropdown of every chart week in the panel.
    out.append("<form method='get' class='small' "
               "style='margin:8px 0;display:flex;gap:8px;"
               "align-items:center;'>")
    out.append(f"<input type='hidden' name='tab' "
               f"value='{html.escape(current_tab)}'>")
    if current_bot:
        out.append(f"<input type='hidden' name='bot' "
                   f"value='{html.escape(current_bot)}'>")
    if period_key and period_key != "all":
        out.append(f"<input type='hidden' name='period' "
                   f"value='{html.escape(period_key)}'>")
    out.append("<label class='gray'>Week:</label>"
               "<select name='week' onchange='this.form.submit()'>")
    sel_all = " selected" if not week else ""
    out.append(f"<option value=''{sel_all}>All weeks</option>")
    for w in weeks:
        sel = " selected" if w == week else ""
        out.append(f"<option value='{html.escape(str(w))}'{sel}>"
                   f"{html.escape(str(w))}</option>")
    out.append("</select></form>")

    defs: Dict[str, Dict[str, str]] = {}
    out.append("<div style='overflow-x:auto;margin-top:8px;'>")
    out.append("<table class='training-data-table'><thead><tr>")
    for key, label, definition in _TD_COLUMNS:
        defs[key] = {"label": label, "def": definition}
        cls = "" if key in _TD_TEXT_COLS else " class='num'"
        out.append(
            f"<th{cls}><button type='button' class='col-def-btn' "
            f"data-col='{html.escape(key)}'>{html.escape(label)}</button>"
            "</th>"
        )
    out.append("</tr></thead><tbody>")
    for r in window:
        label_on = False
        try:
            label_on = float(r.get("in_hot_100") or 0) >= 0.5
        except (TypeError, ValueError):
            pass
        out.append("<tr>")
        for key, _, _ in _TD_COLUMNS:
            cell = _td_fmt(key, r.get(key))
            cls = "" if key in _TD_TEXT_COLS else " class='num'"
            if key == "in_hot_100":
                color = "#3fb950" if label_on else "#f85149"
                cell = f"<b style='color:{color};'>{cell}</b>"
            out.append(f"<td{cls}>{cell}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")

    # ── Pagination (same chrome as the World Cup training panel) ─────
    def _page_link(p: int) -> str:
        params = [("tab", current_tab)]
        if current_bot:
            params.append(("bot", current_bot))
        if period_key and period_key != "all":
            params.append(("period", period_key))
        if week:
            params.append(("week", week))
        params.append(("page", str(p)))
        return "?" + "&".join(f"{k}={v}" for k, v in params)

    out.append("<div class='small' style='margin-top:14px;display:flex;"
               "align-items:center;gap:12px;'>")
    if page > 1:
        out.append(f"<a class='tab-pill' href='{_page_link(page - 1)}'>"
                   "← Prev</a>")
    else:
        out.append("<span class='tab-pill tab-pill-disabled'>← Prev</span>")
    out.append(f"<span>Page <b>{page:,}</b> of <b>{total_pages:,}</b> "
               f"<span class='gray'>({total:,} rows)</span></span>")
    out.append("<form method='get' style='display:inline;'>")
    out.append(f"<input type='hidden' name='tab' "
               f"value='{html.escape(current_tab)}'>")
    if current_bot:
        out.append(f"<input type='hidden' name='bot' "
                   f"value='{html.escape(current_bot)}'>")
    if period_key and period_key != "all":
        out.append(f"<input type='hidden' name='period' "
                   f"value='{html.escape(period_key)}'>")
    if week:
        out.append(f"<input type='hidden' name='week' "
                   f"value='{html.escape(week)}'>")
    out.append("<label class='gray' style='margin-right:6px;'>Jump:</label>")
    if total_pages <= 400:
        out.append("<select name='page' onchange='this.form.submit()'>")
        for p in range(1, total_pages + 1):
            sel = " selected" if p == page else ""
            out.append(f"<option value='{p}'{sel}>{p}</option>")
        out.append("</select>")
    else:
        out.append(
            f"<input type='number' name='page' min='1' "
            f"max='{total_pages}' value='{page}' style='width:80px;'>"
            "<button type='submit'>Go</button>"
        )
    out.append("</form>")
    if page < total_pages:
        out.append(f"<a class='tab-pill' href='{_page_link(page + 1)}'>"
                   "Next →</a>")
    else:
        out.append("<span class='tab-pill tab-pill-disabled'>Next →</span>")
    out.append("</div>")

    # ── Column-definition popover (same pattern as the WC panel) ─────
    out.append(
        "<div id='bb-col-def-pop' class='col-def-pop' hidden>"
        "<div class='col-def-pop-title'></div>"
        "<div class='col-def-pop-body'></div></div>"
    )
    out.append(
        "<style>"
        ".col-def-btn { background:none; border:0; color:inherit; "
        "font:inherit; cursor:pointer; padding:0; "
        "text-decoration:underline dotted; }"
        ".col-def-btn:hover { color:#79c0ff; }"
        ".col-def-pop { position:absolute; z-index:1000; max-width:320px; "
        "padding:10px 12px; background:#0d1117; color:#c9d1d9; "
        "border:1px solid #30363d; border-radius:6px; "
        "box-shadow:0 8px 24px rgba(0,0,0,.4); font-size:12px; }"
        ".col-def-pop-title { font-weight:600; margin-bottom:4px; }"
        "</style>"
    )
    out.append(
        "<script>(function(){"
        f"var defs = {json.dumps(defs)};"
        "var pop = document.getElementById('bb-col-def-pop');"
        "if (!pop) return;"
        "document.querySelectorAll('.training-data-table .col-def-btn')"
        ".forEach(function(btn){"
        "btn.addEventListener('click', function(ev){"
        "ev.stopPropagation();"
        "var d = defs[btn.dataset.col]; if (!d) return;"
        "pop.querySelector('.col-def-pop-title').textContent = d.label;"
        "pop.querySelector('.col-def-pop-body').textContent = d.def;"
        "var r = btn.getBoundingClientRect();"
        "pop.hidden = false;"
        "pop.style.left = Math.min(r.left + window.scrollX, "
        "window.scrollX + document.documentElement.clientWidth - 340) + 'px';"
        "pop.style.top = (r.bottom + window.scrollY + 6) + 'px';"
        "});});"
        "document.addEventListener('click', function(){ pop.hidden = true; });"
        "document.addEventListener('keydown', function(e){"
        "if (e.key === 'Escape') pop.hidden = true; });"
        "})();</script>"
    )
    out.append("</div></section>")
    return "".join(out)
