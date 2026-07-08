"""World Cup Forecast dashboard pages.

Renders the Models tab and Training Data tab for the ``world-cup`` bot.
The bot is advisory-only for now — the model is NOT wired into any live
or sim trading loop — so both pages read static artifacts written by the
World Cup Forecast trainer:

  model_report.json   bake-off metrics, selected/pruned features,
                      calibration, upcoming-match predictions
                      (bot config key: ``model_report_path``)
  training_data.csv   one row per historical World Cup finals match with
                      pre-match features + the who-won label
                      (bot config key: ``training_data_path``)

Both loaders degrade to a friendly explanation when the files are
missing (e.g. before the repo is first pulled onto the droplet).
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #


def load_report(path: str | None) -> Dict[str, Any]:
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


def _load_training_rows(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _fnum(v: Any, decimals: int = 3, signed: bool = False) -> str:
    if v is None or v == "":
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if x == int(x) and abs(x) >= 1000:
        return f"{int(x):,}"
    fmt = f"{{:{'+' if signed else ''}.{decimals}f}}"
    return fmt.format(x)


def model_summary_for_card(model_report_path: str | None,
                           sim_state_path: str | None = None,
                           ) -> Dict[str, Any]:
    """Home-tab bot card summary, shaped like ``fetch_latest_model`` /
    ``tennis.model_summary_for_card`` output so the cross-bot card grid
    renders World Cup with the same cells as every other bot. Model
    metrics come from the bake-off winner on the held-out test slice;
    the actual-win ledger comes from the sim trader's state file.
    """
    report = load_report(model_report_path)
    if not report:
        return {}
    stats: Dict[str, Any] = {}
    if sim_state_path:
        p = Path(sim_state_path)
        if p.exists():
            try:
                stats = (json.loads(p.read_text()) or {}).get("stats") or {}
            except (OSError, json.JSONDecodeError):
                stats = {}
    models = report.get("models") or []
    best = next((m for m in models if m.get("key") == report.get("best_model")),
                models[0] if models else {})
    split = report.get("split") or {}
    return {
        "classifier_accuracy": best.get("accuracy"),
        "training_brier": best.get("brier"),
        "training_log_loss": best.get("log_loss"),
        "training_f1": best.get("f1"),
        "training_precision": best.get("precision"),
        "training_recall": best.get("recall"),
        # ROC AUC of the binary "team1 wins" probability — the closest
        # 3-class analogue to the other bots' binary AUC.
        "training_roc_auc": best.get("roc_auc_team1"),
        "feature_count": len(report.get("selected_features") or []),
        "rows_train": split.get("rows_train"),
        "rows_test": split.get("rows_test"),
        "actual_wins": int(stats.get("wins") or 0),
        "actual_losses": int(stats.get("losses") or 0),
    }


# --------------------------------------------------------------------------- #
# Models tab                                                                  #
# --------------------------------------------------------------------------- #

_FAMILY_LABELS = {
    "logistic": "Logistic regression",
    "tree ensemble": "Tree ensemble",
    "scoreline": "Scoreline (Dixon-Coles)",
    "ensemble": "Calibrated blend",
}


def render_models_panel(bot: dict) -> str:
    """World Cup Models tab — two-section layout shared with every
    other bot: (1) a table of every model the trainer produced with
    the same stats surfaced on the home-page model cards plus Brier,
    and (2) the readable-features panel with permutation-importance
    bars. All other historical sections (calibration table, upcoming
    matches, ensemble breakdown, etc.) have been removed to keep the
    layout identical across sports.
    """
    from .dashboard import (  # local import — avoids cycle
        _render_models_run_table, _render_feature_source_table,
    )
    report = load_report(bot.get("model_report_path"))
    out: List[str] = []

    if not report:
        return (
            "<div class='empty'>World Cup model artifacts not found. "
            "Run <code>src/train_models.py</code> in the World Cup "
            "Forecast repo (writes "
            "<code>data/processed/artifacts/model_report.json</code>) "
            "and pull the repo onto this host.</div>"
        )

    models = report.get("models") or []
    candidates = report.get("candidate_features") or []
    split = report.get("split") or {}
    best_key = report.get("best_model")

    # Rows_train/test come from the split blob (strings like "2010-2014");
    # the shared table also accepts numeric counts — pass through when
    # present so the two units read consistently.
    rows_train = split.get("rows_train")
    rows_test = split.get("rows_test")
    generated = (report.get("generated_at") or "")[:10]

    # Shape the models list into the trainer-metrics.json ``per_model``
    # convention the shared table consumes: one entry per model, keyed
    # by a human name. Sort so the winning model floats to the top.
    per_model: Dict[str, Dict[str, Any]] = {}
    models_sorted = sorted(models,
                            key=lambda m: (float(m.get("log_loss") or 1e9),
                                            0 if m.get("key") == best_key
                                                else 1))
    for m in models_sorted:
        name = m.get("name") or m.get("key") or "?"
        star = " ★" if m.get("key") == best_key else ""
        per_model[f"{name}{star}"] = {
            "accuracy": m.get("accuracy"),
            "log_loss": m.get("log_loss"),
            "brier": m.get("brier"),
            # Macro-averaged across the three outcomes (team1/draw/
            # team2); ROC AUC is the binary "team1 wins" probability —
            # the same numbers the home-page card shows.
            "f1": m.get("f1"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "roc_auc": m.get("roc_auc_team1"),
        }
    metrics_shim: Dict[str, Any] = {
        "per_model": per_model,
        "rows_train": rows_train,
        "rows_test": rows_test,
    }
    out.append(_render_models_run_table(
        metrics_shim,
        feature_count=len(candidates) if candidates else None,
        last_trained=generated or "—",
    ))

    # Features section — shape the candidate list into the shared
    # feature-importance CSV convention (feature / mean_importance /
    # positive_folds / selected) using permutation importance when
    # available, and the selected/candidate flag otherwise.
    perm = {p.get("feature"): p.get("importance") or 0.0
             for p in (report.get("permutation_importance") or [])
             if p.get("feature")}
    feats: List[Dict[str, Any]] = []
    for c in candidates:
        name = c.get("name") or ""
        feats.append({
            "feature": name,
            "mean_importance": float(perm.get(name, 0.0) or 0.0),
            "positive_folds": 5 if c.get("selected") else 0,
            "selected": bool(c.get("selected")),
        })
    out.append(_render_feature_source_table(feats))
    return "".join(out)




# --------------------------------------------------------------------------- #
# Training Data tab                                                           #
# --------------------------------------------------------------------------- #

# (csv column or derived key, header label, definition). Diff columns are
# derived at render time from the per-team CSV columns. The ⚙ MODEL
# FEATURE flag is added dynamically from the report's selected feature
# list so the table stays honest after a retrain changes the set.
_TD_COLUMNS: List[tuple] = [
    ("date", "Date", "Match date, YYYY-MM-DD."),
    ("tournament", "Tournament",
     "Competition the match was played in — FIFA World Cup, World Cup "
     "qualification, continental championships, friendlies, and every "
     "other official international."),
    ("team1", "Team 1",
     "First team as listed in the source data (the nominal 'home' side; "
     "most World Cup matches are on neutral ground)."),
    ("team2", "Team 2", "Second team as listed in the source data."),
    ("winner", "Winner",
     "THE DEPENDENT VARIABLE — who won this match. 'draw' for group-stage "
     "level results; knockout draws show the team that advanced on "
     "penalties with the shootout flag set."),
    ("score", "Score", "Full-time score, team1–team2 (90 min + extra time; "
     "penalty shootouts are not included in the score)."),
    ("shootout", "PSO?", "1 if the match was decided by a penalty shootout."),
    ("country", "Host country", "Country the match was played in."),
    ("team1_host", "T1 host?", "1 if team1 is the host nation."),
    ("team2_host", "T2 host?", "1 if team2 is the host nation."),
    ("neutral", "Neutral?",
     "1 if played at a neutral venue; 0 means team1 was at home."),
    ("team1_elo", "T1 Elo",
     "Team 1's pre-match World Football Elo rating, replayed over every "
     "international since 1872 (K-factor by competition importance, "
     "goal-difference multiplier, +100 home bonus)."),
    ("team2_elo", "T2 Elo", "Team 2's pre-match Elo rating."),
    ("elo_diff", "Elo Δ", "Team 1's Elo minus Team 2's — the single "
     "strongest predictor."),
    ("abs_elo_diff", "|Elo Δ|", "Absolute Elo gap — mismatch size "
     "regardless of side; large gaps suppress the draw probability."),
    ("elo_sum", "Elo Σ",
     "Sum of both teams' Elo — overall match quality; closer, stronger "
     "pairings draw more often."),
    ("form5_diff", "Form 5 Δ",
     "Avg points per match (3/1/0) over each team's last 5 "
     "internationals, team1 − team2."),
    ("form10_diff", "Form 10 Δ", "Same over the last 10 internationals."),
    ("winrate20_diff", "Win% 20 Δ",
     "Win rate over the last 20 internationals, team1 − team2."),
    ("gf10_diff", "Goals for Δ",
     "Goals scored per game over the last 10 internationals, "
     "team1 − team2 — attack strength."),
    ("ga10_diff", "Goals against Δ",
     "Goals conceded per game over the last 10, team1 − team2 — "
     "defensive leak (negative favors team1)."),
    ("rest_diff", "Rest Δ",
     "Days since each team's last international (capped at 60), "
     "team1 − team2."),
    ("matches_diff", "Career Δ",
     "Career internationals played, team1 − team2 — program maturity."),
    ("wc_matches_diff", "WC exp Δ",
     "World Cup finals matches previously played, team1 − team2 — "
     "tournament experience."),
    ("streak_diff", "Streak Δ",
     "Current win/loss streak (+N = N straight wins), team1 − team2."),
    ("h2h_n", "H2H played", "Prior meetings between the two teams "
     "across all competitions."),
    ("h2h_t1_edge", "H2H edge",
     "Team 1's head-to-head win rate minus 0.5 over prior meetings "
     "(0 when they never met)."),
    ("h2h_t1_gd", "H2H GD",
     "Team 1's average goal difference across prior meetings."),
    ("atk_diff", "Attack Δ",
     "Dixon-Coles-style attack rating difference — multiplicative "
     "goal-scoring strength, EWMA-updated against opponent defense."),
    ("defw_diff", "Def leak Δ",
     "Defensive-weakness rating difference (positive = team1 concedes "
     "more than its opposition quality explains)."),
    ("elo_mom_diff", "Elo mom Δ",
     "Elo momentum: rating change over each team's last 10 matches "
     "(diff) — rising vs fading sides."),
    ("sos10_diff", "SoS Δ",
     "Strength of schedule: average opponent Elo over the last 10 "
     "matches (diff) — who earned their recent form."),
    ("cleansheet10_diff", "Clean sheet Δ",
     "Clean-sheet rate over the last 10 matches (diff)."),
    ("blank10_diff", "Blanked Δ",
     "Failed-to-score rate over the last 10 matches (diff)."),
    ("congest30_diff", "Congestion Δ",
     "Matches played in the last 30 days (diff) — fixture load."),
    ("edition_matches_diff", "Edition Δ",
     "Matches already played in this tournament edition (diff)."),
    ("late_stage", "Late stage?",
     "1 when both sides have played 3+ matches in a finals-tournament "
     "edition — a knockout-round proxy."),
    ("importance", "Importance",
     "Match importance tier derived from the Elo K-factor: 4 = WC "
     "finals, ~3.3 = continental finals, ~2.7 = qualifiers, 1.33 = "
     "friendly."),
    ("same_confed", "Same confed?",
     "1 if both teams belong to the same confederation."),
    ("uefa_diff", "UEFA Δ",
     "UEFA membership, team1 − team2 (+1 = only team1 is European)."),
    ("conmebol_diff", "CONMEBOL Δ",
     "CONMEBOL membership, team1 − team2 (+1 = only team1 is South "
     "American)."),
    ("travel_diff_km", "Travel Δ",
     "Distance from each team's home country to the host country "
     "(team1 − team2, km) — positive = team1 travelled farther."),
    ("altitude_m", "Altitude",
     "Venue altitude in metres (0 outside known high-altitude "
     "venues such as Mexico City or La Paz)."),
]

# Map derived diff key -> (team1 csv col, team2 csv col)
_DIFF_SOURCES = {
    "form5_diff": ("team1_form5", "team2_form5"),
    "form10_diff": ("team1_form10", "team2_form10"),
    "winrate20_diff": ("team1_winrate20", "team2_winrate20"),
    "gf10_diff": ("team1_gf10", "team2_gf10"),
    "ga10_diff": ("team1_ga10", "team2_ga10"),
    "rest_diff": ("team1_rest_days", "team2_rest_days"),
    "matches_diff": ("team1_matches", "team2_matches"),
    "wc_matches_diff": ("team1_wc_matches", "team2_wc_matches"),
    "streak_diff": ("team1_streak", "team2_streak"),
}

_TEXT_COLS = {"date", "tournament", "team1", "team2", "winner", "score",
              "country"}

# Segment filter pills: query value -> label. Values match the
# ``segment`` column build_training_data.py writes.
_SEGMENTS = [
    (None, "All"),
    ("wc_finals", "WC finals"),
    ("wc_qualifier", "WC qualifiers"),
    ("continental", "Continental"),
    ("friendly", "Friendlies"),
    ("other", "Other"),
]
_SEGMENT_VALUES = {s for s, _ in _SEGMENTS if s}


def _query_history_db(db_path: str, segment: str | None,
                      offset: int, limit: int):
    """(total, rows) page from the full-grain SQLite, newest first."""
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        where = "WHERE segment = ?" if segment else ""
        args = [segment] if segment else []
        total = con.execute(
            f"SELECT COUNT(*) FROM matches {where}", args).fetchone()[0]
        cur = con.execute(
            f"SELECT * FROM matches {where} "
            f"ORDER BY date DESC, rowid DESC LIMIT ? OFFSET ?",
            args + [limit, offset])
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()
    return total, rows


def _derive(row: Dict[str, Any], key: str) -> Any:
    def f(col):
        v = row.get(col)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    if key in _DIFF_SOURCES:
        a, b = (f(c) for c in _DIFF_SOURCES[key])
        return None if a is None or b is None else a - b
    if key == "elo_diff":
        return f("elo_diff")
    if key == "abs_elo_diff":
        v = f("elo_diff")
        return None if v is None else abs(v)
    if key == "score":
        t1, t2 = row.get("team1_score"), row.get("team2_score")
        if t1 in (None, "") or t2 in (None, ""):
            return None
        return f"{int(float(t1))}–{int(float(t2))}"
    if key == "winner":
        w = row.get("winner")
        return w if w else "draw"
    if key == "h2h_t1_edge":
        v = f("h2h_t1_winrate")
        return None if v is None else v - 0.5
    return row.get(key)


def render_training_data_panel(*, bot: dict, current_bot: str | None,
                                  page: int = 1, page_size: int = 20,
                                  segment: str | None = None,
                                  current_tab: str = "training",
                                  period_key: str = "all") -> str:
    report = load_report(bot.get("model_report_path"))
    selected = set(report.get("selected_features") or [])
    segment = segment if segment in _SEGMENT_VALUES else None

    out: List[str] = []
    out.append("<section class='card'><div class='body'>")
    out.append("<h2>Training Data — World Cup</h2>")

    # Full-grain SQLite is the primary source; the WC-finals CSV keeps
    # the page alive on hosts that predate the history DB.
    db_path = bot.get("training_db_path")
    total = 0
    window: List[Dict[str, Any]] = []
    if db_path and Path(db_path).exists():
        total_pages = 1  # recomputed below once total is known
        page = max(1, page)
        try:
            total, window = _query_history_db(
                db_path, segment, (page - 1) * page_size, page_size)
        except Exception:  # noqa: BLE001
            total, window = 0, []
        if total and not window and page > 1:
            # page beyond the end (e.g. filter changed) — clamp to last
            total_pages = max(1, (total + page_size - 1) // page_size)
            page = total_pages
            _, window = _query_history_db(
                db_path, segment, (page - 1) * page_size, page_size)
    if not window:
        rows = _load_training_rows(bot.get("training_data_path"))
        if not rows:
            out.append(
                "<p class='small gray'>The training dataset hasn't been "
                "generated on this host yet. Run "
                "<code>src/build_training_data.py</code> in the World Cup "
                "Forecast repo (writes <code>data/training_history.db</code>)"
                " and pull the repo here.</p></div></section>"
            )
            return "".join(out)
        segment = None
        total = len(rows)
        view = sorted(rows, key=lambda r: r.get("date", ""), reverse=True)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), total_pages)
        window = view[(page - 1) * page_size:(page - 1) * page_size
                      + page_size]

    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)

    out.append(
        f"<p class='small gray'>Every official men's international "
        f"since 1872 — <b>{total:,}</b> matches"
        f"{' in this slice' if segment else ''}. One row per match; "
        f"<b>Winner</b> is the dependent variable the models are "
        f"trained to predict (3-way: team1 / draw / team2). All "
        f"features are computed strictly from matches played "
        f"<i>before</i> the row's date by replaying the full history in "
        f"order, so nothing leaks the outcome. The shipped model trains "
        f"on {html.escape(report.get('training_slice') or 'the WC-finals slice')}; "
        f"the bake-off races finals-only and full-history variants. "
        f"Sorted newest first. Click a column header for its "
        f"definition; ⚙ MODEL FEATURE marks the {len(selected)} "
        f"features that survived pruning.</p>"
    )

    # No competition filter (removed per operator request) — the table
    # always pages the full grain; the Tournament column still shows
    # each row's competition. The ``seg`` query param stays supported
    # for hand-built URLs but no UI emits it.

    defs: Dict[str, Dict[str, str]] = {}
    out.append("<div style='overflow-x:auto;margin-top:12px;'>")
    out.append("<table class='training-data-table'><thead><tr>")
    for key, label, definition in _TD_COLUMNS:
        if key in selected:
            definition += " ⚙ MODEL FEATURE."
        defs[key] = {"label": label, "def": definition}
        cls = "" if key in _TEXT_COLS else " class='num'"
        out.append(
            f"<th{cls}><button type='button' class='col-def-btn' "
            f"data-col='{html.escape(key)}'>{html.escape(label)}</button>"
            "</th>"
        )
    out.append("</tr></thead><tbody>")

    for r in window:
        out.append("<tr>")
        for key, _, _ in _TD_COLUMNS:
            v = _derive(r, key)
            if v is None or v == "":
                cell = "—"
            elif key in _TEXT_COLS:
                cell = html.escape(str(v))
            elif key in ("team1_host", "team2_host", "neutral", "shootout",
                         "late_stage", "same_confed"):
                cell = "Yes" if str(v) in ("1", "1.0") else "No"
            elif key in ("altitude_m", "importance"):
                cell = _fnum(v, 2 if key == "importance" else 0)
            elif key.endswith("_diff") or key in ("h2h_t1_edge", "h2h_t1_gd"):
                cell = _fnum(v, 2, signed=True)
            elif key in ("team1_elo", "team2_elo"):
                cell = _fnum(v, 0)
            elif key == "h2h_n":
                cell = _fnum(v, 0)
            else:
                cell = _fnum(v, 2)
            cls = "" if key in _TEXT_COLS else " class='num'"
            out.append(f"<td{cls}>{cell}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")

    # ── Pagination ──────────────────────────────────────────────────
    def _page_link(p: int) -> str:
        params = [("tab", current_tab)]
        if current_bot:
            params.append(("bot", current_bot))
        if period_key and period_key != "all":
            params.append(("period", period_key))
        if segment:
            params.append(("seg", segment))
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
               f"<span class='gray'>({total:,} matches)</span></span>")
    out.append("<form method='get' style='display:inline;'>")
    out.append(f"<input type='hidden' name='tab' "
               f"value='{html.escape(current_tab)}'>")
    if current_bot:
        out.append(f"<input type='hidden' name='bot' "
                   f"value='{html.escape(current_bot)}'>")
    if period_key and period_key != "all":
        out.append(f"<input type='hidden' name='period' "
                   f"value='{html.escape(period_key)}'>")
    if segment:
        out.append(f"<input type='hidden' name='seg' "
                   f"value='{html.escape(segment)}'>")
    out.append("<label class='gray' style='margin-right:6px;'>Jump:</label>")
    if total_pages <= 400:
        out.append("<select name='page' onchange='this.form.submit()'>")
        for p in range(1, total_pages + 1):
            sel = " selected" if p == page else ""
            out.append(f"<option value='{p}'{sel}>{p}</option>")
        out.append("</select>")
    else:
        # Full history is ~2,500 pages — a dropdown that size bloats
        # every render, so fall back to a numeric jump box.
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

    # ── Column-definition popover (same pattern as the tennis panel) ─
    out.append(
        "<div id='wc-col-def-pop' class='col-def-pop' hidden>"
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
        "var pop = document.getElementById('wc-col-def-pop');"
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
