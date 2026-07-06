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


# --------------------------------------------------------------------------- #
# Models tab                                                                  #
# --------------------------------------------------------------------------- #

_FAMILY_LABELS = {
    "logistic": "Logistic regression",
    "tree ensemble": "Tree ensemble",
    "scoreline": "Scoreline (Poisson)",
}


def render_models_panel(bot: dict) -> str:
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

    # ── Advisory banner ─────────────────────────────────────────────
    out.append(
        "<p class='small' style='margin:0 0 12px 0;padding:8px 12px;"
        "border:1px solid #d29922;border-radius:6px;color:#d29922;'>"
        "Advisory only — this model is <b>not wired into live or sim "
        "trading</b>. The pages here exist to evaluate the model before "
        "any capital (paper or real) follows it."
        "</p>"
    )

    models = report.get("models") or []
    best_key = report.get("best_model")
    best = next((m for m in models if m.get("key") == best_key),
                models[0] if models else {})
    selected = report.get("selected_features") or []
    candidates = report.get("candidate_features") or []
    split = report.get("split") or {}

    # ── Headline cards ──────────────────────────────────────────────
    out.append(
        "<div class='cards' style='display:grid;"
        "grid-template-columns:repeat(6, 1fr);gap:10px;width:100%;'>"
    )
    cards = [
        ("Best model", best.get("name", "—")),
        ("Test log loss", _fnum(best.get("log_loss"), 4)),
        ("Brier", _fnum(best.get("brier"), 4)),
        ("Accuracy", (f"{float(best.get('accuracy', 0)) * 100:.1f}%"
                      if best.get("accuracy") is not None else "—")),
        ("Features", f"{len(selected)} of {len(candidates)}"),
        ("Held-out matches", f"{int(split.get('rows_test') or 0):,}"),
    ]
    for label, value in cards:
        out.append(
            f"<div class='card'><div class='label'>{html.escape(label)}"
            f"</div><div class='value' style='font-size:15px;'>"
            f"{html.escape(str(value))}</div></div>"
        )
    out.append("</div>")

    # ── Model overview ──────────────────────────────────────────────
    ds = report.get("dataset") or {}
    overview = [
        ("Task", "3-way match outcome: team1 win / draw / team2 win"),
        ("Training matches", split.get("train", "—")),
        ("Validation (feature pruning)", split.get("validation", "—")),
        ("Held-out test", split.get("test", "—")),
        ("History replayed for features",
         f"{ds.get('n_matches', 0):,} World Cup matches "
         f"({(ds.get('date_range') or ['?', '?'])[0][:4]}–"
         f"{(ds.get('date_range') or ['?', '?'])[1][:4]}), features "
         "computed from the full 1872-present international history"),
        ("Data source", ds.get("source", "—")),
        ("Report generated", (report.get("generated_at") or "—")[:19]
         .replace("T", " ") + " UTC"),
    ]
    out.append("<h3 class='subhead'>Model overview</h3>")
    out.append("<dl style='display:grid;grid-template-columns:auto 1fr;"
               "gap:6px 18px;margin:0 0 12px 0;font-size:13px;'>")
    for label, value in overview:
        out.append(
            f"<dt class='gray' style='margin:0;'>{html.escape(label)}</dt>"
            f"<dd style='margin:0;color:#c9d1d9;'>"
            f"{html.escape(str(value))}</dd>"
        )
    out.append("</dl>")

    # ── Bake-off table ──────────────────────────────────────────────
    out.append(
        "<h3 class='subhead'>Model bake-off "
        "<span class='small gray'>(every family scored on the same "
        "untouched 2018–2026 test matches; lower log loss is better)"
        "</span></h3>"
    )
    out.append(
        "<table><thead><tr><th>Model</th><th>Family</th>"
        "<th class='num'>Features</th><th class='num'>Log loss</th>"
        "<th class='num'>Brier</th><th class='num'>Accuracy</th>"
        "<th>Notes</th></tr></thead><tbody>"
    )
    baseline_ll = report.get("class_baseline_log_loss")
    for m in models:
        is_best = m.get("key") == best_key
        style = " style='color:#3fb950;font-weight:600;'" if is_best else ""
        name = html.escape(m.get("name", "?")) + (" ★" if is_best else "")
        out.append(
            f"<tr><td{style}>{name}</td>"
            f"<td>{html.escape(_FAMILY_LABELS.get(m.get('family', ''), m.get('family', '—')))}</td>"
            f"<td class='num'>{m.get('n_features', '—')}</td>"
            f"<td class='num'{style}>{_fnum(m.get('log_loss'), 4)}</td>"
            f"<td class='num'>{_fnum(m.get('brier'), 4)}</td>"
            f"<td class='num'>{float(m.get('accuracy', 0)) * 100:.1f}%</td>"
            f"<td class='small gray'>{html.escape(m.get('note') or '')}</td>"
            "</tr>"
        )
    if baseline_ll is not None:
        out.append(
            f"<tr><td class='gray'>Class priors (know-nothing floor)</td>"
            f"<td class='gray'>baseline</td><td class='num gray'>0</td>"
            f"<td class='num gray'>{_fnum(baseline_ll, 4)}</td>"
            f"<td class='num gray'>—</td><td class='num gray'>—</td>"
            f"<td class='small gray'>Always predicts the historical "
            f"win/draw/loss base rates.</td></tr>"
        )
    out.append("</tbody></table>")

    # ── Feature selection ───────────────────────────────────────────
    out.append(
        "<h3 class='subhead'>Candidate features "
        "<span class='small gray'>(18 candidates → greedy backward "
        "elimination on expanding-window CV log loss; ✓ = survives "
        "pruning and is used by the shipped models)</span></h3>"
    )
    out.append(
        "<table><thead><tr><th></th><th>Feature</th><th>Definition</th>"
        "</tr></thead><tbody>"
    )
    for c in candidates:
        sel = c.get("selected")
        mark = ("<span style='color:#3fb950;font-weight:700;'>✓</span>"
                if sel else "<span class='gray'>✗</span>")
        name_style = "" if sel else " class='gray'"
        out.append(
            f"<tr><td>{mark}</td>"
            f"<td{name_style}><code>{html.escape(c.get('name', ''))}</code></td>"
            f"<td class='small gray'>{html.escape(c.get('description', ''))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")

    prune_history = report.get("prune_history") or []
    if prune_history:
        out.append("<details style='margin:8px 0 12px 0;'>"
                   "<summary class='small gray' style='cursor:pointer;'>"
                   "Pruning order (each step removes the feature whose "
                   "removal most improves CV log loss)</summary>")
        out.append("<table><thead><tr><th>Step</th><th>Removed</th>"
                   "<th class='num'>Features left</th>"
                   "<th class='num'>CV log loss</th></tr></thead><tbody>")
        for i, h in enumerate(prune_history):
            removed = h.get("removed")
            removed_cell = (f"<code>{html.escape(removed)}</code>" if removed
                            else "<span class='gray'>(start — all 18)"
                                 "</span>")
            out.append(
                f"<tr><td>{i}</td><td>{removed_cell}</td>"
                f"<td class='num'>{h.get('n_features', '—')}</td>"
                f"<td class='num'>{_fnum(h.get('cv_log_loss'), 5)}</td></tr>"
            )
        out.append("</tbody></table></details>")

    # ── Logistic coefficients (interpretability) ────────────────────
    coefs = report.get("coefficients_pruned_logistic") or []
    if coefs:
        out.append(
            "<h3 class='subhead'>What moves the prediction "
            "<span class='small gray'>(pruned-logistic coefficients on "
            "standardized features — the interpretable cousin of the "
            "winning forest)</span></h3>"
        )
        out.append("<table><thead><tr><th>Feature</th>"
                   "<th class='num'>team1 win</th><th class='num'>draw</th>"
                   "<th class='num'>team2 win</th></tr></thead><tbody>")
        for c in coefs:
            out.append(
                f"<tr><td><code>{html.escape(c.get('feature', ''))}</code></td>"
                f"<td class='num'>{_fnum(c.get('team1'), 3, signed=True)}</td>"
                f"<td class='num'>{_fnum(c.get('draw'), 3, signed=True)}</td>"
                f"<td class='num'>{_fnum(c.get('team2'), 3, signed=True)}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")

    # ── Calibration ─────────────────────────────────────────────────
    cal = report.get("calibration_team1_win") or []
    filled = [b for b in cal if b.get("n")]
    if filled:
        out.append(
            "<h3 class='subhead'>Calibration — P(team1 wins) "
            "<span class='small gray'>(held-out test set; a "
            "well-calibrated model's observed bar matches its "
            "predicted bar in every bin)</span></h3>"
        )
        out.append("<table><thead><tr><th>Predicted bin</th>"
                   "<th class='num'>Matches</th>"
                   "<th class='num'>Avg predicted</th>"
                   "<th class='num'>Observed win rate</th>"
                   "<th style='width:40%;'>Predicted vs observed</th>"
                   "</tr></thead><tbody>")
        for b in filled:
            pred = float(b["pred"]) if b.get("pred") is not None else 0.0
            obs = float(b["obs"]) if b.get("obs") is not None else 0.0
            bar = (
                "<div style='position:relative;height:14px;'>"
                f"<div style='position:absolute;left:0;top:0;height:6px;"
                f"width:{pred * 100:.0f}%;background:#58a6ff;'></div>"
                f"<div style='position:absolute;left:0;top:8px;height:6px;"
                f"width:{obs * 100:.0f}%;background:#3fb950;'></div></div>"
            )
            out.append(
                f"<tr><td>{b['lo']:.1f}–{b['hi']:.1f}</td>"
                f"<td class='num'>{b['n']}</td>"
                f"<td class='num'>{pred:.2f}</td>"
                f"<td class='num'>{obs:.2f}</td>"
                f"<td>{bar}</td></tr>"
            )
        out.append("</tbody></table>")
        out.append(
            "<p class='small gray' style='margin-top:4px;'>"
            "<span style='color:#58a6ff;'>■</span> predicted&nbsp;&nbsp;"
            "<span style='color:#3fb950;'>■</span> observed</p>"
        )

    # ── Upcoming matches (advisory) ─────────────────────────────────
    upcoming = report.get("upcoming_predictions") or []
    if upcoming:
        out.append(
            "<h3 class='subhead'>Upcoming World Cup matches "
            "<span class='small gray'>(scored by the winning model — "
            "advisory only, nothing is traded)</span></h3>"
        )
        out.append("<table><thead><tr><th>Date</th><th>Match</th>"
                   "<th class='num'>Team 1 wins</th>"
                   "<th class='num'>Draw</th>"
                   "<th class='num'>Team 2 wins</th></tr></thead><tbody>")
        for u in upcoming:
            probs = [float(u.get("p_team1") or 0),
                     float(u.get("p_draw") or 0),
                     float(u.get("p_team2") or 0)]
            hi = probs.index(max(probs))
            cells = []
            for i, p in enumerate(probs):
                style = (" style='color:#3fb950;font-weight:600;'"
                         if i == hi else "")
                cells.append(f"<td class='num'{style}>{p * 100:.0f}%</td>")
            out.append(
                f"<tr><td>{html.escape(u.get('date', ''))}</td>"
                f"<td>{html.escape(u.get('team1', ''))} vs "
                f"{html.escape(u.get('team2', ''))}</td>"
                + "".join(cells) + "</tr>"
            )
        out.append("</tbody></table>")

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

_TEXT_COLS = {"date", "team1", "team2", "winner", "score", "country"}


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
                                  current_tab: str = "training",
                                  period_key: str = "all") -> str:
    rows = _load_training_rows(bot.get("training_data_path"))
    report = load_report(bot.get("model_report_path"))
    selected = set(report.get("selected_features") or [])

    out: List[str] = []
    out.append("<section class='card'><div class='body'>")
    out.append("<h2>Training Data — World Cup</h2>")

    if not rows:
        out.append(
            "<p class='small gray'>The training dataset hasn't been "
            "generated on this host yet. Run "
            "<code>src/build_training_data.py</code> in the World Cup "
            "Forecast repo (writes <code>data/training_data.csv</code>) "
            "and pull the repo here.</p></div></section>"
        )
        return "".join(out)

    total = len(rows)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)

    out.append(
        f"<p class='small gray'>Every FIFA World Cup finals match ever "
        f"played — <b>{total:,}</b> matches, 1930 through the current "
        f"tournament. One row per match; <b>Winner</b> is the dependent "
        f"variable the models are trained to predict (3-way: team1 / "
        f"draw / team2). All features are computed strictly from "
        f"matches played <i>before</i> the row's date by replaying the "
        f"full 49k-match international history, so nothing leaks the "
        f"outcome. Sorted newest first. Click a column header for its "
        f"definition; ⚙ MODEL FEATURE marks the "
        f"{len(selected)} features that survived pruning.</p>"
    )

    # newest first
    view = sorted(rows, key=lambda r: r.get("date", ""), reverse=True)
    start = (page - 1) * page_size
    window = view[start:start + page_size]

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
            elif key in ("team1_host", "team2_host", "neutral", "shootout"):
                cell = "Yes" if str(v) in ("1", "1.0") else "No"
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
    out.append("<label class='gray' style='margin-right:6px;'>Jump:</label>")
    out.append("<select name='page' onchange='this.form.submit()'>")
    for p in range(1, total_pages + 1):
        sel = " selected" if p == page else ""
        out.append(f"<option value='{p}'{sel}>{p}</option>")
    out.append("</select></form>")
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
