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
        ("Winner trains on", report.get("training_slice")
         or split.get("train", "—")),
        ("Training pool", split.get("train", "—")),
        ("Validation (feature pruning)", split.get("validation", "—")),
        ("Held-out test", split.get("test", "—")),
        ("History replayed for features",
         f"{ds.get('n_all_matches') or ds.get('n_matches', 0):,} "
         f"internationals "
         f"({(ds.get('date_range') or ['?', '?'])[0][:4]}–"
         f"{(ds.get('date_range') or ['?', '?'])[1][:4]})"),
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
    if report.get("shipped_note"):
        out.append(
            f"<p class='small gray' style='margin:0 0 12px 0;'>"
            f"{html.escape(report['shipped_note'])}</p>"
        )

    # ── Market benchmark ────────────────────────────────────────────
    mb = report.get("market_benchmark")
    if mb:
        model_ll, mkt_ll = mb.get("model_log_loss"), mb.get("market_log_loss")
        if model_ll is not None and mkt_ll is not None:
            gap = model_ll - mkt_ll
            verdict = ("model beats the market"
                       if gap < 0 else "market still leads")
            color = "#3fb950" if gap < 0 else "#d29922"
            out.append(
                f"<p class='small' style='margin:0 0 12px 0;padding:8px "
                f"12px;border:1px solid #30363d;border-radius:6px;'>"
                f"<b>Market benchmark</b> — on the same "
                f"{mb.get('n_matches', 0)} settled 2026 matches, Kalshi's "
                f"pre-kickoff prices score <b>{mkt_ll:.4f}</b> log loss "
                f"vs the model's <b>{model_ll:.4f}</b> "
                f"(<span style='color:{color};font-weight:600;'>"
                f"{gap:+.4f} — {verdict}</span>). "
                f"<span class='gray'>{html.escape(mb.get('note') or '')}"
                f"</span></p>"
            )

    # ── Ensemble composition ────────────────────────────────────────
    ens = report.get("ensemble") or {}
    comps = ens.get("components") or []
    if comps:
        out.append(
            f"<h3 class='subhead'>Ensemble composition "
            f"<span class='small gray'>(weights searched on 2010–2014 "
            f"validation; per-model temperature calibration; "
            f"Dixon-Coles rho = {ens.get('dc_rho', 0):+.2f}; training "
            f"decay half-life {ens.get('decay_half_life_years', 0):.0f} "
            f"years)</span></h3>"
        )
        out.append("<table><thead><tr><th>Component</th>"
                   "<th class='num'>Blend weight</th>"
                   "<th class='num'>Temperature</th>"
                   "<th class='num'>Features</th></tr></thead><tbody>")
        for c in sorted(comps, key=lambda c: -(c.get("weight") or 0)):
            w = float(c.get("weight") or 0)
            bar = (f"<div style='display:inline-block;height:8px;"
                   f"width:{max(2, round(w * 120))}px;"
                   f"background:#58a6ff;margin-right:6px;"
                   f"vertical-align:middle;'></div>")
            out.append(
                f"<tr><td><code>{html.escape(str(c.get('key')))}</code></td>"
                f"<td class='num'>{bar}{w * 100:.1f}%</td>"
                f"<td class='num'>{_fnum(c.get('temperature'), 2)}</td>"
                f"<td class='num'>{c.get('n_features', '—')}</td></tr>"
            )
        out.append("</tbody></table>")

    # ── Bake-off table ──────────────────────────────────────────────
    out.append(
        "<h3 class='subhead'>Model bake-off "
        "<span class='small gray'>(every family scored on the same "
        "untouched 2018–2026 test matches; lower log loss is better)"
        "</span></h3>"
    )
    out.append(
        "<table><thead><tr><th>Model</th><th>Family</th>"
        "<th class='num'>Features</th>"
        "<th class='num' title='Matches the model was fitted on'>"
        "Train rows</th><th class='num'>Log loss</th>"
        "<th class='num'>Brier</th><th class='num'>Accuracy</th>"
        "<th>Notes</th></tr></thead><tbody>"
    )
    baseline_ll = report.get("class_baseline_log_loss")
    for m in models:
        is_best = m.get("key") == best_key
        style = " style='color:#3fb950;font-weight:600;'" if is_best else ""
        name = html.escape(m.get("name", "?")) + (" ★" if is_best else "")
        rows_train = m.get("rows_train")
        out.append(
            f"<tr><td{style}>{name}</td>"
            f"<td>{html.escape(_FAMILY_LABELS.get(m.get('family', ''), m.get('family', '—')))}</td>"
            f"<td class='num'>{m.get('n_features', '—')}</td>"
            f"<td class='num'>{f'{int(rows_train):,}' if rows_train else '—'}</td>"
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
            f"<td class='num gray'>—</td>"
            f"<td class='num gray'>{_fnum(baseline_ll, 4)}</td>"
            f"<td class='num gray'>—</td><td class='num gray'>—</td>"
            f"<td class='small gray'>Always predicts the historical "
            f"win/draw/loss base rates.</td></tr>"
        )
    out.append("</tbody></table>")

    # ── Feature selection ───────────────────────────────────────────
    out.append(
        f"<h3 class='subhead'>Candidate features "
        f"<span class='small gray'>({len(candidates)} candidates → "
        f"pruned per track: backward elimination on finals CV, "
        f"permutation importance on the full history; ✓ = used by the "
        f"shipped winner)</span></h3>"
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
                            else "<span class='gray'>(start — all "
                                 f"{len(candidates)})</span>")
            out.append(
                f"<tr><td>{i}</td><td>{removed_cell}</td>"
                f"<td class='num'>{h.get('n_features', '—')}</td>"
                f"<td class='num'>{_fnum(h.get('cv_log_loss'), 5)}</td></tr>"
            )
        out.append("</tbody></table></details>")

    perm = report.get("permutation_importance") or []
    if perm:
        out.append("<details style='margin:8px 0 12px 0;'>"
                   "<summary class='small gray' style='cursor:pointer;'>"
                   "Permutation importance (all-matches track — mean "
                   "log-loss impact of shuffling each feature on "
                   "2010–2014 validation)</summary>")
        out.append("<table><thead><tr><th>Feature</th>"
                   "<th class='num'>Importance</th></tr></thead><tbody>")
        for p in perm:
            v = p.get("importance")
            style = "" if (v or 0) > 0 else " class='gray'"
            out.append(
                f"<tr{style}><td><code>{html.escape(p.get('feature', ''))}"
                f"</code></td>"
                f"<td class='num'>{_fnum(v, 5, signed=True)}</td></tr>")
        out.append("</tbody></table></details>")

    # ── Logistic coefficients (interpretability) ────────────────────
    coefs = report.get("coefficients_pruned_logistic") or []
    if coefs:
        coef_src = ("full-history logistic"
                    if report.get("coefficients_source") == "logit_all"
                    else "pruned finals logistic")
        out.append(
            f"<h3 class='subhead'>What moves the prediction "
            f"<span class='small gray'>({coef_src} — the interpretable "
            f"cousin of the winning model; 'per +1 SD' compares feature "
            f"strength, 'per unit' is the true coefficient in natural "
            f"units, e.g. per Elo point)</span></h3>"
        )
        out.append("<table><thead><tr><th>Feature</th>"
                   "<th class='num'>team1 win / +1 SD</th>"
                   "<th class='num'>team1 win / unit</th>"
                   "<th class='num'>draw / +1 SD</th>"
                   "<th class='num'>team2 win / +1 SD</th>"
                   "</tr></thead><tbody>")
        for c in coefs:
            out.append(
                f"<tr><td><code>{html.escape(c.get('feature', ''))}</code></td>"
                f"<td class='num'>{_fnum(c.get('team1'), 3, signed=True)}</td>"
                f"<td class='num'>{_fnum(c.get('team1_per_unit'), 4, signed=True)}</td>"
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
