"""Hormuz Forecast dashboard pages — Models tab + Training Data tab.

The Hormuz Forecast bot (repo: Port Forecast → /root/port-forecast) is a
standard sim.db macro bot: its watchlist, home card, and positions all
flow through the shared ``dashboard_type: standard`` path. Only these two
tabs are bespoke, because the user asked for a specific layout:

  Models tab
    Part 1 — the model bake-off + the deployed Ridge model's coefficients
    Part 2 — every feature that feeds the model, with name, description,
             current value, and a bar visualising its influence

  Training Data tab
    The full weekly panel: every independent feature column plus the
    dependent variable (``peak`` — the week's highest daily transit count).

Both read committed artifacts written by the bot:
  model_card.json    (bot config key ``model_report_path``)
  training_data.csv  (bot config key ``training_data_path``)

Both degrade to a friendly message when the files are missing (e.g. before
the repo is first pulled onto the droplet).
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List

# Feature keys shown with 0-decimal integer formatting on the training
# table (ship counts); everything else gets 2 decimals.
_INT_FEATURES = {"peak", "peak_lag1", "peak_lag2", "peak_lag4", "peak_ma4",
                 "momentum_4w", "regime"}


def _load_card(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_rows(path: str | None) -> List[Dict[str, Any]]:
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


def _fnum(v: Any, decimals: int = 2) -> str:
    if v is None or v == "":
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if decimals == 0:
        return f"{int(round(x)):,}"
    return f"{x:.{decimals}f}"


_CSS = (
    "<style>"
    ".hz-note{background:#161b22;border:1px solid #30363d;border-radius:6px;"
    "padding:10px 12px;margin:8px 0 16px;font-size:12px;color:#8b949e;}"
    ".hz-stat-row{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 14px;}"
    ".hz-stat{background:#0d1117;border:1px solid #30363d;border-radius:8px;"
    "padding:10px 14px;min-width:120px;}"
    ".hz-stat .k{font-size:11px;color:#8b949e;text-transform:uppercase;"
    "letter-spacing:.04em;}"
    ".hz-stat .v{font-size:20px;font-weight:600;color:#e6edf3;margin-top:2px;}"
    ".hz-table{width:100%;border-collapse:collapse;font-size:13px;}"
    ".hz-table th,.hz-table td{padding:7px 10px;border-bottom:1px solid #21262d;"
    "text-align:left;}"
    ".hz-table th{color:#8b949e;font-weight:600;font-size:11px;"
    "text-transform:uppercase;letter-spacing:.03em;}"
    ".hz-table td.num,.hz-table th.num{text-align:right;font-variant-numeric:"
    "tabular-nums;}"
    ".hz-table tr.best td{background:rgba(63,185,80,.08);}"
    ".hz-bar-wrap{display:flex;align-items:center;gap:8px;min-width:150px;}"
    ".hz-bar-track{position:relative;flex:1;height:8px;background:#21262d;"
    "border-radius:4px;overflow:hidden;}"
    ".hz-bar-fill{position:absolute;top:0;bottom:0;border-radius:4px;}"
    ".hz-bar-num{font-variant-numeric:tabular-nums;font-size:12px;"
    "color:#c9d1d9;min-width:52px;text-align:right;}"
    ".hz-pos{background:#3fb950;}.hz-neg{background:#f85149;}"
    ".hz-sec-title{font-size:13px;font-weight:600;color:#e6edf3;"
    "margin:20px 0 8px;}"
    "</style>"
)


# --------------------------------------------------------------------------- #
# Home-card summary (cross-bot grid uses fetch_latest_model for standard bots,
# so no bespoke summary is needed — the sim.db model_snapshots row drives it.)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Models tab
# --------------------------------------------------------------------------- #

_BAKEOFF_LABELS = {
    "ridge": "Ridge regression (Hormuz-only)",
    "pooled": "Pooled 28-chokepoint GBM",
    "trend_extrap": "Trend extrapolation",
    "ma4": "4-week moving average",
    "persistence": "Persistence (last week)",
}


def render_models_panel(out: List[str], bot: dict) -> None:
    card = _load_card(bot.get("model_report_path"))
    if not card:
        out.append(
            "<div class='empty'>Hormuz model artifacts not found. Run "
            "<code>python run.py --train</code> in the Port Forecast repo "
            "(writes <code>data/model_card.json</code>) and pull the repo "
            "onto this host.</div>"
        )
        return

    out.append(_CSS)

    metrics = card.get("metrics") or {}
    forecast = card.get("forecast_peak")
    psf = card.get("peak_so_far")
    tws = card.get("target_week_start") or "—"
    sib = (card.get("sibling_context") or {}).get("implied_peak")

    # ── Headline stats ──────────────────────────────────────────────
    out.append("<div class='hz-stat-row'>")
    def _stat(k, v):
        out.append(f"<div class='hz-stat'><div class='k'>{html.escape(k)}</div>"
                   f"<div class='v'>{v}</div></div>")
    _stat("Forecast peak", _fnum(forecast, 1))
    _stat("Peak so far", _fnum(psf, 0))
    _stat("Residual σ", _fnum(card.get("residual_std"), 1))
    _stat("Walk-fwd MAE", _fnum(metrics.get("mae"), 1))
    _stat("Dir. accuracy",
          "—" if metrics.get("directional_accuracy") is None
          else f"{float(metrics['directional_accuracy'])*100:.0f}%")
    if sib is not None:
        _stat("Market-implied", _fnum(sib, 0))
    out.append("</div>")

    out.append(
        f"<div class='hz-note'>Forecasting the peak daily transit count for "
        f"the contract week starting <b>{html.escape(str(tws))}</b>. "
        f"{html.escape(card.get('notes') or '')} Settlement source: "
        f"{html.escape(card.get('data_source') or 'IMF PortWatch')}.</div>"
    )

    # ── Part 1a: model bake-off ─────────────────────────────────────
    out.append("<div class='hz-sec-title'>Models — walk-forward bake-off</div>")
    out.append("<table class='hz-table'><thead><tr>"
               "<th>Model</th><th class='num'>MAE (ships)</th>"
               "<th class='num'>RMSE</th><th class='num'>Dir. acc</th>"
               "<th class='num'>Weeks</th></tr></thead><tbody>")
    _deployed = (metrics.get("deployed") or "ridge")
    for b in card.get("bakeoff") or []:
        key = b.get("model")
        label = _BAKEOFF_LABELS.get(key, key or "—")
        if key == _deployed:
            label += " (deployed) ★"
        cls = " class='best'" if key == _deployed else ""
        dacc = b.get("dir_acc")
        out.append(
            f"<tr{cls}><td>{html.escape(label)}</td>"
            f"<td class='num'>{_fnum(b.get('mae'),2)}</td>"
            f"<td class='num'>{_fnum(b.get('rmse'),2)}</td>"
            f"<td class='num'>"
            f"{'—' if dacc is None else f'{float(dacc)*100:.0f}%'}</td>"
            f"<td class='num'>{b.get('n','—')}</td></tr>"
        )
    out.append("</tbody></table>")

    # ── Part 1b: deployed model coefficients ────────────────────────
    feats = card.get("features") or []
    coefs = [(f.get("name") or f.get("key"), f.get("key"),
              float(f.get("coefficient") or 0.0)) for f in feats]
    max_c = max((abs(c) for _, _, c in coefs), default=1.0) or 1.0
    if _deployed == "pooled":
        _coef_title = ("Deployed model — pooled GBM feature importances"
                       " <span style='color:#8b949e;font-weight:400;'>"
                       "(split-gain share; trained on every chokepoint)</span>")
        _coef_col = "Importance"
    else:
        _coef_title = ("Deployed model — Ridge coefficients"
                       " <span style='color:#8b949e;font-weight:400;'>"
                       f"(standardized; intercept "
                       f"{_fnum(card.get('intercept'),2)})</span>")
        _coef_col = "Coefficient"
    out.append(f"<div class='hz-sec-title'>{_coef_title}</div>")
    out.append("<table class='hz-table'><thead><tr><th>Feature</th>"
               f"<th class='num'>{_coef_col}</th><th>Relative influence</th>"
               "</tr></thead><tbody>")
    for name, key, c in sorted(coefs, key=lambda t: -abs(t[2])):
        pct = abs(c) / max_c * 100.0
        cls = "hz-pos" if c >= 0 else "hz-neg"
        out.append(
            f"<tr><td>{html.escape(str(name))}</td>"
            f"<td class='num'>{c:+.3f}</td>"
            f"<td><div class='hz-bar-wrap'><div class='hz-bar-track'>"
            f"<div class='hz-bar-fill {cls}' style='width:{pct:.1f}%;'></div>"
            f"</div></div></td></tr>"
        )
    out.append("</tbody></table>")

    # ── Part 2: features that feed the model (value + bar) ──────────
    out.append("<div class='hz-sec-title'>Features — current values feeding "
               "the forecast</div>")
    vals = [(f.get("name") or f.get("key"), f.get("description") or "",
             f.get("value")) for f in feats]
    # Bar scales each feature's current value against the max abs value
    # across features so the reader sees relative magnitude at a glance.
    max_v = max((abs(float(v)) for _, _, v in vals
                 if v is not None), default=1.0) or 1.0
    out.append("<table class='hz-table'><thead><tr><th>Feature</th>"
               "<th>Description</th><th class='num'>Current value</th>"
               "<th>Magnitude</th></tr></thead><tbody>")
    for name, desc, v in vals:
        if v is None:
            barcell = "<span style='color:#8b949e;'>n/a</span>"
            valcell = "—"
        else:
            fv = float(v)
            pct = abs(fv) / max_v * 100.0
            cls = "hz-pos" if fv >= 0 else "hz-neg"
            barcell = (f"<div class='hz-bar-wrap'><div class='hz-bar-track'>"
                       f"<div class='hz-bar-fill {cls}' "
                       f"style='width:{pct:.1f}%;'></div></div></div>")
            valcell = _fnum(fv, 2)
        out.append(
            f"<tr><td>{html.escape(str(name))}</td>"
            f"<td style='color:#8b949e;'>{html.escape(str(desc))}</td>"
            f"<td class='num'>{valcell}</td>"
            f"<td>{barcell}</td></tr>"
        )
    out.append("</tbody></table>")


# --------------------------------------------------------------------------- #
# Training Data tab
# --------------------------------------------------------------------------- #

# (csv column, header, definition). `peak` is the dependent variable; the
# rest are the independent features build_feature_table emits.
_TD_COLUMNS: List[tuple] = [
    ("week_start", "Week (Mon)", "Monday of the Mon–Sun contract week."),
    ("peak", "Peak ships",
     "THE DEPENDENT VARIABLE — the week's highest single-day transit-call "
     "count through the Strait of Hormuz, per IMF PortWatch. This is what "
     "the Kalshi KXHORMUZPEAK market settles on."),
    ("peak_lag1", "Peak −1wk", "Previous week's peak (autoregressive)."),
    ("peak_lag2", "Peak −2wk", "Peak two weeks back."),
    ("peak_lag4", "Peak −4wk", "Peak four weeks back — trend anchor."),
    ("peak_ma4", "Peak MA4", "4-week moving average of the weekly peak."),
    ("mean_lag1", "Mean −1wk", "Previous week's mean daily transit calls."),
    ("trend_4w", "Trend 4wk",
     "Least-squares slope (ships/week) of the peak over the last 4 weeks."),
    ("momentum_4w", "Momentum",
     "Peak one week ago minus peak four weeks ago."),
    ("tanker_share_lag1", "Tanker %",
     "Prior-week tanker fraction of all transits."),
    ("regime", "Regime",
     "1 from the March-2026 structural break onward, else 0."),
    ("weeks_since_break", "Wks since break",
     "Weeks elapsed in the disrupted regime (recovery clock)."),
    ("brent_lag1", "Brent −1wk",
     "Prior-week Brent crude price — oil war-risk premium proxy (FRED)."),
    ("brent_chg_lag1", "Brent Δ",
     "Prior-week change in Brent crude price."),
    ("gdelt_lag1", "Tension −1wk",
     "Prior-week GDELT article volume for 'Strait of Hormuz' (geopolitical "
     "tension)."),
    # Scale-free features used by the pooled 28-chokepoint model.
    ("trailing_med13", "Med13",
     "Trailing 13-week median peak — the level everything below is "
     "normalised by."),
    ("r1", "Ratio −1wk", "Last week's peak ÷ trailing 13-week median."),
    ("r2", "Ratio −2wk", "Peak two weeks back ÷ trailing median."),
    ("r4", "Ratio −4wk", "Peak four weeks back ÷ trailing median."),
    ("ratio_ma4", "Ratio MA4", "4-week average of the level ratio."),
    ("trend_r4", "Ratio trend", "4-week slope of the level ratio."),
    ("mom_r4", "Ratio mom", "Level ratio 1wk back minus 4wks back."),
    ("disruption_depth", "Disrupt depth",
     "13-week ÷ 52-week median — how suppressed traffic is vs the long "
     "run."),
    ("recovery_ratio", "Recovery",
     "4-week ÷ 13-week median — recovering when above 1."),
    ("vol_r4", "Ratio vol", "4-week std of the level ratio."),
    ("peakiness_lag1", "Peakiness",
     "Prior-week peak ÷ mean daily transits — burstiness."),
]
_TEXT_COLS = {"week_start"}


def render_training_data_panel(*, bot: dict, current_bot: str | None,
                               page: int = 1, page_size: int = 20,
                               current_tab: str = "training",
                               period_key: str = "all") -> str:
    rows = _load_rows(bot.get("training_data_path"))
    out: List[str] = []
    out.append("<section class='card'><div class='body'>")
    out.append("<h2>Training Data — Hormuz Forecast</h2>")

    if not rows:
        out.append(
            "<p class='small gray'>The weekly training panel hasn't been "
            "generated on this host yet. Run <code>python run.py --train</code>"
            " in the Port Forecast repo (writes "
            "<code>data/training_data.csv</code>) and pull the repo here."
            "</p></div></section>"
        )
        return "".join(out)

    # Present columns = the intersection of our schema with what's in the
    # CSV (so a schema change doesn't 500 the page).
    present = set(rows[0].keys())
    cols = [(k, lbl, d) for (k, lbl, d) in _TD_COLUMNS if k in present]

    total = len(rows)
    view = sorted(rows, key=lambda r: r.get("week_start", ""), reverse=True)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)
    window = view[(page - 1) * page_size:(page - 1) * page_size + page_size]

    out.append(
        f"<p class='small gray'>One row per Mon–Sun week of IMF PortWatch "
        f"Strait-of-Hormuz history — <b>{total:,}</b> weeks, historical data "
        f"and the current snapshot combined in one table. <b>Peak ships</b> "
        f"is the dependent variable (the week's highest daily transit count); "
        f"the top row is the open contract week the model is forecasting, so "
        f"its peak is still blank. Every other column is an independent "
        f"feature, all computed strictly from data <i>before</i> the row's "
        f"week so nothing leaks the outcome. Sorted newest first. Click a "
        f"column header for its definition.</p>"
    )

    defs: Dict[str, Dict[str, str]] = {}
    out.append("<div style='overflow-x:auto;margin-top:12px;'>")
    out.append("<table class='training-data-table'><thead><tr>")
    for key, label, definition in cols:
        defs[key] = {"label": label, "def": definition}
        cls = "" if key in _TEXT_COLS else " class='num'"
        out.append(
            f"<th{cls}><button type='button' class='col-def-btn' "
            f"data-col='{html.escape(key)}'>{html.escape(label)}</button></th>"
        )
    out.append("</tr></thead><tbody>")
    for r in window:
        out.append("<tr>")
        for key, _, _ in cols:
            v = r.get(key)
            if v is None or v == "":
                cell = "—"
            elif key in _TEXT_COLS:
                cell = html.escape(str(v))
            else:
                cell = _fnum(v, 0 if key in _INT_FEATURES else 2)
            cls = "" if key in _TEXT_COLS else " class='num'"
            # Highlight the dependent variable column.
            if key == "peak":
                cls = " class='num' style='color:#3fb950;font-weight:600;'"
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
        out.append(f"<a class='tab-pill' href='{_page_link(page - 1)}'>← Prev</a>")
    else:
        out.append("<span class='tab-pill tab-pill-disabled'>← Prev</span>")
    out.append(f"<span>Page <b>{page:,}</b> of <b>{total_pages:,}</b> "
               f"<span class='gray'>({total:,} weeks)</span></span>")
    if page < total_pages:
        out.append(f"<a class='tab-pill' href='{_page_link(page + 1)}'>Next →</a>")
    else:
        out.append("<span class='tab-pill tab-pill-disabled'>Next →</span>")
    out.append("</div>")

    # ── Column-definition popover (same pattern as the WC/tennis panel) ─
    out.append(
        "<div id='hz-col-def-pop' class='col-def-pop' hidden>"
        "<div class='col-def-pop-title'></div>"
        "<div class='col-def-pop-body'></div></div>"
    )
    out.append(
        "<style>"
        ".col-def-btn{background:none;border:0;color:inherit;font:inherit;"
        "cursor:pointer;padding:0;text-decoration:underline dotted;}"
        ".col-def-btn:hover{color:#79c0ff;}"
        ".col-def-pop{position:absolute;z-index:1000;max-width:320px;"
        "padding:10px 12px;background:#0d1117;color:#c9d1d9;"
        "border:1px solid #30363d;border-radius:6px;"
        "box-shadow:0 8px 24px rgba(0,0,0,.4);font-size:12px;}"
        ".col-def-pop-title{font-weight:600;margin-bottom:4px;}"
        "</style>"
    )
    out.append(
        "<script>(function(){"
        f"var defs = {json.dumps(defs)};"
        "var pop = document.getElementById('hz-col-def-pop');"
        "if (!pop) return;"
        "document.querySelectorAll('.training-data-table .col-def-btn')"
        ".forEach(function(btn){btn.addEventListener('click',function(ev){"
        "ev.stopPropagation();"
        "var d = defs[btn.dataset.col]; if (!d) return;"
        "pop.querySelector('.col-def-pop-title').textContent = d.label;"
        "pop.querySelector('.col-def-pop-body').textContent = d.def;"
        "var r = btn.getBoundingClientRect(); pop.hidden = false;"
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
