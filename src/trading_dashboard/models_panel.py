"""Models tab — per-bot model deep-dive (pregame + in-game views)."""
from __future__ import annotations

import csv
import html
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from .data import _conn
from .panels import _render_bot_unavailable


# ── Models page (per-bot deep-dive) ──────────────────────────────────────
def _find_training_artifact(db_path: str, *names: str) -> Path:
    """Locate a trainer-written file for a bot. Different bots stage
    their training artifacts in different siblings of ``data/sim.db``:

    - newer bots (nba, cpi, claims) land them next to sim.db (``data/``)
    - retail-gas writes into ``artifacts/`` instead
    - natural-gas / peak-load writes models into ``models/``

    Returns the first existing path; falls back to the ``data/`` location
    if nothing is found so that callers can pass it to readers that
    handle missing files (they'll just return empty results).
    """
    base = Path(db_path).parent
    candidates: List[Path] = []
    for name in names:
        candidates.extend([
            base / name,
            base.parent / "artifacts" / name,
            base.parent / "models" / name,
        ])
    for p in candidates:
        if p.exists():
            return p
    return Path(db_path).parent / names[0]


def _read_feature_importance(csv_path: str) -> List[dict]:
    """Parse the bot's feature_importance.csv into a list of feature
    dicts: {feature, mean_importance, positive_folds, selected}.

    Tolerates missing/empty files by returning an empty list — the
    renderer shows an empty-state in that case rather than crashing.

    Two on-disk schemas exist in the wild:
      • Newer bots (nba, cpi, claims, natural-gas): a clean
        ``feature, mean_importance, positive_folds, selected`` header.
      • Retail-gas (legacy): per-fold columns + a ``mean`` /
        ``positive_folds`` / ``eligible`` triple, with an unnamed
        first column carrying the feature name.

    The reader picks whichever set of columns exists so every bot's
    feature_importance.csv renders the same way on the dashboard.
    """
    p = Path(csv_path)
    if not p.exists():
        return []
    out: List[dict] = []
    try:
        with p.open("r") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames or [])
            name_key = ("feature" if "feature" in fields
                        else ("" if "" in fields
                              else (fields[0] if fields else None)))
            imp_key = ("mean_importance" if "mean_importance" in fields
                       else ("mean" if "mean" in fields else None))
            sel_key = ("selected" if "selected" in fields
                       else ("eligible" if "eligible" in fields else None))
            for row in rd:
                feat_name = (row.get(name_key) or "") if name_key is not None else ""
                try:
                    imp = float(row.get(imp_key) or 0.0) if imp_key else 0.0
                except (TypeError, ValueError):
                    imp = 0.0
                try:
                    pf = int(float(row.get("positive_folds") or 0))
                except (TypeError, ValueError):
                    pf = 0
                sel_raw = row.get(sel_key) if sel_key else None
                sel = str(sel_raw or "").strip().lower() in (
                    "true", "1", "yes",
                )
                out.append({
                    "feature": feat_name,
                    "mean_importance": imp,
                    "positive_folds": pf,
                    "selected": sel,
                })
    except (OSError, csv.Error):
        return []
    return out


def _holdout_confidence(pairs: List[Tuple[float, int]]) -> dict:
    """Translate the trainer's held-out predictions into a
    sample-size-driven confidence tier for the metrics on the model
    page. Returns a dict with ``tier`` (none/low/moderate/good/high),
    a CSS colour, a one-word label, and a sentence-long ``reason``
    that explains *why* the user should (or shouldn't) trust the
    headline numbers.

    The thresholds borrow standard rules-of-thumb for binary-classifier
    holdout sample sizes:
      • <30 predictions or minority class <5 → noisy, can flip 5+ pts
      • <100 / minority <20 → directionally meaningful, ±2-3 pts
      • <500 → stable to ~1 pt
      • ≥500 → calibration deciles each carry enough data
    """
    n = len(pairs)
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = n - n_pos
    minority = min(n_pos, n_neg) if n else 0
    if n == 0:
        return {
            "tier": "none", "color": "#8b949e",
            "label": "No held-out data",
            "reason": ("This bot's trainer hasn't written a "
                       "holdout_predictions.csv yet — the metrics on "
                       "this page can't be confidence-graded."),
            "n": 0, "n_pos": 0, "n_neg": 0,
        }
    if n < 30 or minority < 5:
        return {
            "tier": "low", "color": "#f85149",
            "label": "Low confidence",
            "reason": (f"Only {n} held-out predictions"
                       + (f" (minority class = {minority})"
                          if minority < 5 else "")
                       + " — the accuracy / ROC / calibration figures "
                       "below are noisy at this sample size and can "
                       "swing 5+ percentage points across retrains."),
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
        }
    if n < 100 or minority < 20:
        return {
            "tier": "moderate", "color": "#d29922",
            "label": "Moderate confidence",
            "reason": (f"{n} held-out predictions ({n_pos} positives / "
                       f"{n_neg} negatives) — directionally meaningful "
                       "but the per-decile calibration bins still carry "
                       "wide error bars. Treat headline metrics as "
                       "±2-3 pts."),
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
        }
    if n < 500:
        return {
            "tier": "good", "color": "#3fb950",
            "label": "Good confidence",
            "reason": (f"{n} held-out predictions ({n_pos} positives / "
                       f"{n_neg} negatives) — sample size is large "
                       "enough that the headline accuracy / ROC AUC "
                       "are stable to within ~1 pt across retrains."),
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
        }
    return {
        "tier": "high", "color": "#3fb950",
        "label": "High confidence",
        "reason": (f"{n:,} held-out predictions ({n_pos:,} positives / "
                   f"{n_neg:,} negatives) — enough data per "
                   "calibration decile to read at face value."),
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
    }


def _read_holdout_predictions(csv_path: str) -> List[Tuple[float, int]]:
    """Load (predicted_prob, actual_label) pairs from a bot's
    holdout_predictions.csv. The trainer writes this file on each
    retrain — it carries the model's evaluation against the held-out
    historical test set, which is what the user sees as "the
    model's accuracy" on the Models tab.

    Returns an empty list when the file is missing or unreadable
    (e.g. a bot whose trainer hasn't been redeployed yet).
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
                    prob = float(row.get("predicted_prob") or 0.0)
                    label = int(float(row.get("actual_label") or 0))
                except (TypeError, ValueError):
                    continue
                out.append((prob, 1 if label else 0))
    except (OSError, csv.Error):
        return []
    return out


def _svg_calibration(bins: List[dict],
                       live_bins: List[dict] | None = None) -> str:
    """Reliability diagram — predicted-prob bin midpoint on X, observed
    win-rate on Y, point size scales with bin sample count. Diagonal
    reference line shows perfect calibration.

    ``live_bins`` (optional) overlays a second series sourced from the
    bot's live closed-bet ledger. Holdout = blue (training-time
    expectation); live = orange (what the bot is actually getting).
    Divergence between the two is the drift signal.
    """
    populated = [b for b in bins if b.get("n", 0) > 0]
    live_populated = [b for b in (live_bins or []) if b.get("n", 0) > 0]
    if not populated and not live_populated:
        return ("<div class='empty'>Not enough closed bets yet to "
                "draw a calibration curve.</div>")
    width, height = 460, 320
    pad_l, pad_r, pad_t, pad_b = 50, 20, 24, 36
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    parts: List[str] = []
    parts.append(
        f"<svg viewBox='0 0 {width} {height}' "
        f"style='width:100%;height:auto;max-height:340px;"
        f"display:block;background:#0d1117;border:1px solid #21262d;"
        f"border-radius:6px;'>"
    )
    # Axes — 0..1 both directions.
    parts.append(
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' "
        f"y2='{pad_t + inner_h}' stroke='#21262d'/>"
        f"<line x1='{pad_l}' y1='{pad_t + inner_h}' "
        f"x2='{pad_l + inner_w}' y2='{pad_t + inner_h}' stroke='#21262d'/>"
    )
    # Gridlines + labels at each decile.
    for k in range(0, 11, 2):
        frac = k / 10.0
        x = pad_l + frac * inner_w
        y = pad_t + (1 - frac) * inner_h
        parts.append(
            f"<line x1='{x}' x2='{x}' y1='{pad_t}' "
            f"y2='{pad_t + inner_h}' stroke='#161b22'/>"
            f"<text x='{x}' y='{pad_t + inner_h + 14}' fill='#8b949e' "
            f"font-size='10' text-anchor='middle'>{int(frac*100)}%</text>"
            f"<line x1='{pad_l}' x2='{pad_l + inner_w}' "
            f"y1='{y}' y2='{y}' stroke='#161b22'/>"
            f"<text x='{pad_l - 6}' y='{y + 3}' fill='#8b949e' "
            f"font-size='10' text-anchor='end'>{int(frac*100)}%</text>"
        )
    # Diagonal: perfect calibration.
    parts.append(
        f"<line x1='{pad_l}' y1='{pad_t + inner_h}' "
        f"x2='{pad_l + inner_w}' y2='{pad_t}' stroke='#484f58' "
        f"stroke-dasharray='4,3'/>"
    )
    # Polyline through populated bins so the reliability shape is easy
    # to follow even when some deciles are sparsely populated.
    pts: List[Tuple[float, float]] = []
    n_total = sum(b.get("n", 0) for b in populated) or 1
    for b in populated:
        mid = (b["lo"] + b["hi"]) / 2.0
        rate = b["wins"] / b["n"]
        x = pad_l + mid * inner_w
        y = pad_t + (1 - rate) * inner_h
        pts.append((x, y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    parts.append(
        f"<polyline points='{poly}' fill='none' "
        f"stroke='#58a6ff' stroke-width='2'/>"
    )
    # Points sized by bin n. Tooltip shows the raw count + win rate.
    for b, (x, y) in zip(populated, pts):
        size = max(3, min(14, (b["n"] / n_total) * 60))
        parts.append(
            f"<g><title>{b['lo']*100:.0f}–{b['hi']*100:.0f}%: "
            f"{b['wins']}/{b['n']} won "
            f"({b['wins']/b['n']*100:.0f}%)</title>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{size:.1f}' "
            f"fill='#58a6ff' fill-opacity='0.7' stroke='#58a6ff'/></g>"
        )
    # Live-bets overlay: same polyline + circle treatment in orange so
    # the user can see drift at a glance — when the orange line drops
    # below the blue (holdout) line, the model is losing more bets
    # than it expected to in that bucket.
    if live_populated:
        live_pts: List[Tuple[float, float]] = []
        n_live_total = sum(b.get("n", 0) for b in live_populated) or 1
        for b in live_populated:
            mid = (b["lo"] + b["hi"]) / 2.0
            rate = b["wins"] / b["n"]
            x = pad_l + mid * inner_w
            y = pad_t + (1 - rate) * inner_h
            live_pts.append((x, y))
        live_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in live_pts)
        parts.append(
            f"<polyline points='{live_poly}' fill='none' "
            f"stroke='#e3934d' stroke-width='2' stroke-dasharray='5,3'/>"
        )
        for b, (x, y) in zip(live_populated, live_pts):
            size = max(3, min(14, (b["n"] / n_live_total) * 60))
            parts.append(
                f"<g><title>Live · {b['lo']*100:.0f}–{b['hi']*100:.0f}%: "
                f"{b['wins']}/{b['n']} resolved YES "
                f"({b['wins']/b['n']*100:.0f}%)</title>"
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{size:.1f}' "
                f"fill='#e3934d' fill-opacity='0.7' stroke='#e3934d'/></g>"
            )
    # Legend — placed inside the chart so the two series stay visually
    # tied to the lines they label.
    legend_y = pad_t + 6
    parts.append(
        f"<g><line x1='{pad_l + 6}' x2='{pad_l + 26}' y1='{legend_y}' "
        f"y2='{legend_y}' stroke='#58a6ff' stroke-width='2'/>"
        f"<text x='{pad_l + 30}' y='{legend_y + 3}' fill='#8b949e' "
        f"font-size='10'>Holdout</text></g>"
    )
    if live_populated:
        parts.append(
            f"<g><line x1='{pad_l + 90}' x2='{pad_l + 110}' "
            f"y1='{legend_y}' y2='{legend_y}' stroke='#e3934d' "
            f"stroke-width='2' stroke-dasharray='5,3'/>"
            f"<text x='{pad_l + 114}' y='{legend_y + 3}' fill='#8b949e' "
            f"font-size='10'>Live</text></g>"
        )
    # Axis labels.
    parts.append(
        f"<text x='{pad_l + inner_w/2}' y='{height - 6}' fill='#8b949e' "
        f"font-size='11' text-anchor='middle'>Predicted probability</text>"
        f"<text x='15' y='{pad_t + inner_h/2}' fill='#8b949e' "
        f"font-size='11' text-anchor='middle' "
        f"transform='rotate(-90 15 {pad_t + inner_h/2})'>"
        f"Observed win rate</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# Rich feature metadata: source label + colour + plain-English
# description + canonical source URL. Drives both the legend / chart
# colour-coding AND the "all features and their sources" table on
# every bot's Models tab, so the user can audit not just *what* a
# feature is but *where it came from* in one place.
#
# Each rule is a substring matcher. Order matters: more specific keys
# (tennis "form_last", "h2h_") sit above more generic ones (NBA
# "diff_") so e.g. ``diff_form_last5`` lands in the tennis bucket
# rather than the NBA bucket. FRED ids and ETF tickers sit above the
# catch-all "derived transform" bucket for the same reason.
FEATURE_RULES: List[dict] = [
    # ── Billboard Hot 100: real weekly chart history from the
    # utdata/rwd-billboard-data public mirror of billboard.com.
    # All 20 features in the Hot 100 membership + #1 models are
    # derived entirely from this one source. Split into five logical
    # groups so the chart legend and table can show users which part
    # of the chart panel each feature came from.
    {"patterns": ("artist_prior_top10_songweeks",
                   "artist_weeks_since_last_top10",
                   "artist_prior_top10_weeks",
                   "artist_prior_no1_weeks",
                   # legacy names, pre-2026-07-15 artifacts
                   "artist_total_prior_top10s",
                   "artist_prior_top10_count"),
     "label": "Billboard Hot 100 (artist history)", "color": "#58a6ff",
     "description": "How often this artist has had top-10 songs and #1s in the past, and how recently. An artist with prior #1 weeks is far more likely to debut another song at the top.",
     "link": "https://www.billboard.com/charts/hot-100/"},
    {"patterns": ("peak_position_so_far", "weeks_on_chart",
                   "debut_rank", "weeks_since_debut",
                   "last_seen_rank", "weeks_since_last_on_chart",
                   "best_3wk_rank", "rank_change_last_week",
                   "weeks_in_top10_so_far", "weeks_in_top40_so_far",
                   "weeks_at_no1_so_far"),
     "label": "Billboard Hot 100 (song trajectory)", "color": "#58a6ff",
     "description": "The song's own chart history strictly before this week — how high it has climbed, its most recent rank and how long since it last charted, how it has moved week-to-week, and how many weeks it has spent in the top 10 / top 40 / at #1.",
     "link": "https://www.billboard.com/charts/hot-100/"},
    {"patterns": ("debut_month_sin", "debut_month_cos", "debut_dow"),
     "label": "Billboard Hot 100 (release timing)", "color": "#58a6ff",
     "description": "When the song first appeared on the Hot 100 (month of the year). Captures seasonal patterns in which release windows produce chart hits.",
     "link": "https://www.billboard.com/charts/hot-100/"},
    {"patterns": ("competition_last_week", "competition_count"),
     "label": "Billboard Hot 100 (weekly competition)", "color": "#58a6ff",
     "description": "How crowded the most recent chart week's top 40 was with fresh debuts. A busy release window means more competition for chart slots.",
     "link": "https://www.billboard.com/charts/hot-100/"},
    {"patterns": ("is_new_to_pool",),
     "label": "Billboard Hot 100 (panel structure)", "color": "#58a6ff",
     "description": "Flags rows for songs that debuted or re-entered from outside the trailing 12-week popular pool — the top-10 model uses it to treat album-bomb debuts differently from established chart songs.",
     "link": "https://www.billboard.com/charts/hot-100/"},

    # ── Tennis / Table tennis: Jeff Sackmann dataset + bot-computed Elo
    {"patterns": ("surface_elo", "style_elo"),
     "label": "Elo (bot-computed)", "color": "#bc8cff",
     "description": "How strong the player is on this specific court surface or against this style of opponent. Higher number means a better player.",
     "link": "https://en.wikipedia.org/wiki/Elo_rating_system"},
    {"patterns": ("h2h_",),
     "label": "Head-to-head (Sackmann)", "color": "#58a6ff",
     "description": "How the two players have done against each other in their past matches.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("form_last",),
     "label": "Form (Sackmann)", "color": "#58a6ff",
     "description": "What fraction of their recent matches the player won — a measure of how hot they are right now.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("serve_pts", "return_pts", "bp_saved",
                   "deuce_win_pct", "closing_win_pct",
                   "point_win_pct", "game_margin",
                   "comeback_rate"),
     "label": "Match stats (Sackmann)", "color": "#58a6ff",
     "description": "How well the player has been serving, returning, and winning tight points in their recent matches.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("days_rest", "matches_last_7d"),
     "label": "Schedule (Sackmann)", "color": "#58a6ff",
     "description": "How rested the player is — days since their last match and how many matches they've played this week.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("hand_matchup", "diff_hand"),
     "label": "Player profile (Sackmann)", "color": "#58a6ff",
     "description": "Whether the matchup is lefty-vs-righty, lefty-vs-lefty, etc — these matchups play out differently.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("rank_diff",),
     "label": "ATP/WTA rankings", "color": "#58a6ff",
     "description": "Gap between the two players' official world rankings.",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    {"patterns": ("round_rank", "level_rank", "is_bo7"),
     "label": "Tournament metadata", "color": "#58a6ff",
     "description": "How big the tournament is and how late in the bracket the match sits (later rounds in bigger tournaments play differently).",
     "link": "https://github.com/JeffSackmann/tennis_atp"},
    # ── NBA / generic Elo: pattern is "_elo" or "elo_" (not plain
    # "elo") so it doesn't false-match the substring inside words
    # like "below" / "above" / "develop".
    {"patterns": ("_elo", "elo_"),
     "label": "Elo (bot-computed)", "color": "#bc8cff",
     "description": "How strong this team or player has been recently. Goes up after wins and down after losses, scaled by the margin of victory.",
     "link": "https://en.wikipedia.org/wiki/Elo_rating_system"},
    {"patterns": ("_b2b", "b2b_"),
     "label": "Schedule (bot-computed)", "color": "#bc8cff",
     "description": "Whether the team is playing on the second night of a back-to-back — a fatigue signal that affects performance.",
     "link": "https://github.com/swar/nba_api"},
    # ── NBA: nba_api advanced box-score stats ───────────────────────
    {"patterns": ("_off_rating", "_def_rating", "_net_rating",
                   "_efg_pct", "_oreb_pct", "_tov_pct", "_ft_per_fga",
                   "_fg3m", "_pace", "_win_r", "_team_win",
                   "_team_"),
     "label": "nba_api advanced stats", "color": "#58a6ff",
     "description": "How efficiently the team has been scoring, defending, shooting threes, rebounding, etc — averaged over their recent games.",
     "link": "https://github.com/swar/nba_api"},
    # ── FRED macro series — one entry per series so the link points at
    # the exact series page. Narrow patterns first.
    {"patterns": ("nonfarm_payrolls", "payems"),
     "label": "FRED PAYEMS", "color": "#3fb950",
     "description": "How many people are employed in the US (excluding farm workers). Updated monthly by the government — the headline jobs number.",
     "link": "https://fred.stlouisfed.org/series/PAYEMS"},
    {"patterns": ("treasury_10y", "dgs10"),
     "label": "FRED DGS10", "color": "#3fb950",
     "description": "Interest rate on a 10-year US government bond. Higher means borrowing is more expensive across the economy.",
     "link": "https://fred.stlouisfed.org/series/DGS10"},
    {"patterns": ("treasury_2y", "dgs2"),
     "label": "FRED DGS2", "color": "#3fb950",
     "description": "Interest rate on a 2-year US government bond. Reflects what markets expect short-term rates to do.",
     "link": "https://fred.stlouisfed.org/series/DGS2"},
    {"patterns": ("wti_oil", "dcoilwtico"),
     "label": "FRED DCOILWTICO", "color": "#3fb950",
     "description": "Price of US benchmark crude oil per barrel.",
     "link": "https://fred.stlouisfed.org/series/DCOILWTICO"},
    {"patterns": ("henry_hub", "mhhngsp"),
     "label": "FRED MHHNGSP", "color": "#3fb950",
     "description": "Benchmark price of US natural gas.",
     "link": "https://fred.stlouisfed.org/series/MHHNGSP"},
    {"patterns": ("vix", "vixcls"),
     "label": "FRED VIXCLS", "color": "#3fb950",
     "description": "How much volatility traders expect in the stock market over the next month — known as the 'fear gauge'.",
     "link": "https://fred.stlouisfed.org/series/VIXCLS"},
    {"patterns": ("unemployment_rate", "unrate"),
     "label": "FRED UNRATE", "color": "#3fb950",
     "description": "Percentage of Americans who want a job but don't have one.",
     "link": "https://fred.stlouisfed.org/series/UNRATE"},
    {"patterns": ("continuing_claims", "ccsa"),
     "label": "FRED CCSA", "color": "#3fb950",
     "description": "Number of people still collecting jobless benefits this week (continuing claims).",
     "link": "https://fred.stlouisfed.org/series/CCSA"},
    {"patterns": ("initial_claims", "icsa"),
     "label": "FRED ICSA", "color": "#3fb950",
     "description": "Number of new initial jobless claims filed this week — the thing this bot predicts.",
     "link": "https://fred.stlouisfed.org/series/ICSA"},
    {"patterns": ("ppi", "ppiaco"),
     "label": "FRED PPIACO", "color": "#3fb950",
     "description": "How much prices changed at the wholesale level (what factories charge stores). Leads consumer prices.",
     "link": "https://fred.stlouisfed.org/series/PPIACO"},
    {"patterns": ("headline_cpi", "cpiaucsl"),
     "label": "FRED CPIAUCSL", "color": "#3fb950",
     "description": "Overall consumer price level — what a typical basket of goods and services costs Americans.",
     "link": "https://fred.stlouisfed.org/series/CPIAUCSL"},
    {"patterns": ("core_cpi", "core_mom", "cpilfesl"),
     "label": "FRED CPILFESL", "color": "#3fb950",
     "description": "Consumer prices excluding food and gas (which swing a lot month to month). A cleaner read on underlying inflation.",
     "link": "https://fred.stlouisfed.org/series/CPILFESL"},
    {"patterns": ("used_cars_cpi", "cuur0000seta02"),
     "label": "FRED CUUR0000SETA02", "color": "#3fb950",
     "description": "How much used-car prices have changed.",
     "link": "https://fred.stlouisfed.org/series/CUUR0000SETA02"},
    {"patterns": ("fed_funds_rate", "fedfunds"),
     "label": "FRED FEDFUNDS", "color": "#3fb950",
     "description": "The interest rate the Federal Reserve targets. Sets the floor for borrowing costs across the economy.",
     "link": "https://fred.stlouisfed.org/series/FEDFUNDS"},
    {"patterns": ("industrial_production", "industrial_prod", "indpro"),
     "label": "FRED INDPRO", "color": "#3fb950",
     "description": "How much US factories, mines, and utilities are producing.",
     "link": "https://fred.stlouisfed.org/series/INDPRO"},
    {"patterns": ("umich_inflation", "mich"),
     "label": "FRED MICH", "color": "#3fb950",
     "description": "How much inflation regular Americans expect over the next year (University of Michigan survey).",
     "link": "https://fred.stlouisfed.org/series/MICH"},
    {"patterns": ("consumer_sentiment", "umcsent"),
     "label": "FRED UMCSENT", "color": "#3fb950",
     "description": "How optimistic regular Americans feel about the economy (University of Michigan survey).",
     "link": "https://fred.stlouisfed.org/series/UMCSENT"},
    {"patterns": ("cleveland_expinf", "expinf1yr"),
     "label": "FRED EXPINF1YR", "color": "#3fb950",
     "description": "How much inflation experts expect over the next year (Cleveland Fed model).",
     "link": "https://fred.stlouisfed.org/series/EXPINF1YR"},
    {"patterns": ("m2_yoy", "m2sl"),
     "label": "FRED M2SL", "color": "#3fb950",
     "description": "How much money is circulating in the US economy (cash, checking, savings accounts).",
     "link": "https://fred.stlouisfed.org/series/M2SL"},
    {"patterns": ("retail_gas", "gasregw"),
     "label": "FRED GASREGW", "color": "#3fb950",
     "description": "Average price at the pump for regular gas in the US.",
     "link": "https://fred.stlouisfed.org/series/GASREGW"},
    {"patterns": ("jolts_layoffs", "jtsldl"),
     "label": "FRED JTSLDL (JOLTS)", "color": "#3fb950",
     "description": "How many people were laid off or fired across the US in the latest month.",
     "link": "https://fred.stlouisfed.org/series/JTSLDL"},
    {"patterns": ("jolts_hires", "jtshil"),
     "label": "FRED JTSHIL (JOLTS)", "color": "#3fb950",
     "description": "How many people were hired across the US in the latest month.",
     "link": "https://fred.stlouisfed.org/series/JTSHIL"},
    {"patterns": ("jolts_quits", "jtsqul"),
     "label": "FRED JTSQUL (JOLTS)", "color": "#3fb950",
     "description": "How many people quit their job in the latest month. Higher means workers feel confident they can find another job.",
     "link": "https://fred.stlouisfed.org/series/JTSQUL"},
    {"patterns": ("jolts_openings", "jtsjol"),
     "label": "FRED JTSJOL (JOLTS)", "color": "#3fb950",
     "description": "How many job openings are posted across the US right now.",
     "link": "https://fred.stlouisfed.org/series/JTSJOL"},
    {"patterns": ("unemp_5_14", "uemp5to14"),
     "label": "FRED UEMP5TO14", "color": "#3fb950",
     "description": "How many people have been unemployed for between 5 and 14 weeks.",
     "link": "https://fred.stlouisfed.org/series/UEMP5TO14"},
    {"patterns": ("unemp_27plus", "uemp27ov"),
     "label": "FRED UEMP27OV", "color": "#3fb950",
     "description": "How many people have been unemployed for 27 weeks or more — the long-term unemployed.",
     "link": "https://fred.stlouisfed.org/series/UEMP27OV"},
    # NOTE: pattern requires the trailing underscore so "unemployment"
    # in the Google Trends search-term "filed_for_unemployment" doesn't
    # get misattributed to a duration bucket.
    {"patterns": ("uemp_", "unemp_"),
     "label": "FRED UEMP* (duration buckets)", "color": "#3fb950",
     "description": "How many people are unemployed, grouped by how long they've been out of work.",
     "link": "https://fred.stlouisfed.org/categories/12"},
    {"patterns": ("durable_orders", "dgorder"),
     "label": "FRED DGORDER", "color": "#3fb950",
     "description": "Orders for big-ticket items expected to last three years or more — cars, appliances, machinery. A sign of business investment.",
     "link": "https://fred.stlouisfed.org/series/DGORDER"},
    {"patterns": ("policy_uncertainty", "usepuindxd"),
     "label": "FRED USEPUINDXD", "color": "#3fb950",
     "description": "How uncertain US government policy is right now, measured from news coverage of policy disputes.",
     "link": "https://fred.stlouisfed.org/series/USEPUINDXD"},
    {"patterns": ("trade_weighted_dollar", "dtwexbgs"),
     "label": "FRED DTWEXBGS", "color": "#3fb950",
     "description": "How strong the US dollar is, measured against a basket of other countries' currencies.",
     "link": "https://fred.stlouisfed.org/series/DTWEXBGS"},
    # ── Alt-data / non-FRED sources ─────────────────────────────────
    {"patterns": ("google_trends",),
     "label": "Google Trends", "color": "#d29922",
     "description": "How often Americans are searching Google for terms like 'how to file jobless claims' or 'laid off'. A real-time signal of job losses.",
     "link": "https://trends.google.com/trends/"},
    {"patterns": ("bfs_total_apps", "busappwnsaus"),
     "label": "Census BFS (via FRED BUSAPPWNSAUS)", "color": "#3fb950",
     "description": "Number of business applications filed in the US this week. Slowing applications mean fewer new employers about to start hiring.",
     "link": "https://fred.stlouisfed.org/series/BUSAPPWNSAUS"},
    {"patterns": ("bfs_high_wage_apps", "wbusappwnsaus"),
     "label": "Census BFS high-wage (via FRED WBUSAPPWNSAUS)", "color": "#3fb950",
     "description": "Business applications filed by people who plan to pay wages — the subset most likely to actually hire workers.",
     "link": "https://fred.stlouisfed.org/series/WBUSAPPWNSAUS"},
    {"patterns": ("bfs_high_propensity", "hbusappwnsaus"),
     "label": "Census BFS high-propensity (via FRED HBUSAPPWNSAUS)", "color": "#3fb950",
     "description": "Business applications that the Census Bureau identifies as highly likely to turn into employer businesses within the next year.",
     "link": "https://fred.stlouisfed.org/series/HBUSAPPWNSAUS"},
    {"patterns": ("eia_diesel_demand", "wdiupus2"),
     "label": "EIA WDIUPUS2 (diesel demand)", "color": "#d29922",
     "description": "Total US diesel fuel consumed this week — a real-time proxy for freight and trucking activity. Slowing freight tends to lead layoffs.",
     "link": "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WDIUPUS2&f=W"},
    {"patterns": ("eia_gasoline_demand", "wgfupus2"),
     "label": "EIA WGFUPUS2 (gasoline demand)", "color": "#d29922",
     "description": "Total US gasoline consumed this week — a proxy for how much people are commuting to work.",
     "link": "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WGFUPUS2&f=W"},
    {"patterns": ("noaa_cdo_hdd", "noaa_cdo_cdd",
                   "noaa_cdo_cold_snap", "noaa_cdo_heat_wave",
                   "noaa_cdo_heavy_precip", "noaa_cdo_tmin",
                   "noaa_cdo_tmax", "noaa_cdo"),
     "label": "NOAA CDO (station weather)", "color": "#d29922",
     "description": "Daily temperature and precipitation from five major-city weather stations (NYC, LAX, ORD, ATL, DFW), aggregated into heating/cooling-degree days, cold-snap flags, heat-wave flags, and heavy-precipitation flags. Lets the model see when weather (not the labor market) is driving claims spikes.",
     "link": "https://www.ncei.noaa.gov/cdo-web/"},
    # ── Survivor elimination: show / game / social signal ──────────
    # Order matters — survivor's reddit_* patterns sit above the
    # generic `reddit` rule so unemployment's reddit_layoffs_* still
    # falls through to r/layoffs.
    {"patterns": ("reddit_mention", "reddit_boot", "reddit_sentiment",
                   "reddit_visibility", "reddit_target"),
     "label": "Reddit r/survivor", "color": "#d29922",
     "description": "Signal scraped from the r/survivor subreddit — how often a contestant is mentioned, how often the community picks them as the next boot, and how positive or negative the discussion around them is.",
     "link": "https://reddit.com/r/survivor"},
    {"patterns": ("season", "episode", "remaining", "is_finale",
                   "pre_merge_phase", "merged", "swap_phase",
                   "tribe_size", "starting_tribe_size",
                   "episode_share"),
     "label": "Show structure", "color": "#bc8cff",
     "description": "Where we are in the season — which episode, how many contestants are left, whether tribes have merged or swapped.",
     "link": "https://survivor.fandom.com/wiki/Main_Page"},
    {"patterns": ("immunity_won", "tribe_immunity", "has_idol",
                   "advantages_held", "idols_played_this_ep",
                   "vote_steals_active"),
     "label": "Game state / advantages", "color": "#3fb950",
     "description": "Whether the contestant has an idol, an advantage, or won immunity this episode — concrete protections that change boot risk.",
     "link": "https://survivor.fandom.com/wiki/Hidden_Immunity_Idol"},
    {"patterns": ("confessional_count", "confessional_share",
                   "visibility_score", "visibility_spike",
                   "negative_edit_score", "narrative_intensity",
                   "strategic_isolation"),
     "label": "Edit / on-show signal", "color": "#58a6ff",
     "description": "Signal extracted from the episode edit — how much screen time the contestant gets, how positive or negative the framing is, and whether the editors are setting them up as the boot.",
     "link": "https://survivor.fandom.com/wiki/Survivor_(franchise)"},
    {"patterns": ("in_main_alliance", "prior_votes_against",
                   "times_targeted", "swing_vote_potential",
                   "voting_minority_score",
                   "same_starting_tribe_remaining"),
     "label": "Alliance / voting state", "color": "#d29922",
     "description": "Where the contestant sits politically — whether they're in the majority alliance, how many votes they've taken in past tribals, and how often they've been targeted.",
     "link": "https://survivor.fandom.com/wiki/Alliance"},
    {"patterns": ("is_returnee", "season_returnee_count",
                   "prior_perf_score", "is_returnee_first_three"),
     "label": "Returnee history", "color": "#8b949e",
     "description": "Whether the contestant has played Survivor before and how they did — returnees behave (and get edited) differently from first-timers.",
     "link": "https://survivor.fandom.com/wiki/Returning_player"},
    {"patterns": ("reddit",),
     "label": "Reddit r/layoffs", "color": "#d29922",
     "description": "How many posts about losing a job were submitted to the r/layoffs subreddit.",
     "link": "https://reddit.com/r/layoffs"},
    # ── Retail-gas / energy: futures, ETFs, EIA Weekly Status ───────
    {"patterns": ("rbob_gasoline_futures", "rbob_"),
     "label": "CME RBOB futures", "color": "#58a6ff",
     "description": "Wholesale gasoline price (what gas stations pay to buy gas). Moves before pump prices.",
     "link": "https://www.cmegroup.com/markets/energy/refined-products/rbob-gasoline-physical.html"},
    {"patterns": ("brent_futures", "brent_spot", "brent_wti_spread"),
     "label": "ICE Brent crude", "color": "#58a6ff",
     "description": "Price of Brent crude oil — the international benchmark used to price most of the world's oil.",
     "link": "https://www.theice.com/products/219/Brent-Crude-Futures"},
    {"patterns": ("wti_futures", "wti_spot", "wti_term_structure"),
     "label": "NYMEX WTI crude", "color": "#58a6ff",
     "description": "Price of West Texas Intermediate — the US benchmark for crude oil.",
     "link": "https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.html"},
    {"patterns": ("natural_gas_futures",),
     "label": "CME Henry Hub NG futures", "color": "#58a6ff",
     "description": "Price of natural gas futures — what wholesale buyers pay for delivery in the coming month.",
     "link": "https://www.cmegroup.com/markets/energy/natural-gas/natural-gas.html"},
    {"patterns": ("heating_oil_futures",),
     "label": "NYMEX heating-oil futures", "color": "#58a6ff",
     "description": "Price of heating oil futures (also used to price diesel).",
     "link": "https://www.cmegroup.com/markets/energy/refined-products/heating-oil.html"},
    {"patterns": ("dxy_dollar_index",),
     "label": "ICE Dollar Index (DXY)", "color": "#58a6ff",
     "description": "How strong the US dollar is against a basket of major currencies (euro, yen, pound, etc).",
     "link": "https://www.theice.com/products/194/US-Dollar-Index-Futures"},
    {"patterns": ("ovx",),
     "label": "CBOE Oil VIX (OVX)", "color": "#58a6ff",
     "description": "How volatile traders expect oil prices to be over the next month — oil's version of the stock-market 'fear gauge'.",
     "link": "https://www.cboe.com/tradable_products/vix/oil_volatility/"},
    {"patterns": ("energy_sector_etf", "xle"),
     "label": "XLE ETF", "color": "#58a6ff",
     "description": "Price of the ETF that holds the big US energy stocks (Exxon, Chevron, etc) — a proxy for how the oil & gas sector is doing.",
     "link": "https://finance.yahoo.com/quote/XLE"},
    {"patterns": ("uga_gasoline_etf", "uga_"),
     "label": "UGA ETF", "color": "#58a6ff",
     "description": "Price of an ETF that directly tracks gasoline prices.",
     "link": "https://finance.yahoo.com/quote/UGA"},
    {"patterns": ("uso_oil_etf",),
     "label": "USO ETF", "color": "#58a6ff",
     "description": "Price of an ETF that directly tracks the price of oil.",
     "link": "https://finance.yahoo.com/quote/USO"},
    {"patterns": ("usl_12mo_oil",),
     "label": "USL ETF", "color": "#58a6ff",
     "description": "Price of an ETF that tracks oil at several future delivery dates (smoother than tracking just one).",
     "link": "https://finance.yahoo.com/quote/USL"},
    {"patterns": ("refinery_utilization", "crude_imports",
                   "crude_stocks", "gasoline_imports",
                   "gasoline_stocks", "gasoline_product_supplied",
                   "distillate_crack"),
     "label": "EIA Weekly Petroleum Status Report", "color": "#3fb950",
     "description": "Weekly government data on how much oil and gasoline is being produced, refined, imported, and held in storage.",
     "link": "https://www.eia.gov/petroleum/supply/weekly/"},
    {"patterns": ("natgas_to_oil", "rbob_minus_brent",
                   "rbob_minus_wti", "rbob_to_wti"),
     "label": "Energy spread (bot-computed)", "color": "#bc8cff",
     "description": "Price difference (or ratio) between two energy products — used to spot when one is cheap relative to the other.",
     "link": "https://www.cmegroup.com/markets/energy/refined-products/rbob-gasoline-physical.html"},
    {"patterns": ("hurricane",),
     "label": "NOAA / NHC hurricane data", "color": "#3fb950",
     "description": "Whether a hurricane is currently active in the Atlantic. Hurricanes shut down Gulf oil rigs and refineries.",
     "link": "https://www.nhc.noaa.gov/"},
    {"patterns": ("summer_driving", "memorial_july4"),
     "label": "Seasonal / calendar", "color": "#d29922",
     "description": "Whether we're in summer driving season or near a major holiday weekend (both push up gas demand).",
     "link": ""},
    {"patterns": ("gas_price_anchor", "gas_pct_above", "gas_pct_below",
                   "gas_range", "gas_zscore", "gas_change_consistency"),
     "label": "Retail-gas target derivative (bot-computed)", "color": "#8b949e",
     "description": "How today's gas price compares to its recent history — its average, range, and how far above or below normal it sits.",
     "link": "https://fred.stlouisfed.org/series/GASREGW"},
    # ── Natural-gas-specific data ───────────────────────────────────
    {"patterns": ("ng_storage_bcf", "storage_change_wow", "storage_lag"),
     "label": "EIA NG Weekly Storage Report", "color": "#3fb950",
     "description": "How much natural gas is being held in underground storage tanks across the US. Low storage means tight supply.",
     "link": "https://www.eia.gov/dnav/ng/ng_stor_wkly_s1_w.htm"},
    {"patterns": ("ng_production", "production_lag", "production_yoy"),
     "label": "EIA NG production", "color": "#3fb950",
     "description": "How much natural gas is being pumped out of US wells.",
     "link": "https://www.eia.gov/naturalgas/production/"},
    {"patterns": ("region_gulf", "region_midwest", "region_northeast",
                   "region_south", "region_west"),
     "label": "NOAA regional weather", "color": "#3fb950",
     "description": "Temperature and weather in a specific region of the US. Hot or cold weather drives heating and cooling demand for natural gas.",
     "link": "https://www.ncei.noaa.gov/access/monitoring/dyk/heating-cooling-degree-information"},
    {"patterns": ("gulf_wind", "gulf_storm", "gulf_max_wind"),
     "label": "NOAA Gulf-of-Mexico weather", "color": "#3fb950",
     "description": "Wind and storm activity in the Gulf of Mexico, where much of the US's oil and gas is produced.",
     "link": "https://www.nhc.noaa.gov/"},
    {"patterns": ("lng_wind", "lng_storm", "lng_temp", "lng_terminal"),
     "label": "NOAA LNG-terminal weather", "color": "#3fb950",
     "description": "Weather near the terminals that ship US natural gas overseas. Bad weather can disrupt exports.",
     "link": "https://www.eia.gov/naturalgas/storage/dashboard/"},
    {"patterns": ("national_avg_temp", "national_cdd", "national_hdd",
                   "national_humidity", "national_wind"),
     "label": "NOAA national weather", "color": "#3fb950",
     "description": "Average US-wide temperature, humidity, and wind, weighted so populated areas count more.",
     "link": "https://www.ncei.noaa.gov/access/monitoring/dyk/heating-cooling-degree-information"},
    {"patterns": ("cdd", "hdd"),
     "label": "NOAA HDD/CDD", "color": "#3fb950",
     "description": "A measure of how cold or hot the weather is — basically the size of the gap between the day's temperature and 65°F. Predicts heating and cooling demand.",
     "link": "https://www.ncei.noaa.gov/access/monitoring/dyk/heating-cooling-degree-information"},
    {"patterns": ("humidity", "temp_lag", "wind_lag", "wind_rolling",
                   "heat_wave_days", "cold_wave_days"),
     "label": "NOAA weather", "color": "#3fb950",
     "description": "Local temperature, humidity, wind speed, or extreme-weather flag from US weather stations.",
     "link": "https://www.ncei.noaa.gov/access/monitoring/dyk/heating-cooling-degree-information"},
    # ── Time-of-period / seasonal flags ─────────────────────────────
    {"patterns": ("week_sin", "week_cos", "week_of_year",
                   "month_sin", "month_cos", "month", "quarter",
                   "day_of_year", "dow_sin", "dow_cos", "day_of_week",
                   "holiday", "is_holiday", "is_weekend", "is_thursday",
                   "is_winter", "is_summer", "is_shoulder", "winter"),
     "label": "Seasonal / calendar", "color": "#d29922",
     "description": "What time of year, day of week, or whether it's a holiday — lets the model pick up seasonal patterns.",
     "link": ""},
    # ── Bot-computed transforms of the target itself ────────────────
    {"patterns": ("target_lag", "target_rolling"),
     "label": "Target derivative (bot-computed)", "color": "#8b949e",
     "description": "What the thing we're predicting has done in the recent past — its previous values and rolling averages.",
     "link": ""},
    {"patterns": ("log_return", "roc_", "trend_dev", "trend_sma"),
     "label": "Derived transform (bot-computed)", "color": "#8b949e",
     "description": "How fast and in what direction the target has been changing — its momentum and trend.",
     "link": ""},
    {"patterns": ("_lag_", "rolling_", "ma13_", "ma52_",
                   "_change_", "_zscore", "_mean_", "_std_",
                   "rolling_mean", "_diff", "_surprise"),
     "label": "Derived transform (bot-computed)", "color": "#8b949e",
     "description": "A past value, recent average, or 'surprise vs expectations' calculated from one of the inputs above.",
     "link": ""},
    # ── NBA: catch-all for derived diff / home / away features. Sits
    # AFTER the tennis-specific rules so ``diff_form_last5`` already
    # got bucketed into "Form (Sackmann)" by then.
    {"patterns": ("diff_", "home_", "away_"),
     "label": "nba_api derived (diff)", "color": "#58a6ff",
     "description": "How the home and away teams compare on a specific stat (home value minus away value).",
     "link": "https://github.com/swar/nba_api"},
]


# Per-base feature descriptions. Maps the un-transformed feature root
# (e.g. ``rbob_gasoline_futures_last``) to a plain-English sentence
# describing what the raw data series is. The transform suffix (lag,
# return, volatility, rolling, etc.) is parsed off and appended at
# render time, so each fully-named feature ends up with a unique
# description even when several share a base.
#
# Order matters: more specific keys first (e.g.
# ``rbob_gasoline_futures_last`` before ``rbob_``). Lookup is by
# longest matching prefix.
_FEATURE_BASES: List[Tuple[str, str]] = [
    # ── Tennis / Table-tennis match-level features ──────────────────
    ("diff_surface_elo_pre",  "Pre-match Elo rating gap between the two players on this specific court surface — accounts for surface specialists."),
    ("diff_style_elo_pre",    "Pre-match Elo rating gap between the two players against opponents of this play style."),
    ("diff_elo_pre",          "Pre-match Elo rating gap between the two players. Higher = player A is more likely to win."),
    ("diff_days_rest",        "Gap in days since each player's last match. Positive = player A has had more rest."),
    ("diff_avg_serve_pts_won_10",  "Difference between the two players in % of points won on serve, averaged over each player's last 10 matches."),
    ("diff_avg_return_pts_won_10", "Difference in % of points won on return, averaged over each player's last 10 matches."),
    ("diff_avg_bp_saved_10",  "Difference in % of break points saved (when serving from behind), averaged over each player's last 10 matches."),
    ("diff_avg_game_margin_10",  "Difference in average game margin (how decisively each player wins) over their last 10 matches."),
    ("diff_avg_point_win_pct_10","Difference in overall point-win % between the two players over their last 10 matches."),
    ("diff_std_game_margin_10",  "Difference in how consistent each player's game margins have been over their last 10 matches. Lower = steadier."),
    ("diff_std_point_win_pct_10","Difference in how consistent each player's point-win rates have been over their last 10 matches."),
    ("diff_closing_win_pct_10",  "Difference in clutch factor: % of close games each player has won over their last 10 matches."),
    ("diff_deuce_win_pct_10",    "Difference in % of deuce points each player has won over their last 10 matches."),
    ("diff_comeback_rate_20",    "Difference in how often each player comes back from behind to win, over their last 20 matches."),
    ("diff_form_last5",  "Difference in win rate over each player's most recent 5 matches."),
    ("diff_form_last10", "Difference in win rate over each player's most recent 10 matches."),
    ("diff_form_last20", "Difference in win rate over each player's most recent 20 matches."),
    ("diff_matches_last_7d", "Difference in how many matches each player has played in the past 7 days. Heavy recent schedule = potential fatigue."),
    ("diff_hand_left", "Whether the two players' handedness pairing includes a leftie (lefties play differently)."),
    ("h2h_a_wins_last5",  "How many of their last 5 meetings player A has won against player B."),
    ("h2h_a_wins_minus_b_wins", "Net head-to-head record across all past meetings (A's wins minus B's wins)."),
    ("hand_matchup_lr", "Whether this is a lefty-vs-righty matchup. Lefties have a small structural advantage on tour."),
    ("is_bo7", "Whether the match is best-of-7 games (vs the standard best-of-5). Longer formats favour the more consistent player."),
    ("rank_diff",   "Gap between the two players' official world rankings (ATP/WTA)."),
    ("level_rank",  "How prestigious the tournament is (e.g. Grand Slam > Masters > 250). Bigger events draw stronger fields and play more conservatively."),
    ("round_rank",  "How deep in the bracket the match sits (1st round = early, final = late). Later rounds tend to be tighter."),

    # ── NBA matchup features ────────────────────────────────────────
    ("home_elo_pre", "Pre-game Elo rating of the home team."),
    ("away_elo_pre", "Pre-game Elo rating of the away team."),
    ("diff_elo",     "Home minus away Elo rating before the game."),
    ("home_b2b",     "Whether the home team is playing on the second night of a back-to-back."),
    ("away_b2b",     "Whether the away team is playing on the second night of a back-to-back."),
    ("diff_off_rating", "Difference between home and away in offensive efficiency (points scored per 100 possessions)."),
    ("diff_def_rating", "Difference between home and away in defensive efficiency (points allowed per 100 possessions)."),
    ("diff_net_rating", "Difference between home and away in net rating (offense minus defense per 100 possessions)."),
    ("diff_pace",       "Difference between home and away in pace of play (possessions per 48 minutes)."),
    ("diff_efg_pct",    "Difference in effective field-goal % (gives extra credit for 3-pointers)."),
    ("diff_oreb_pct",   "Difference in offensive rebounding rate."),
    ("diff_tov_pct",    "Difference in turnover rate."),
    ("diff_ft_per_fga", "Difference in how often the team gets to the free-throw line per shot attempt."),
    ("diff_fg3m",       "Difference in 3-pointers made per game."),
    ("diff_margin",     "Difference in average scoring margin in recent games."),
    ("diff_team_win",   "Difference in recent win rate between the two teams."),

    # ── Natural-gas-specific raw series ─────────────────────────────
    ("ng_storage_bcf",     "Total natural gas held in US underground storage tanks (billions of cubic feet). Low = supply is tight."),
    ("ng_production_bcfd", "Total US natural gas production, in billions of cubic feet per day."),
    ("region_gulf_temp_f",      "Average temperature in the Gulf region (Fahrenheit)."),
    ("region_midwest_temp_f",   "Average temperature in the Midwest region (Fahrenheit)."),
    ("region_northeast_temp_f", "Average temperature in the Northeast region (Fahrenheit)."),
    ("region_south_temp_f",     "Average temperature in the southern region (Fahrenheit)."),
    ("region_west_temp_f",      "Average temperature in the western region (Fahrenheit)."),
    ("region_gulf_cdd",      "Cooling-degree-days in the Gulf region — how warm it's been there."),
    ("region_midwest_cdd",   "Cooling-degree-days in the Midwest — how warm it's been there."),
    ("region_northeast_cdd", "Cooling-degree-days in the Northeast — how warm it's been there."),
    ("region_south_cdd",     "Cooling-degree-days in the southern region — how warm it's been there."),
    ("region_west_cdd",      "Cooling-degree-days in the western region — how warm it's been there."),
    ("region_gulf_hdd",      "Heating-degree-days in the Gulf region — how cold it's been there."),
    ("region_midwest_hdd",   "Heating-degree-days in the Midwest — how cold it's been there."),
    ("region_northeast_hdd", "Heating-degree-days in the Northeast — how cold it's been there."),
    ("region_south_hdd",     "Heating-degree-days in the southern region — how cold it's been there."),
    ("region_west_hdd",      "Heating-degree-days in the western region — how cold it's been there."),
    ("gulf_wind",       "Wind speed over the Gulf of Mexico — high winds disrupt offshore oil and gas operations."),
    ("gulf_storm",      "Whether a named storm is currently active in the Gulf of Mexico."),
    ("gulf_max_wind",   "Peak wind gust recorded over the Gulf of Mexico."),
    ("lng_wind",        "Wind speed at US LNG export terminals — high winds halt tanker loading."),
    ("lng_storm",       "Whether a storm is currently affecting US LNG export terminals."),
    ("lng_temp",        "Temperature at US LNG export terminals."),
    ("lng_terminal_avg",   "Average operating conditions across US LNG export terminals."),
    ("lng_terminal_storm", "Whether any storm has touched a US LNG terminal."),
    ("lng_terminal_wind",  "Wind speed at US LNG export terminals."),
    ("national_avg_temp",     "Average US temperature, weighted so heavily-populated areas count more."),
    ("national_cdd",          "US-wide cooling-degree-days, population-weighted."),
    ("national_hdd",          "US-wide heating-degree-days, population-weighted."),
    ("national_humidity_pct", "Average US humidity, population-weighted."),
    ("national_wind_mph",     "Average US wind speed (mph), population-weighted."),
    ("cdd_sum_3d",     "Total cooling-degree-days over the past 3 days."),
    ("hdd_sum_3d",     "Total heating-degree-days over the past 3 days."),
    ("cold_wave_days", "Number of recent consecutive days flagged as a cold wave (extreme cold)."),
    ("heat_wave_days", "Number of recent consecutive days flagged as a heat wave (extreme heat)."),
    ("gulf_storm_count",   "Number of named storms currently active in the Gulf of Mexico."),
    ("gulf_storm_active",  "1 if any named storm is active in the Gulf, else 0."),
    ("lng_storm_count",    "Number of storms currently affecting US LNG export terminals."),
    ("cdd",      "Cooling-degree-days — a measure of how warm the day was (how far the average temperature sat above 65°F). Predicts AC / power demand."),
    ("hdd",      "Heating-degree-days — a measure of how cold the day was (how far the average temperature sat below 65°F). Predicts heating demand."),
    ("humidity", "Local humidity."),
    ("temp",     "Local temperature."),
    ("wind",     "Local wind speed."),
    ("is_winter",   "Whether it's currently winter (drives natural-gas heating demand)."),
    ("is_summer",   "Whether it's currently summer (drives cooling / electricity demand)."),
    ("is_shoulder", "Whether we're in the shoulder season between winter and summer (mild weather, low energy demand)."),
    ("is_thursday", "Whether today is Thursday — the day EIA releases its weekly natural-gas storage report. Markets often move ahead of the print."),
    ("is_weekend",  "Whether today is Saturday or Sunday."),
    ("is_holiday",  "Whether today is a US federal holiday."),
    ("holiday",     "Whether today is a US federal holiday."),
    ("day_of_week", "Day of the week (0 = Monday … 6 = Sunday)."),
    ("day_of_year", "Day of the year (1–366)."),
    ("week_of_year","Week of the year (1–52)."),
    ("dow_sin",   "Day-of-week encoded as a sine wave so the model can pick up weekly cycles."),
    ("dow_cos",   "Day-of-week encoded as a cosine wave (paired with dow_sin to mark position in the week)."),
    ("week_sin",  "Week-of-year encoded as a sine wave so the model can pick up seasonal patterns."),
    ("week_cos",  "Week-of-year encoded as a cosine wave (paired with week_sin to mark position in the year)."),
    ("month_sin", "Month-of-year encoded as a sine wave for seasonality."),
    ("month_cos", "Month-of-year encoded as a cosine wave for seasonality."),
    ("quarter",   "What quarter of the year it is (1–4)."),
    ("month",     "What month it is (1 = January … 12 = December)."),
    ("winter",         "Whether it's currently winter."),
    ("summer_driving", "Whether we're in the US summer driving season (Memorial Day → Labor Day). Gasoline demand peaks here."),
    ("memorial_july4", "Whether we're near Memorial Day or July 4th — major holiday weekends spike gasoline demand."),
    ("hurricane",      "Whether a hurricane is currently active in the Atlantic basin."),
    ("log_return_abs",   "Absolute size of the most recent daily price moves (regardless of direction)."),
    ("log_return_accel", "Whether the rate-of-change in price is speeding up or slowing down."),
    ("log_return_std",   "How spread-out recent daily returns have been."),
    ("log_return_vol",   "Realized volatility of recent daily price returns."),
    ("log_return",       "Daily log return of the price (a smoother measure than % change)."),
    ("roc_7",  "Rate of change over the past 7 days (% change over a week)."),
    ("roc_30", "Rate of change over the past 30 days (% change over a month)."),
    ("roc_90", "Rate of change over the past 90 days (% change over a quarter)."),
    ("trend_dev_30", "How far the price has drifted away from its 30-day moving average."),
    ("trend_dev_90", "How far the price has drifted away from its 90-day moving average."),
    ("trend_sma7_minus_sma30",  "7-day moving average minus the 30-day moving average. Positive = short-term uptrend."),
    ("trend_sma30_minus_sma90", "30-day moving average minus the 90-day moving average. Positive = medium-term uptrend."),
    ("target_rolling_90_std",  "Standard deviation of natural-gas price over the past 90 days — a measure of recent volatility."),
    ("target",      "Natural-gas closing price."),
    ("storage_change_wow", "Week-over-week change in US natural-gas storage levels."),
    ("storage",     "US natural-gas storage level."),
    ("production_yoy", "Year-over-year change in US natural-gas production."),
    ("production",  "US natural-gas production (billions of cubic feet per day)."),

    # ── Retail-gas / energy raw series ──────────────────────────────
    ("rbob_gasoline_futures_last",  "Wholesale gasoline price — RBOB futures close. What gas stations pay to buy gasoline; moves before pump prices do."),
    ("rbob_gasoline_futures_mean",  "Average wholesale gasoline price (RBOB futures) over the trading week."),
    ("rbob_minus_brent_per_gallon", "Gap (per gallon) between US wholesale gasoline and Brent crude. Widens when refining margins are healthy."),
    ("rbob_minus_wti_per_gallon",   "Gap (per gallon) between US wholesale gasoline and WTI crude — essentially the gasoline refining margin."),
    ("rbob_to_wti_per_gallon_ratio","Ratio of wholesale gasoline to WTI crude (per gallon). Indicates the relative profit margin for refining."),
    ("brent_spot",         "Brent crude oil spot price — the international benchmark for oil."),
    ("brent_futures_last", "Brent crude oil futures closing price."),
    ("brent_wti_spread",   "Price gap between Brent and WTI crude. Widens when it's harder to export US oil."),
    ("wti_spot",            "West Texas Intermediate (WTI) crude oil spot price — the US benchmark."),
    ("wti_futures_last",    "WTI crude oil futures closing price."),
    ("wti_term_structure",  "Shape of the WTI futures curve — whether near-term oil is cheaper or pricier than longer-dated. Reveals supply tightness."),
    ("crude_imports",        "Volume of crude oil imported into the US."),
    ("crude_stocks_ex_spr",  "US crude oil inventories, excluding the Strategic Petroleum Reserve."),
    ("gasoline_imports",     "Volume of gasoline imported into the US."),
    ("gasoline_stocks_total","Total US gasoline inventories held by refiners and distributors."),
    ("gasoline_product_supplied", "How much gasoline US refiners delivered to the market this week — a proxy for consumer demand."),
    ("distillate_crack",     "Profit margin from refining crude oil into distillates (diesel, heating oil)."),
    ("heating_oil_futures_last", "Heating oil futures closing price (also used to price diesel)."),
    ("natural_gas_futures_last", "Natural gas futures closing price."),
    ("dxy_dollar_index_last",   "US Dollar Index (DXY) close — how strong the dollar is against major currencies. Stronger dollar tends to push oil down."),
    ("trade_weighted_dollar",   "Broad measure of the US dollar against many trading-partner currencies."),
    ("ovx_level",               "Oil VIX — how volatile traders expect oil prices to be over the next month."),
    ("energy_sector_etf_last",  "XLE ETF closing price — tracks the big US energy companies (Exxon, Chevron, etc)."),
    ("uga_gasoline_etf_last",   "UGA ETF closing price — directly tracks gasoline futures."),
    ("uso_oil_etf_last",        "USO ETF closing price — directly tracks oil futures."),
    ("usl_12mo_oil_etf_last",   "USL ETF closing price — tracks oil at 12 future delivery dates (smoother than just front-month)."),
    ("refinery_utilization",    "Percentage of US refinery capacity currently in use."),
    ("industrial_production",   "US industrial production index — how busy factories, mines and utilities are."),
    ("natgas_to_oil_ratio",     "Ratio of natural-gas to crude-oil prices. Reveals which energy source is cheap relative to the other."),
    ("gas_pct_above_26w_high",  "How far today's retail gas price sits above its 26-week high (% above)."),
    ("gas_pct_above_13w_high",  "How far today's retail gas price sits above its 13-week high (% above)."),
    ("gas_pct_above_4w_high",   "How far today's retail gas price sits above its 4-week high (% above)."),
    ("gas_pct_below_26w_high",  "How far today's retail gas price sits below its 26-week high (% below)."),
    ("gas_pct_below_13w_high",  "How far today's retail gas price sits below its 13-week high (% below)."),
    ("gas_pct_below_4w_high",   "How far today's retail gas price sits below its 4-week high (% below)."),
    ("gas_pct_below_26w_low",   "How far today's retail gas price sits below its 26-week low (% below)."),
    ("gas_pct_below_13w_low",   "How far today's retail gas price sits below its 13-week low (% below)."),
    ("gas_pct_above_26w_low",   "How far today's retail gas price sits above its 26-week low (% above)."),
    ("gas_pct_above_13w_low",   "How far today's retail gas price sits above its 13-week low (% above)."),
    ("gas_pct_above_4w_low",    "How far today's retail gas price sits above its 4-week low (% above)."),
    ("gas_range_4w",            "Spread (high minus low) of retail gas price over the past 4 weeks."),
    ("gas_range_13w",           "Spread of retail gas price over the past 13 weeks."),
    ("gas_zscore_13w",          "How many standard deviations today's gas price sits from its 13-week average."),
    ("gas_zscore_26w",          "How many standard deviations today's gas price sits from its 26-week average."),
    ("gas_zscore_52w",          "How many standard deviations today's gas price sits from its 52-week average."),
    ("gas_price_anchor",        "Recent retail gas price used as the starting point for projecting future prices."),
    ("gas_change_consistency",  "Whether recent week-over-week gas-price moves have all gone the same direction (consistent trend) or zig-zagged."),

    # ── Jobless-claims alt-data ─────────────────────────────────────
    ("google_trends_filed_for_unemployment", "How often Americans Google 'filed for unemployment' — a real-time read on new jobless-claims filers."),
    ("google_trends_unemployment_benefits",  "How often Americans Google 'unemployment benefits' — signals would-be jobless-claims filers."),
    ("google_trends_how_to_file_unemployment", "How often Americans Google 'how to file unemployment' — likely first-time jobless-claims filers."),
    ("google_trends_laid_off",   "How often Americans Google 'laid off'."),
    ("google_trends_lost_my_job","How often Americans Google 'lost my job'."),
    ("google_trends",            "Volume of jobless-claims-related Google searches across the US."),
    ("reddit",       "Number of posts about losing a job submitted to r/layoffs."),
    ("bfs_total_apps",      "Number of business applications filed in the US this week (Census BFS via FRED). Slowing applications mean fewer new employers about to hire."),
    ("bfs_high_wage_apps",  "Business applications from people who plan to pay wages — the subset most likely to actually hire."),
    ("bfs_high_propensity", "Business applications the Census Bureau flags as highly likely to become employer businesses within a year."),
    ("eia_diesel_demand",   "Total US diesel fuel consumed this week — proxy for freight/trucking activity."),
    ("eia_gasoline_demand", "Total US gasoline consumed this week — proxy for commuting activity."),
    ("noaa_cdo_hdd_mean",   "Heating-degree days across five major US cities — how much colder than 65°F it was this week."),
    ("noaa_cdo_cdd_mean",   "Cooling-degree days across five major US cities — how much hotter than 65°F it was this week."),
    ("noaa_cdo_cold_snap_flag", "Flag for whether any of five major US cities had a day with low temp under 20°F this week."),
    ("noaa_cdo_heat_wave_flag", "Flag for whether any of five major US cities had a day with high temp over 95°F this week."),
    ("noaa_cdo_heavy_precip_flag", "Flag for whether any of five major US cities had more than 1.5 inches of precipitation in a single day."),
    ("noaa_cdo_tmin_min",   "Coldest daily-minimum temperature across five major US cities this week."),
    ("noaa_cdo_tmax_max",   "Hottest daily-maximum temperature across five major US cities this week."),

    # ── Jobless-claims / macro raw series (FRED friendly names) ────
    ("nonfarm_payrolls",  "Total US employment (excluding farm workers) — the headline jobs number."),
    ("treasury_10y",      "Interest rate on a 10-year US government bond."),
    ("treasury_2y",       "Interest rate on a 2-year US government bond."),
    ("wti_oil",           "WTI crude oil price — the US benchmark."),
    ("henry_hub",         "Henry Hub natural-gas benchmark price."),
    ("vix",               "Stock-market expected volatility over the next month — the 'fear gauge'."),
    ("unemployment_rate", "% of Americans who want a job but don't have one."),
    ("continuing_claims", "Number of people still collecting jobless benefits (continuing claims)."),
    ("initial_claims",    "Number of new initial jobless claims filed this week — the thing this bot predicts."),
    ("ppi",               "Producer Price Index — what wholesalers charge stores."),
    ("headline_cpi",      "Headline consumer price level (everything in the basket)."),
    ("core_mom",          "Month-over-month change in core inflation (excluding food and energy)."),
    ("core_cpi",          "Consumer price level excluding food and gas."),
    ("used_cars_cpi",     "Consumer-price-index sub-component for used cars and trucks."),
    ("fed_funds_rate",    "The Federal Reserve's target interest rate."),
    ("industrial_prod",   "US industrial production index."),
    ("umich_inflation",   "1-year inflation expectations from the U-Michigan consumer survey."),
    ("consumer_sentiment","U-Michigan consumer sentiment index — how Americans feel about the economy."),
    ("cleveland_expinf",  "Cleveland Fed model's expectation for inflation 1 year out."),
    ("m2_yoy",            "Year-over-year change in M2 money supply (cash, checking, savings accounts circulating in the US economy)."),
    ("retail_gas",        "US average pump price for regular gasoline."),
    ("jolts_layoffs",     "How many workers were laid off across the US in the latest month."),
    ("jolts_hires",       "How many workers were hired across the US in the latest month."),
    ("jolts_quits",       "How many workers quit their job in the latest month."),
    ("jolts_openings",    "How many job openings are posted across the US."),
    ("unemp_5_14_weeks",  "Number of Americans who have been unemployed for 5–14 weeks."),
    ("unemp_27plus_weeks","Number of Americans who have been unemployed for 27+ weeks (long-term unemployed)."),
    ("durable_orders",    "Orders for big-ticket items expected to last 3+ years."),
    ("policy_uncertainty","Measure of US economic-policy uncertainty from news coverage of policy disputes."),

    # ── Jobless-claims-bot base names not covered elsewhere ─────────
    ("yield_curve_spread", "Gap between long-term and short-term Treasury yields. An inverted (negative) spread has historically preceded recessions."),
    ("claims",   "Number of new jobless claims filed this week (the thing we're predicting)."),
    ("change",   "Week-over-week change in the target (initial jobless claims count)."),
    ("jolts_layoffs_to_hires", "Ratio of layoffs to hires from the JOLTS report — how loose vs tight the job market is."),

    # ── CPI-bot raw-series base names ───────────────────────────────
    ("cleveland_expinf_1y", "Cleveland Fed model's 1-year-ahead expected inflation."),
    ("umich_inflation_1y",  "1-year-ahead inflation expectations from the U-Michigan consumer survey."),
    ("headline_cpi_mom",    "Month-over-month change in the overall consumer price level (the thing we're predicting on the headline CPI bot)."),
    ("core_mom",            "Month-over-month change in core inflation (excluding food and energy) — the thing we're predicting on the core-CPI bot."),
    ("retail_gas_mom",      "Month-over-month change in US average pump prices for regular gasoline."),
    ("used_cars_cpi_mom",   "Month-over-month change in the used-car CPI sub-component."),
    ("wti_oil_mom",         "Month-over-month change in WTI crude oil price."),
    ("ppi_mom",             "Month-over-month change in the Producer Price Index."),

    # ── NBA non-rolling features ────────────────────────────────────
    ("h2h_wins_before", "Number of times these two teams have met before this season — head-to-head sample size."),
    ("elo_win_prob_home", "Pre-game probability the home team wins, implied by the Elo ratings and home-court bonus."),
    ("elo_diff",        "Pre-game Elo rating gap between the two teams (home minus away), plus the home-court bonus."),
    ("rest_diff",       "Home team's days of rest minus the away team's."),
    ("b2b_diff",        "Home team's back-to-back flag minus the away team's. ±1 = one team is on a back-to-back, the other isn't."),
    ("home_elo_pre",    "Pre-game Elo rating of the home team."),
    ("away_elo_pre",    "Pre-game Elo rating of the away team."),
    ("home_days_rest",  "Days since the home team's last game."),
    ("away_days_rest",  "Days since the away team's last game."),
    ("home_b2b",        "1 if the home team is on the second night of a back-to-back, else 0."),
    ("away_b2b",        "1 if the away team is on the second night of a back-to-back, else 0."),
    ("home_long_rest",  "1 if the home team has had 4+ days of rest, else 0."),
    ("away_long_rest",  "1 if the away team has had 4+ days of rest, else 0."),
    ("home_games_into_season", "How many games the home team has played this season so far."),
    ("away_games_into_season", "How many games the away team has played this season so far."),
]


# Per-stat NBA descriptions. Keys are the lowercased stat names
# (matches whatever sits between HOME_TEAM_ / AWAY_TEAM_ / DIFF_ and
# the rolling-window suffix). Values are the human-readable name for
# the stat, ready to drop into a sentence like
# "Home team's <stat>, averaged over their last 10 games."
_NBA_STAT_DESCRIPTIONS: dict = {
    "off_rating":  "offensive efficiency (points scored per 100 possessions)",
    "def_rating":  "defensive efficiency (points allowed per 100 possessions)",
    "net_rating":  "net rating (offense minus defense per 100 possessions)",
    "pace":        "pace of play (possessions per 48 minutes)",
    "efg_pct":     "effective field-goal percentage (gives extra credit for made 3-pointers)",
    "tov_pct":     "turnover rate (turnovers per possession)",
    "oreb_pct":    "offensive-rebound rate (% of own missed shots rebounded)",
    "ft_per_fga":  "free-throw trips per field-goal attempt (how often the team draws fouls)",
    "margin":      "scoring margin (points scored minus points allowed)",
    "win":         "win rate",
    "fg3m":        "3-pointers made per game",
    "fg3a":        "3-point attempts per game",
}


def _describe_nba_rolling(name: str) -> str:
    """If ``name`` is an NBA rolling-stat feature
    (``HOME_TEAM_<stat>_R<N>``, ``AWAY_TEAM_<stat>_R<N>``, or
    ``DIFF_<stat>_R<N>``), return a feature-specific description.
    Returns "" otherwise so the caller can fall through.
    """
    n = (name or "").lower()
    m = re.match(
        r"^(home_team_|away_team_|diff_)(.+?)_r(\d+)$", n
    )
    if not m:
        return ""
    prefix, stat, window = m.group(1), m.group(2), int(m.group(3))
    stat_desc = _NBA_STAT_DESCRIPTIONS.get(stat)
    if not stat_desc:
        return ""
    if prefix == "home_team_":
        return (f"Home team's {stat_desc}, averaged over "
                f"their last {window} games.")
    if prefix == "away_team_":
        return (f"Away team's {stat_desc}, averaged over "
                f"their last {window} games.")
    # diff_
    return (f"Home minus away differential in {stat_desc}, "
            f"averaged over each team's last {window} games.")


def _period_unit(base: str, plural: bool = True,
                  full_name: str = "",
                  cadence: str = "") -> str:
    """Pick the right time-unit (day / week / month) for transforms
    attached to this base.

    Daily-cadence bots (natural-gas) lag and roll in DAYS. Monthly-
    cadence bot (CPI) lags and rolls in MONTHS. Weekly bots (retail-
    gas, unemployment) lag and roll in WEEKS.

    Resolution order:
      1. ``cadence`` argument (e.g. "months") — most reliable when
         the caller has scanned the whole feature list.
      2. Per-feature monthly markers (``_mom`` / ``_1y_`` / etc.)
      3. Daily vocabulary in the base.
      4. Default to weekly.
    """
    if cadence:
        if cadence.startswith("month"):
            return "months" if plural else "month"
        if cadence.startswith("day"):
            return "days" if plural else "day"
        if cadence.startswith("week"):
            return "weeks" if plural else "week"

    daily_exact = {
        "target", "production", "storage", "log_return",
        "log_return_abs", "log_return_accel", "log_return_std",
        "log_return_vol", "temp", "wind", "humidity",
        "cdd", "hdd",
    }
    daily_prefixes = (
        "target_", "production_", "storage_", "log_return_",
        "roc_", "roc", "trend_dev", "trend_sma",
        "region_", "gulf_", "lng_", "national_", "ng_",
        "cdd_", "hdd_", "humidity_", "temp_", "wind_",
        "cold_wave", "heat_wave",
    )
    # CPI features carry a ``_mom`` / ``_1y_`` tag in their full
    # name — fall back to that when the batch cadence isn't passed.
    if full_name and any(t in full_name for t in (
        "_mom", "_1y_", "headline_cpi", "core_mom",
        "cleveland_expinf", "umich_inflation_1y",
    )):
        return "months" if plural else "month"

    is_daily = base in daily_exact or any(
        base.startswith(p) for p in daily_prefixes
    )
    if is_daily:
        return "days" if plural else "day"
    return "weeks" if plural else "week"


def _detect_bot_cadence(features: List[dict]) -> str:
    """Detect the bot's cadence by scanning the whole feature list.

    Returns one of ``"days"``, ``"weeks"``, ``"months"``. This is
    far more reliable than per-feature detection because shared
    bases (e.g. ``vix_zscore_N``) appear in both monthly (CPI) and
    weekly (unemployment) bots — only the surrounding feature set
    tells you which.
    """
    names = " ".join((f.get("feature") or "").lower() for f in features)
    if any(t in names for t in (
        "_mom", "headline_cpi_mom", "core_mom",
        "cleveland_expinf", "umich_inflation_1y",
    )):
        return "months"
    if any(t in names for t in (
        "target_lag", "target_rolling", "ng_storage",
        "ng_production", "region_", "log_return", "trend_dev",
    )):
        return "days"
    return "weeks"


def _strip_transform_suffix(name: str, cadence: str = "") -> Tuple[str, str]:
    """Peel off the transform suffix and return (base, transform_text).

    Recognised suffixes (in the order they're stripped off the right
    end of the name):

      * ``_lag_N``                       →  "lagged N {unit}"
      * ``_change_lag_N``                →  "week-/month-over-period change, lagged N"
      * ``_surprise_vs_Nw_avg``          →  "vs its N-week average"
      * ``_anomaly_30d``                 →  "anomaly vs the 30-day norm"
      * ``_zscore_N``                    →  "z-scored vs the past N-{unit} window"
      * ``_dev_ma_N`` / ``_dev_N``       →  "deviation from N-{unit} moving average"
      * ``_rolling_N``                   →  "N-{unit} rolling average"
      * ``_mean_N``                      →  "N-{unit} mean"
      * ``_return_Nw`` / ``_volatility_Nw`` /
        ``_change_Nw`` / ``_change_Nm``  →  "{N}-week / month return etc"
      * ``_yoy``                         →  "year-over-year change"
    """
    transforms: List[Any] = []
    PLACEHOLDER_LAG = "<lag>"
    PLACEHOLDER_ROLL = "<roll>"
    PLACEHOLDER_ZSCORE = "<zscore>"
    PLACEHOLDER_DEVMA = "<devma>"
    PLACEHOLDER_MEAN = "<mean>"
    PLACEHOLDER_CHANGE_LAG = "<changelag>"
    original = name

    # _lag_N — the outermost transform. Unit is resolved later once
    # we know the base.
    m = re.search(r"_lag_(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_LAG, int(m.group(1))))
        name = name[: m.start()]

    # _change_lag_N is sometimes written ``_change_lag_N`` (no week
    # qualifier) — interpret as "1-period change, lagged N".
    m = re.search(r"_change$", name)
    if m and transforms and isinstance(transforms[-1], tuple) and transforms[-1][0] == PLACEHOLDER_LAG:
        # We just stripped a _lag_N off; if what remains ends in
        # _change, treat the pair as a single change-then-lag transform.
        lag_n = transforms.pop()[1]
        transforms.append((PLACEHOLDER_CHANGE_LAG, lag_n))
        name = name[: m.start()]

    # _surprise_vs_Nw_avg — sits between the base and the lag.
    m = re.search(r"_surprise_vs_(\d+)w_avg$", name)
    if m:
        transforms.append(f"surprise vs the {m.group(1)}-week average")
        name = name[: m.start()]

    # _anomaly_30d
    m = re.search(r"_anomaly_30d$", name)
    if m:
        transforms.append("anomaly vs the 30-day norm")
        name = name[: m.start()]

    # _zscore_N — z-scored vs an N-period look-back window.
    m = re.search(r"_zscore_(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_ZSCORE, int(m.group(1))))
        name = name[: m.start()]

    # _dev_ma_N — deviation from N-period moving average. NB: we do
    # NOT match a bare ``_dev_N`` here, because some bases end in
    # ``trend_dev_30`` / ``trend_dev_90`` and the literal ``_30``
    # suffix is part of the base, not a transform.
    m = re.search(r"_dev_ma(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_DEVMA, int(m.group(1))))
        name = name[: m.start()]

    # _rolling_N — unit is resolved against the base.
    m = re.search(r"_rolling_(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_ROLL, int(m.group(1))))
        name = name[: m.start()]

    # _mean_N — N-period rolling mean (no explicit unit in name).
    m = re.search(r"_mean_(\d+)$", name)
    if m:
        transforms.append((PLACEHOLDER_MEAN, int(m.group(1))))
        name = name[: m.start()]

    # _return_Nw / _volatility_Nw / _change_Nw / _change_Nm
    # (units explicit in name — no period-unit resolution needed).
    for pat, fmt in (
        (r"_return_(\d+)w$",     "{n}-week log return"),
        (r"_volatility_(\d+)w$", "{n}-week realized volatility"),
        (r"_change_(\d+)w$",     "{n}-week % change"),
        (r"_change_(\d+)m$",     "{n}-month % change"),
    ):
        m = re.search(pat, name)
        if m:
            transforms.append(fmt.format(n=m.group(1)))
            name = name[: m.start()]
            break

    # NB: ``_yoy`` is intentionally NOT stripped here. It's usually
    # baked into the BASE name (``m2_yoy``, ``production_yoy``) — the
    # underlying series is already a year-over-year measure — so
    # stripping it would lose the "_yoy" suffix from the base lookup.

    base = name

    # Resolve placeholders now that we know the base + original name.
    def _unit(plural):
        return _period_unit(base, plural=plural, full_name=original,
                             cadence=cadence)
    resolved: List[str] = []
    for t in transforms:
        if isinstance(t, tuple) and t[0] == PLACEHOLDER_LAG:
            n = t[1]
            unit = _unit(plural=(n != 1))
            resolved.append(f"lagged {n} {unit}")
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_CHANGE_LAG:
            n = t[1]
            unit = _unit(plural=(n != 1))
            resolved.append(f"1-{unit.rstrip('s')} change, lagged {n} {unit}")
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_ROLL:
            n = t[1]
            unit = _unit(plural=True)
            resolved.append(f"{n}-{unit.rstrip('s')} rolling average")
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_MEAN:
            n = t[1]
            unit = _unit(plural=True)
            resolved.append(f"{n}-{unit.rstrip('s')} rolling mean")
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_ZSCORE:
            n = t[1]
            unit = _unit(plural=True)
            resolved.append(
                f"z-scored against its past {n}-{unit.rstrip('s')} "
                f"window (how many standard deviations from the mean)"
            )
        elif isinstance(t, tuple) and t[0] == PLACEHOLDER_DEVMA:
            n = t[1]
            unit = _unit(plural=True)
            resolved.append(
                f"deviation from its {n}-{unit.rstrip('s')} moving average"
            )
        else:
            resolved.append(t)

    # Reverse so the inner-most transform reads first.
    resolved.reverse()
    if not resolved:
        return base, ""
    return base, " — " + ", ".join(resolved) + "."


def _base_description(base: str) -> str:
    """Look up the plain-English description for a feature's base.

    Picks the LONGEST matching prefix so e.g. ``headline_cpi_mom``
    resolves to the ``headline_cpi_mom`` entry (not the shorter
    ``headline_cpi`` one), independent of declaration order. Returns
    an empty string when nothing matches — caller falls back to the
    rule's generic description.
    """
    best_prefix = ""
    best_desc = ""
    for prefix, desc in _FEATURE_BASES:
        if base == prefix or base.startswith(prefix + "_"):
            if len(prefix) > len(best_prefix):
                best_prefix, best_desc = prefix, desc
    return best_desc


def _describe_feature(name: str, cadence: str = "") -> str:
    """Produce a unique, feature-specific description.

    The base-prefix description + the parsed transform suffix combine
    to a sentence like: "Wholesale gasoline price (RBOB futures
    close) ... — 4-week log return, lagged 1 week."

    NBA rolling-stat features (``HOME_TEAM_OFF_RATING_R10`` and
    friends) are handled by a dedicated parser first, since their
    naming convention doesn't fit the base-prefix model.
    """
    # NBA rolling stats — try the dedicated parser before falling
    # through to the generic base + transform pipeline.
    nba = _describe_nba_rolling(name)
    if nba:
        return nba

    lower = (name or "").lower()
    base, transform = _strip_transform_suffix(lower, cadence=cadence)
    desc = _base_description(base)
    if not desc:
        # No specific base match — return empty so feature_metadata
        # falls back to the rule's generic description.
        return ""
    # Trim the trailing period on the base before appending the
    # transform fragment so the punctuation reads cleanly.
    if transform:
        if desc.endswith("."):
            desc = desc[:-1]
        return desc + transform
    return desc


def feature_metadata(name: str, cadence: str = "") -> dict:
    """Map a feature name to its source label, colour, plain-English
    description, and link to where the raw data comes from.

    ``cadence`` is an optional hint ("days" / "weeks" / "months")
    derived from the surrounding feature set — pass it when a
    feature's name alone is ambiguous (e.g. ``vix_zscore_3`` could
    be monthly in CPI's panel or weekly in unemployment's). When
    omitted, the function falls back to per-feature heuristics.
    """
    n = (name or "").lower()
    for rule in FEATURE_RULES:
        for pat in rule["patterns"]:
            if pat in n:
                specific = _describe_feature(name, cadence=cadence)
                return {
                    "label": rule["label"],
                    "color": rule["color"],
                    "description": specific or rule["description"],
                    "link": rule["link"],
                }
    specific = _describe_feature(name, cadence=cadence)
    return {"label": "Other", "color": "#6e7681",
            "description": specific or
                "Source for this feature hasn't been documented yet.",
            "link": ""}


def _readable_feature_name(name: str) -> str:
    """Render a raw feature identifier in a more reader-friendly form.

    Replaces underscores with spaces and capitalizes the first letter;
    leaves embedded numbers / abbreviations alone. The raw name still
    appears beneath it in monospace for users who need the canonical
    identifier.
    """
    s = (name or "").replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else ""


def _render_models_run_table(
    metrics: Dict[str, Any] | None,
    *,
    feature_count: int | None = None,
    last_trained: str = "—",
    fallback_rows: List[Tuple[str, Dict[str, Any]]] | None = None,
) -> str:
    """Unified 'Models run' table used on every bot's Models tab.

    One row per model the trainer produced; columns: Accuracy / F1 /
    Precision / Recall / ROC AUC / Brier / Last trained. (Features and
    train/test row counts were dropped per operator request —
    ``feature_count`` stays in the signature so callers don't churn.)

    ``metrics`` is the trainer's metrics.json dict. Tennis-shape trainers
    populate ``per_model`` (dict of model_name → metric dict) plus an
    ``elo_only`` baseline and separate ``ensemble`` / ``blended`` rollups.
    sim.db-shape trainers store just the final classifier's numbers at
    the top level (``blended``); callers pass those via ``fallback_rows``
    when the ``per_model`` dict is missing.
    """
    metrics = metrics or {}
    rows_source: List[Tuple[str, Dict[str, Any]]] = []

    # Order rows: baseline first (grounds every other number), then
    # each per-model entry in a stable order, then the rollups.
    if metrics.get("elo_only"):
        rows_source.append(("Elo baseline", metrics["elo_only"]))
    per_model = metrics.get("per_model") or {}
    if isinstance(per_model, dict):
        # Stable order — the trainer key order isn't guaranteed by
        # json.load, so we sort by best Brier ascending so the strongest
        # component sits at the top of its group.
        pm_sorted = sorted(
            per_model.items(),
            key=lambda kv: (float(kv[1].get("brier") or 1.0), kv[0]),
        )
        for name, block in pm_sorted:
            if isinstance(block, dict):
                rows_source.append((name.upper(), block))
    if metrics.get("ensemble"):
        rows_source.append(("Ensemble", metrics["ensemble"]))
    if metrics.get("blended"):
        rows_source.append(("Blended (final)", metrics["blended"]))
    if not rows_source and fallback_rows:
        rows_source = list(fallback_rows)

    rows_train = metrics.get("rows_train")
    rows_test = metrics.get("rows_test")

    def _pct(v: Any, decimals: int = 1) -> str:
        try:
            return f"{float(v) * 100:.{decimals}f}%"
        except (TypeError, ValueError):
            return "—"

    def _num(v: Any, fmt: str = "{:.4f}") -> str:
        try:
            return fmt.format(float(v))
        except (TypeError, ValueError):
            return "—"

    def _int_str(v: Any) -> str:
        try:
            return f"{int(v):,}"
        except (TypeError, ValueError):
            return "—"

    parts: List[str] = []
    parts.append(
        "<h3 class='subhead'>Models run "
        "<span class='small gray'>(held-out test set — same stats as "
        "the home-page model cards plus Brier)</span></h3>"
    )
    parts.append(
        "<div style='overflow-x:auto;margin-bottom:14px;'>"
        "<table class='models-run-table' "
        "style='width:100%;border-collapse:collapse;font-size:12.5px;'>"
    )
    parts.append(
        "<thead><tr>"
        "<th style='text-align:left;'>Model</th>"
        "<th class='num'>Accuracy</th>"
        "<th class='num'>F1</th>"
        "<th class='num'>Precision</th>"
        "<th class='num'>Recall</th>"
        "<th class='num'>ROC AUC</th>"
        "<th class='num'>Brier</th>"
        "<th class='num'>Last trained</th>"
        "</tr></thead><tbody>"
    )
    if not rows_source:
        parts.append(
            "<tr><td colspan='8' class='gray' "
            "style='text-align:center;padding:10px;'>"
            "No model metrics available — the trainer has not written "
            "metrics.json yet.</td></tr>"
        )
    else:
        for label, block in rows_source:
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(label))}</td>"
                f"<td class='num'>{_pct(block.get('accuracy'))}</td>"
                f"<td class='num'>{_pct(block.get('f1'))}</td>"
                f"<td class='num'>{_pct(block.get('precision'))}</td>"
                f"<td class='num'>{_pct(block.get('recall'))}</td>"
                f"<td class='num'>{_pct(block.get('roc_auc'))}</td>"
                f"<td class='num'>{_num(block.get('brier'))}</td>"
                f"<td class='num'>{html.escape(str(last_trained))}</td>"
                "</tr>"
            )
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _render_feature_source_table(features: List[dict]) -> str:
    """Aligned feature table with the importance bar on the right.

    Columns: readable feature name, plain-English description, source
    name (clickable link to the canonical source page), and importance
    (a colour-coded horizontal bar with the scalar beside it). Rows
    are sorted by importance descending; only features the
    walk-forward stability filter kept are shown.
    """
    if not features:
        return ""
    kept = [f for f in features if f.get("selected")]
    if not kept:
        return ("<div class='empty' style='margin-top:12px;'>"
                "No features survived the stability filter on the "
                "last retrain — the model is in degenerate state.</div>")
    kept.sort(key=lambda f: f.get("mean_importance") or 0.0, reverse=True)
    cadence = _detect_bot_cadence(features)
    max_imp = max(
        (abs(f.get("mean_importance") or 0.0) for f in kept),
        default=1.0,
    ) or 1.0

    rows: List[str] = []
    for f in kept:
        name = f.get("feature") or ""
        imp = float(f.get("mean_importance") or 0.0)
        md = feature_metadata(name, cadence=cadence)
        link_url = md.get("link") or ""
        bar_pct = (abs(imp) / max_imp) * 100.0
        readable = _readable_feature_name(name)
        if link_url:
            src_cell = (
                f"<a href='{html.escape(link_url)}' target='_blank' "
                f"rel='noopener noreferrer' class='ft-src'>"
                f"<span class='ft-dot' style='background:"
                f"{html.escape(md['color'])};'></span>"
                f"{html.escape(md['label'])} ↗</a>"
            )
        else:
            src_cell = (
                f"<span class='ft-src ft-src-nolink'>"
                f"<span class='ft-dot' style='background:"
                f"{html.escape(md['color'])};'></span>"
                f"{html.escape(md['label'])}</span>"
            )
        rows.append(
            f"<div class='ft-row' "
            f"title='{html.escape(name)} · imp {imp:.4f}'>"
            f"<div class='ft-name-cell' title='{html.escape(name)}'>"
            f"{html.escape(readable)}</div>"
            f"<div class='ft-desc'>{html.escape(md['description'])}"
            f"</div>"
            f"<div class='ft-src-cell'>{src_cell}</div>"
            f"<div class='ft-bar-cell'>"
            f"<div class='ft-bar-track'>"
            f"<div class='ft-bar-fill' style='width:{bar_pct:.1f}%;"
            f"background:{html.escape(md['color'])};'></div>"
            f"</div>"
            f"<div class='ft-bar-imp'>{imp:.4f}</div>"
            f"</div>"
            f"</div>"
        )

    parts = [
        f"<h3 class='subhead'>Feature importance "
        f"<span class='small gray'>({len(kept)} kept by the "
        f"stability filter)</span></h3>",
        "<div class='ft-layout'>",
        "<div class='ft-head'>"
        "<div>Feature</div>"
        "<div>Description</div>"
        "<div>Source</div>"
        "<div>Importance</div>"
        "</div>",
        "<div class='ft-body'>",
        *rows,
        "</div>",
        "</div>",
        _FEATURE_DETAIL_CSS_JS,
    ]
    return "".join(parts)


_FEATURE_DETAIL_CSS_JS = """
<style>
.ft-layout {
  border: 1px solid #21262d;
  border-radius: 6px;
  overflow: hidden;
  margin-top: 4px;
  background: #0d1117;
}
.ft-head {
  display: grid;
  grid-template-columns: 200px 1fr 170px 220px;
  gap: 16px;
  padding: 8px 12px;
  background: #0d1117;
  border-bottom: 1px solid #21262d;
  color: #8b949e;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  position: sticky;
  top: 0;
  z-index: 1;
}
.ft-body {
  max-height: 600px;
  overflow-y: auto;
}
.ft-row {
  display: grid;
  grid-template-columns: 200px 1fr 170px 220px;
  gap: 16px;
  padding: 10px 12px;
  border-bottom: 1px solid #161b22;
  align-items: center;
}
.ft-row:last-child { border-bottom: none; }
.ft-row:hover { background: #161b22; }
.ft-name-cell {
  font-size: 13px;
  color: #c9d1d9;
  font-weight: 500;
  line-height: 1.3;
  min-width: 0;
}
.ft-desc {
  color: #c9d1d9;
  font-size: 12px;
  line-height: 1.45;
}
.ft-src-cell { font-size: 12px; }
.ft-src {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #58a6ff;
  text-decoration: none;
}
.ft-src:hover { text-decoration: underline; }
.ft-src-nolink { color: #8b949e; }
.ft-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 2px;
  flex: 0 0 auto;
}
.ft-bar-cell {
  display: grid;
  grid-template-columns: 1fr 56px;
  gap: 8px;
  align-items: center;
  /* visual divider so the importance side reads as its own column */
  border-left: 1px solid #21262d;
  padding-left: 12px;
  margin-left: -8px;
}
.ft-head > div:last-child {
  border-left: 1px solid #21262d;
  padding-left: 12px;
  margin-left: -8px;
}
.ft-bar-track {
  position: relative;
  height: 10px;
  background: #161b22;
  border-radius: 2px;
  overflow: hidden;
}
.ft-bar-fill {
  position: absolute;
  top: 0; left: 0; bottom: 0;
  border-radius: 2px;
}
.ft-bar-imp {
  font-size: 11px;
  color: #8b949e;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
@media (max-width: 900px) {
  .ft-head, .ft-row {
    grid-template-columns: 160px 1fr 140px 160px;
    gap: 10px;
  }
}
@media (max-width: 720px) {
  .ft-head { display: none; }
  .ft-row {
    grid-template-columns: 1fr;
    gap: 6px;
  }
  .ft-bar-cell {
    border-left: none;
    padding-left: 0;
    margin-left: 0;
  }
}
</style>
"""


def _render_model_view_toggle(out: List[str], bot_key: str,
                                 current_view: str,
                                 current_bot: str) -> None:
    """Pre-game / In-game pill toggle for sport bots. Selecting a tab
    swaps the ``?model_view=`` query param via a full-page nav so the
    URL stays shareable.
    """
    bot_qs = (f"&bot={html.escape(current_bot)}" if current_bot else "")
    options = [
        ("pregame", "Pre-game"),
        ("ingame", "In-game"),
    ]
    out.append("<div class='model-view-toggle'>")
    for key, label in options:
        active = " model-view-active" if key == current_view else ""
        href = f"?tab=models{bot_qs}&model_view={key}"
        out.append(
            f"<a class='model-view-pill{active}' "
            f"href='{html.escape(href)}'>{html.escape(label)}</a>"
        )
    out.append("</div>")


def _ingame_proxy_metrics(bot: dict,
                            threshold: float = 0.15) -> Optional[dict]:
    """Proxy classifier metrics for the in-game model's divergence
    feature, the only part of the model that can be replayed from
    the data we currently log.

    For each closed bet:
        prediction = 1 if divergence_at_entry > ``threshold`` else 0
                     (heuristic: 'this bet should win because the
                      market was overreacting at entry')
        actual     = 1 if realized_pnl > 0 else 0

    Returns ``{accuracy, precision, recall, f1, roc_auc, n}`` or
    ``None`` when there aren't enough closed bets to score.

    Live-state features (NBA lead/time, tennis live_prob_*) can't
    be replayed without historical state snapshots and are not
    measured here. The Models view labels this clearly.
    """
    rows = _ingame_backtest_rows(bot)
    if not rows or len(rows) < 5:
        return None
    tp = fp = tn = fn = 0
    pos_scores: List[float] = []
    neg_scores: List[float] = []
    for r in rows:
        pred = 1 if r["divergence"] > threshold else 0
        actual = 1 if r["pnl_cents"] > 0 else 0
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 0:
            tn += 1
        else:
            fn += 1
        if actual == 1:
            pos_scores.append(r["divergence"])
        else:
            neg_scores.append(r["divergence"])
    n = len(rows)
    accuracy = (tp + tn) / n
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = ((2.0 * precision * recall) / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    # Wilcoxon-Mann-Whitney AUC: P(random positive > random
    # negative) using divergence as the score.
    pairs = 0
    correct = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            pairs += 1
            if ps > ns:
                correct += 1.0
            elif ps == ns:
                correct += 0.5
    auc = (correct / pairs) if pairs > 0 else None
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "n": n,
        "threshold": threshold,
    }


def _render_ingame_model_view(out: List[str], bot: dict,
                                 active_bets: List[dict],
                                 closed_positions: List[dict] | None = None,
                                 ) -> None:
    """In-game model view for a sport bot's Models tab.

    Shows three sections:
      1. What the in-game model does for this sport (description).
      2. Currently-open positions with their live in-game prediction.
      3. The features the model uses, with sources called out.

    The pre-game model's content (Accuracy / F1 / ROC / etc.) is on
    the Pre-game tab — completely separate.
    """
    bot_key = bot.get("key", "")
    name = bot.get("name", bot_key)
    SPORT_DESC = {
        "nba": (
            "Live game state pulled from ESPN's public scoreboard "
            "every 30 seconds. Win probability derived from the "
            "canonical basketball logistic on lead / √(seconds "
            "remaining), then overlaid with cross-sport market "
            "features (velocity, volatility, divergence from "
            "pre-game prior). High confidence after Q1; low while "
            "the live state is still volatile."
        ),
        "tennis": (
            "Trusts the bot's own watchlist.json live_prob fields "
            "(computed per-point from the live data feed) as the "
            "authoritative live signal. Layers market overreaction "
            "detection on top — when the market has moved sharply "
            "away from a stable live estimate, the model says "
            "HOLD. Confidence ramps from set 1 through set 3+."
        ),
        "table-tennis": (
            "Same architecture as tennis — the bot's per-point "
            "live model is authoritative. Confidence ladder is "
            "tighter since table-tennis sets settle quickly."
        ),
        "darts": (
            "Same architecture as tennis. Confidence bumps higher "
            "after a single set because darts settles faster within "
            "a set than tennis does."
        ),
    }
    desc = SPORT_DESC.get(bot_key, "")

    # ── Headline metric cards (proxy classifier on the divergence
    # feature; the live-state portion isn't backtestable yet). Same
    # 6-card layout as the pre-game view so the user can scan
    # accuracy / F1 / precision / recall / ROC AUC / features side
    # by side.
    proxy = _ingame_proxy_metrics(bot)

    def _pct(v: object, decimals: int = 0) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v)*100:.{decimals}f}%"
        except (TypeError, ValueError):
            return "—"

    # Count features the heuristic actually uses today (the "Live"
    # rows in the SPORT_FEATURES table below).
    feat_table = {
        # Counts the green "Live" rows in SPORT_FEATURES below.
        "nba": 17,           # score, time, velocity, vol, divergence,
                              # espn_proj, injuries, foul trouble,
                              # box-score gaps, home/away, news,
                              # vegas, recent_form, cdn_pace,
                              # cdn_ft_rate, cdn_fouls, cdn_starter_pm
        "tennis": 9,         # live_prob, score, injury_flag, pre_game,
                              # divergence (now real via snapshotter),
                              # bot_confidence/volatility, bot_rec,
                              # comeback_prob, espn_news
        "table-tennis": 7,   # tennis features minus comeback + espn_news
        "darts": 7,          # same as table-tennis
    }
    live_feature_count = feat_table.get(bot_key, 0)

    out.append("<div class='row compact'>")
    if proxy is None:
        out.append(
            "<div class='card'><div class='label'>Sample size</div>"
            "<div class='value gray'>too small</div></div>"
        )
        for _ in range(5):
            out.append(
                "<div class='card'><div class='label'>—</div>"
                "<div class='value gray'>—</div></div>"
            )
    else:
        n_bets_title = (f"Proxy backtest on {proxy['n']} closed bets "
                         f"(divergence > {proxy['threshold']*100:.0f}% "
                         f"→ predict win).")
        for label, value, kind in [
            ("Accuracy", proxy["accuracy"], "pct"),
            ("F1", proxy["f1"], "pct"),
            ("Precision", proxy["precision"], "pct"),
            ("Recall", proxy["recall"], "pct"),
            ("ROC AUC", proxy["roc_auc"], "pct"),
            ("Features", live_feature_count, "count"),
        ]:
            if kind == "count":
                shown = str(value)
                title_attr = ("Number of features the in-game heuristic "
                               "currently consumes (the Live rows in the "
                               "Features in play table below).")
            else:
                shown = _pct(value, 0) if value is not None else "—"
                title_attr = n_bets_title
            out.append(
                f"<div class='card'><div class='label' "
                f"title='{html.escape(title_attr)}'>"
                f"{html.escape(label)}</div>"
                f"<div class='value'>{html.escape(shown)}</div></div>"
            )
    out.append("</div>")
    out.append(
        "<p class='small gray' style='margin:-6px 0 14px 0;'>"
        "Metrics are computed on a proxy classifier — "
        "<strong>divergence-at-entry &gt; "
        f"{(proxy['threshold']*100 if proxy else 15):.0f}%</strong> as "
        "the predictor of 'this bet will win'. The live-state "
        "features (NBA lead/time, tennis live_prob) need historical "
        "snapshots we don't yet log; those will start showing up "
        "in these numbers once snapshotting lands.</p>"
    )

    out.append(
        "<h3 class='subhead'>How the in-game model works "
        "<span class='small gray'>(heuristic baseline — see "
        "in_game/README.md for the training-pipeline roadmap)"
        "</span></h3>"
    )
    if desc:
        out.append(f"<p class='small' style='color:#c9d1d9;"
                    f"margin:0 0 14px 0;'>{html.escape(desc)}</p>")

    # ── Section 2: current live predictions ─────────────────────────
    live_rows: List[dict] = []
    for ab in (active_bets or []):
        ig = ab.get("_in_game") or {}
        if not ig:
            continue
        live_rows.append({"bet": ab, "ig": ig})
    out.append(
        f"<h3 class='subhead'>Currently-open positions "
        f"<span class='small gray'>({len(live_rows)} with a live "
        f"prediction)</span></h3>"
    )
    if not live_rows:
        out.append(
            "<div class='empty'>No open positions with a confident "
            "live prediction right now. The in-game model only "
            "speaks for sport positions whose match is past the "
            "start gate; pre-tip or pre-match positions show up "
            "here once the game is well underway.</div>"
        )
    else:
        out.append(
            "<table><thead><tr>"
            "<th>Ticker</th><th>Side</th>"
            "<th class='num'>Entry</th>"
            "<th class='num'>Live prob</th>"
            "<th class='num'>Confidence</th>"
            "<th>Action</th><th>Reason</th>"
            "</tr></thead><tbody>"
        )
        ACTION_LABEL = {
            "exit_now": ("EXIT", "red"),
            "let_run": ("RUN", "green"),
            "hold": ("HOLD", "yellow"),
            "neutral": ("—", "gray"),
        }
        for r in live_rows:
            b = r["bet"]
            ig = r["ig"]
            ticker = html.escape(b.get("ticker") or
                                   b.get("match_id") or "—")
            side = html.escape((b.get("side") or "").upper())
            entry_c = b.get("entry_price_cents")
            entry_str = (f"{int(entry_c)}¢" if entry_c is not None
                          else "—")
            lp = ig.get("live_prob_yes") or 0.0
            conf = ig.get("confidence") or 0.0
            action = (ig.get("action") or "neutral").lower()
            label, cls = ACTION_LABEL.get(action, ("—", "gray"))
            reason = html.escape(ig.get("reason") or "")
            out.append(
                f"<tr><td class='mono'>{ticker}</td>"
                f"<td>{side}</td>"
                f"<td class='num'>{entry_str}</td>"
                f"<td class='num'>{lp*100:.0f}%</td>"
                f"<td class='num'>{conf*100:.0f}%</td>"
                f"<td><span class='in-game-pill ig-{cls}'>"
                f"{label}</span></td>"
                f"<td class='small gray'>{reason}</td></tr>"
            )
        out.append("</tbody></table>")

    # ── Section 3: features the model uses ──────────────────────────
    SPORT_FEATURES = {
        "nba": [
            ("Score differential", "Live", "ESPN scoreboard"),
            ("Time remaining", "Live", "ESPN scoreboard (period + clock)"),
            ("Market velocity", "Live", "market_views history (cents/min)"),
            ("Market volatility", "Live", "market_views history (stdev)"),
            ("Divergence vs pre-game", "Live",
             "market_views + position.model_yes_prob_at_entry"),
            ("ESPN win projection", "Live",
             "ESPN /summary?event= predictor block"),
            ("Critical injury counts (per team)", "Live",
             "ESPN /summary injuries block"),
            ("Foul trouble (players w/ ≥4 PF)", "Live",
             "ESPN /summary boxscore.players[*]"),
            ("Live FG% / FT% / 3P% / AST / REB / TO gaps", "Live",
             "ESPN /summary boxscore.teams[*].statistics"),
            ("Home-court advantage (time-decayed)", "Live",
             "ESPN scoreboard homeAway flag"),
            ("Recent injury news mentions (12h)", "Live",
             "ESPN /news?limit=40 keyword scan (in_game/news_signals)"),
            ("Vegas consensus win prob (moneyline)", "Live",
             "ESPN /summary pickcenter.{home,away}TeamOdds.moneyLine"),
            ("Recent form (W-L in last 5)", "Live",
             "ESPN /summary lastFiveGames.events[].gameResult"),
            ("Live pace (possessions / 48 min)", "Live",
             "NBA.com CDN /boxscore.homeTeam.statistics.pace"),
            ("Free-throw rate gap", "Live",
             "NBA.com CDN boxscore: FTA / FGA per team"),
            ("Fouled-out + foul-trouble counts (CDN)", "Live",
             "NBA.com CDN boxscore.players foulsPersonal"),
            ("Starter vs bench +/- splits", "Live",
             "NBA.com CDN boxscore.players starter + plusMinusPoints"),
            ("Shooting % vs xFG", "TODO",
             "shot-chart endpoint + offline xFG training set "
             "(richer than the raw FG% we already have)"),
            ("Lineup combinations on floor", "TODO",
             "NBA.com CDN playbyplay sub events → reconstruct "
             "5-man unit. Doable next session."),
            ("Per-player minutes restriction", "TODO",
             "team injury report + minutes feed"),
        ],
        "tennis": [
            ("Live win probability", "Live",
             "watchlist.json live_prob_a / live_prob_b"),
            ("Current score state", "Live",
             "watchlist.json current_score"),
            ("Injury news flag", "Live",
             "watchlist.json injury_news_flag"),
            ("Pre-game prior", "Live",
             "watchlist.json pre_match_prob_a"),
            ("Market divergence", "Live",
             "watchlist.json yes_ask_cents_a/_b"),
            ("Bot confidence + volatility score", "Live",
             "watchlist.json confidence_score + volatility_score"),
            ("Bot recommended action (signal only)", "Live",
             "watchlist.json recommended_action"),
            ("Comeback probability from set state", "Live",
             "Klaassen-Magnus style heuristic on parsed set wins"),
            ("Recent ESPN injury news mentions (12h)", "Live",
             "ESPN tennis /news feed keyword scan"),
            ("Breakpoint conversion %", "TODO",
             "live point-by-point feed (the bot has it but doesn't "
             "yet expose it on the watchlist row)"),
            ("Serve velocity decline", "TODO",
             "radar gun data; only via broadcast-paired feeds"),
        ],
        "table-tennis": [
            ("Live win probability", "Live",
             "watchlist.json live_prob_a / live_prob_b"),
            ("Current score state", "Live",
             "watchlist.json current_score"),
            ("Reaction-time decay", "TODO",
             "would require frame-level video analysis"),
            ("Spin/style matchup", "TODO",
             "historical match outcomes by player style"),
        ],
        "darts": [
            ("Live win probability", "Live",
             "watchlist.json live_prob_a / live_prob_b"),
            ("Set state", "Live", "watchlist.json current_score"),
            ("Checkout %", "TODO",
             "live leg-by-leg feed (bot has access; not yet exposed "
             "on watchlist row)"),
            ("Leg-streak persistence", "TODO",
             "per-tick watchlist with running leg stats"),
        ],
    }
    feats = SPORT_FEATURES.get(bot_key, [])
    if feats:
        out.append(
            "<h3 class='subhead'>Features in play "
            "<span class='small gray'>(Live = currently used; "
            "TODO = would need an additional feed or trained "
            "model — see in_game/README.md)</span></h3>"
        )
        out.append(
            "<table><thead><tr>"
            "<th>Feature</th><th>Status</th><th>Source</th>"
            "</tr></thead><tbody>"
        )
        for label, status, source in feats:
            status_cls = "green" if status == "Live" else "gray"
            out.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td class='{status_cls}'>{html.escape(status)}</td>"
                f"<td class='small gray'>{html.escape(source)}</td>"
                f"</tr>"
            )
        out.append("</tbody></table>")

    # ── Coefficients (what weights the heuristic uses today) ────────
    _render_ingame_coefficients(out, bot_key)

    # ── Historical backtest of the testable features ────────────────
    _render_ingame_backtest(out, bot)

    # ── Recent predictions log + outcome reconciliation ─────────────
    _render_ingame_predictions_log(out, bot, closed_positions or [])


# ──────────────────────────────────────────────────────────────────
# In-game model coefficients — the weights and constants baked into
# the heuristic implementations under ``in_game/``. Listed here so
# the user can see them at a glance and (later) compare them to
# what a trained model would set.
#
# Each entry: (label, current_weight, role description, status)
# Status values:
#   ``tuned``    — hand-set in code; defensible heuristic
#   ``learned``  — would come from a trained model; currently
#                  hand-set as a placeholder
# ──────────────────────────────────────────────────────────────────
_INGAME_COEFFICIENTS: Dict[str, List[tuple]] = {
    "nba": [
        ("Lead / √(sec remaining) coefficient", 0.045,
         "Logistic weight in win_prob = σ(c · lead / √(time)). "
         "Canonical basketball value (Brian Burke).", "tuned"),
        ("Confidence past Q1, low volatility", 0.70,
         "Output confidence when game is past Q1 and live "
         "market vol < 1.5", "tuned"),
        ("Confidence past Q1, high volatility", 0.45,
         "Output confidence when past Q1 but vol ≥ 1.5", "tuned"),
        ("Confidence pre-Q1", 0.25,
         "Output confidence within the first 12 game-minutes — "
         "still too noisy to trust the lead-based model",
         "tuned"),
        ("Market velocity cap", 1.00,
         "Max absolute cents/min movement the velocity feature "
         "reports (clipped to keep linear combos sane)",
         "tuned"),
        ("Divergence reversion damp", 0.50,
         "Fraction of (current_market − pre_game) the in-game "
         "model pulls back toward the pre-game prior",
         "tuned"),
        ("Volatility damp ceiling", 4.00,
         "Volatility above this completely cancels the "
         "reversion pull (market is in too much flux to trust "
         "pre-game prior)",
         "tuned"),
        ("ESPN win-projection weight", 0.25,
         "Weight assigned to ESPN's own predicted win % when "
         "blending it as a third opinion alongside state_prob "
         "and reversion-adjusted prior",
         "tuned"),
        ("Injury / foul-trouble nudge (per player)", 0.015,
         "Each critical injury OR foul-troubled key player on "
         "our team drops live_prob by this; same magnitude on "
         "the opp team raises it",
         "tuned"),
        ("Box-score gap nudge (per pp)", 0.001,
         "Per-pp coefficient on live FG% / FT% / 3P% / AST gap "
         "vs opponent. Turnover gap weighted 3×, rebound gap "
         "weighted 1.5×",
         "tuned"),
        ("Home-court advantage (peak)", 0.025,
         "Maximum nudge applied when our team is at home "
         "(decays from full strength at tip to 30% in the "
         "final minute of regulation)",
         "tuned"),
        ("Injury news nudge (per article, 12h)", 0.010,
         "Each ESPN news article mentioning our team in an "
         "injury context within the last 12 hours nudges "
         "live_prob down by this; same on the opp team raises it",
         "tuned"),
        ("Vegas (pickcenter) blend weight", 0.15,
         "Sportsbook consensus moneyline → implied win prob, "
         "blended in at this weight as a fourth opinion. "
         "Smaller than ESPN's 25% because the pre-game model "
         "already implicitly factors in line movement",
         "tuned"),
        ("Recent-form nudge (per 5 games)", 0.015,
         "Per (our_wins − opp_wins) / 5 differential nudge from "
         "lastFiveGames. Captures momentum carry-over",
         "tuned"),
        ("CDN: FT-rate gap nudge (per pp)", 0.003,
         "NBA.com CDN: per-pp coefficient on (our FT/FGA − opp "
         "FT/FGA). Getting to the line is a stable team edge.",
         "tuned"),
        ("CDN: fouled-out player nudge", 0.030,
         "Each one of our players with 6+ PF drops live_prob by "
         "this. Sourced from NBA.com CDN boxscore.players[*]",
         "tuned"),
        ("CDN: foul-trouble (4-5 PF) nudge", 0.010,
         "Smaller nudge than fouled-out; reflects coach managing "
         "minutes on a player nearing disqualification",
         "tuned"),
        ("CDN: starter +/- amplification", 0.001,
         "Per-point coefficient on average starter plus-minus. "
         "Negative starter +/- with positive bench +/- means "
         "rotation is working against us",
         "tuned"),
        ("CDN: pace × lead amplification", 0.0005,
         "Per (pace − 100 league avg) × sign(lead). High pace "
         "amplifies existing leads (fewer variance possessions "
         "left); low pace gives trailing team more comeback room",
         "tuned"),
        ("EXIT_NOW threshold", 0.30,
         "Recommend exit when our_side_prob falls below this "
         "with confidence ≥ 0.5",
         "tuned"),
        ("LET_RUN threshold", 0.10,
         "Recommend let-run when our_side_prob exceeds entry "
         "by this much",
         "tuned"),
    ],
    "tennis": [
        ("Confidence in set 1", 0.35, "Low — first set is noisy",
         "tuned"),
        ("Confidence in set 2", 0.55, "Medium — pattern emerging",
         "tuned"),
        ("Confidence in set 3+", 0.75, "High — match well-determined",
         "tuned"),
        ("Injury-flag confidence haircut", 0.25,
         "Subtract this from confidence when injury_news_flag is set",
         "tuned"),
        ("Divergence floor (overreaction)", 0.15,
         "Minimum |market − pre_game| before the reversion pull "
         "kicks in",
         "tuned"),
        ("Reversion pull strength", 0.30,
         "Fraction of (market − pre_game) the in-game estimate "
         "pulls back",
         "tuned"),
        ("Divergence threshold for HOLD", 0.20,
         "|market − pre_game| above this with low live volatility "
         "→ market overreaction → HOLD",
         "tuned"),
        ("Bot confidence cap (over our model)", 0.10,
         "In-game confidence can't exceed the tennis bot's own "
         "confidence_score by more than this; caps our claims "
         "when the bot itself is unsure",
         "tuned"),
        ("Bot-volatility haircut threshold", 0.50,
         "Tennis bot's volatility_score above this triggers a "
         "confidence haircut (calibrated to bot's own threshold)",
         "tuned"),
        ("Bot-volatility haircut weight", 0.40,
         "Multiplier on (bot_volatility − threshold) when "
         "applying the haircut; capped at 0.30 max",
         "tuned"),
        ("Comeback-prob exit threshold", 0.10,
         "Recommend EXIT when historical comeback probability "
         "from current set state falls below this (match "
         "essentially decided)",
         "tuned"),
        ("Injury-news confidence haircut (per article)", 0.10,
         "Each recent ESPN tennis article matching a player name + "
         "injury keyword drops in-game confidence by this; "
         "max 3 articles compound",
         "tuned"),
        ("EXIT_NOW threshold", 0.30,
         "Recommend exit when our_bet_prob falls below this with "
         "confidence ≥ 0.5",
         "tuned"),
        ("LET_RUN threshold", 0.10,
         "Recommend let-run when our_bet_prob exceeds entry by "
         "this much",
         "tuned"),
    ],
    "table-tennis": [
        ("Same coefficients as tennis", 0.0,
         "Table tennis uses the tennis heuristic with no overrides",
         "tuned"),
    ],
    "darts": [
        ("Confidence bump after 1 set", 0.10,
         "Add this to the tennis-derived confidence once one "
         "set has completed (darts sets settle faster)",
         "tuned"),
        ("Inherits all tennis coefficients", 0.0,
         "Confidence ladder + reversion logic from tennis.py",
         "tuned"),
    ],
}


def _render_ingame_coefficients(out: List[str], bot_key: str) -> None:
    """Coefficient table + bar chart of relative weights for the
    sport bot's in-game heuristic. Mirrors the pre-game model's
    top-features visual idiom so the two views read consistently.
    """
    coefs = _INGAME_COEFFICIENTS.get(bot_key, [])
    out.append(
        "<h3 class='subhead'>Coefficients "
        "<span class='small gray'>(heuristic — hand-tuned today, "
        "would be learned in a trained version)</span></h3>"
    )
    if not coefs:
        out.append(
            "<div class='empty'>No coefficient table registered "
            "for this sport.</div>"
        )
        return
    # Table view — full detail.
    out.append(
        "<table><thead><tr>"
        "<th>Coefficient</th>"
        "<th class='num'>Value</th>"
        "<th>Role</th>"
        "<th>Status</th>"
        "</tr></thead><tbody>"
    )
    for label, value, role, status in coefs:
        status_cls = "green" if status == "tuned" else "yellow"
        status_label = ("Hand-tuned" if status == "tuned"
                          else "TODO: learn from data")
        v_str = (f"{value:.3f}" if isinstance(value, float)
                  else str(value))
        out.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td class='num'>{html.escape(v_str)}</td>"
            f"<td class='small gray'>{html.escape(role)}</td>"
            f"<td class='{status_cls}'>{html.escape(status_label)}</td>"
            f"</tr>"
        )
    out.append("</tbody></table>")


def _render_ingame_backtest(out: List[str], bot: dict) -> None:
    """Historical backtest of the in-game model's most testable
    feature: pre-game/market divergence at entry. Closed positions
    are bucketed by |pre_game_prob − market_implied_prob| at entry,
    and we report the realized P&L per bucket. A real 'market
    overreaction' signal shows up as high-divergence buckets that
    correlate with positive realized P&L.

    Coefficients that depend on live game state (lead/time, set
    score, live volatility) can't be replayed from the data we
    currently store. They're marked as not-yet-backtestable above
    in the coefficient table.
    """
    bot_key = bot.get("key", "")
    out.append(
        "<h3 class='subhead'>Historical backtest "
        "<span class='small gray'>(divergence-at-entry feature, "
        "closed bets only — the live-state features can't be "
        "replayed without snapshot data we don't yet log)"
        "</span></h3>"
    )
    rows = _ingame_backtest_rows(bot)
    if not rows:
        out.append(
            "<div class='empty'>No closed bets with the data we "
            "need (pre-game model probability + entry market "
            "price) on file yet.</div>"
        )
        return
    # Buckets in 5pp width up to 0.30, then a tail bucket for huge
    # divergences. Each row: bets, mean divergence, win rate,
    # mean realized cents per contract, total P&L.
    buckets = [
        ("0–5%",    0.00, 0.05),
        ("5–10%",   0.05, 0.10),
        ("10–15%",  0.10, 0.15),
        ("15–20%",  0.15, 0.20),
        ("20–30%",  0.20, 0.30),
        ("30%+",    0.30, 1.01),
    ]
    out.append(
        "<table><thead><tr>"
        "<th>Divergence @ entry</th>"
        "<th class='num'>Bets</th>"
        "<th class='num'>Mean div.</th>"
        "<th class='num'>Win %</th>"
        "<th class='num'>¢/contract</th>"
        "<th class='num'>Total P&amp;L</th>"
        "</tr></thead><tbody>"
    )
    any_data = False
    for label, lo, hi in buckets:
        bucket = [r for r in rows if lo <= r["divergence"] < hi]
        n = len(bucket)
        if n == 0:
            out.append(
                f"<tr><td>{html.escape(label)}</td>"
                f"<td class='num gray'>0</td>"
                f"<td class='num gray'>—</td>"
                f"<td class='num gray'>—</td>"
                f"<td class='num gray'>—</td>"
                f"<td class='num gray'>—</td></tr>"
            )
            continue
        any_data = True
        mean_div = sum(r["divergence"] for r in bucket) / n * 100.0
        wins = sum(1 for r in bucket if r["pnl_cents"] > 0)
        win_pct = wins / n
        cents_per = sum(r["pnl_per_contract"] for r in bucket) / n
        total = sum(r["pnl_cents"] for r in bucket)
        win_cls = ("green" if win_pct > 0.5
                    else ("red" if win_pct < 0.5 else "gray"))
        cents_cls = ("green" if cents_per > 0
                      else ("red" if cents_per < 0 else "gray"))
        total_cls = ("green" if total > 0
                      else ("red" if total < 0 else "gray"))
        cents_sign = "+" if cents_per > 0 else ("−" if cents_per < 0 else "")
        total_sign = "+" if total > 0 else ("−" if total < 0 else "")
        out.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td class='num'>{n}</td>"
            f"<td class='num'>{mean_div:.1f}%</td>"
            f"<td class='num {win_cls}'>{win_pct*100:.0f}%</td>"
            f"<td class='num {cents_cls}'>"
            f"{cents_sign}{abs(cents_per):.2f}¢</td>"
            f"<td class='num {total_cls}'>"
            f"{total_sign}${abs(total)/100:.2f}</td></tr>"
        )
    out.append("</tbody></table>")
    if not any_data:
        out.append(
            "<p class='small gray' style='margin-top:8px;'>"
            "No closed bets fell into any divergence bucket yet — "
            "the backtest populates once the bot closes positions "
            "with both a pre-game model probability and an entry "
            "market price on record.</p>"
        )


def _ingame_backtest_rows(bot: dict) -> List[dict]:
    """Pull closed bets for the bot and compute the divergence /
    realized-P&L pairs the backtest needs. Empty when the bot has
    no closed bets or the schema is missing required columns.
    """
    bot_key = bot.get("key", "")
    rows: List[dict] = []
    if bot.get("dashboard_type") == "sport":
        # Tennis-shape: closed_positions in sim_state.json. The rollup
        # shape exposes entry_price_cents (market view) + the bot's
        # pre-game model probability under model_yes_prob_at_entry.
        # Divergence = |model − market| on the side bet (tennis rows
        # are always YES-side in the rollup output).
        from . import tennis as _tennis
        for p in _tennis.closed_positions_for_rollup(
                bot.get("sim_state_path"), limit=500):
            entry_cents = p.get("entry_price_cents")
            entry_model = p.get("model_yes_prob_at_entry")
            pnl = p.get("realized_pnl_cents")
            contracts = p.get("contracts") or 1
            if (entry_cents is None or entry_model is None
                    or pnl is None):
                continue
            try:
                market_prob = float(entry_cents) / 100.0
                div = abs(float(entry_model) - market_prob)
                pnl_int = int(pnl)
            except (TypeError, ValueError):
                continue
            rows.append({
                "divergence": div,
                "pnl_cents": pnl_int,
                "pnl_per_contract": pnl_int / max(1, int(contracts)),
            })
        return rows
    # Standard sim.db (NBA).
    db_path = bot.get("db_path") or ""
    if not db_path or not Path(db_path).exists():
        return []
    try:
        with closing(_conn(db_path)) as c:
            cols = {r["name"] for r in
                    c.execute("PRAGMA table_info(positions)").fetchall()}
            if "model_yes_prob_at_entry" not in cols:
                return []
            db_rows = c.execute(
                "SELECT side, entry_price_cents, contracts, "
                "       realized_pnl_cents, model_yes_prob_at_entry "
                "FROM positions WHERE status = 'closed' "
                "  AND realized_pnl_cents IS NOT NULL "
                "  AND entry_price_cents IS NOT NULL "
                "  AND model_yes_prob_at_entry IS NOT NULL"
            ).fetchall()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []
    for r in db_rows:
        side = (r["side"] or "").upper()
        try:
            entry_c = int(r["entry_price_cents"])
            pnl_int = int(r["realized_pnl_cents"])
            contracts = int(r["contracts"] or 1)
            model_yes = float(r["model_yes_prob_at_entry"])
        except (TypeError, ValueError):
            continue
        market_yes_for_side = (entry_c / 100.0 if side == "YES"
                                 else (100 - entry_c) / 100.0)
        model_yes_for_side = (model_yes if side == "YES"
                                else 1.0 - model_yes)
        div = abs(model_yes_for_side - market_yes_for_side)
        rows.append({
            "divergence": div,
            "pnl_cents": pnl_int,
            "pnl_per_contract": pnl_int / max(1, contracts),
        })
    return rows


def _render_ingame_predictions_log(out: List[str], bot: dict,
                                       closed_positions: List[dict]
                                       ) -> None:
    """Recent predictions panel for the In-game view.

    Reads the tail of ``data/in_game_predictions.jsonl`` filtered to
    this bot, joins each entry against the closed-bet ledger
    (matching on ticker), and surfaces the outcome (WON / LOST /
    OPEN) so the user can see whether the model's calls held up.
    """
    bot_key = bot.get("key", "")
    out.append(
        "<h3 class='subhead'>Recent predictions "
        "<span class='small gray'>(every confident action "
        "transition the in-game model has logged for this bot, "
        "newest first)</span></h3>"
    )
    try:
        from .in_game import logger as _ig_logger
        entries = _ig_logger.read_for_bot(bot_key, limit=40)
    except Exception:  # noqa: BLE001
        entries = []
    if not entries:
        out.append(
            "<div class='empty'>No predictions logged yet. The log "
            "populates once the in-game model issues a confident "
            "EXIT / RUN / HOLD action and that action changes from "
            "the previous one for the same ticker.</div>"
        )
        return
    # Index closed positions by ticker for outcome lookup. We use
    # the most-recent close per ticker so re-traded contracts use
    # the latest realized P&L.
    by_ticker: Dict[str, dict] = {}
    for c in (closed_positions or []):
        t = c.get("ticker") or ""
        if not t:
            continue
        # Newer close wins (later exited_at).
        prev = by_ticker.get(t)
        if prev is None:
            by_ticker[t] = c
            continue
        if (c.get("exited_at") or "") > (prev.get("exited_at") or ""):
            by_ticker[t] = c
    out.append(
        "<table><thead><tr>"
        "<th>When</th><th>Ticker</th><th>Side</th>"
        "<th>Action</th>"
        "<th class='num'>Live prob</th>"
        "<th class='num'>Confidence</th>"
        "<th>Outcome</th><th>Reason</th>"
        "</tr></thead><tbody>"
    )
    ACTION_LABEL = {
        "exit_now": ("EXIT", "ig-red"),
        "let_run": ("RUN", "ig-green"),
        "hold": ("HOLD", "ig-yellow"),
    }
    for e in entries:
        ts = (e.get("ts") or "")[:19].replace("T", " ")
        ticker = e.get("ticker") or ""
        side = (e.get("side") or "—")
        action = (e.get("action") or "").lower()
        a_label, a_cls = ACTION_LABEL.get(action, ("—", "ig-gray"))
        lp = e.get("live_prob_yes") or 0.0
        conf = e.get("confidence") or 0.0
        reason = e.get("reason") or ""
        match = by_ticker.get(ticker)
        if match:
            pnl = int(match.get("realized_pnl_cents") or 0)
            if pnl > 0:
                outcome_label = "WON"
                outcome_cls = "green"
            elif pnl < 0:
                outcome_label = "LOST"
                outcome_cls = "red"
            else:
                outcome_label = "FLAT"
                outcome_cls = "gray"
            outcome_html = (
                f"<span class='{outcome_cls}' "
                f"title='Realized P&amp;L: "
                f"{'+' if pnl > 0 else ('−' if pnl < 0 else '')}"
                f"${abs(pnl)/100:.2f}'>{outcome_label}</span>"
            )
        else:
            outcome_html = "<span class='gray'>OPEN</span>"
        out.append(
            f"<tr><td class='small gray'>{html.escape(ts)}</td>"
            f"<td class='mono'>{html.escape(ticker)}</td>"
            f"<td>{html.escape(str(side))}</td>"
            f"<td><span class='in-game-pill {a_cls}'>"
            f"{a_label}</span></td>"
            f"<td class='num'>{lp*100:.0f}%</td>"
            f"<td class='num'>{conf*100:.0f}%</td>"
            f"<td>{outcome_html}</td>"
            f"<td class='small gray'>{html.escape(reason)}</td></tr>"
        )
    out.append("</tbody></table>")


def _render_models_panel(out: List[str], bot: dict, model: dict | None,
                          display: dict | None,
                          available_bots: List[dict],
                          current_bot: str,
                          model_view: str = "pregame",
                          bot_active_bets: List[dict] | None = None,
                          bot_closed_positions: List[dict] | None = None,
                          ) -> None:
    """Per-bot Models tab content. Standard sim.db bots get the full
    deep-dive (headline metrics card row, full feature list bar chart,
    calibration curve, predicted-vs-realized EV, hedge audit, etc.).
    Tennis dispatches into its own renderer.

    Sport bots (NBA, tennis, table-tennis, darts) get a Pre-game /
    In-game toggle at the top so the user can switch between the
    standard pre-game view and the in-game advisory model's view.
    The two are completely separate — toggling never mixes the
    populations.
    """
    bot_key = (bot or {}).get("key", "")
    is_sport_bot = bot_key in {"nba", "wnba", "tennis", "table-tennis", "darts"}
    # Every model page uses the same section-header layout so the
    # "Model" title and the body content sit at the same vertical
    # position regardless of bot. Sport bots fill the right-hand
    # slot with the real Pre-game / In-game toggle; non-sport bots
    # render an invisible toggle of identical dimensions so the
    # header row has the same height byte-for-byte. (A bare
    # min-height won't do it — the toggle's actual rendered height
    # depends on the page's font / line-height, which we can't
    # predict precisely from CSS alone.)
    out.append("<div class='section'>")
    out.append("<div class='section-header'><h2>Model</h2>")
    if is_sport_bot and bot:
        _render_model_view_toggle(out, bot_key, model_view, current_bot)
    else:
        out.append(
            "<div class='model-view-toggle' "
            "style='visibility:hidden;' aria-hidden='true'>"
            "<span class='model-view-pill'>Pre-game</span>"
            "<span class='model-view-pill'>In-game</span>"
            "</div>"
        )
    out.append("</div>")
    out.append("<div class='body'>")
    # Bot filter moved above the tab bar (per user request).
    if not bot:
        out.append("<div class='empty'>Bot not found.</div>")
        out.append("</div></div>")
        return
    if is_sport_bot and model_view == "ingame":
        _render_ingame_model_view(out, bot, bot_active_bets or [],
                                     bot_closed_positions or [])
        out.append("</div></div>")
        return
    # World Cup is advisory-only (no trading loop yet) — its Models tab
    # renders the offline bake-off report instead of the live-metrics
    # card the other sport bots get.
    if bot_key == "world-cup":
        from . import world_cup as _world_cup
        out.append(_world_cup.render_models_panel(bot))
        out.append("</div></div>")
        return
    # Tennis-shape bots (tennis / table-tennis / darts) don't have a
    # sim.db — they keep their model artifacts in metrics.json +
    # coefficients.json. Delegate to the tennis renderer; Phase 2b
    # will replace this with a unified section-by-section layout.
    if bot.get("dashboard_type") == "sport":
        from . import tennis as _tennis
        metrics = _tennis.load_metrics(bot.get("metrics_path"))
        coefficients = _tennis.load_coefficients(bot.get("coefficients_path"))
        sim_state = _tennis.load_sim_state(bot.get("sim_state_path"))
        out.append(_tennis._render_tennis_models_page(
            metrics, coefficients, sim_state,
            metrics_path=bot.get("metrics_path"),
        ))
        out.append("</div></div>")
        return
    # Billboard also has no sim.db — same JSON-source pattern as
    # tennis. The billboard renderer reproduces the SAME visual
    # sections this function produces for sim.db bots (headline
    # metrics cards → top features → ROC → calibration → empty-state
    # stubs for the closed-bet-driven sections), so the Models tab
    # is visually identical to retail-gas-prices'.
    if bot.get("dashboard_type") == "billboard":
        from . import billboard as _billboard
        _billboard.render_models_panel(out, bot)
        out.append("</div></div>")
        return
    db_path = bot.get("db_path") or ""
    if not db_path or not Path(db_path).exists():
        _render_bot_unavailable(out, bot.get("key", ""))
        out.append("</div></div>")
        return

    # Holdout predictions drive the confidence tier (rendered down
    # next to the ROC + Confusion + Calibration block, where the user
    # naturally looks for held-out context) and the chart data. Load
    # once here so the page only reads the file a single time.
    holdout_path = _find_training_artifact(
        db_path, "holdout_predictions.csv")
    pairs = _read_holdout_predictions(str(holdout_path))
    conf = _holdout_confidence(pairs)

    # Training artifacts — feature-importance drives the readable
    # features section; ``model`` (already loaded above) provides the
    # blended-model metrics for the fallback row in the models table.
    fi_path = _find_training_artifact(
        db_path, "feature_importance.csv")
    feats = _read_feature_importance(str(fi_path))

    # sim.db-shape trainers only store the final classifier's numbers
    # (there's no per_model breakdown), so we shape a single-row
    # fallback for the shared table using the same field names.
    fallback_rows: List[Tuple[str, Dict[str, Any]]] = []
    if model:
        fallback_rows.append(("Blended (final)", {
            "accuracy":  model.get("classifier_accuracy"),
            "f1":        model.get("training_f1"),
            "precision": model.get("training_precision"),
            "recall":    model.get("training_recall"),
            "roc_auc":   model.get("training_roc_auc"),
            "brier":     model.get("training_brier"),
        }))
    metrics_shim: Dict[str, Any] = {
        "rows_train": (model or {}).get("rows_train"),
        "rows_test":  (model or {}).get("rows_test"),
    }
    captured = (model or {}).get("captured_at") or ""
    last_trained = str(captured)[:10] if captured else "—"

    # 1) Table of models run.
    out.append(_render_models_run_table(
        metrics_shim,
        feature_count=(int((model or {}).get("feature_count"))
                        if (model or {}).get("feature_count") is not None
                        else (len(feats) if feats else None)),
        last_trained=last_trained,
        fallback_rows=fallback_rows,
    ))

    # 2) Features with definitions and bars.
    out.append(_render_feature_source_table(feats))

    out.append("</div></div>")
