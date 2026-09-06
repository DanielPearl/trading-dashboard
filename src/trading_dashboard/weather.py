"""Weather bot dashboard adapter.

The bot (Weather Forecast repo) prices Kalshi's daily weather
contracts off published forecasts — NWS ``probabilityOfPrecipitation``
for the rain markets, the NWS official daily high plus multi-model
ensemble spread for the temperature ladders. There is no trained
model, so this adapter mirrors the reality-leaks one: synthesise
standard watchlist rows from the JSON, return no metrics, and let the
shared renderers do the rest. Positions come from the standard sim.db
readers.

Weather-specific columns (watchlist_panel's is_weather_bot branch):
City, Question, Forecast, and a Source column carrying the published
forecast the probability came from.
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.weather")


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
    return {}


def load_coefficients(path: str | None) -> Dict[str, Any]:
    return {}


def load_sim_state(path: str | None) -> Dict[str, Any]:
    return {}


def is_available(metrics_path: str | None) -> bool:
    return True


def model_summary_for_card(metrics_path: str | None,
                            sim_state_path: str | None = None
                            ) -> Dict[str, Any]:
    """No trained model — the home card shows ledger stats only."""
    return {}


def summary_for_rollup(sim_state_path: str | None) -> Dict[str, Any]:
    return {
        "open_count": 0, "active_contracts": 0,
        "period_bets_made": 0, "period_net_pnl_cents": 0,
        "period_wins": 0, "period_losses": 0,
        "period_money_spent_cents": 0, "period_money_gained_cents": 0,
        "potential_gain_cents": 0, "active_money_spent_cents": 0,
        "total_bets": 0, "realized_pnl_cents": 0,
        "wins_lifetime": 0, "losses_lifetime": 0,
    }


def active_bets_for_rollup(sim_state_path: str | None,
                             watchlist_path: str | None = None
                             ) -> List[Dict[str, Any]]:
    return []


def closed_positions_for_rollup(sim_state_path: str | None,
                                  limit: int = 100,
                                  real_only: bool = False
                                  ) -> List[Dict[str, Any]]:
    return []


def build_standard_watchlist_rows(payload: Dict[str, Any]
                                    ) -> List[Dict[str, Any]]:
    """Translate weather rows into the shared watchlist schema.

    Weather-specific fields ride along under ``_``-prefixed keys:

        _city          settlement city
        _station       ASOS/CLI station the contract settles on
        _kind          rain | temp
        _forecast_f    NWS official daily high (temp rows)
        _sigma_f       within-model ensemble spread (temp rows)
        _daily_pop     blended NWS PoP (rain rows)
        _calibrated    station has fitted bias/dispersion constants
        _contract_day  the local calendar day being settled
    """
    raw = payload.get("rows") or []
    out: List[Dict[str, Any]] = []
    for r in raw:
        is_rain = r.get("series") == "KXRAIN"
        out.append({
            "ticker": r.get("ticker") or "",
            "title": r.get("title") or "",
            # Rain markets have no strike ladder — floor_strike is 0
            # ("precipitation strictly greater than 0 inches"), which
            # the shared Question renderer would format against the
            # temperature display config as a bare "0F". Blank the
            # strikes and give the column the real question instead.
            "direction": ('Rain \u2265 0.01\u2033' if is_rain
                          else (r.get("question") or "")),
            "_city": r.get("city") or "",
            "_station": r.get("station") or "",
            "_kind": "rain" if is_rain else "temp",
            "_forecast_f": r.get("forecast_max_f"),
            "_sigma_f": r.get("sigma_f"),
            "_daily_pop": r.get("daily_pop"),
            "_calibrated": bool(r.get("calibrated")),
            "_contract_day": r.get("contract_day"),
            "strike_low": None if is_rain else r.get("strike_low"),
            "strike_high": None if is_rain else r.get("strike_high"),
            "yes_ask_cents": r.get("yes_ask_cents"),
            "no_ask_cents": r.get("no_ask_cents"),
            "spread_cents": r.get("spread_cents"),
            "volume": r.get("volume"),
            "open_interest": r.get("open_interest"),
            "model_prob_yes": r.get("model_prob"),
            "raw_model_prob_yes": r.get("model_prob"),
            "bot_verdict": r.get("verdict") or "SKIP",
            "rejection_reason": r.get("rejection_reason") or "",
            "minutes_to_close": r.get("minutes_to_close"),
            "rules_primary": r.get("rules_primary"),
        })
    return out


def render_models_panel(out: List[str], bot: Dict[str, Any]) -> None:
    payload = load_watchlist(bot.get("watchlist_json_path"))
    rows = payload.get("rows") or []
    temp_on = payload.get("temp_enabled")
    rain_on = payload.get("rain_enabled")
    calibrated = payload.get("calibrated_stations") or []
    sums = payload.get("ladder_sums") or {}

    if payload.get("slate_guard_tripped"):
        out.append(
            "<p style='color:#d29922;border:1px solid #d29922;"
            "border-radius:6px;padding:10px 12px;'>⚠ <b>Slate-bias guard "
            "tripped this tick.</b> Signals came back one-signed across "
            "the whole slate, which is the signature of an estimator "
            "offset rather than many independent edges. All entries were "
            "suppressed.</p>")
    if not temp_on:
        out.append(
            "<p style='color:#8b949e;border:1px solid #30363d;"
            "border-radius:6px;padding:10px 12px;'>Temperature ladders are "
            "<b>WATCH-only</b>. Raw published forecasts carry a per-station "
            "bias and an ensemble spread that is too wide once models are "
            "pooled; trading them uncalibrated means trading the bias. "
            "Rain is unaffected — NWS PoP is already calibrated.</p>")

    out.append(
        "<h3 class='subhead' style='margin-top:18px;'>Probability "
        "sources</h3>"
        "<p class='small gray'>The weather bot has no trained model. "
        "Each contract's probability is read off a published forecast."
        "</p>"
        "<div style='overflow-x:auto;'><table style='width:100%;"
        "border-collapse:collapse;font-size:12.5px;'><thead><tr>"
        "<th style='text-align:left;padding:4px 10px 4px 0;'>Contract</th>"
        "<th style='text-align:left;padding:4px 10px;'>Source</th>"
        "<th style='text-align:left;padding:4px 10px;'>How it maps</th>"
        "<th style='text-align:left;padding:4px 0 4px 10px;'>Trading</th>"
        "</tr></thead><tbody>"
        "<tr style='border-top:1px solid #21262d;'>"
        "<td style='padding:5px 10px 5px 0;'><b>KXRAIN</b></td>"
        "<td class='small' style='padding:5px 10px;'>NWS "
        "<code>probabilityOfPrecipitation</code></td>"
        "<td class='small' style='padding:5px 10px;'>Officially "
        "calibrated P(&ge;0.01in at a point) — the same definition the "
        "contract settles on, trace counted as zero. The day's two "
        "12-hour periods are blended into one calendar-day "
        f"probability (blend={payload.get('pop_blend')}).</td>"
        f"<td style='padding:5px 0 5px 10px;color:"
        f"{'#3fb950' if rain_on else '#8b949e'};'>"
        f"{'ARMED' if rain_on else 'off'}</td></tr>"
        "<tr style='border-top:1px solid #21262d;'>"
        "<td style='padding:5px 10px 5px 0;'><b>KXHIGH*</b></td>"
        "<td class='small' style='padding:5px 10px;'>NWS official daily "
        "high + Open-Meteo ensemble spread</td>"
        "<td class='small' style='padding:5px 10px;'>Centre from the "
        "NWS forecast, width from the <i>median of per-model</i> member "
        "spreads (never pooled — pooling four models inflates sigma "
        "~1.5&times; because inter-model bias is not forecast "
        "uncertainty). Integer-rounded strikes.</td>"
        f"<td style='padding:5px 0 5px 10px;color:"
        f"{'#3fb950' if temp_on else '#8b949e'};'>"
        f"{'ARMED' if temp_on else 'WATCH-only'}</td></tr>"
        "</tbody></table></div>")

    off = {k: v for k, v in sums.items() if abs(float(v) - 1.0) > 0.02}
    out.append(
        "<h3 class='subhead' style='margin-top:18px;'>Ladder partition "
        "check</h3>"
        "<p class='small gray'>Each temperature ladder is a clean "
        "partition (one &lt;K, four 2&deg;F buckets, one &gt;K). Model "
        "probabilities across a ladder must sum to 1.00 — a ladder that "
        "doesn't is mis-parsed or missing a leg and its rows should not "
        f"be trusted. <b>{len(sums)}</b> ladders checked, "
        f"<b style='color:{'#3fb950' if not off else '#f85149'};'>"
        f"{len(off)}</b> off-partition.</p>")
    if off:
        out.append("<p class='small' style='color:#f85149;'>" +
                   html.escape(", ".join(f"{k}={v:.3f}"
                                          for k, v in list(off.items())[:8]))
                   + "</p>")

    out.append(
        "<h3 class='subhead' style='margin-top:18px;'>Station "
        "calibration</h3>"
        f"<p class='small gray'><b>{len(calibrated)}</b> of "
        f"<b>{payload.get('stations', 0)}</b> settlement stations have "
        "fitted bias / dispersion constants (30 observed settlements "
        "required). Until a station is calibrated its temperature "
        "contracts stay WATCH-only. Kalshi settles on The Weather "
        "Company's published value, which nobody archives — the bot "
        "logs every station-day so the constants can be fitted from "
        "data that would otherwise be lost.</p>")
    if calibrated:
        out.append("<p class='small gray'>Calibrated: " +
                   html.escape(", ".join(calibrated)) + "</p>")

    gen = payload.get("generated_at")
    if gen:
        out.append(f"<p class='small gray' style='margin-top:8px;'>"
                   f"Last tick: {html.escape(str(gen))} — "
                   f"{len(rows)} contracts priced.</p>")
