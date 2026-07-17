"""Reality Leaks dashboard adapter.

The bot (Reality Leaks repo) paper-trades reality-TV contracts off
public leak channels — Reddit spoiler communities, Reality Steve's
RSS, UK tabloid feeds. Its exporter writes watchlist.json (one row
per open Kalshi market across every configured show, annotated with
leak status/source) and a standard-schema sim.db paper ledger.

This module is the seam between those artifacts and the shared
renderers, mirroring the billboard adapter: the GET handler
synthesises standard watchlist rows from the JSON, sets model = None,
and falls through to the shared render_page. Positions come from the
standard sim.db readers (server.py / data.py "reality" branches).

Reality-TV-specific columns (rendered by watchlist_panel's
is_reality_bot branch): Show, Contestant, and a Leak column carrying
status + source, with the leak headline linked underneath.
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("dashboard.reality-leaks")


# --------------------------------------------------------------------------- #
# JSON loaders — same signatures as the billboard adapter                     #
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
    """No trained model — the 'model' is the leak-confidence mapping.
    Return {} so the home card shows the ledger stats only."""
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
    """Positions come from sim.db via the standard readers — the
    server's reality branch never calls this. Stub for signature
    parity with the other adapters."""
    return []


def closed_positions_for_rollup(sim_state_path: str | None,
                                  limit: int = 100,
                                  real_only: bool = False
                                  ) -> List[Dict[str, Any]]:
    return []


# --------------------------------------------------------------------------- #
# Watchlist row adapter                                                       #
# --------------------------------------------------------------------------- #

def build_standard_watchlist_rows(payload: Dict[str, Any]
                                    ) -> List[Dict[str, Any]]:
    """Translate reality-leaks watchlist rows into the schema the
    shared ``_render_watchlist`` expects. Reality-specific fields ride
    along under ``_``-prefixed keys for the is_reality_bot columns:

        _show           show name (e.g. "Big Brother US")
        _contestant     contestant the contract is about
        _leak_status    none | rumor | confirmed
        _leak_source    e.g. "r/BigBrother", "Reality Steve"
        _leak_url       link to the leak post/article
        _leak_title     leak headline
        _leak_age_hours age of the leak
        _market_kind    winner | elimination | rank | other
    """
    raw = payload.get("rows") or []
    out: List[Dict[str, Any]] = []
    for r in raw:
        out.append({
            "ticker": r.get("ticker") or "",
            "title": r.get("title") or "",
            "direction": r.get("contestant") or "",
            "_show": r.get("show") or "",
            "_contestant": r.get("contestant") or "",
            "_market_kind": r.get("market_kind") or "",
            "_leak_status": r.get("leak_status") or "none",
            "_leak_source": r.get("leak_source"),
            "_leak_url": r.get("leak_url"),
            "_leak_title": r.get("leak_title"),
            "_leak_age_hours": r.get("leak_age_hours"),
            "strike_low": None,
            "strike_high": None,
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


# --------------------------------------------------------------------------- #
# Models tab — no trained model; explain the leak-confidence mapping          #
# and show per-show leak coverage from the exporter's coverage block.         #
# --------------------------------------------------------------------------- #

def render_models_panel(out: List[str], bot: Dict[str, Any]) -> None:
    payload = load_watchlist(bot.get("watchlist_json_path"))
    coverage = payload.get("coverage") or []
    out.append(
        "<h3 class='subhead' style='margin-top:18px;'>Leak-confidence "
        "model</h3>"
        "<p class='small gray'>Reality Leaks has no trained model — "
        "the probability source is the leak itself. Public leak "
        "channels (Reddit spoiler communities, Reality Steve, UK "
        "tabloids) are polled every tick; when a post names a "
        "contestant plus an outcome keyword near that name, the "
        "contract's model probability is set by leak confidence: "
        "<b>confirmed</b> (spoiler-outlet headline, high-upvote "
        "thread, or explicit 'confirmed / spoiler' language) &rarr; "
        "93%; <b>rumor</b> &rarr; 72%. Direction is market-aware: an "
        "elimination leak implies YES on that contestant's "
        "elimination contract and NO on their winner contract. Buys "
        "require the leak-implied edge to clear 8&cent; after price "
        "caps — a leak the market has already priced is a SKIP. "
        "Paper trading only.</p>"
    )
    if not coverage:
        out.append("<p class='small gray'>No coverage snapshot yet — "
                   "waiting for the first exporter tick.</p>")
        return
    out.append(
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;"
        "font-size:12.5px;'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:4px 10px 4px 0;'>Show</th>"
        "<th style='text-align:left;padding:4px 10px;'>Leak channel</th>"
        "<th style='text-align:left;padding:4px 10px;'>Tier</th>"
        "<th class='num' style='padding:4px 10px;'>Open markets</th>"
        "<th class='num' style='padding:4px 10px;'>Posts scanned</th>"
        "<th class='num' style='padding:4px 0 4px 10px;'>Leaks matched</th>"
        "</tr></thead><tbody>"
    )
    for c in coverage:
        tier = c.get("tier") or ""
        tier_color = "#3fb950" if tier == "reliable" else "#d29922"
        out.append(
            "<tr style='border-top:1px solid #21262d;'>"
            f"<td style='padding:5px 10px 5px 0;'><b>"
            f"{html.escape(str(c.get('show') or ''))}</b></td>"
            f"<td class='small' style='padding:5px 10px;'>"
            f"{html.escape(str(c.get('leak_channel') or ''))}</td>"
            f"<td style='padding:5px 10px;color:{tier_color};'>"
            f"{html.escape(tier)}</td>"
            f"<td class='num' style='padding:5px 10px;'>"
            f"{c.get('open_markets', 0)}</td>"
            f"<td class='num' style='padding:5px 10px;'>"
            f"{c.get('posts_scanned', 0)}</td>"
            f"<td class='num' style='padding:5px 0 5px 10px;'>"
            f"{c.get('leaks_matched', 0)}</td>"
            "</tr>"
        )
    out.append("</tbody></table></div>")
    gen = payload.get("generated_at")
    if gen:
        out.append(f"<p class='small gray' style='margin-top:8px;'>"
                   f"Last tick: {html.escape(str(gen))}</p>")
