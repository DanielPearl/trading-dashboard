"""Top-level page renderer — chrome, tabs, panels, live-update JS."""
from __future__ import annotations

import html
import json
from datetime import datetime
from datetime import timezone
from typing import List
from .css import CSS
from .data import build_kalshi_cross_bot_history, filter_history_by_period
from .fmt import _favicon_link
from .models_panel import _render_models_panel
from .panels import (
    PERIOD_OPTIONS,
    _period_days,
    _render_bet_history_block,
    _render_bot_cards,
    _render_bot_filter,
    _render_bot_unavailable,
    _render_history_attribution,
    _render_history_chart,
    _render_notifications_panel,
    _render_seasons_panel,
    _render_summary,
    _render_summary_cards,
)
from .watchlist_panel import _render_watchlist

import logging
log = logging.getLogger("dashboard")


def render_page(
    model: dict | None,
    global_summary: dict,
    global_active_bets: List[dict],
    global_history: List[dict],
    latest_active: dict | None,
    bot_active_bets: List[dict] | None,
    bot_closed_positions: List[dict],
    watchlist: List[dict],
    underlying_history: List[dict],
    display: dict,
    kalshi_history: List[dict],
    atm_market: dict | None,
    contract_open_ts: float | None,
    contract_close_ts: float | None,
    event_title: str | None,
    risk_caps: dict,
    edge_cfg: dict,
    validator_cfg: dict,
    hedge_cfg: dict,
    available_bots: List[dict],
    current_bot: str,
    period_key: str = "all",
    tab_key: str = "home",
    bot_models: List[dict] | None = None,
    prob_history: List[dict] | None = None,
    model_view: str = "pregame",
    threshold_source: dict | None = None,
    extra_cfg: dict | None = None,
    mode: str = "sim",
    live_state_paths: list[str] | None = None,
    query_string: str = "",
) -> str:
    # Mode-driven page chrome. ``mode`` is "sim" (paper-trading
    # default) or "live" (real-money dashboard). Anything not
    # explicitly "live" renders as sim — safety default. See
    # dashboard.yaml's ``mode:`` field and config.py's load_config
    # for where this comes from.
    is_live = (mode == "live")
    page_title = ("Kalshi LIVE Trading"
                   if is_live else "Kalshi Simulation Dashboard")
    meta_warning = ("LIVE TRADING — real money at risk"
                     if is_live else "DRY-RUN mode (no real orders)")
    # Mode-pill text + which port the pill links to. Sim → 8081
    # (live), live → 8080 (sim). The pill's href is rewritten by the
    # tiny inline script below so it picks up the current
    # window.location.host (works on localhost dev and the droplet
    # interchangeably) and preserves the query string (so toggling
    # sides doesn't drop your selected bot / tab / period).
    peer_port = 8080 if is_live else 8081
    pill_text = ("← Back to SIMULATION"
                  if is_live else "→ Switch to LIVE (real money)")

    out: List[str] = []
    out.append("<!doctype html><html><head>")
    out.append("<meta charset='utf-8'>")
    # No meta-refresh — JS at the bottom of the page polls /api/snapshot
    # every 5s and patches live cells in place. The page never reloads.
    out.append(f"<title>{html.escape(page_title)}</title>")
    out.append(_favicon_link())
    out.append(f"<style>{CSS}</style>")
    # data-mode hook so CSS can target either palette without touching
    # any render code. Today both modes use the default GitHub-dark
    # palette; the live theme (red-tinted) will land in CSS in the
    # next chapter (the fork itself).
    out.append(f"</head><body data-mode='{html.escape(mode)}'>")
    out.append(
        f"<h1>{html.escape(page_title)}"
        f"<a class='mode-pill' data-peer-port='{peer_port}' "
        f"href='#'>{html.escape(pill_text)}</a></h1>"
    )
    out.append(
        f"<div class='meta'>Loaded "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
        f" · live updates every 5s · {html.escape(meta_warning)}</div>"
    )
    # Inline mode-pill href rewriter. Runs before the rest of the
    # page JS executes so the user never sees the placeholder '#'
    # href in any context (link preview, right-click "copy URL", etc).
    # Preserves window.location.pathname + search so toggling between
    # sim and live keeps the user on the same bot / tab / period.
    out.append(
        "<script>"
        "(function(){"
        "  document.querySelectorAll('.mode-pill').forEach(function(a){"
        "    var p=a.dataset.peerPort; if(!p) return;"
        "    var u=new URL(window.location.href); u.port=p;"
        "    a.href=u.toString();"
        "  });"
        "})();"
        "</script>"
    )

    # ── Top-level page tabs ───────────────────────────────────────────
    # 2026-07-13 redesign: four top-level tabs — Home, Contracts,
    # History, Seasons. The old Watchlist / Models / Training Data
    # tabs became sub-tabs INSIDE the Contracts tab (watchlist is the
    # default sub-tab). Legacy ?tab=watchlist|models|training URLs
    # keep working — they land on Contracts with that sub-tab active,
    # and sub-tab clicks keep writing those same legacy keys to the
    # URL so every deep link elsewhere in the codebase stays valid.
    tabs = [
        ("home", "Home"),
        ("contracts", "Contracts"),
        ("history", "History"),
        ("seasons", "Seasons"),
    ]
    subtabs = [
        ("watchlist", "Watchlist"),
        ("models", "Model"),
        ("training", "Training Data"),
    ]
    subtab_keys = {k for k, _ in subtabs}
    if tab_key in subtab_keys:
        active_tab = "contracts"
        active_subtab = tab_key
    elif tab_key == "contracts":
        active_tab = "contracts"
        active_subtab = "watchlist"
    elif tab_key in {k for k, _ in tabs}:
        active_tab = tab_key
        active_subtab = "watchlist"
    else:
        active_tab = "home"
        active_subtab = "watchlist"

    out.append("<div class='tab-bar'>")
    for k, label in tabs:
        cls = "tab-pill" + (" tab-pill-active" if k == active_tab else "")
        out.append(
            f"<a class='{cls}' data-tab='{html.escape(k)}' "
            f"href='#tab-{html.escape(k)}'>{html.escape(label)}</a>"
        )
    out.append("</div>")

    def _open_panel(name: str) -> None:
        cls = "tab-panel" + (" tab-panel-active" if name == active_tab else "")
        out.append(f"<div class='{cls}' data-panel='{html.escape(name)}'>")

    def _open_subpanel(name: str) -> None:
        cls = ("subtab-panel"
               + (" subtab-panel-active" if name == active_subtab else ""))
        out.append(f"<div class='{cls}' data-subpanel='{html.escape(name)}'>")

    period_label = next(
        (lbl for k, lbl, _ in PERIOD_OPTIONS if k == period_key),
        "All-time",
    )

    # ── HOME tab — summary cards + active bets + per-bot perf cards ──
    # Performance and Home were merged per user request; the bot-card
    # grid sits below the summary section as a "what's in each bot"
    # overview that doubles as a click-through to each bot's Watchlist.
    _open_panel("home")
    _render_summary(out, global_summary, global_active_bets, global_history,
                     period_key=period_key, current_bot=current_bot,
                     available_bots=available_bots,
                     hedge_cfg=hedge_cfg)
    # The Active bets table inside _render_summary already aggregates
    # open positions across every bot — in live mode those ARE the
    # real-money positions on Kalshi (the sim_state feed and the
    # bot dbs each point at the live artifacts when mode == "live").
    # A separate "Open Real-Money Positions" section would just show
    # the same rows with a subset of columns, so we render it once.
    out.append("<div class='section'><h2>Model performance</h2>"
               "<div class='body'>")
    _render_notifications_panel(out)
    _render_bot_cards(out, global_summary, bot_models, period_label)
    out.append("</div></div>")
    out.append("</div>")  # /home panel

    # ── CONTRACTS tab — sub-tabs: Watchlist / Model / Training Data ──
    # The bot filter sits at the top of the panel, ABOVE the sub-tab
    # bar (user 2026-07-13), and applies to all three sub-pages —
    # switching bots keeps the active sub-tab via the legacy
    # ?tab=<subtab> key baked into the option URLs (the JS overrides
    # it with whichever sub-tab is visible at click time).
    _open_panel("contracts")
    # No "All bots" entry (user 2026-07-13) — each bot trades its own
    # contract series, so a cross-bot view of the Contracts sub-pages
    # doesn't exist; the dropdown always names a specific bot.
    if available_bots:
        _render_bot_filter(out, available_bots,
                            current_bot=current_bot,
                            period_key=period_key,
                            select_id="bot-select-top",
                            tab_key=active_subtab)
    out.append("<div class='subtab-bar'>")
    for k, label in subtabs:
        cls = ("subtab-pill"
               + (" subtab-pill-active" if k == active_subtab else ""))
        out.append(
            f"<a class='{cls}' data-subtab='{html.escape(k)}' "
            f"href='#tab-{html.escape(k)}'>{html.escape(label)}</a>"
        )
    out.append("</div>")

    # ── Watchlist sub-tab — chart + strike ladder + Kalshi rules ─────
    _open_subpanel("watchlist")
    if (not watchlist and not latest_active
            and not [b for b in available_bots
                     if b["key"] == current_bot and b.get("available")]):
        _render_bot_unavailable(out, current_bot)
    else:
        # Bot dropdown is rendered inside _render_watchlist (below the
        # section title, above the current-prediction card) so it sits
        # with the section it scopes.
        _render_watchlist(out, watchlist, model,
                          underlying_history=underlying_history,
                          display=display,
                          latest_active=latest_active,
                          bot_active_bets=bot_active_bets or [],
                          kalshi_history=kalshi_history,
                          prob_history=prob_history or [],
                          atm_market=atm_market,
                          contract_open_ts=contract_open_ts,
                          contract_close_ts=contract_close_ts,
                          event_title=event_title,
                          edge_cfg=edge_cfg,
                          validator_cfg=validator_cfg,
                          risk_caps=risk_caps,
                          hedge_cfg=hedge_cfg,
                          extra_cfg=extra_cfg,
                          threshold_source=threshold_source,
                          available_bots=available_bots,
                          current_bot=current_bot,
                          period_key=period_key)
        # Standalone Kalshi-rules section retired 2026-07-08 — every
        # watchlist row now carries its own Rules ``i`` button in the
        # Rules column that opens a popover with that ticker's
        # ``rules_primary``. Non-sport bots still get the section
        # value because their strike ladder shares one rule across
        # every row, but the popover-per-row idiom scales cleanly to
        # the sport bots' one-rule-per-match reality.
    out.append("</div>")  # /watchlist subpanel

    # ── Model sub-tab — per-bot model deep-dive ──────────────────────
    _open_subpanel("models")
    current_bot_dict = next(
        (b for b in available_bots if b.get("key") == current_bot),
        None,
    )
    _render_models_panel(
        out,
        bot=current_bot_dict or {},
        model=model,
        display=display,
        available_bots=available_bots,
        current_bot=current_bot,
        model_view=model_view,
        bot_active_bets=bot_active_bets,
        bot_closed_positions=bot_closed_positions,
    )
    out.append("</div>")  # /models subpanel

    # ── Training Data sub-tab — training panel + Kalshi outcomes ─────
    # Sourced from data/training_history.db on the tennis-forecast
    # droplet. Currently tennis-only since it's the only bot whose
    # trainer writes to the DB; tab still renders for other bots with
    # an explanatory message rather than blank.
    _open_subpanel("training")
    try:
        from . import tennis as _tennis_mod
        from urllib.parse import parse_qs as _parse_qs
        _td_qs = _parse_qs(query_string)
        try:
            _page = max(1, int(_td_qs.get("page", ["1"])[0]))
        except (TypeError, ValueError):
            _page = 1
        if current_bot == "world-cup":
            # World Cup ships its own training panel — the full
            # historical WC-finals match grain with the who-won label.
            from . import world_cup as _wc_mod
            out.append(_wc_mod.render_training_data_panel(
                bot=current_bot_dict or {},
                current_bot=current_bot,
                page=_page,
                page_size=20,
                segment=_td_qs.get("seg", [None])[0],
                current_tab="training",
                period_key=period_key,
            ))
        elif current_bot == "billboard":
            # Billboard ships its own training panel — the full
            # (song × chart-week) popular-pool grain with the
            # in_hot_100 label, paged from training_data.db.
            from . import billboard as _bb_mod
            out.append(_bb_mod.render_training_data_panel(
                bot=current_bot_dict or {},
                current_bot=current_bot,
                page=_page,
                page_size=100,
                week=_td_qs.get("week", [None])[0],
                current_tab="training",
                period_key=period_key,
            ))
        else:
            out.append(_tennis_mod.render_training_data_panel(
                current_bot=current_bot,
                page=_page,
                page_size=20,
                tour_filter=_td_qs.get("tour", [None])[0],
                split_filter=_td_qs.get("split", [None])[0],
                current_tab="training",
                period_key=period_key,
            ))
    except Exception:  # noqa: BLE001
        log.exception("training data panel failed to render")
        out.append("<div class='empty'>Training Data unavailable — "
                    "see dashboard log for details.</div>")
    out.append("</div>")  # /training subpanel
    out.append("</div>")  # /contracts panel

    # ── HISTORY tab — closed-bet history across all bots ──────────────
    _open_panel("history")
    out.append(
        f"<div class='section'><h2>Contract history "
        f"<span class='small gray'>({html.escape(period_label)})"
        f"</span></h2>"
        f"<div class='body'>"
    )
    # 2026-07-11 (updated): History tab sources every row from Kalshi's
    # real ``/portfolio/settlements`` + ``/portfolio/fills`` endpoints
    # on BOTH LIVE and SIM. Per user 2026-07-11 "only show actual closed
    # contracts in history. no paper bets should be in here." — the
    # earlier SIM branch that retained ``global_history`` (paper closes)
    # was retired. Both dashboards now show the same real-Kalshi ledger.
    kalshi_history: List[dict] = build_kalshi_cross_bot_history(
        available_bots)
    # Period filter — same rule as ``global_history``. None keeps all.
    kalshi_history = filter_history_by_period(
        kalshi_history, _period_days(period_key))

    # Cards synthesised from the same list every other block reads.
    _kh_wins = sum(1 for h in kalshi_history
                    if (h.get("realized_pnl_cents") or 0) > 0)
    _kh_total_cents = sum(int(h.get("realized_pnl_cents") or 0)
                           for h in kalshi_history)
    _kh_spent_cents = sum(
        int((h.get("entry_price_cents") or 0)
             * (h.get("contracts") or 1))
        for h in kalshi_history)
    kalshi_rollup = {
        "total_opened": len(kalshi_history),
        "total_closed": len(kalshi_history),
        "wins": _kh_wins,
        "losses": len(kalshi_history) - _kh_wins,
        "win_rate": (_kh_wins / len(kalshi_history))
                     if kalshi_history else None,
        "total_realized_pnl": _kh_total_cents / 100.0,
        "total_staked": _kh_spent_cents / 100.0,
        "active_bets": (global_summary.get("active_bets", 0)
                         if global_summary else 0),
    }
    _render_summary_cards(out, kalshi_rollup or global_summary,
                           id_suffix="-history",
                           show_closed_contracts=True)
    _render_history_chart(out, kalshi_history,
                            period_key=period_key,
                            current_bot=current_bot)
    _render_history_attribution(out, kalshi_history)
    _ledger_hint = (
        "(from Kalshi /portfolio/settlements + /portfolio/fills "
        "— every bot, every settled contract)" if is_live else
        "(every bot's closed paper bets — the sim ledger)")
    out.append("<h3 class='subhead'>All Bets "
                "<span class='small gray'>"
                f"{_ledger_hint}</span></h3>")
    out.append("<div class='history-scroll'>")
    if kalshi_history:
        _render_bet_history_block(out, kalshi_history, heading="",
                                    shown_initially=25)
    elif is_live:
        out.append("<div class='empty'>No settled Kalshi contracts yet "
                    "— the ledger will populate as bots trade live and "
                    "positions settle.</div>")
    else:
        out.append("<div class='empty'>No closed paper bets yet — the "
                    "ledger fills in as the simulators' positions "
                    "settle.</div>")
    out.append("</div>")
    out.append("</div></div>")
    out.append("</div>")  # /history panel

    # ── SEASONS tab — one card per bot with a real-world season ──────
    _open_panel("seasons")
    _render_seasons_panel(out, available_bots)
    out.append("</div>")  # /seasons panel

    # Live-update JS: polls /api/snapshot every 5s and patches summary
    # cards + watchlist cells in place. Pass the period so the live
    # cards keep matching the user's filter selection between polls.
    # Shared "Why?" modal — single instance, populated dynamically by
    # the JS hook when any .criteria-btn is clicked.
    out.append(
        "<div id='criteria-overlay' class='criteria-overlay' hidden></div>"
        "<div id='criteria-modal' class='criteria-modal' hidden>"
        "  <div class='criteria-modal-head'>"
        "    <div>"
        "      <h3>Why was this bet chosen?</h3>"
        "      <div class='ticker' id='criteria-modal-ticker'></div>"
        "    </div>"
        "    <button type='button' id='criteria-close' "
        "      class='criteria-modal-close' aria-label='Close'>×</button>"
        "  </div>"
        "  <div class='criteria-modal-body' id='criteria-modal-body'></div>"
        "</div>"
    )
    # Daily droplet diagnosis modal — single shared instance, opened by
    # any .diagnosis-btn click. Body is populated from /api/diagnosis/latest
    # at click time so it always reflects the latest scheduled report
    # without bloating the initial page render.
    out.append(
        "<div id='diagnosis-overlay' class='diagnosis-overlay' hidden></div>"
        "<div id='diagnosis-modal' class='diagnosis-modal' hidden>"
        "  <div class='diagnosis-modal-head'>"
        "    <div>"
        "      <h3>Daily droplet diagnosis</h3>"
        "      <div class='diagnosis-meta' id='diagnosis-modal-meta'></div>"
        "    </div>"
        "    <button type='button' id='diagnosis-close' "
        "      class='diagnosis-modal-close' aria-label='Close'>×</button>"
        "  </div>"
        "  <div class='diagnosis-modal-body' id='diagnosis-modal-body'></div>"
        "</div>"
    )
    # Stash the gating config in a window global so the per-bet popup
    # can list "validators that were met" without bloating every
    # criteria-btn payload. These are global rules (same for all bots),
    # so one publish per page is enough.
    buy_criteria_payload = json.dumps({
        "edge": edge_cfg or {},
        "validators": validator_cfg or {},
        "risk": risk_caps or {},
        "hedge": hedge_cfg or {},
        "_source": threshold_source or {"source": "fallback",
                                          "captured_at": None,
                                          "missing_keys": []},
    }, separators=(",", ":"), default=str)
    out.append(
        f"<script>window.__BUY_CRITERIA__ = {buy_criteria_payload};</script>"
    )
    out.append(_BOT_TOGGLE_JS)
    out.append(_HISTORY_CHART_JS)
    out.append(_SEASON_COUNTDOWN_JS)
    out.append(_live_update_script(current_bot, period_key=period_key))
    # Freshness stamp — makes a stale browser tab self-evident: if this
    # timestamp is old, the page predates the latest deploy/restart and
    # needs a reload. Renders on every tab of both sites.
    out.append(
        f"<div class='small gray' style='margin:18px 0 8px 0;"
        f"text-align:center;opacity:0.7;'>page rendered "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        f" · reload for latest</div>"
    )
    out.append("</body></html>")
    return "".join(out)


# Click handler for the homepage bot-card toggle. Hits the
# /api/bot/toggle POST endpoint with the bot key, then updates the
# card's data-enabled state + label / PAUSED badge without a reload.
# event.preventDefault stops the parent <a class='bot-card'> from
# following its href when the user clicks the toggle itself.
_BOT_TOGGLE_JS = """<script>
function toggleBotState(ev, btn) {
  ev.preventDefault();
  ev.stopPropagation();
  const key = btn.dataset.botKey;
  if (!key) return;
  btn.disabled = true;
  fetch('/api/bot/toggle?bot=' + encodeURIComponent(key), {method: 'POST'})
    .then(function (r) { return r.json(); })
    .then(function (data) {
      const enabled = !!data.enabled;
      btn.dataset.enabled = enabled ? '1' : '0';
      btn.setAttribute('aria-pressed', enabled ? 'true' : 'false');
      const card = btn.closest('.bot-card');
      if (card) {
        card.classList.toggle('bot-card-paused', !enabled);
        // Add or remove the PAUSED badge to match the new state.
        const nameEl = card.querySelector('.bot-name');
        if (nameEl) {
          const existing = nameEl.querySelector('.paused-badge');
          if (!enabled && !existing) {
            const pill = document.createElement('span');
            pill.className = 'paused-badge';
            pill.title = 'Bot is paused — toggle on to resume taking bets.';
            pill.textContent = 'PAUSED';
            nameEl.appendChild(pill);
          } else if (enabled && existing) {
            existing.remove();
          }
        }
      }
    })
    .catch(function () { /* swallow — keep current visual state */ })
    .finally(function () { btn.disabled = false; });
}
</script>"""


# Seasons-tab live countdown. Each card carries data-start / data-end
# (ISO datetimes) — we tick once a second and update the headline +
# remaining-time fields. Only two states are surfaced:
#   • Before start  → "Starts in …" (yellow)
#   • Between       → "Ends in …"   (green)
# A card whose season has already ended is hidden server-side (the
# renderer doesn't emit it), so the JS doesn't need an "over" branch.
_SEASON_COUNTDOWN_JS = """<script>
(function () {
  function fmt(ms) {
    if (ms <= 0) return "0d 0h 0m 0s";
    const s = Math.floor(ms / 1000);
    const days = Math.floor(s / 86400);
    const hours = Math.floor((s % 86400) / 3600);
    const mins = Math.floor((s % 3600) / 60);
    const secs = s % 60;
    return days + "d " + hours + "h " + mins + "m " + secs + "s";
  }
  function pct(now, start, end) {
    if (now <= start) return 0;
    if (now >= end) return 100;
    const span = end - start;
    if (span <= 0) return 100;
    return Math.max(0, Math.min(100, ((now - start) / span) * 100));
  }
  function tick() {
    const now = Date.now();
    document.querySelectorAll('[data-season-card]').forEach(function (card) {
      const start = parseInt(card.dataset.start, 10);
      const end = parseInt(card.dataset.end, 10);
      if (!isFinite(start) || !isFinite(end)) return;
      const statusEl = card.querySelector('[data-season-status]');
      const labelEl = card.querySelector('[data-season-countdown-label]');
      const valueEl = card.querySelector('[data-season-countdown-value]');
      const fillEl = card.querySelector('[data-season-progress-fill]');
      let status, label, value, color;
      if (now < start) {
        status = 'Upcoming';
        label = 'Starts in';
        value = fmt(start - now);
        color = 'yellow';
      } else if (now < end) {
        status = 'In season';
        label = 'Ends in';
        value = fmt(end - now);
        color = 'green';
      } else {
        // Season just ticked past its end window while the page was
        // open. Hide the card rather than flashing "season over" —
        // matches the server-side behaviour of dropping ended cards.
        card.style.display = 'none';
        return;
      }
      if (statusEl) {
        statusEl.textContent = status;
        statusEl.classList.remove('green', 'yellow', 'gray');
        statusEl.classList.add(color);
      }
      if (labelEl) labelEl.textContent = label;
      if (valueEl) {
        valueEl.textContent = value;
        valueEl.classList.remove('green', 'yellow', 'gray');
        valueEl.classList.add(color);
      }
      if (fillEl) fillEl.style.width = pct(now, start, end).toFixed(2) + '%';
    });
  }
  tick();
  setInterval(tick, 1000);
})();
</script>"""

# History-tab daily P&L chart renderer. Reads the closed-bet ledger
# embedded as JSON on the SVG node, filters by selected bot + date
# range, buckets bets by UTC day, then plots each day's net realized
# P&L in cents. The line crosses zero naturally when winning vs. losing
# days alternate; a dashed baseline at $0 makes the sign obvious.
_HISTORY_CHART_JS = """<script>
(function () {
  const svg = document.querySelector('[data-history-chart]');
  if (!svg) return;
  let raw = [];
  try { raw = JSON.parse(svg.dataset.points || '[]'); } catch (e) {}
  const W = 800, H = 260;
  const PAD_L = 56, PAD_R = 14, PAD_T = 14, PAD_B = 28;
  const INNER_W = W - PAD_L - PAD_R;
  const INNER_H = H - PAD_T - PAD_B;
  function fmtSignedDollars(cents) {
    const v = cents / 100;
    const sign = v > 0 ? '+' : (v < 0 ? '-' : '');
    return sign + '$' + Math.abs(v).toFixed(2);
  }
  function fmtDate(epoch) {
    const d = new Date(epoch * 1000);
    return d.toLocaleDateString(undefined,
      {month: 'short', day: 'numeric'});
  }
  function render() {
    const now = new Date();
    const todayMidnight = Math.floor(Date.UTC(
      now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) / 1000);
    // Bucket every closed bet by UTC day, summing realized P&L in
    // cents per bucket. Each series point is (UTC-midnight epoch,
    // day net).
    const daily = new Map();
    raw.forEach(function (p) {
      const d = new Date(p[0] * 1000);
      const dayEpoch = Math.floor(Date.UTC(
        d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) / 1000);
      daily.set(dayEpoch, (daily.get(dayEpoch) || 0) + p[1]);
    });
    // Ensure today is always the rightmost point. If no trades
    // settled today, the bucket is 0 — same daily-delta semantics
    // (today made $0 of new realized P&L).
    if (!daily.has(todayMidnight)) {
      daily.set(todayMidnight, 0);
    }
    const series = Array.from(daily.entries())
      .sort(function (a, b) { return a[0] - b[0]; });
    svg.innerHTML = '';
    if (series.length === 0) return;
    // X range: first closed-bet day → today (UTC midnight). Using
    // today's midnight (not "now") keeps each day a full equal slot
    // and parks the last data point exactly at the right edge.
    const tMin = series[0][0];
    const tMax = todayMidnight;
    const tSpan = Math.max(1, tMax - tMin);
    // Y range: include 0 so the zero baseline always shows. 8% pad.
    let vals = series.map(function (s) { return s[1]; });
    vals.push(0);
    let yMin = Math.min.apply(null, vals);
    let yMax = Math.max.apply(null, vals);
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const yPad = (yMax - yMin) * 0.08;
    yMin -= yPad; yMax += yPad;
    function x(t) {
      return PAD_L + ((t - tMin) / tSpan) * INNER_W;
    }
    function y(v) {
      return PAD_T + (1 - (v - yMin) / (yMax - yMin)) * INNER_H;
    }
    const NS = 'http://www.w3.org/2000/svg';
    function el(name, attrs, text) {
      const n = document.createElementNS(NS, name);
      for (const k in attrs) n.setAttribute(k, attrs[k]);
      if (text != null) n.textContent = text;
      return n;
    }
    // Horizontal gridlines + Y labels (5 ticks).
    for (let i = 0; i <= 4; i++) {
      const yv = yMin + (i / 4) * (yMax - yMin);
      const py = y(yv);
      svg.appendChild(el('line', {
        x1: PAD_L, y1: py, x2: W - PAD_R, y2: py,
        stroke: '#1f2530', 'stroke-width': '1'
      }));
      svg.appendChild(el('text', {
        x: PAD_L - 6, y: py + 4, fill: '#8b949e',
        'font-size': '10', 'text-anchor': 'end'
      }, fmtSignedDollars(yv)));
    }
    // Zero baseline — always drawn so the user can see at a glance
    // when the daily line is above (gains) vs. below (losses).
    const py0 = y(0);
    svg.appendChild(el('line', {
      x1: PAD_L, y1: py0, x2: W - PAD_R, y2: py0,
      stroke: '#6e7681', 'stroke-width': '1'
    }));
    // X-axis date ticks — one label per actual day in the series,
    // never a fractional position that rounds to a duplicate date.
    // When the series is long, downsample to ~6 labels by even-stride
    // selection over the real day epochs.
    const MAX_TICKS = 6;
    const dayEpochs = series.map(function (s) { return s[0]; });
    const stride = Math.max(1, Math.ceil(dayEpochs.length / MAX_TICKS));
    const tickEpochs = [];
    for (let i = 0; i < dayEpochs.length; i += stride) {
      tickEpochs.push(dayEpochs[i]);
    }
    // Always include the rightmost (today) so the user sees today's
    // label even if the stride skipped it.
    if (tickEpochs[tickEpochs.length - 1] !== dayEpochs[dayEpochs.length - 1]) {
      tickEpochs.push(dayEpochs[dayEpochs.length - 1]);
    }
    tickEpochs.forEach(function (t, i) {
      const px = x(t);
      const last = i === tickEpochs.length - 1;
      svg.appendChild(el('text', {
        x: px, y: H - 8, fill: '#8b949e',
        'font-size': '10',
        'text-anchor': i === 0 ? 'start' : (last ? 'end' : 'middle')
      }, fmtDate(t)));
    });
    // Split the line into colored segments — green when the value is
    // >= 0 (gains), red when < 0 (losses). When two consecutive points
    // straddle zero, interpolate the crossing so the color flips
    // exactly at the baseline.
    const GREEN = '#3fb950', RED = '#f85149';
    const colorOf = function (v) { return v >= 0 ? GREEN : RED; };
    for (let i = 1; i < series.length; i++) {
      const a = series[i - 1], b = series[i];
      const sameSide = (a[1] >= 0) === (b[1] >= 0);
      if (sameSide) {
        svg.appendChild(el('polyline', {
          points: x(a[0]).toFixed(1) + ',' + y(a[1]).toFixed(1) + ' ' +
            x(b[0]).toFixed(1) + ',' + y(b[1]).toFixed(1),
          fill: 'none', stroke: colorOf(a[1]),
          'stroke-width': '2',
          'stroke-linejoin': 'round', 'stroke-linecap': 'round'
        }));
      } else {
        // Interpolate t where the line crosses zero.
        const t = a[1] / (a[1] - b[1]);
        const xc = a[0] + t * (b[0] - a[0]);
        svg.appendChild(el('polyline', {
          points: x(a[0]).toFixed(1) + ',' + y(a[1]).toFixed(1) + ' ' +
            x(xc).toFixed(1) + ',' + y(0).toFixed(1),
          fill: 'none', stroke: colorOf(a[1]),
          'stroke-width': '2',
          'stroke-linejoin': 'round', 'stroke-linecap': 'round'
        }));
        svg.appendChild(el('polyline', {
          points: x(xc).toFixed(1) + ',' + y(0).toFixed(1) + ' ' +
            x(b[0]).toFixed(1) + ',' + y(b[1]).toFixed(1),
          fill: 'none', stroke: colorOf(b[1]),
          'stroke-width': '2',
          'stroke-linejoin': 'round', 'stroke-linecap': 'round'
        }));
      }
    }
    // Dots on each daily point, colored by sign of that day's net.
    series.forEach(function (s) {
      svg.appendChild(el('circle', {
        cx: x(s[0]), cy: y(s[1]), r: '2.5',
        fill: colorOf(s[1])
      }));
    });
  }
  render();
})();
</script>"""


def _live_update_script(current_bot: str, period_key: str = "all") -> str:
    """Self-contained JS block that fetches /api/snapshot every 5s
    and patches DOM cells with new values. Highlights changed cells
    briefly so updates are visible.
    """
    bot_param = html.escape(current_bot)
    period_param = html.escape(period_key)
    return f"""<script>
(function () {{
  const BOT = "{bot_param}";
  const PERIOD = "{period_param}";
  const POLL_MS = 5000;

  // Format helpers — must mirror the server-side rendering in render_page.
  function fmtSignedCents(c) {{
    if (c === null || c === undefined) return "—";
    const dollars = Math.abs(c) / 100.0;
    const sign = c > 0 ? "+" : (c < 0 ? "−" : "");
    return sign + "$" + dollars.toFixed(2);
  }}
  function fmtPct(p, hasData) {{
    if (!hasData || p === null || p === undefined) return "—";
    return Math.round(p * 100) + "%";
  }}
  function fmtEv(ev) {{
    if (ev === null || ev === undefined) return "0";
    const rounded = Math.round(ev * 100) / 100;
    if (rounded === 0) return "0";
    const sign = rounded >= 0 ? "+" : "−";
    return sign + "$" + Math.abs(rounded).toFixed(2);
  }}
  function evClass(ev, minEv) {{
    if (ev === null || ev === undefined) return "gray";
    if (ev >= minEv) return "green";
    if (ev > 0) return "yellow";
    return "red";
  }}
  function flash(el) {{
    if (!el) return;
    el.classList.add("cell-flash");
    setTimeout(function () {{ el.classList.remove("cell-flash"); }}, 800);
  }}
  function patch(id, newText, newClass) {{
    const el = document.getElementById(id);
    if (!el) return;
    if (el.textContent !== newText) {{
      el.textContent = newText;
      flash(el);
    }}
    if (newClass !== undefined) {{
      // Replace any of green/red/yellow/gray, keep other classes.
      el.classList.remove("green", "red", "yellow", "gray");
      if (newClass) el.classList.add(newClass);
    }}
  }}
  function patchCell(td, newText, newClass) {{
    if (!td) return;
    if (td.textContent !== newText) {{
      td.textContent = newText;
      flash(td);
    }}
    if (newClass !== undefined) {{
      td.classList.remove("green", "red", "yellow", "gray");
      if (newClass) td.classList.add(newClass);
    }}
  }}

  function applySnapshot(snap) {{
    // ── Summary cards ──────────────────────────────────────────────
    // Home tab: Active bets | Active contracts | Active bots
    // | Money spent | Potential gain | Week change %.
    // All values are live, never period-scoped.
    const s = snap.summary || {{}};
    patch("card-active-bets", String(s.active_bets ?? 0));
    patch("card-active-contracts", String(s.active_contracts ?? 0));
    patch("card-active-bots", String(s.active_bots ?? 0));
    patch("card-money-spent",
          (s.active_money_spent_cents ?? 0) === 0
            ? "$0.00"
            : fmtSignedCents(-(s.active_money_spent_cents ?? 0)));
    patch("card-potential-earnings",
          "+" + fmtSignedCents(s.potential_gain_cents).replace(/^[+−-]/, ""),
          "green");
    // Week change %: (this_week / |net - this_week|) * 100. Mirrors
    // the Python _week_change_pct so the polled value matches the
    // server-rendered first paint.
    {{
      const tw = s.this_week_pnl_cents ?? 0;
      const lt = s.net_pnl_cents ?? 0;
      const wa = lt - tw;
      let text, cls;
      if (wa === 0) {{ text = "—"; cls = "gray"; }}
      else {{
        const pct = (tw / Math.abs(wa)) * 100;
        const sign = pct > 0 ? "+" : (pct < 0 ? "−" : "");
        text = sign + Math.abs(pct).toFixed(1) + "%";
        cls = pct > 0 ? "green" : (pct < 0 ? "red" : "gray");
      }}
      patch("card-week-change", text, cls);
    }}

    // ── Watchlist rows ─────────────────────────────────────────────
    const minEv = snap.min_ev || 0.03;
    // Sport bots split their watchlist into two tables (Active bets +
    // Model vs market); non-sport bots use a single ``watchlist-tbody``.
    // Query both ids and merge — rows without a match are skipped
    // silently either way.
    const tbodies = ["watchlist-tbody", "watchlist-tbody-active"]
      .map(function (id) {{ return document.getElementById(id); }})
      .filter(Boolean);
    if (tbodies.length && snap.watchlist) {{
      const rowsByTicker = {{}};
      tbodies.forEach(function (tb) {{
        tb.querySelectorAll("tr[data-ticker]").forEach(function (tr) {{
          rowsByTicker[tr.getAttribute("data-ticker")] = tr;
        }});
      }});
      // Keep the "row-bought" highlight in sync with the active-bets
      // list — if a position opens or closes mid-poll, the held strike
      // gets/loses its blue rail without a full page reload. The
      // BOUGHT pill itself is server-rendered; if it's missing the
      // user can refresh to pick it up.
      const boughtBySide = {{}};
      // Only REAL Kalshi holdings paint a row as bought (user
      // 2026-07-10) — paper/sim actives no longer drive the
      // highlight, matching the server-rendered rule.
      (snap.kalshi_held || []).forEach(function (ab) {{
        if (ab && ab.ticker) {{
          const s = (ab.side || "").toUpperCase();
          boughtBySide[ab.ticker] = (s === "NO") ? "no" : "yes";
        }}
      }});
      tbodies.forEach(function (tbody) {{ tbody.querySelectorAll("tr[data-ticker]").forEach(function (tr) {{
        const t = tr.getAttribute("data-ticker");
        const side = boughtBySide[t];
        tr.classList.remove("row-bought", "bought-yes", "bought-no");
        if (side) {{
          tr.classList.add("row-bought", "bought-" + side);
          tr.classList.remove("row-suspect");
        }} else {{
          // Per user request: only HOLDING rows render full-bright
          // white. When a position closes mid-poll, re-apply the
          // greyed style so the row drops back to dimmed without
          // a full page reload.
          tr.classList.add("row-suspect");
        }}
      }}); }});
      // Patch a single side-span inside one of the combined cells
      // (Kalshi / My / Edge / EV). Each cell has two spans flanking
      // a "/" separator; we update them in place so the polled
      // refresh keeps the per-side colour without re-rendering.
      function patchSide(cell, side, text, cls) {{
        if (!cell) return;
        const span = cell.querySelector("span[data-side='" + side + "']");
        if (!span) return;
        if (span.textContent !== text) {{
          span.textContent = text;
          flash(cell);
        }}
        if (cls !== undefined) {{
          span.classList.remove("green", "red", "yellow", "gray");
          if (cls) span.classList.add(cls);
        }}
      }}
      function fmtPctEdge(e) {{
        if (e === null || e === undefined) return "0";
        const pp = Math.round(e * 100);
        if (pp === 0) return "0";
        return (pp >= 0 ? "+" : "") + pp + "%";
      }}
      function edgeClass(e) {{
        if (e === null || e === undefined) return "gray";
        if (e >= 0.05) return "green";
        if (e > 0) return "yellow";
        if (e <= -0.02) return "red";
        return "gray";
      }}
      snap.watchlist.forEach(function (r) {{
        const tr = rowsByTicker[r.ticker];
        if (!tr) return;  // server added a new row — page reload would catch
        const ya = r.kalshi_yes, na = r.kalshi_no;
        const kyes = (ya !== null && ya !== undefined) ? (ya + "%")
                   : (na !== null && na !== undefined) ? ((100 - na) + "%")
                   : "—";
        const kno  = (na !== null && na !== undefined) ? (na + "%")
                   : (ya !== null && ya !== undefined) ? ((100 - ya) + "%")
                   : "—";
        // My % column removed from the visual set; model_prob_yes
        // still lives on ``r`` for downstream consumers.
        // Edge (reference prob − Kalshi ask, no half-spread). Reference
        // prob prefers Pinnacle when the row ships it (same rule the
        // server-side render uses); falls back to the bot's model.
        const refProb = (r.pinnacle_prob_yes !== null && r.pinnacle_prob_yes !== undefined)
          ? r.pinnacle_prob_yes : r.model_prob_yes;
        const edgeYes = (refProb !== null && refProb !== undefined
                          && ya !== null && ya !== undefined)
          ? (refProb - ya / 100) : null;
        const edgeNo = (refProb !== null && refProb !== undefined
                         && na !== null && na !== undefined)
          ? ((1 - refProb) - na / 100) : null;
        patchCell(tr.querySelector("[data-field='oi']"),
                  r.open_interest !== null && r.open_interest !== undefined
                    ? Math.round(Number(r.open_interest)).toLocaleString() : "—");
        // "kalshi" is the live-Kalshi column; My % / Kalshi-entry %
        // columns were dropped from the visual set (model_prob_yes
        // stays on the payload for JSON consumers).
        const kalshiCell = tr.querySelector("[data-field='kalshi']");
        patchSide(kalshiCell, 'yes', kyes);
        patchSide(kalshiCell, 'no',  kno);
        const edgeCell = tr.querySelector("[data-field='edge']");
        patchSide(edgeCell, 'yes', fmtPctEdge(edgeYes), edgeClass(edgeYes));
        patchSide(edgeCell, 'no',  fmtPctEdge(edgeNo),  edgeClass(edgeNo));
        const evCell = tr.querySelector("[data-field='ev']");
        patchSide(evCell, 'yes', fmtEv(r.ev_yes), evClass(r.ev_yes, minEv));
        patchSide(evCell, 'no',  fmtEv(r.ev_no),  evClass(r.ev_no, minEv));
      }});
    }}
  }}

  function poll() {{
    fetch("/api/snapshot?bot=" + encodeURIComponent(BOT)
          + "&period=" + encodeURIComponent(PERIOD),
          {{cache: "no-store"}})
      .then(function (r) {{ return r.ok ? r.json() : null; }})
      .then(function (snap) {{ if (snap) applySnapshot(snap); }})
      .catch(function () {{ /* swallow — try again next tick */ }});
  }}

  // Initial fetch on load + recurring poll.
  poll();
  setInterval(poll, POLL_MS);

  // ── Bot dropdown (Watchlist tab) + Period dropdowns ─────────────
  // Each <option>'s value carries the target URL; on change we
  // navigate there. Same destinations as the old pill links — the
  // dropdowns are just a quieter UI for the same action. The Period
  // selector appears on both Home and History tabs (one instance
  // each, marked with [data-period-select] so we can wire them all).
  // Bot-selectors (one on Home, one on each per-bot Watchlist tab).
  // All marked with [data-bot-select]; on change we navigate to the
  // option's value. The option values bake in the SERVER-rendered tab
  // (from ?tab= at page load), but tab pills swap panels client-side
  // via history.replaceState — so the option's URL goes stale once
  // the user changes tabs. Re-read the current tab from the URL bar
  // (or fall back to the active tab pill) and override the option's
  // ?tab= so the bot switch keeps the user on whichever tab is
  // currently visible.
  // Browsers default to "auto" scroll restoration on history navigation
  // but they do NOT restore scroll on cross-URL navigation (which is
  // what a bot-select change triggers). Stash the current scrollY in
  // sessionStorage so the next page load can re-apply it.
  if ("scrollRestoration" in history) {{
    history.scrollRestoration = "manual";
  }}
  const _SCROLL_KEY = "dashboardBotSwitchScrollY";
  try {{
    const saved = sessionStorage.getItem(_SCROLL_KEY);
    if (saved !== null) {{
      sessionStorage.removeItem(_SCROLL_KEY);
      // Defer one frame so the layout settles before scrolling — the
      // body keeps growing as deferred SVG charts paint in.
      requestAnimationFrame(function () {{
        window.scrollTo(0, parseInt(saved, 10) || 0);
      }});
    }}
  }} catch (err) {{ /* sessionStorage disabled — ignore */ }}

  document.querySelectorAll("[data-bot-select]").forEach(function (sel) {{
    sel.addEventListener("change", function () {{
      let target = sel.value;
      if (!target) return;
      // The visible tab pill is the authoritative source of truth for
      // what panel the user is looking at — tab clicks swap panels
      // client-side and then call history.replaceState, but that
      // replaceState can lag or skip in edge cases (modal interaction,
      // browser quirks). The pill's .tab-pill-active class is always
      // in sync with the visible panel.
      let currentTab = null;
      const activePill = document.querySelector(".tab-pill-active");
      if (activePill) currentTab = activePill.getAttribute("data-tab");
      // Contracts is a container — the meaningful destination is the
      // active sub-tab (watchlist / models / training). Those legacy
      // keys are what the server maps back to Contracts + sub-tab.
      if (currentTab === "contracts") {{
        const activeSub = document.querySelector(".subtab-pill-active");
        currentTab = (activeSub && activeSub.getAttribute("data-subtab"))
                       || "watchlist";
      }}
      if (!currentTab) {{
        try {{
          currentTab = new URL(window.location.href)
            .searchParams.get("tab");
        }} catch (err) {{ /* old browser */ }}
      }}
      try {{
        // Resolve against the current origin so we can use
        // URLSearchParams regardless of whether the option value
        // starts with "?" or "/". Per-bot options carry ?bot=X — for
        // those we always inject the active tab (overwriting whatever
        // tab the server rendered into the option) so the user stays
        // on the same tab. The "All bots" entry has no ?bot= and is
        // intentionally left alone — it lands on the cross-bot Home.
        const u = new URL(target, window.location.origin);
        if (currentTab && u.searchParams.has("bot")) {{
          u.searchParams.set("tab", currentTab);
          target = u.pathname + u.search;
        }}
      }} catch (err) {{ /* fall through to raw target */ }}
      try {{
        sessionStorage.setItem(_SCROLL_KEY, String(window.scrollY));
      }} catch (err) {{ /* ignore */ }}
      window.location.href = target;
    }});
  }});
  // Period-selectors (Home + History tabs).
  document.querySelectorAll("[data-period-select]").forEach(function (sel) {{
    sel.addEventListener("change", function () {{
      const url = sel.value;
      if (url) window.location.href = url;
    }});
  }});

  // ── "Why?" modal — bet criteria popup ────────────────────────
  // Each .criteria-btn carries data-criteria with the entry-time
  // snapshot. On click we populate one shared modal at the bottom
  // of the page and reveal it; click overlay or × to dismiss.
  const critOverlay = document.getElementById("criteria-overlay");
  const critModal   = document.getElementById("criteria-modal");
  const critBody    = document.getElementById("criteria-modal-body");
  const critTicker  = document.getElementById("criteria-modal-ticker");
  const critClose   = document.getElementById("criteria-close");
  function fmtPct(v) {{
    if (v === null || v === undefined || !isFinite(v)) return "—";
    return (v * 100).toFixed(0) + "%";
  }}
  function fmtCents3(v) {{
    if (v === null || v === undefined || !isFinite(v)) return "—";
    const sign = v >= 0 ? "+" : "−";
    return sign + "$" + Math.abs(v).toFixed(2);
  }}
  function buildCriteriaHTML(c) {{
    // Every value in this popup is rendered green: the bet only
    // exists because each criterion cleared, so every line is a
    // "this passed" datapoint.
    let html = "<div class='crit-section'><h4>Why we took it</h4><dl>";
    html += "<dt>Model probability</dt><dd class='green'>"
         + fmtPct(c.model_p) + "</dd>";
    html += "<dt>Market probability</dt><dd class='green'>"
         + fmtPct(c.kalshi_p) + "</dd>";
    const edgeStr = (c.edge_pts === null || !isFinite(c.edge_pts))
      ? "—"
      : (c.edge_pts >= 0 ? "+" : "−")
        + Math.abs(c.edge_pts).toFixed(0) + " pts";
    html += "<dt>Edge</dt><dd class='green'>" + edgeStr + "</dd>";
    html += "<dt>Entry EV / contract</dt><dd class='green'>"
         + fmtCents3(c.entry_ev) + "</dd>";
    html += "<dt>Break-even probability</dt><dd class='green'>"
         + fmtPct(c.break_even) + "</dd>";
    html += "<dt>Validators met</dt><dd class='green'>100%</dd>";
    html += "</dl></div>";
    return html;
  }}
  function showCriteria(btn) {{
    if (!critOverlay || !critModal) return;
    let data = {{}};
    try {{ data = JSON.parse(btn.dataset.criteria || "{{}}"); }} catch (e) {{}}
    if (critTicker) critTicker.textContent = data.ticker || "";
    if (critBody)   critBody.innerHTML     = buildCriteriaHTML(data);
    critOverlay.hidden = false;
    critModal.hidden   = false;
  }}
  function hideCriteria() {{
    if (critOverlay) critOverlay.hidden = true;
    if (critModal)   critModal.hidden   = true;
  }}
  // Build the "buy criteria + validators" reference popup body from
  // the bot's edge/validator/risk/hedge configs serialised on the
  // button as data-rules.
  function fmtCents(c) {{
    if (c === null || c === undefined || !isFinite(c)) return "—";
    return "$" + (c / 100).toFixed(2);
  }}
  function fmtMin(m) {{
    if (m === null || m === undefined || !isFinite(m)) return "—";
    if (m >= 1440) return (m / 1440).toFixed(0) + "d";
    if (m >= 60)   return (m / 60).toFixed(0) + "h";
    return m + "min";
  }}
  function buildRulesHTML(r) {{
    // Bullet-point overview of every gate the bot runs before buying
    // and exiting. Each bullet renders the gate description AND the
    // actual numeric value the bot uses (pulled from the bot's
    // ``data/effective_config.json`` when present, else the dashboard
    // YAML — the source banner at the top tells the user which).
    // Use the per-position Why? button to see the actual values that
    // cleared each gate at entry-time for a specific bet.
    const ed  = (r && r.edge)       || {{}};
    const va  = (r && r.validators) || {{}};
    const rk  = (r && r.risk)       || {{}};
    const hg  = (r && r.hedge)      || {{}};
    const src = (r && r._source)    || {{}};
    const xt  = (r && r.extra)      || {{}};

    function fmtNum(v, suffix) {{
      if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) {{
        return "—";
      }}
      if (typeof v === "boolean") return v ? "on" : "off";
      if (Array.isArray(v))      return v.join("–");
      return v + (suffix || "");
    }}
    function fmtCash(c) {{
      if (c === null || c === undefined || !isFinite(c)) return "—";
      return "$" + (c / 100).toFixed(2);
    }}
    function fmtPctF(v) {{
      if (v === null || v === undefined || !isFinite(v)) return "—";
      return (v * 100).toFixed(0) + "%";
    }}
    function fmtMinH(m) {{
      if (m === null || m === undefined || !isFinite(m)) return "—";
      if (m >= 1440) return (m / 1440).toFixed(0) + "d";
      if (m >= 60)   return (m / 60).toFixed(0) + "h";
      return m + "min";
    }}
    function fmtSec(s) {{
      if (s === null || s === undefined || !isFinite(s)) return "—";
      if (s >= 3600) return (s / 3600).toFixed(1) + "h";
      if (s >= 60)   return (s / 60).toFixed(0) + "min";
      return s + "s";
    }}
    function valSpan(s) {{
      return "<span style='font-variant-numeric:tabular-nums;"
           + "color:#f0f6fc;font-weight:600;'>" + s + "</span>";
    }}

    // ── New card-based layout ────────────────────────────────────
    // Every rule is a (label, description, chip-value) triple. Cards
    // group related rules under an icon + heading. Chip class hints
    // colour ("pos" green, "neg" red, "info" blue, default neutral).
    // Rules with a null value drop silently, so a bot that doesn't
    // set a given field just doesn't show that row.
    function ruleRow(label, desc, chip, chipCls) {{
      const chipHtml = chip
        ? "<span class='crit-chip " + (chipCls || "") + "'>" + chip + "</span>"
        : "";
      const descHtml = desc
        ? "<p class='crit-rule-desc'>" + desc + "</p>"
        : "";
      return "<div class='crit-rule'>"
           + "<div class='crit-rule-head'>"
           + "<span class='crit-rule-label'>" + label + "</span>"
           + chipHtml
           + "</div>"
           + descHtml
           + "</div>";
    }}
    function ruleCard(iconLetter, iconCls, title, sub, rules) {{
      // Filter out rules where chip is missing/"—"/undefined.
      const kept = rules.filter(function (r) {{
        return r && r[2] != null && r[2] !== "—" && r[2] !== "";
      }});
      if (!kept.length) return "";
      let inner = kept.map(function (r) {{
        return ruleRow(r[0], r[1], r[2], r[3]);
      }}).join("");
      const subHtml = sub
        ? "<span class='crit-card-sub'>" + sub + "</span>"
        : "";
      return "<div class='crit-card'>"
           + "<div class='crit-card-head'>"
           + "<span class='crit-card-icon " + iconCls + "'>" + iconLetter + "</span>"
           + "<span class='crit-card-title'>" + title + "</span>"
           + subHtml
           + "</div>"
           + inner
           + "</div>";
    }}
    let html = "";

    // Compact source pill — was a full-width banner before; now a
    // small chip at the top so it doesn't dominate the modal.
    if (src.source === "live") {{
      const ts = src.captured_at ? " · reported " + src.captured_at : "";
      html += "<div class='crit-source-pill live'>"
           + "<span class='dot'></span>"
           + "Live config" + ts
           + "</div>";
    }} else {{
      html += "<div class='crit-source-pill fallback'>"
           + "<span class='dot'></span>"
           + "Dashboard defaults — bot hasn't reported its live config"
           + "</div>";
    }}

    const isTennis = xt && xt.kind === "tennis";
    const pb = va.prob_bounds_cents;
    const pbStr = (Array.isArray(pb) && pb.length === 2)
      ? pb[0] + "¢–" + pb[1] + "¢" : null;

    // Hero card at the top (tennis-specific for now — it summarises
    // the whole bot at a glance: what's the reference, what's the
    // edge floor, and where's the skip ceiling).
    if (isTennis) {{
      const edgeFloorPct = (ed.min_prob_edge_over_breakeven != null)
        ? fmtPctF(ed.min_prob_edge_over_breakeven) : "—";
      const skipCapPct = (xt.max_edge_skip != null)
        ? fmtPctF(xt.max_edge_skip) : "—";
      const refLabel = (xt.reference_book === "pinnacle_devigged")
        ? "Pinnacle (devigged)" : (xt.reference_book || "—");
      html += "<div class='crit-hero'>"
           + "<p class='crit-hero-lead'>"
           + "This bot buys when Kalshi disagrees with a sharp "
           + "reference book. <b>Every column labelled &lsquo;Edge&rsquo; "
           + "on the watchlist is <span style='color:#3fb950;'>"
           + "reference &minus; Kalshi</span>.</b>"
           + "</p>"
           + "<div class='crit-hero-stats'>"
           + "<div class='crit-hero-stat'>"
           + "<div class='crit-hero-stat-label'>Reference</div>"
           + "<div class='crit-hero-stat-value'>" + refLabel + "</div>"
           + "</div>"
           + "<div class='crit-hero-stat'>"
           + "<div class='crit-hero-stat-label'>Edge floor</div>"
           + "<div class='crit-hero-stat-value'>&ge; " + edgeFloorPct + "</div>"
           + "</div>"
           + "<div class='crit-hero-stat'>"
           + "<div class='crit-hero-stat-label'>Skip ceiling</div>"
           + "<div class='crit-hero-stat-value'>&gt; " + skipCapPct + "</div>"
           + "</div>"
           + "</div></div>";
    }}

    // ── Card 1: Edge & EV ─────────────────────────────────────────
    const edgeRules = [];
    if (isTennis) {{
      // Tennis rewrites label + descriptions to talk about Pinnacle,
      // not "the model". Same underlying threshold — different framing.
      edgeRules.push(
        ["Min edge (Pinnacle vs Kalshi)",
         "The Pinnacle-vs-Kalshi gap on the buy side has to clear this floor. Below it we treat the disagreement as noise and skip.",
         "≥ " + fmtPctF(ed.min_prob_edge_over_breakeven), "pos"],
        ["Strong-edge threshold",
         "The signal label flips SMALL_EDGE → STRONG_EDGE once the gap clears this. Purely a labelling cutoff — doesn't gate trades unless require_strong_edge is on.",
         xt.strong_edge_min != null ? "≥ " + fmtPctF(xt.strong_edge_min) : null,
         "info"],
        ["Max edge skip",
         "Hard ceiling on the gap. Above this we assume one book is stale or broken, not that we've found a huge real edge, and skip the trade entirely.",
         xt.max_edge_skip != null ? "&gt; " + fmtPctF(xt.max_edge_skip) + " → skip" : null,
         "neg"],
        ["Min EV per contract",
         "Expected $ return on a $1 contract after subtracting half-spread and Kalshi's entry fee. Filters trades where fees eat the edge.",
         ed.min_ev_per_contract != null ? "≥ $" + fmtNum(ed.min_ev_per_contract) : null,
         "pos"],
        ["Max entry price",
         "Hard cap on the per-contract price the bot will pay. Above this the loss-if-wrong ($) vs gain-if-right ($) ratio is too punishing even on positive EV.",
         ed.max_entry_price_cents != null ? "≤ " + fmtCash(ed.max_entry_price_cents) : null,
         "neg"],
      );
    }} else {{
      // Non-tennis: original model-based framing.
      edgeRules.push(
        ["Min model confidence",
         "Skip band around 50/50 — the model's blended probability has to land outside this before the bot considers either side.",
         ed.min_model_confidence != null
           ? "skip if p ∈ [" + fmtPctF(ed.min_model_confidence) + ", "
             + fmtPctF(1 - (ed.min_model_confidence || 0)) + "]"
           : null, "info"],
        ["Min EV per contract",
         "Expected $ return on a $1 contract after half-spread. Filters thin-margin trades where slippage eats the edge.",
         ed.min_ev_per_contract != null ? "≥ $" + fmtNum(ed.min_ev_per_contract) : null,
         "pos"],
        ["Min edge over break-even",
         "Buffer above the price-implied break-even probability — the model has to win meaningfully more often than the price says it has to.",
         ed.min_prob_edge_over_breakeven != null ? "≥ " + fmtPctF(ed.min_prob_edge_over_breakeven) : null,
         "pos"],
        ["Min raw model edge",
         "Raw (un-blended) model probability has to clear the ask by this much, so a market-dominated blend can't mask a thin underlying edge.",
         ed.min_raw_model_edge != null ? "≥ " + fmtPctF(ed.min_raw_model_edge) : null,
         "pos"],
        ["Max entry price",
         "Hard cap on the per-contract price the bot will pay.",
         ed.max_entry_price_cents != null ? "≤ " + fmtCash(ed.max_entry_price_cents) : null,
         "neg"],
      );
    }}
    html += ruleCard("1", "edge", "When to buy",
      isTennis ? "Pinnacle-vs-Kalshi edge + EV" : "Model edge + EV",
      edgeRules);

    // ── Card 2: Market health ─────────────────────────────────────
    html += ruleCard("2", "market", "Market must look healthy",
      "Liquidity + price sanity",
      [
        ["Max spread",
         "Ceiling on YES ask − NO ask. Wide spreads mean the book isn't quoting a real price; the bot won't bet into them.",
         va.max_spread_cents != null ? "≤ " + fmtNum(va.max_spread_cents, "¢") : null,
         "info"],
        ["Kalshi price band",
         "Only buy sides currently priced inside this range. Anything cheaper than the low end is deep-underdog / stale-book territory; anything above the high end can't earn enough on a win to justify the risk.",
         pbStr, "info"],
        ["Min open interest",
         "Real positions held by other traders on this contract. Confirms there are counterparties, not just the bot's own echo on a thin book.",
         va.min_open_interest != null ? "≥ " + fmtNum(va.min_open_interest) : null,
         "info"],
        ["Min book depth",
         "Total contracts resting across YES + NO within 3¢ of the touch. Avoids markets where our own order would move the price.",
         va.min_book_depth_contracts != null
           ? "≥ " + fmtNum(va.min_book_depth_contracts) + " contracts" : null,
         "info"],
        ["Min volume",
         "Minimum contracts traded so far. Brand-new markets with zero volume have unreliable mid prices.",
         va.min_volume != null ? "≥ " + fmtNum(va.min_volume) : null, "info"],
        ["Time-to-close window",
         "Trade only when the contract has enough time to play out but not so much that the edge erodes before settle.",
         (va.min_minutes_to_close != null && va.max_minutes_to_close != null)
           ? fmtMinH(va.min_minutes_to_close) + " – " + fmtMinH(va.max_minutes_to_close)
           : null, "info"],
      ]);

    // ── Card 3: Risk / portfolio ──────────────────────────────────
    html += ruleCard("3", "risk", "Portfolio + risk caps",
      "How much, how often",
      [
        ["Fixed bet size",
         "Every position the bot opens is the same $ size, not scaled by edge magnitude.",
         rk.bet_size_cents != null ? fmtCash(rk.bet_size_cents) : null,
         "info"],
        ["Max concurrent positions",
         "Ceiling on simultaneous open contracts. Prevents racking up correlated exposure across the ladder.",
         rk.max_open_positions != null ? "≤ " + fmtNum(rk.max_open_positions) : null,
         "info"],
        ["Max total exposure",
         "$ ceiling on the combined entry cost of all open positions.",
         rk.max_total_exposure_cents != null ? "≤ " + fmtCash(rk.max_total_exposure_cents) : null,
         "info"],
        ["Max bets per day",
         "Throttle on how many fresh positions the bot can open in 24h.",
         rk.max_bets_per_day != null ? "≤ " + fmtNum(rk.max_bets_per_day) : null,
         "info"],
        ["Cooldown on same market",
         "Minimum wait before re-entering a contract after closing it.",
         rk.cooldown_seconds_same_market != null ? "≥ " + fmtSec(rk.cooldown_seconds_same_market) : null,
         "info"],
      ]);

    // ── Card 4: Exit rules ────────────────────────────────────────
    const exitRules = [];
    if (isTennis) {{
      // Tennis exits use a market-prob threshold (profit lock at 95¢+
      // on the side you hold), not the macro-bot cent-lift threshold.
      exitRules.push(
        ["Profit lock (mark price)",
         "Close any open position once its side has drifted to this Kalshi mark or higher. At 95+¢ the remaining upside is rounding error vs the variance of holding to settle.",
         hg.profit_lock_market_prob != null ? "≥ " + fmtPctF(hg.profit_lock_market_prob) : null,
         "pos"],
      );
    }} else {{
      exitRules.push(
        ["Auto-hedger",
         "Kill switch for the exit monitor. Off = positions ride to settlement.",
         hg.enabled != null ? (hg.enabled ? "on" : "off") : null, "info"],
        ["Profit-lock",
         "Close once the mark has gained this many cents above entry.",
         hg.profit_lock_cents != null ? "+" + fmtNum(hg.profit_lock_cents, "¢") : null,
         "pos"],
        ["Stop-loss",
         "Close once the mark has dropped this many cents below entry.",
         hg.stop_loss_cents != null ? "−" + fmtNum(hg.stop_loss_cents, "¢") : null,
         "neg"],
        ["Hedge size fraction",
         "Fraction of the position to close when a trigger fires. Full-exit vs partial scale-off.",
         hg.hedge_size_fraction != null ? fmtPctF(hg.hedge_size_fraction) : null, "info"],
      );
    }}
    html += ruleCard("4", "exit", "When we exit", "", exitRules);

    // Footer — micro note about the per-bet Why? popup that shows the
    // actual entry-time values for a specific position.
    html += "<div class='crit-foot'>"
         + "Every check above must pass on the chosen side before a "
         + "buy fires. Click <b>Why?</b> on an open position to see "
         + "the exact numbers that cleared each gate for that bet at "
         + "entry-time."
         + "</div>";

    return html;
  }}
  function showRules(btn) {{
    if (!critOverlay || !critModal) return;
    // Prefer per-button data-rules payload; fall back to the global
    // window.__BUY_CRITERIA__ stash so callers without explicit
    // configs still open the same modal.
    let data = (window.__BUY_CRITERIA__ || {{}});
    try {{
      const local = btn.dataset && btn.dataset.rules
        ? JSON.parse(btn.dataset.rules) : null;
      if (local && Object.keys(local).length) data = local;
    }} catch (e) {{}}
    const h3 = critModal.querySelector("h3");
    if (h3) h3.textContent = "Buy criteria & validators";
    if (critTicker) critTicker.textContent = "";
    if (critBody)   critBody.innerHTML = buildRulesHTML(data);
    critOverlay.hidden = false;
    critModal.hidden   = false;
  }}

  // Contract-rules popup body. Kalshi's API only exposes a short
  // rules_primary string (and sometimes a rules_secondary block);
  // the *full* rules live in Kalshi's web UI. Render a clear link
  // to the market's "View full rules" page on Kalshi as the primary
  // affordance, with the cached short text shown as quick context.
  function buildContractRulesHTML(d) {{
    let html = "";
    if (d.kalshi_url) {{
      html += "<div class='crit-section'>"
           + "<a href='" + d.kalshi_url + "' target='_blank' "
           + "rel='noopener noreferrer' "
           + "style='display:inline-block;padding:8px 14px;"
           + "background:#1f6feb;color:#fff;text-decoration:none;"
           + "border-radius:6px;font-weight:600;font-size:13px;'>"
           + "View full rules on Kalshi ↗</a>"
           + "<div class='gray' style='font-size:11px;margin-top:6px;'>"
           + "Kalshi's web UI shows the complete rules text, including "
           + "settlement sources and edge cases.</div></div>";
    }}
    if (d.primary) {{
      html += "<div class='crit-section'><h4>Quick summary</h4>"
           + "<div style='font-size:13px;line-height:1.6;color:#c9d1d9;'>"
           + d.primary.split(/\\n/).map(function (p) {{
               return "<p style='margin:0 0 10px 0;'>" + p + "</p>";
             }}).join("")
           + "</div></div>";
    }}
    if (d.secondary) {{
      html += "<div class='crit-section'><h4>Additional details</h4>"
           + "<div style='font-size:13px;line-height:1.6;color:#c9d1d9;'>"
           + d.secondary.split(/\\n/).map(function (p) {{
               return "<p style='margin:0 0 10px 0;'>" + p + "</p>";
             }}).join("")
           + "</div></div>";
    }}
    if (!html) {{
      html = "<div class='crit-section'>"
           + "<span class='gray'>No rules cached yet.</span></div>";
    }}
    return html;
  }}
  function showContractRules(btn) {{
    if (!critOverlay || !critModal) return;
    let data = {{}};
    try {{
      data = JSON.parse(btn.dataset.contractRules || "{{}}");
    }} catch (e) {{}}
    const h3 = critModal.querySelector("h3");
    if (h3) h3.textContent = "Contract rules";
    if (critTicker) critTicker.textContent = "";
    if (critBody)   critBody.innerHTML = buildContractRulesHTML(data);
    critOverlay.hidden = false;
    critModal.hidden   = false;
  }}

  document.addEventListener("click", function (e) {{
    const contractBtn = e.target.closest(".contract-rules-btn");
    if (contractBtn) {{
      e.preventDefault();
      showContractRules(contractBtn);
      return;
    }}
    const ruleBtn = e.target.closest(".criteria-rules-btn");
    if (ruleBtn) {{
      e.preventDefault();
      showRules(ruleBtn);
      return;
    }}
    const btn = e.target.closest(".criteria-btn");
    if (btn) {{
      e.preventDefault();
      // Restore the per-bet header — the rules popup may have changed it.
      const h3 = critModal && critModal.querySelector("h3");
      if (h3) h3.textContent = "Why was this bet chosen?";
      showCriteria(btn);
    }}
  }});
  if (critOverlay) critOverlay.addEventListener("click", hideCriteria);
  if (critClose)   critClose.addEventListener("click", hideCriteria);
  document.addEventListener("keydown", function (e) {{
    if (e.key === "Escape") hideCriteria();
  }});

  // ── Daily droplet diagnosis modal ───────────────────────────────
  // Fetched on-demand from /api/diagnosis/latest (the dashboard's
  // do_GET reads diagnosis/latest.json on disk — written by the
  // scheduled daily-droplet-diagnosis agent). Renders an empty state
  // when no report exists yet so the button works on a fresh install
  // before the first run completes.
  const diagOverlay = document.getElementById("diagnosis-overlay");
  const diagModal   = document.getElementById("diagnosis-modal");
  const diagBody    = document.getElementById("diagnosis-modal-body");
  const diagMeta    = document.getElementById("diagnosis-modal-meta");
  const diagClose   = document.getElementById("diagnosis-close");
  function diagEsc(s) {{
    if (s === null || s === undefined) return "";
    return String(s).replace(/[&<>"']/g, function (c) {{
      return ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}})[c];
    }});
  }}
  function diagRenderItems(title, items) {{
    if (!items || !items.length) return "";
    let h = "<div class='diagnosis-section'><h4>" + diagEsc(title) + "</h4>";
    for (const it of items) {{
      h += "<div class='diagnosis-item'>";
      // Header row: WHAT (bold) + count badge if grouped + last-seen
      // time if known. Service name moved to the right so the
      // important text starts at the left edge.
      h += "<div class='diagnosis-what-row'>";
      h += "<span class='diagnosis-what'>" + diagEsc(it.what || "")
         + "</span>";
      if (it.count && it.count > 1) {{
        h += " <span class='diagnosis-count' title='Times this exact "
           + "error has fired since the last service restart'>"
           + diagEsc(String(it.count)) + "\\u00d7</span>";
      }}
      if (it.last_seen) {{
        h += " <span class='diagnosis-last-seen' title='Most recent "
           + "occurrence'>last seen "
           + diagEsc(it.last_seen.replace("T", " ")) + "</span>";
      }}
      h += "</div>";
      if (it.where) h += "<div class='diagnosis-where'>" + diagEsc(it.where) + "</div>";
      if (it.evidence) h += "<details class='diagnosis-evidence-wrap'><summary>Show details</summary><pre class='diagnosis-evidence'>" + diagEsc(it.evidence) + "</pre></details>";
      // Suggested_fix only renders when there's actually a useful
      // suggestion. The pre-rewrite collector emitted boilerplate on
      // every row ("(automated collector — no fix proposed)") which
      // added noise without information.
      if (it.suggested_fix) h += "<div class='diagnosis-fix'>\\u21b3 " + diagEsc(it.suggested_fix) + "</div>";
      h += "</div>";
    }}
    h += "</div>";
    return h;
  }}
  function diagRenderHeadline(d) {{
    // Plain-English top line so the user knows the state at a glance
    // without scanning the table. Pluralisation matters here — "1
    // service healthy" reads weird, but so does "0 issues" with
    // emphasis. Cover the three common shapes explicitly.
    const audited = d.services_audited || 0;
    const healthy = d.services_healthy || 0;
    const issues  = d.issues_found || 0;
    const broken  = audited - healthy;
    let icon, palette, line;
    if (audited === 0) {{
      icon = "\\u2014"; palette = "neutral";
      line = "No services audited yet.";
    }} else if (broken === 0 && issues === 0) {{
      icon = "\\u2713"; palette = "ok";
      line = audited === 1
        ? "Everything looks healthy."
        : "All " + audited + " services healthy.";
    }} else if (broken === 0) {{
      icon = "!"; palette = "warn";
      line = issues + " issue" + (issues === 1 ? "" : "s")
           + " worth a look. Services themselves are running.";
    }} else {{
      icon = "!"; palette = "bad";
      line = broken + " of " + audited + " service"
           + (audited === 1 ? "" : "s") + " unhealthy"
           + (issues ? ", " + issues + " issue"
                       + (issues === 1 ? "" : "s") + " surfaced." : ".");
    }}
    return "<div class='diagnosis-headline diagnosis-headline-"
         + palette + "'><span class='diagnosis-headline-icon'>"
         + icon + "</span> " + diagEsc(line) + "</div>";
  }}
  function diagRenderServices(services) {{
    if (!services || !services.length) return "";
    let h = "<div class='diagnosis-section'><h4>Service health</h4>";
    h += "<table class='diagnosis-services-table'><thead><tr>";
    h += "<th>Service</th><th>Status</th><th>Restarts (24h)</th><th>Notable</th>";
    h += "</tr></thead><tbody>";
    for (const s of services) {{
      const status = s.status || "unknown";
      h += "<tr>";
      h += "<td>" + diagEsc(s.name) + "</td>";
      h += "<td class='status-" + diagEsc(status) + "'>" + diagEsc(status) + "</td>";
      h += "<td>" + (s.restarts_24h == null ? "\\u2014" : s.restarts_24h) + "</td>";
      h += "<td>" + diagEsc(s.notable || "") + "</td>";
      h += "</tr>";
    }}
    h += "</tbody></table></div>";
    return h;
  }}
  function diagBuildHTML(d) {{
    if (!d || d.status === "no_report_yet") {{
      return "<div class='diagnosis-empty'>No diagnosis report yet."
           + "<br><br>The scheduled <code>daily-droplet-diagnosis</code> "
           + "agent hasn't run yet, or <code>diagnosis/latest.json</code> "
           + "is missing on this host.</div>";
    }}
    if (d.status === "skipped_no_engagement") {{
      let h = "<div class='diagnosis-empty'>"
            + "Skipped \\u2014 no commits to any monitored repo since the "
            + "last report. Waiting on you to act on prior recommendations."
            + "<br><br>Last full report: "
            + diagEsc(d.generated_at || "\\u2014") + "</div>";
      if (d.github_issue_url) {{
        h += "<div style='text-align:center'><a class='diagnosis-github-link' href='"
           + diagEsc(d.github_issue_url) + "' target='_blank' rel='noopener'>"
           + "View previous report on GitHub \\u2192</a></div>";
      }}
      return h;
    }}
    if (d.status === "failed") {{
      return "<div class='diagnosis-empty' style='color:#f85149'>"
           + "Diagnosis run failed.<br><br>"
           + diagEsc(d.error || "(no error message)") + "</div>";
    }}
    let h = "";
    h += diagRenderHeadline(d);
    h += diagRenderServices(d.services);
    h += diagRenderItems("Errors caught since last restart", d.bugs);
    h += diagRenderItems("Recommended changes", d.recommended_changes);
    h += diagRenderItems("Noisy log patterns", d.streamlining);
    const noFindings = (!d.bugs || !d.bugs.length)
                    && (!d.recommended_changes || !d.recommended_changes.length)
                    && (!d.streamlining || !d.streamlining.length);
    if (noFindings) {{
      h += "<div class='diagnosis-empty'>No further findings \\u2014 nothing else to report.</div>";
    }}
    if (d.github_issue_url) {{
      h += "<div style='margin-top:14px'><a class='diagnosis-github-link' href='"
         + diagEsc(d.github_issue_url) + "' target='_blank' rel='noopener'>"
         + "View this report on GitHub \\u2192</a></div>";
    }}
    return h;
  }}
  function diagUpdateDots(d) {{
    let cls = "stale";
    if (d && d.status === "completed") {{
      cls = (d.issues_found && d.issues_found > 0) ? "has-issues" : "healthy";
    }} else if (d && d.status === "skipped_no_engagement") {{
      cls = "stale";
    }} else if (d && d.status === "failed") {{
      cls = "has-issues";
    }}
    document.querySelectorAll("[data-diagnosis-dot]").forEach(function (el) {{
      el.classList.remove("healthy", "has-issues", "stale");
      el.classList.add(cls);
    }});
  }}
  function diagShow() {{
    if (!diagOverlay || !diagModal) return;
    if (diagBody) diagBody.innerHTML = "<div class='diagnosis-empty'>Loading\\u2026</div>";
    if (diagMeta) diagMeta.textContent = "";
    diagOverlay.hidden = false;
    diagModal.hidden = false;
    fetch("/api/diagnosis/latest", {{cache: "no-store"}})
      .then(function (r) {{ return r.json(); }})
      .then(function (d) {{
        if (diagBody) diagBody.innerHTML = diagBuildHTML(d);
        if (diagMeta) {{
          const when = d.generated_at || d.last_checked_at || "\\u2014";
          let counts = "";
          if (typeof d.issues_found === "number") {{
            counts = " \\u00b7 " + d.issues_found + " issue"
                   + (d.issues_found === 1 ? "" : "s");
          }}
          diagMeta.textContent = "Generated " + when + counts;
        }}
        diagUpdateDots(d);
      }})
      .catch(function (e) {{
        if (diagBody) diagBody.innerHTML =
          "<div class='diagnosis-empty' style='color:#f85149'>"
          + "Failed to load diagnosis: " + diagEsc(String(e)) + "</div>";
      }});
  }}
  function diagHide() {{
    if (diagOverlay) diagOverlay.hidden = true;
    if (diagModal)   diagModal.hidden   = true;
  }}
  document.querySelectorAll("[data-diagnosis-trigger]").forEach(function (btn) {{
    btn.addEventListener("click", diagShow);
  }});
  if (diagClose)   diagClose.addEventListener("click", diagHide);
  if (diagOverlay) diagOverlay.addEventListener("click", diagHide);
  document.addEventListener("keydown", function (e) {{
    if (e.key === "Escape") diagHide();
  }});
  // Prime the status dot on page load so the button reflects current
  // health before the user clicks. Same endpoint, no UI side-effects.
  fetch("/api/diagnosis/latest", {{cache: "no-store"}})
    .then(function (r) {{ return r.json(); }})
    .then(diagUpdateDots)
    .catch(function () {{}});

  // ── Tab switcher ────────────────────────────────────────────────
  // Clicks on a tab pill toggle the .tab-pill-active class on the bar
  // and the .tab-panel-active class on the matching panel. Updates
  // ?tab=X via history.replaceState so reloads + the period filter
  // preserve the active tab.
  const tabBar = document.querySelector(".tab-bar");
  if (tabBar) {{
    tabBar.querySelectorAll(".tab-pill").forEach(function (pill) {{
      pill.addEventListener("click", function (e) {{
        e.preventDefault();
        const key = pill.getAttribute("data-tab");
        if (!key) return;
        // Special-case History: full-page navigate WITHOUT preserving
        // the period — the History tab should default to all-time
        // every time the user opens it. Other tabs JS-swap (snappy
        // and preserves period from elsewhere on the page).
        if (key === "history") {{
          window.location.href = "?tab=history";
          return;
        }}
        // Seasons is also cross-bot — no per-bot view, so navigate to
        // a clean ?tab=seasons URL (drops ?bot= and ?period=). Avoids
        // showing an empty panel when the user clicks Seasons from a
        // bot-scoped page that doesn't render the panel.
        if (key === "seasons") {{
          window.location.href = "?tab=seasons";
          return;
        }}
        tabBar.querySelectorAll(".tab-pill").forEach(function (p) {{
          p.classList.toggle("tab-pill-active",
                              p.getAttribute("data-tab") === key);
        }});
        document.querySelectorAll(".tab-panel").forEach(function (panel) {{
          panel.classList.toggle("tab-panel-active",
                                   panel.getAttribute("data-panel") === key);
        }});
        // Selecting Contracts always lands on the Watchlist sub-tab
        // (per user request) — reset the sub-tab state on every
        // top-level Contracts click, not just the first.
        let urlKey = key;
        if (key === "contracts") {{
          urlKey = "watchlist";
          document.querySelectorAll(".subtab-pill").forEach(function (p) {{
            p.classList.toggle("subtab-pill-active",
                                p.getAttribute("data-subtab") === "watchlist");
          }});
          document.querySelectorAll(".subtab-panel").forEach(function (sp) {{
            sp.classList.toggle("subtab-panel-active",
                                 sp.getAttribute("data-subpanel") === "watchlist");
          }});
        }}
        try {{
          const url = new URL(window.location.href);
          url.searchParams.set("tab", urlKey);
          history.replaceState(null, "", url.toString());
        }} catch (err) {{ /* old browser; skip */ }}
      }});
    }});
  }}

  // ── Contracts sub-tab switcher ──────────────────────────────────
  // Watchlist / Model / Training Data live inside the Contracts
  // panel. Clicks swap the visible sub-panel client-side and write
  // the LEGACY key (?tab=watchlist|models|training) to the URL so
  // reloads and every existing deep link land on the right sub-tab.
  document.querySelectorAll(".subtab-pill").forEach(function (pill) {{
    pill.addEventListener("click", function (e) {{
      e.preventDefault();
      const key = pill.getAttribute("data-subtab");
      if (!key) return;
      document.querySelectorAll(".subtab-pill").forEach(function (p) {{
        p.classList.toggle("subtab-pill-active",
                            p.getAttribute("data-subtab") === key);
      }});
      document.querySelectorAll(".subtab-panel").forEach(function (sp) {{
        sp.classList.toggle("subtab-panel-active",
                             sp.getAttribute("data-subpanel") === key);
      }});
      try {{
        const url = new URL(window.location.href);
        url.searchParams.set("tab", key);
        history.replaceState(null, "", url.toString());
      }} catch (err) {{ /* old browser; skip */ }}
    }});
  }});

  // ── Hover crosshair on the underlying chart ───────────────────
  // The SVG carries data-* attrs with t_min/t_max + chart geometry.
  // On mousemove we draw a vertical line and position a "May 1 at 9 AM"
  // tooltip; on mouseleave we hide them. Pure DOM, no chart library.
  document.querySelectorAll(".wl-chart-wrap").forEach(function (wrap) {{
    const svg = wrap.querySelector("svg");
    const tip = wrap.querySelector(".wl-chart-tooltip");
    if (!svg || !tip) return;
    const tmin = parseFloat(wrap.dataset.tmin);
    const tmax = parseFloat(wrap.dataset.tmax);
    const padL = parseFloat(wrap.dataset.padl);
    const innerW = parseFloat(wrap.dataset.innerw);
    const padT = parseFloat(wrap.dataset.padt);
    const padB = parseFloat(wrap.dataset.padb);
    const h = parseFloat(wrap.dataset.h);
    const vbW = parseFloat(wrap.dataset.vbw);
    if (![tmin, tmax, padL, innerW, padT, padB, h, vbW].every(isFinite)) return;

    const ns = "http://www.w3.org/2000/svg";
    // Dim overlay covers the right portion of the chart (everything
    // past the cursor) so the line in that region greys out as the
    // user "scrubs" through. Appended before the cursor line so it
    // sits beneath it in z-order. Using the panel background color
    // (#0d1117) at 0.65 opacity gives a clean greyed-out look without
    // hiding the line entirely.
    const dimRect = document.createElementNS(ns, "rect");
    dimRect.setAttribute("y", padT);
    dimRect.setAttribute("height", h - padB - padT);
    dimRect.setAttribute("fill", "#0d1117");
    dimRect.setAttribute("opacity", "0");
    dimRect.setAttribute("pointer-events", "none");
    svg.appendChild(dimRect);

    const cursor = document.createElementNS(ns, "line");
    cursor.setAttribute("stroke", "#c9d1d9");
    cursor.setAttribute("stroke-width", "1");
    cursor.setAttribute("stroke-dasharray", "2,3");
    cursor.setAttribute("opacity", "0");
    cursor.setAttribute("pointer-events", "none");
    svg.appendChild(cursor);

    // The hero forecast price + change-indicator elements (top-left
    // of the chart card). While the user scrubs the chart, we swap
    // the price for the value at the cursor AND the change indicator
    // for (cursor − earliest); on mouseleave we restore both from
    // their data-current-text / data-current-class attrs.
    const hero = wrap.closest(".wl-hero");
    const heroPrice = hero ? hero.querySelector(".wl-hero-price") : null;
    const heroPriceText = heroPrice ? heroPrice.querySelector(
        ".wl-hero-price-text") : null;
    const heroChange = hero ? hero.querySelector(".wl-hero-change") : null;

    function fmtTs(ts) {{
      const d = new Date(ts * 1000);
      const months = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"];
      const month = months[d.getUTCMonth()];
      const day = d.getUTCDate();
      let hour = d.getUTCHours();
      const minute = d.getUTCMinutes();
      const ampm = hour >= 12 ? "PM" : "AM";
      hour = hour % 12 || 12;
      const minStr = minute === 0 ? "" : ":" + (minute < 10 ? "0" : "") + minute;
      return month + " " + day + " at " + hour + minStr + " " + ampm;
    }}

    // (ts, value) pairs + the bot's display formatting for the hover
    // tooltip. Always snaps to the nearest recorded point so scrubbing
    // anywhere across the chart shows the time + value of the closest
    // forecast Kalshi recorded — no tolerance check, no interpolation.
    let points = [];
    let fmt = {{ divisor: 1.0, decimals: 2, unit: "", unit_position: "prefix" }};
    try {{ points = JSON.parse(wrap.dataset.points || "[]"); }} catch (e) {{}}
    try {{ fmt = Object.assign(fmt, JSON.parse(wrap.dataset.fmt || "{{}}")); }}
    catch (e) {{}}
    // Earliest recorded value — anchor for the (Δ from start of chart)
    // delta the change indicator displays. Computed AFTER points are
    // parsed (let-declared above; would TDZ-throw if accessed earlier).
    const earliestValue = points.length ? points[0][1] : null;

    function fmtValue(raw) {{
      if (raw === null || raw === undefined || !isFinite(raw)) return "—";
      const v = raw / (fmt.divisor || 1);
      const n = v.toLocaleString("en-US", {{
        minimumFractionDigits: fmt.decimals,
        maximumFractionDigits: fmt.decimals,
      }});
      if (fmt.unit_position === "prefix") return (fmt.unit || "") + n;
      if (fmt.unit_position === "suffix") return n + (fmt.unit || "");
      return n;
    }}

    // Snap to the nearest recorded point; always returns one as long
    // as the points array is non-empty. The cursor's mapping is to the
    // closest recorded forecast — both the popup date stamp and value
    // come from that point, so they always agree.
    function nearestPoint(ts) {{
      if (!points.length) return null;
      let lo = 0, hi = points.length - 1;
      if (ts <= points[lo][0]) {{
        return {{ ts: points[lo][0], value: points[lo][1] }};
      }}
      if (ts >= points[hi][0]) {{
        return {{ ts: points[hi][0], value: points[hi][1] }};
      }}
      while (hi - lo > 1) {{
        const mid = (lo + hi) >> 1;
        if (points[mid][0] <= ts) lo = mid; else hi = mid;
      }}
      const dLo = Math.abs(ts - points[lo][0]);
      const dHi = Math.abs(ts - points[hi][0]);
      const closer = dLo <= dHi ? points[lo] : points[hi];
      return {{ ts: closer[0], value: closer[1] }};
    }}

    // Format an absolute delta in the bot's units, no sign. The arrow
    // (▲/▼) is added separately so we can color it via class.
    function fmtDeltaAbs(raw) {{
      if (raw === null || raw === undefined || !isFinite(raw)) return "—";
      const v = Math.abs(raw) / (fmt.divisor || 1);
      const n = v.toLocaleString("en-US", {{
        minimumFractionDigits: fmt.decimals,
        maximumFractionDigits: fmt.decimals,
      }});
      if (fmt.unit_position === "prefix") return (fmt.unit || "") + n;
      if (fmt.unit_position === "suffix") return n + (fmt.unit || "");
      return n;
    }}

    function restoreHero() {{
      if (heroPrice && heroPriceText) {{
        heroPriceText.textContent = heroPrice.dataset.currentText || "";
      }}
      if (heroChange) {{
        heroChange.textContent = heroChange.dataset.currentText || "";
        const cls = heroChange.dataset.currentClass || "";
        heroChange.className = "wl-hero-change" + (cls ? " " + cls : "");
      }}
    }}

    svg.addEventListener("mousemove", function (e) {{
      const rect = svg.getBoundingClientRect();
      // Cursor's x in viewBox space (the SVG scales to the wrap's width).
      const x = (e.clientX - rect.left) * vbW / rect.width;
      if (x < padL || x > padL + innerW) {{
        cursor.setAttribute("opacity", "0");
        dimRect.setAttribute("opacity", "0");
        tip.hidden = true;
        restoreHero();
        return;
      }}
      cursor.setAttribute("x1", x);
      cursor.setAttribute("x2", x);
      cursor.setAttribute("y1", padT);
      cursor.setAttribute("y2", h - padB);
      cursor.setAttribute("opacity", "0.7");
      // Grey out the line to the right of the cursor.
      dimRect.setAttribute("x", x);
      dimRect.setAttribute("width", Math.max(0, padL + innerW - x));
      dimRect.setAttribute("opacity", "0.65");

      const frac = (x - padL) / innerW;
      const cursorTs = tmin + frac * (tmax - tmin);
      const np = nearestPoint(cursorTs);
      // When a recorded point is in range, stamp the popup AND swap
      // the hero forecast price for the value at the cursor. When no
      // point is near (gap in the data) we still show the cursor date
      // in the popup, but leave the hero on the live current so we
      // never display an unsourced value up top.
      if (np !== null) {{
        tip.innerHTML =
          "<div class='wl-chart-tip-time'>" + fmtTs(np.ts) + "</div>"
          + "<div class='wl-chart-tip-value'>" + fmtValue(np.value) + "</div>";
        if (heroPriceText) heroPriceText.textContent = fmtValue(np.value);
        // Update the ▲/▼ change indicator to (cursor value − earliest)
        // — same Δ semantics as the live indicator, just anchored to
        // wherever the user is hovering instead of "now".
        if (heroChange && earliestValue !== null) {{
          const delta = np.value - earliestValue;
          const arrow = delta >= 0 ? "▲" : "▼";
          const cls = delta >= 0 ? "pos" : "neg";
          heroChange.textContent = arrow + " " + fmtDeltaAbs(delta);
          heroChange.className = "wl-hero-change " + cls;
        }}
      }} else {{
        tip.innerHTML =
          "<div class='wl-chart-tip-time'>" + fmtTs(cursorTs) + "</div>";
        restoreHero();
      }}
      tip.hidden = false;
      // Anchor the tooltip in pixel space relative to the wrap so it
      // tracks the cursor regardless of how the SVG is scaled.
      const ratio = rect.width / vbW;
      tip.style.left = (x * ratio) + "px";
    }});

    svg.addEventListener("mouseleave", function () {{
      cursor.setAttribute("opacity", "0");
      dimRect.setAttribute("opacity", "0");
      restoreHero();
      tip.hidden = true;
    }});
  }});
}})();
</script>"""
