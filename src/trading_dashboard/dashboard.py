"""Compatibility facade for the split dashboard package.

The original 14k-line dashboard.py was split into data / fmt /
css / page / panels / models_panel / watchlist_panel / server
modules (2026-07-13). Sibling modules and the droplet entry
script import names from ``trading_dashboard.dashboard`` —
this facade re-exports every top-level name so those imports
keep working unchanged.
"""
from __future__ import annotations

from .data import (  # noqa: F401
    DEFAULT_BANKROLL_CENTS,
    _KALSHI_HISTORY_CACHE,
    _KALSHI_HISTORY_TTL_S,
    _bot_ticker_prefix_index,
    _build_global_active_bets,
    _compute_active_bets_totals,
    _conn,
    _fetch_all_kalshi_history,
    _iso_to_unix,
    _kalshi_held_for_snapshot,
    _live_kalshi_held_tickers,
    _load_sim_state_enrichment,
    _merge_kalshi_with_local,
    _safe_query,
    _sibling_sim_db,
    _summarize_fills_by_ticker,
    _tennis_like_snapshot,
    _ticker_to_bot,
    _watchlist_from_kalshi,
    bot_regime_status,
    build_kalshi_cross_bot_history,
    build_snapshot,
    fetch_active_bets_with_marks,
    fetch_bet_history,
    fetch_bot_effective_config,
    fetch_global_summary,
    fetch_latest_model,
    fetch_latest_open_position,
    fetch_summary,
    fetch_ticker_yes_prob_history,
    fetch_underlying_history,
    fetch_watchlist,
    pick_recent_market_view_ticker,
    resolve_bot_thresholds,
)
from .fmt import (  # noqa: F401
    _BASKETBALL_EVENT_RE,
    _MONTH_MAP,
    _RULES_DATE_RE,
    _TENNIS_EVENT_RE,
    _TICKER_DATE_RE,
    _basketball_event_label,
    _empty_chart_frame,
    _ev_status,
    _favicon_link,
    _market_date_label,
    _match_text_from_ticker,
    _side_tricode_from_ticker,
    _sport_event_label,
    _tennis_event_label,
    cents_or_dash,
    fmt_signed_cents,
    fmt_underlying,
    kalshi_fee_cents,
    minutes_to_close_from_ticker,
    question_str,
    svg_kalshi_chart,
    ticker_cell_html,
    ticker_link_html,
    time_left_str,
    time_to_close_str,
    unrealized_pnl_cents,
)
from .css import (  # noqa: F401
    CSS,
)
from .page import (  # noqa: F401
    _BOT_TOGGLE_JS,
    _HISTORY_CHART_JS,
    _SEASON_COUNTDOWN_JS,
    _live_update_script,
    render_page,
)
from .panels import (  # noqa: F401
    PERIOD_OPTIONS,
    _humanize_countdown,
    _humanize_duration,
    _parse_season_dt,
    _period_days,
    _render_active_bets_table,
    _render_bet_history_block,
    _render_bot_cards,
    _render_bot_filter,
    _render_bot_unavailable,
    _render_history_attribution,
    _render_history_chart,
    _render_home_summary_cards,
    _render_notifications_panel,
    _render_period_filter,
    _render_seasons_panel,
    _render_summary,
    _render_summary_cards,
    _week_change_pct,
)
from .models_panel import (  # noqa: F401
    FEATURE_RULES,
    _FEATURE_BASES,
    _FEATURE_DETAIL_CSS_JS,
    _INGAME_COEFFICIENTS,
    _NBA_STAT_DESCRIPTIONS,
    _base_description,
    _describe_feature,
    _describe_nba_rolling,
    _detect_bot_cadence,
    _find_training_artifact,
    _holdout_confidence,
    _ingame_backtest_rows,
    _ingame_proxy_metrics,
    _period_unit,
    _read_feature_importance,
    _read_holdout_predictions,
    _readable_feature_name,
    _render_feature_source_table,
    _render_ingame_backtest,
    _render_ingame_coefficients,
    _render_ingame_model_view,
    _render_ingame_predictions_log,
    _render_model_view_toggle,
    _render_models_panel,
    _render_models_run_table,
    _strip_transform_suffix,
    _svg_calibration,
    feature_metadata,
)
from .watchlist_panel import (  # noqa: F401
    _EV_INFO_POPOVER_HTML_JS,
    _RULES_INFO_POPOVER_HTML_JS,
    _WATCHLIST_ROW_CLICK_JS,
    _fmt_signed_underlying,
    _render_current_prediction,
    _render_watchlist,
    _render_watchlist_hero,
)
from .server import (  # noqa: F401
    Handler,
    main,
    serve,
)

import logging
log = logging.getLogger("dashboard")
