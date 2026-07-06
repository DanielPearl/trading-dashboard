"""Dashboard config loader.

The dashboard is a read-only viewer over each bot's SQLite DB. It does
not need Kalshi credentials, model artifacts, or any of the gas-bot's
runtime config. All it needs is:

    - a list of bots (key, display name, paths to sim.db / decisions log)
    - the display thresholds shown on the buy-criteria panel
    - HTTP host/port

Everything is loaded from a single YAML (default: config/dashboard.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class DisplayCfg:
    """Per-bot display knobs for the watchlist hero header / chart.

    The dashboard's `model_snapshots.current_gas_price` column is shared
    across bots but the underlying it represents differs (USD/gal,
    USD/MMBtu, thousands of jobless claims). Each bot tells the
    dashboard how to format its value here.
    """
    underlying_label: str = "Underlying"
    underlying_unit: str = "$"
    underlying_decimals: int = 2
    # "prefix" — render as "$2.759"; "suffix" — "189.0K"; "none" — bare number.
    unit_position: str = "prefix"
    # Divide the raw model_snapshots.current_gas_price by this before
    # formatting. Lets unemployment-claims (which stores 189000) render
    # as "189K" without changing the bot's writing schema.
    divisor: float = 1.0
    # Chart sample period in minutes. 60 = hourly Kalshi candles. Set
    # lower (e.g., 15) and the dashboard fetches Kalshi's minute-level
    # candles and downsamples server-side. Use only for series where
    # intraday market activity is dense enough to populate finer bars.
    chart_period_minutes: int = 60
    # Forecast cadence label used in "Predicted ___" cards on the
    # watchlist page. Weekly bots (gas, claims) keep the default; the
    # monthly CPI bot overrides this to "next month".
    prediction_period_label: str = "next week"
    # When true, the watchlist Title column uses the Kalshi event
    # title (e.g. "Initial jobless claims for the week ending May 9,
    # 2026") instead of the per-strike market title. Used by the
    # unemployment-claims bot where every row in the table is the
    # same event and the per-strike "200K"-shorthand title would
    # repeat the strike already shown in the Question column.
    watchlist_title_use_event: bool = False
    # Watchlist Question column format. Default uses the bot's
    # underlying_unit / divisor pair via fmt_underlying. Set to
    # "at_least_full" for "at least 200,000" / "below 200,000" /
    # "200,000 – 205,000" idiom (raw count, comma-separated, no
    # divisor / unit).
    question_format: str = ""


@dataclass
class SeasonCfg:
    """One tournament / season window for the Seasons tab.

    Bots whose markets follow a real-world calendar (NBA regular season,
    Survivor cycle, PDC events, ATP/WTA majors, etc.) declare one or
    more windows here. The Seasons tab renders one card per (bot,
    window) pair plus a live countdown that flips from "starts in" →
    "ends in" → "season over" as time crosses ``start`` and ``end``.
    ``start`` / ``end`` accept any ISO-8601 string PyYAML parses to a
    datetime (e.g. ``2026-02-25T01:00:00Z``).
    """
    name: str
    start: str
    end: str


@dataclass
class BotEntry:
    key: str
    name: str
    db_path: str
    decisions_path: str
    # "standard" → render the macro-bot page (model card, watchlist,
    # positions) reading from sim.db.
    # "sport" → render the head-to-head match page driven by
    # watchlist_json_path + sim_state_path. Used by tennis,
    # table-tennis, darts (which write the JSONs directly) and NBA
    # (whose sim.db is translated to the sport schema by
    # bots/_sport_adapter after each tick).
    # "survivor" → JSON-source shape with a per-contestant table.
    # "billboard" → JSON-source shape with a song-rank table.
    dashboard_type: str = "standard"
    # Sport / survivor / billboard inputs — JSON-shaped bots write
    # these to their data/outputs/ dirs on their own cadence; the
    # dashboard just reads them.
    watchlist_json_path: str | None = None
    metrics_path: str | None = None
    coefficients_path: str | None = None
    sim_state_path: str | None = None
    # World Cup inputs — offline bake-off report + training-dataset
    # files written by the world-cup repo's scripts (the sim trader
    # doesn't write these; see world_cup.py). training_db_path is the
    # full ~49k-match SQLite grain; training_data_path is the WC-finals
    # CSV kept as a fallback.
    model_report_path: str | None = None
    training_data_path: str | None = None
    training_db_path: str | None = None
    # Kalshi series_ticker for the family the bot trades (e.g.
    # "KXAAAGASW", "KXNATGASD", "KXJOBLESSCLAIMS"). The dashboard uses
    # this to fetch live candlesticks straight from Kalshi for the
    # watchlist hero chart, independent of whether the bot itself is up.
    series_ticker: str | None = None
    # Human-readable name of the upstream data source surfaced on each
    # bot's model card and detail page (e.g. "FRED ICSA", "Billboard
    # Hot 100 weekly chart"). Falls through from config; render code
    # reads it via ``bot.get("data_source")``.
    data_source: str | None = None
    display: DisplayCfg = field(default_factory=DisplayCfg)
    # Zero or more season / tournament windows. Empty list → bot is
    # omitted from the Seasons tab. Sport bots typically list multiple
    # tournaments (NBA regular season + playoffs, ATP/WTA Grand Slams,
    # the PDC darts calendar, etc.); Survivor lists one window per
    # cycle.
    seasons: List[SeasonCfg] = field(default_factory=list)


@dataclass
class RiskCfg:
    bet_size_cents: int
    max_open_positions: int
    max_total_exposure_cents: int
    max_bets_per_day: int
    cooldown_seconds_same_market: int


@dataclass
class EdgeCfg:
    """Display-side defaults for the buy-criteria panel.

    Field names line up with ``kalshi_sdk.validators.select_side_by_ev``
    so what the dashboard renders here is what the bots actually gate
    on. ``min_model_accuracy`` stays as an optional extra — it's a
    macro-bot-specific floor (gas / claims / CPI) that runs before the
    shared SDK gates.

    These are *fallback* values. When a bot has written
    ``data/effective_config.json`` at startup the panel renders the
    bot's reported live values instead; missing bot fields fall back to
    these defaults so the panel never reads "—".
    """
    min_model_confidence: float
    min_ev_per_contract: float
    min_prob_edge_over_breakeven: float
    min_raw_model_edge: float = 0.06
    max_entry_price_cents: int = 70
    min_model_accuracy: float = 0.0


@dataclass
class HedgeCfg:
    enabled: bool
    profit_lock_cents: int
    stop_loss_cents: int
    hedge_size_fraction: float


@dataclass
class ValidatorCfg:
    min_book_depth_contracts: int
    max_spread_cents: int
    min_minutes_to_close: int
    max_minutes_to_close: int
    prob_bounds_cents: Tuple[int, int]
    min_volume: int = 0
    min_open_interest: int = 0
    min_depth_at_best_ask: int = 0
    basis_risk_strike_window_dollars: float = 0.0
    basis_risk_max_hours_to_close: float = 0.0


@dataclass
class DashboardConfig:
    host: str
    port: int
    # Mode flag — see dashboard.yaml for semantics. Defaults to "sim"
    # so legacy configs without a ``mode:`` key keep their pre-split
    # behaviour. The live fork sets ``mode: live`` in its
    # dashboard-live.yaml and inherits a different theme + header
    # text from this one knob.
    mode: str
    bots: List[BotEntry]
    risk: RiskCfg
    edge: EdgeCfg
    hedge: HedgeCfg
    validators: ValidatorCfg
    raw: dict = field(default_factory=dict)


def load_config(path: str | Path = "config/dashboard.yaml") -> DashboardConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    validators_raw = dict(raw["validators"])
    validators_raw["prob_bounds_cents"] = tuple(validators_raw["prob_bounds_cents"])

    def _coerce_season(item: dict) -> SeasonCfg:
        s = dict(item)
        # YAML may parse the ISO date strings into datetime objects
        # depending on quoting. Normalise to ISO string so the
        # downstream JS layer always receives text.
        for k in ("start", "end"):
            v = s.get(k)
            if hasattr(v, "isoformat"):
                s[k] = v.isoformat()
            elif v is not None:
                s[k] = str(v)
        return SeasonCfg(**s)

    bots: List[BotEntry] = []
    for b in raw["bots"]:
        b = dict(b)
        if "display" in b and isinstance(b["display"], dict):
            b["display"] = DisplayCfg(**b["display"])
        # Accept either ``seasons:`` (list — preferred) or a legacy
        # singular ``season:`` block (one dict). Both end up as a
        # ``seasons`` list of SeasonCfg.
        raw_seasons = b.pop("seasons", None)
        if raw_seasons is None and "season" in b:
            raw_seasons = [b.pop("season")]
        elif "season" in b:
            del b["season"]
        if raw_seasons:
            b["seasons"] = [_coerce_season(s) for s in raw_seasons]
        bots.append(BotEntry(**b))

    # ``mode`` is freeform-but-validated: anything other than the two
    # recognised values flips back to "sim" with a warning rather than
    # raising. The renderer treats anything-not-"live" as sim, so the
    # worst case from a typo is "looks like sim" — never "accidentally
    # places real-money trades because of a typo".
    mode_raw = str(raw.get("mode", "sim")).strip().lower()
    if mode_raw not in ("sim", "live"):
        import logging
        logging.getLogger("dashboard.config").warning(
            "dashboard.yaml mode=%r is not 'sim' or 'live' — falling "
            "back to 'sim' for safety.", mode_raw,
        )
        mode_raw = "sim"

    return DashboardConfig(
        host=raw.get("host", "0.0.0.0"),
        port=int(raw.get("port", 8080)),
        mode=mode_raw,
        bots=bots,
        risk=RiskCfg(**raw["risk"]),
        edge=EdgeCfg(**raw["edge"]),
        hedge=HedgeCfg(**raw["hedge"]),
        validators=ValidatorCfg(**validators_raw),
        raw=raw,
    )
