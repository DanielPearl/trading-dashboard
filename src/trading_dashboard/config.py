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


@dataclass
class BotEntry:
    key: str
    name: str
    db_path: str
    decisions_path: str
    # "standard" → render the gas-bot-style page (model card, watchlist,
    # positions). "whale" → render the signal-analysis page driven by
    # signals_path / orders_path instead of a sim.db.
    dashboard_type: str = "standard"
    signals_path: str | None = None
    orders_path: str | None = None
    # Kalshi series_ticker for the family the bot trades (e.g.
    # "KXAAAGASW", "KXNATGASD", "KXJOBLESSCLAIMS"). The dashboard uses
    # this to fetch live candlesticks straight from Kalshi for the
    # watchlist hero chart, independent of whether the bot itself is up.
    series_ticker: str | None = None
    display: DisplayCfg = field(default_factory=DisplayCfg)


@dataclass
class RiskCfg:
    bet_size_cents: int
    max_open_positions: int
    max_total_exposure_cents: int
    max_bets_per_day: int
    cooldown_seconds_same_market: int


@dataclass
class EdgeCfg:
    min_edge_yes: float
    min_edge_no: float
    min_model_confidence: float
    min_confidence: float
    min_model_accuracy: float
    min_ev_per_contract: float
    min_prob_edge_over_breakeven: float


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

    bots: List[BotEntry] = []
    for b in raw["bots"]:
        b = dict(b)
        if "display" in b and isinstance(b["display"], dict):
            b["display"] = DisplayCfg(**b["display"])
        bots.append(BotEntry(**b))

    return DashboardConfig(
        host=raw.get("host", "0.0.0.0"),
        port=int(raw.get("port", 8080)),
        bots=bots,
        risk=RiskCfg(**raw["risk"]),
        edge=EdgeCfg(**raw["edge"]),
        hedge=HedgeCfg(**raw["hedge"]),
        validators=ValidatorCfg(**validators_raw),
        raw=raw,
    )
