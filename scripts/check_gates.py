#!/usr/bin/env python3
"""Gate-parity check — every bot connects to the shared buy gates.

User 2026-09-11: "the same validations and gates should exist for all
bots. if i were making a new one they should also connect to the
validations and gates that already exist." This script is that rule's
enforcement, born from the billboard hole (its validator had the edge
FLOORS but never the MAX_EDGE ceiling, and bought a 42pp-gap NO
nobody's gates ever saw).

Three tiers, exit 1 on any failure so deploy.sh can gate on it:

1. BEHAVIOURAL — probe ``kalshi_sdk`` itself: the canonical gates
   must refuse the canonical bad trades (edge below floor, edge above
   the suspect ceiling, price outside the 30-70¢ band, started
   match). Catches a regression inside the shared module, which every
   wiring check below would otherwise inherit silently.

2. WIRING — every bot in the REGISTRY must show its declared
   connection to the shared module (pattern scan of its repo's
   trading/decision files). A dormant bot (no live order path or
   toggle off) warns instead of failing, but FAILS if it is armed
   while unwired.

3. COVERAGE — every bot key in the dashboard config must appear in
   the REGISTRY. A brand-new bot card without a registry entry fails
   the deploy with instructions, so "connect to the shared gates,
   then register it here" is a forced step of adding a bot.

Usage:  python3 scripts/check_gates.py [config/dashboard-live.yaml]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

CONFIG_DEFAULT = "/root/trading-dashboard/config/dashboard-live.yaml"
BOT_STATES = "/root/trading-dashboard/data/bot_states_live.json"

# bot key -> (repo dir, mechanism, [required regex patterns in src/])
#
# Mechanisms:
#   sport_executor — trades ONLY through the dashboard's shared
#       SportLiveExecutor (clamped to kalshi_sdk.buy_criteria, fail-
#       closed prematch rule); repo must still reference the shared
#       module so its exporter gates can't drift.
#   macro_gate     — repo entry path calls kalshi_sdk.buy_criteria.
#       macro_entry_gate (band + floor + calibrated ceiling in one).
#   shared_tokens  — repo wires the shared constants/functions
#       directly (weather, hormuz, billboard).
#   dormant        — no live trading today; must adopt shared gates
#       BEFORE being armed. Warns while dormant, fails if armed.
REGISTRY: dict[str, tuple[str, str, list[str]]] = {
    "tennis":        ("/root/tennis-forecast", "sport_executor",
                      [r"buy_criteria|evaluate_row_gates"]),
    "table-tennis":  ("/root/table-tennis-forecast", "sport_executor",
                      [r"buy_criteria|evaluate_row_gates"]),
    "darts":         ("/root/darts-forecast", "sport_executor",
                      [r"buy_criteria|evaluate_row_gates"]),
    "basketball":    ("/root/nba", "sport_executor",
                      [r"buy_criteria|exporter_defaults"]),
    "mlb":           ("/root/baseball", "sport_executor",
                      [r"buy_criteria|exporter_defaults"]),
    "world-cup":     ("/root/world-cup", "sport_executor",
                      [r"buy_criteria|exporter_defaults"]),
    "cpi":           ("/root/cpi", "macro_gate", [r"macro_entry_gate"]),
    "pce":           ("/root/cpi", "macro_gate", [r"macro_entry_gate"]),
    "gdp":           ("/root/cpi", "macro_gate", [r"macro_entry_gate"]),
    "unemployment-claims": ("/root/unemployment-claims", "macro_gate",
                            [r"macro_entry_gate"]),
    "hormuz":        ("/root/port-forecast", "shared_tokens",
                      [r"select_side_by_ev", r"MAX_EDGE"]),
    "rain":          ("/root/weather-forecast", "shared_tokens",
                      [r"buy_criteria", r"max_edge|MAX_EDGE"]),
    "temp":          ("/root/weather-forecast", "shared_tokens",
                      [r"buy_criteria", r"max_edge|MAX_EDGE"]),
    "billboard":     ("/root/billboard-charts", "shared_tokens",
                      [r"buy_criteria", r"MAX_EDGE|edge_exceeds_max"]),
    # Dormant / advisory — unwired legacy or no live order path.
    # Arming any of these without first wiring the shared gates (and
    # upgrading the mechanism above) is exactly what this check
    # exists to catch.
    "gas-prices":    ("/root/gas-prices", "dormant", []),
    "natural-gas":   ("/root/peak-load", "dormant", []),
    "survivor":      ("/root/survivor-elimination", "dormant", []),
    "reality-leaks": ("/root/reality-leaks", "dormant", []),
}


def behavioural() -> list[str]:
    fails: list[str] = []
    from kalshi_sdk.buy_criteria import (macro_entry_gate,
                                         kelly_contracts,
                                         clamp_executor_cfg,
                                         KELLY_MAX_CONTRACTS)
    from kalshi_sdk.validators import evaluate_row_gates

    ok, _ = macro_entry_gate(0.55, 42)
    if not ok:
        fails.append("macro_entry_gate refuses a clean 13pp trade")
    for name, args, kw in (
            ("edge floor", (0.46, 42), {}),
            ("suspect ceiling (calibrated)", (0.99, 50), {}),
            ("suspect ceiling (uncalibrated)",
             (0.80, 50), {"calibrated_source": False}),
            ("price band low", (0.60, 20), {}),
            ("price band high", (0.95, 80), {}),
    ):
        ok, _ = macro_entry_gate(*args, **kw)
        if ok:
            fails.append(f"macro_entry_gate PASSED the {name} probe")

    base = dict(live_prob_a=0.60, market_prob_a=0.45, ev_a=0.05,
                recommended_action="EDGE")
    d = evaluate_row_gates(dict(base, kickoff="2020-01-01T00:00:00Z"),
                           small_edge_min=0.09, min_ev=0.0,
                           min_market_prob=0.30, max_market_prob=0.70,
                           tradeable_labels={"EDGE"})
    if d.eligible or "prematch" not in d.blockers:
        fails.append("evaluate_row_gates PASSED a started match")
    d = evaluate_row_gates(dict(base, live_prob_a=0.50),
                           small_edge_min=0.09, min_ev=0.0,
                           min_market_prob=0.30, max_market_prob=0.70,
                           tradeable_labels={"EDGE"})
    if d.eligible:
        fails.append("evaluate_row_gates PASSED a 5pp edge")
    if kelly_contracts(0.99, 40, 10_000_000) > KELLY_MAX_CONTRACTS:
        fails.append("kelly_contracts exceeded its hard cap")
    me, lo, hi = clamp_executor_cfg(min_edge=0.01,
                                    min_entry_price_cents=5,
                                    max_entry_price_cents=99)
    if me < 0.09 or lo < 30 or hi > 70:
        fails.append("clamp_executor_cfg failed to clamp a loose config")
    return fails


def _grep(repo: str, pattern: str) -> bool:
    src = Path(repo) / "src"
    if not src.exists():
        return False
    try:
        r = subprocess.run(
            ["grep", "-rlE", pattern, str(src), "--include=*.py"],
            capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def _enabled(bot: str) -> bool:
    try:
        states = json.load(open(BOT_STATES)).get("states", {})
        e = states.get(bot)
        return bool(e and e.get("enabled", True))
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    failures: list[str] = []
    warnings: list[str] = []

    for f in behavioural():
        failures.append(f"SDK: {f}")

    # Coverage: every configured bot key must be registered.
    keys: list[str] = []
    try:
        text = Path(cfg_path).read_text()
        keys = re.findall(r"^\s*-\s*key:\s*([\w-]+)", text, re.M)
    except OSError as e:
        failures.append(f"config unreadable ({cfg_path}): {e}")
    for k in keys:
        if k not in REGISTRY:
            failures.append(
                f"{k}: NOT in the shared-gates registry — a new bot "
                f"must wire kalshi_sdk.buy_criteria gates into its "
                f"entry path, then register its mechanism in "
                f"scripts/check_gates.py")

    # Wiring per registered bot.
    for bot, (repo, mech, patterns) in REGISTRY.items():
        if mech == "dormant":
            has_orders = _grep(repo, r"place_order|submit_ioc")
            armed = _enabled(bot)
            if armed and has_orders:
                failures.append(
                    f"{bot}: ARMED with a live order path but no "
                    f"shared-gate wiring — wire "
                    f"kalshi_sdk.buy_criteria before enabling")
            elif has_orders:
                warnings.append(
                    f"{bot}: dormant, unwired order path — wire the "
                    f"shared gates before ever arming")
            continue
        missing = [p for p in patterns if not _grep(repo, p)]
        if missing:
            failures.append(
                f"{bot}: shared-gate wiring missing in {repo}/src "
                f"(pattern(s) {missing})")

    for w in warnings:
        print(f"WARN  {w}")
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        print(f"\n{len(failures)} gate-parity failure(s)")
        return 1
    print(f"gate parity OK — SDK probes pass, {len(REGISTRY)} bots "
          f"wired, {len(keys)} configured keys covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
