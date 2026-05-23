"""Sim.db → sport-watchlist JSON adapter.

The dashboard's "sport" page renderer (used by tennis, table-tennis,
darts) reads two files per bot:

  * ``<repo>/data/outputs/watchlist.json`` — one row per match with
    per-side model + market probabilities, edges, EV, buy verdict
  * ``<repo>/data/outputs/sim_state.json`` — open / closed positions
    + aggregate stats

The "sport-shape" bots (tennis et al.) write these directly. The
"standard-shape" bots (NBA, gas, unemployment, cpi, natural-gas)
write to a ``sim.db`` SQLite database instead, with one row per
market in ``market_views`` (so a head-to-head game generates TWO
rows — one for each side's YES contract).

This adapter is the bridge that lets a sim.db-shape bot render
through the unified sport page. It:

  1. Reads the latest snapshot per ticker from ``market_views``.
  2. Collapses each pair of "Team A wins" / "Team B wins" markets
     into a single match row with both sides populated.
  3. Pulls open + closed positions from ``positions`` + ``trades``
     and aggregates the sim_state stats.
  4. Atomically writes watchlist.json + sim_state.json.

It's deliberately generic: ``ticker_parser`` is a per-bot callable
that decodes a ticker into ``{away, home, team_being_asked}``, and
``tournament_label`` / ``surface_label`` let the caller tag the
match. Future sim.db-shape sport bots (e.g. baseball, hockey) reuse
the adapter with their own parser.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


log = logging.getLogger("dashboard.sport_adapter")


# A parsed ticker — what each bot's ticker_parser returns.
# ``base_id`` is the match-level identifier shared by both sides
# (e.g. for NBA's ``KXNBAGAME-26MAY25NYKCLE-CLE``, base_id is
# ``KXNBAGAME-26MAY25NYKCLE`` so both -CLE and -NYK collapse into
# one watchlist row).
ParsedTicker = dict  # {"base_id": str, "away": str, "home": str,
                      #  "team_being_asked": str}

TickerParser = Callable[[str], Optional[ParsedTicker]]


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` via a tempfile + rename. Concurrent
    dashboard readers never see a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".tmp", dir=str(path.parent), text=True,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        # If anything failed, clean up the tempfile.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _verdict_to_action(verdict: str, edge_yes: Optional[float],
                        edge_no: Optional[float],
                        strong_edge_min: float = 0.05) -> str:
    """Map the bot's BUY_YES / BUY_NO / SKIP verdict + edge size to
    the sport page's signal label set (STRONG_EDGE / SMALL_EDGE /
    NO_TRADE / WATCH). Tennis-side uses thresholds on the edge
    magnitude; we mirror that here using the same default cut.
    """
    if verdict in ("BUY_YES", "BUY_NO"):
        e = edge_yes if verdict == "BUY_YES" else edge_no
        e_abs = abs(e) if e is not None else 0.0
        if e_abs >= strong_edge_min:
            return "STRONG_EDGE"
        return "SMALL_EDGE"
    return "NO_TRADE"


def _confidence_score(raw_prob: Optional[float]) -> float:
    """Distance from 0.5, scaled to [0, 1]. Used as a directional
    confidence number when the upstream bot doesn't track its own.
    """
    if raw_prob is None:
        return 0.0
    return min(1.0, max(0.0, 2.0 * abs(raw_prob - 0.5)))


def _latest_snapshots(con: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return the most recent market_views row per ticker — the
    "current state" of every market the bot is watching."""
    cur = con.execute(
        """
        SELECT m.*
        FROM market_views m
        INNER JOIN (
            SELECT ticker, MAX(captured_at) AS latest
            FROM market_views
            GROUP BY ticker
        ) t ON m.ticker = t.ticker AND m.captured_at = t.latest
        """,
    )
    return cur.fetchall()


def _row_pair_to_match(parsed_a: ParsedTicker, row_a: sqlite3.Row,
                        parsed_b: Optional[ParsedTicker],
                        row_b: Optional[sqlite3.Row],
                        tournament_label: str,
                        surface_label: str) -> dict:
    """Collapse the (Team A wins YES) and (Team B wins YES) market_views
    rows into one sport-shape watchlist row. If only one side is
    present (the other market hasn't snapshotted yet), the other
    side's probability is inferred as 1 - p_a (which is what the
    bot does internally for binary YES/NO markets).
    """
    away = parsed_a["away"]
    home = parsed_a["home"]
    # By convention, player_a = away, player_b = home.
    # Find the per-side YES probability + market data.
    def _yes_view_for(team: str) -> tuple[Optional[sqlite3.Row],
                                            Optional[ParsedTicker]]:
        if parsed_a["team_being_asked"] == team:
            return row_a, parsed_a
        if parsed_b and parsed_b["team_being_asked"] == team:
            return row_b, parsed_b
        return None, None

    row_for_a, _ = _yes_view_for(away)
    row_for_b, _ = _yes_view_for(home)

    def _prob(row: Optional[sqlite3.Row]) -> Optional[float]:
        if row is None:
            return None
        v = row["model_prob_yes"]
        return float(v) if v is not None else None

    def _market_prob(row: Optional[sqlite3.Row]) -> Optional[float]:
        if row is None:
            return None
        ask = row["yes_ask_cents"]
        return (float(ask) / 100.0) if ask is not None else None

    def _edge_yes(row: Optional[sqlite3.Row]) -> Optional[float]:
        if row is None:
            return None
        v = row["edge_yes"]
        return float(v) if v is not None else None

    def _raw_prob(row: Optional[sqlite3.Row]) -> Optional[float]:
        if row is None:
            return None
        v = row["raw_model_prob_yes"]
        return float(v) if v is not None else None

    # Each side's "win probability" is its own market's YES prob.
    # If that side's market hasn't snapshotted yet, fall back to
    # (1 - other_side_yes_prob).
    live_prob_a = _prob(row_for_a)
    live_prob_b = _prob(row_for_b)
    if live_prob_a is None and live_prob_b is not None:
        live_prob_a = 1.0 - live_prob_b
    if live_prob_b is None and live_prob_a is not None:
        live_prob_b = 1.0 - live_prob_a

    market_prob_a = _market_prob(row_for_a)
    market_prob_b = _market_prob(row_for_b)

    edge_a = _edge_yes(row_for_a)
    edge_b = _edge_yes(row_for_b)

    # Which side has the better verdict? Pick the row whose
    # bot_verdict is BUY_YES with the larger edge. If neither is
    # BUY_YES, fall back to NO_TRADE.
    def _verdict(row: Optional[sqlite3.Row]) -> str:
        return (row["bot_verdict"] or "SKIP") if row is not None else "SKIP"

    verdict_a = _verdict(row_for_a)
    verdict_b = _verdict(row_for_b)

    buy_side: Optional[str] = None
    buy_side_edge = 0.0
    buy_eligible = False
    if verdict_a == "BUY_YES":
        buy_side, buy_side_edge, buy_eligible = "A", (edge_a or 0.0), True
    elif verdict_b == "BUY_YES":
        buy_side, buy_side_edge, buy_eligible = "B", (edge_b or 0.0), True

    # Use the side-with-an-edge for the row-level action label;
    # fall back to whichever side has a non-SKIP verdict.
    if buy_side == "A":
        action = _verdict_to_action(verdict_a, edge_a, edge_b)
    elif buy_side == "B":
        action = _verdict_to_action(verdict_b, edge_b, edge_a)
    else:
        action = "NO_TRADE"

    # Match-level confidence: average of the two sides' raw model
    # confidence. The "spread" between model and market is what the
    # dashboard's volatility column shows for sport bots; we leave
    # volatility at 0 since the macro bots don't track it.
    confidence_score = max(
        _confidence_score(_raw_prob(row_for_a)),
        _confidence_score(_raw_prob(row_for_b)),
    )

    # Market metadata — take whichever row has them.
    src = row_for_a if row_for_a is not None else row_for_b
    open_interest = src["open_interest"] if src is not None else None
    volume = src["volume"] if src is not None else None
    spread_cents = src["spread_cents"] if src is not None else None
    event_title = (src["event_title"] if src is not None else None) or ""

    return {
        "match_id": parsed_a["base_id"],
        "player_a": away,
        "player_b": home,
        "side_player_a": away,
        "side_player_b": home,
        "tournament": tournament_label,
        "surface": surface_label,
        "round_label": event_title or f"{away} @ {home}",
        # Probabilities
        "pre_match_prob_a": live_prob_a if live_prob_a is not None else 0.5,
        "pre_match_prob_b": live_prob_b if live_prob_b is not None else 0.5,
        "live_prob_a": live_prob_a if live_prob_a is not None else 0.5,
        "live_prob_b": live_prob_b if live_prob_b is not None else 0.5,
        "market_prob_a": market_prob_a,
        "market_prob_b": market_prob_b,
        # Edges (already net of slippage in the source)
        "edge_a": edge_a,
        "edge_b": edge_b,
        "ev_a": edge_a,  # NBA doesn't split edge vs EV; surface what we have
        "ev_b": edge_b,
        # Signal
        "confidence_score": round(confidence_score, 4),
        "volatility_score": 0.0,
        "injury_news_flag": False,
        "recommended_action": action,
        "reason_for_signal": (src["rejection_reason"] if src is not None
                                else "") or "",
        "current_score": "",  # NBA bot doesn't track in-game scores
        "last_updated": (src["captured_at"]
                          if src is not None else
                          datetime.now(timezone.utc).isoformat()),
        # Liquidity
        "open_interest": int(open_interest) if open_interest is not None
                          else None,
        "volume": int(volume) if volume is not None else None,
        "spread_cents": int(spread_cents) if spread_cents is not None
                          else None,
        "yes_ask_cents_a": (int(row_for_a["yes_ask_cents"])
                              if row_for_a is not None
                              and row_for_a["yes_ask_cents"] is not None
                              else None),
        "yes_ask_cents_b": (int(row_for_b["yes_ask_cents"])
                              if row_for_b is not None
                              and row_for_b["yes_ask_cents"] is not None
                              else None),
        # Titles
        "title_a": f"Will {away} beat {home}?",
        "title_b": f"Will {home} beat {away}?",
        "title": event_title or f"{away} @ {home}",
        "event_title": event_title or f"{away} @ {home}",
        # Buy gate
        "buy_eligible": buy_eligible,
        "buy_score": round(abs(buy_side_edge), 6),
        "buy_side": buy_side,
        "buy_side_edge": buy_side_edge,
        "buy_side_ev": buy_side_edge if buy_side else None,
        "buy_gates": [],
        "buy_blockers": [],
    }


def _build_watchlist(con: sqlite3.Connection,
                      ticker_parser: TickerParser,
                      tournament_label: str,
                      surface_label: str) -> list[dict]:
    """Collapse latest market_views snapshots into match-level rows."""
    rows = _latest_snapshots(con)
    # Group by (base_id, away, home) so each match collects its two
    # sides. If only one side has snapshotted yet, the match still
    # gets a row with the other side inferred.
    groups: dict[str, list[tuple[ParsedTicker, sqlite3.Row]]] = defaultdict(list)
    for r in rows:
        parsed = ticker_parser(r["ticker"])
        if parsed is None:
            log.debug("sport_adapter — skipping unparseable ticker %s",
                      r["ticker"])
            continue
        groups[parsed["base_id"]].append((parsed, r))

    matches: list[dict] = []
    for base_id, pair in groups.items():
        if len(pair) == 1:
            (parsed_a, row_a) = pair[0]
            matches.append(_row_pair_to_match(
                parsed_a, row_a, None, None,
                tournament_label, surface_label,
            ))
        else:
            # Two sides — pass both; the collapser figures out which
            # row corresponds to player_a (away) vs player_b (home).
            (parsed_a, row_a), (parsed_b, row_b) = pair[0], pair[1]
            matches.append(_row_pair_to_match(
                parsed_a, row_a, parsed_b, row_b,
                tournament_label, surface_label,
            ))
    return matches


def _build_sim_state(con: sqlite3.Connection,
                      ticker_parser: TickerParser,
                      tournament_label: str,
                      surface_label: str) -> dict:
    """Translate positions + position_marks + trades into the
    sim_state.json structure the sport renderer reads.
    """
    open_positions: list[dict] = []
    closed_positions: list[dict] = []
    total_realized_pnl = 0.0
    total_unrealized_pnl = 0.0
    total_staked = 0.0
    wins = 0
    losses = 0

    pos_cur = con.execute(
        """
        SELECT p.*, m.yes_ask_cents AS mark_yes_ask,
                m.no_ask_cents AS mark_no_ask,
                m.mid_cents AS mark_mid_cents,
                m.updated_at AS mark_updated_at
        FROM positions p
        LEFT JOIN position_marks m ON m.position_id = p.id
        """,
    )
    for p in pos_cur.fetchall():
        parsed = ticker_parser(p["ticker"])
        if parsed is None:
            continue
        away, home = parsed["away"], parsed["home"]
        team = parsed["team_being_asked"]
        side_label = "PLAYER_A" if team == away else "PLAYER_B"
        side_player = team
        # Stake: contracts × entry_price_cents (cents) ÷ 100 → dollars
        contracts = int(p["contracts"] or 1)
        entry_cents = int(p["entry_price_cents"] or 0)
        stake = (contracts * entry_cents) / 100.0
        total_staked += stake

        entry_market_prob = (float(p["kalshi_yes_prob_at_entry"])
                              if p["kalshi_yes_prob_at_entry"] is not None
                              else float(entry_cents) / 100.0)
        entry_model_prob = (float(p["model_yes_prob_at_entry"])
                             if p["model_yes_prob_at_entry"] is not None
                             else entry_market_prob)

        record = {
            "position_id": str(p["id"]),
            "match_id": parsed["base_id"],
            "tournament": tournament_label,
            "surface": surface_label,
            "player_a": away,
            "player_b": home,
            "side": side_label,
            "side_player": side_player,
            "title": f"Will {team} beat "
                       f"{home if team == away else away}?",
            "event_title": f"{away} @ {home}",
            "entry_market_prob": entry_market_prob,
            "entry_model_prob": entry_model_prob,
            "stake": stake,
            "slippage": 0.0,
            "opened_at": p["opened_at"] or "",
            "label_at_open": "STRONG_EDGE",  # macro bots don't tag this
            "reason_at_open": "",
        }

        if p["status"] == "open":
            mid = p["mark_mid_cents"]
            current_market_prob = (float(mid) / 100.0
                                     if mid is not None
                                     else entry_market_prob)
            # Mark-to-market P&L: side==YES means we win when prob ↑
            side = p["side"] or "YES"
            if side == "YES":
                mtm_cents = (current_market_prob * 100.0) - entry_cents
            else:
                mtm_cents = (100.0 - current_market_prob * 100.0) \
                              - (100 - entry_cents)
            unrealized = (mtm_cents / 100.0) * contracts
            total_unrealized_pnl += unrealized
            record.update({
                "current_market_prob": current_market_prob,
                "current_model_prob": entry_model_prob,
                "unrealized_pnl": unrealized,
            })
            open_positions.append(record)
        else:  # closed
            exit_cents = int(p["exit_price_cents"] or entry_cents)
            settle_market_prob = float(exit_cents) / 100.0
            realized = float(p["realized_pnl_cents"] or 0) / 100.0
            total_realized_pnl += realized
            if realized > 0:
                wins += 1
            elif realized < 0:
                losses += 1
            record.update({
                "closed_at": p["exited_at"] or "",
                "settle_market_prob": settle_market_prob,
                "realized_pnl": realized,
                "close_reason": p["error_type"] or "settled",
                "result": "SETTLED",
                "won": realized > 0,
                "winner_side": side_label if realized > 0
                                else ("PLAYER_B" if side_label == "PLAYER_A"
                                        else "PLAYER_A"),
            })
            closed_positions.append(record)

    total_closed = len(closed_positions)
    win_rate = (wins / total_closed) if total_closed > 0 else None
    roi = (total_realized_pnl / total_staked) if total_staked > 0 else None

    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "started_at": now_iso,  # we don't track start time across restarts
        "last_tick_at": now_iso,
        "open_positions": open_positions,
        "closed_positions": closed_positions,
        "stats": {
            "total_opened": len(open_positions) + total_closed,
            "total_closed": total_closed,
            "open_count": len(open_positions),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_realized_pnl": total_realized_pnl,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_staked": total_staked,
            "roi": roi,
        },
        "last_settled_at_by_match_id": {},
    }


def translate(db_path: str | Path,
               watchlist_out: str | Path,
               sim_state_out: str | Path,
               ticker_parser: TickerParser,
               tournament_label: str,
               surface_label: str = "") -> None:
    """Read ``db_path`` and write sport-format watchlist.json +
    sim_state.json to the configured output paths. Intended to be
    called by a sim.db-shape bot module after each tick.

    Best-effort: any error is logged and swallowed — a failing
    adapter shouldn't break the bot's trading loop, only its
    rendering on the sport page (which falls back to the previously
    written file if present).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        log.warning("sport_adapter: %s does not exist; skipping translate",
                    db_path)
        return
    try:
        con = sqlite3.connect(str(db_path), isolation_level=None)
        con.row_factory = sqlite3.Row
        try:
            rows = _build_watchlist(con, ticker_parser, tournament_label,
                                     surface_label)
            state = _build_sim_state(con, ticker_parser, tournament_label,
                                      surface_label)
        finally:
            con.close()
        watchlist = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        }
        _atomic_write_json(Path(watchlist_out), watchlist)
        _atomic_write_json(Path(sim_state_out), state)
        log.info("sport_adapter — wrote %d matches → %s",
                 len(rows), watchlist_out)
    except Exception:  # noqa: BLE001
        log.exception("sport_adapter: translation failed for %s", db_path)
