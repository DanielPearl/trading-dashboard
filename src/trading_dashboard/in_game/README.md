# In-game model layer

A second model that fires after a sport match has gone live. The
pre-game model is each bot's existing classifier (untouched by this
package). Both coexist in the dashboard; the in-game model is read-
only against pre-game outputs and never writes back.

## Public API

```python
from trading_dashboard import in_game

pred = in_game.predict(bot, position, market_view=None)
# Returns LivePrediction(live_prob_yes, confidence, recommended_action,
#                        reason, features) or None.
```

`recommended_action` values: `"exit_now"`, `"let_run"`, `"hold"`,
`"neutral"`. The hedge monitor consults this when evaluating sport
positions:

- `exit_now` (confidence ≥ 0.5) ⇒ pre-empt a close before any
  threshold triggers; logged with reason `"ingame_exit"`.
- `let_run` / `hold` (confidence ≥ 0.5) ⇒ suppress a threshold-
  triggered hedge close.
- `neutral` or low confidence ⇒ defer to the existing
  profit-lock / stop-loss thresholds.

## What's heuristic today

| Sport | Source for live state | Model |
| --- | --- | --- |
| NBA | ESPN scoreboard + `/summary?event=` (predictor, injuries, live box score) | Logistic on lead / √(seconds remaining), blended 75/25 with ESPN's own win projection, then nudged by injury counts, foul trouble (≥4 PF), and live FG/FT/3P/AST/REB/TO gaps |
| Tennis | bot's `watchlist.json` (`live_prob_a`, `live_prob_b`, `current_score`, `injury_news_flag`) | Trust the bot's own live estimate; layer divergence + reversion pull from market |
| Table tennis | same as tennis | same |
| Darts | same as tennis (sets settle faster) | tennis logic with a confidence bump after the first set |

The cross-sport `features.py` module provides three signals all
sport modules use:
- `market_velocity` — recent cents/minute drift in YES price
- `volatility` — stdev of recent first-differences
- `divergence` — |pre-game prior − current market prob|
- `expected_reversion_pull` — divergence damped by volatility

## What this layer does NOT do yet (gaps + extension plan)

Each row below is a feature on the user's wish list that requires
either an external data feed or a trained model. The path to enable
it is sketched.

### NBA — what's Live now vs. what's still TODO

| Feature | Status | Source / TODO |
| --- | --- | --- |
| Score differential | **Live** | ESPN scoreboard |
| Time remaining | **Live** | ESPN scoreboard (period + clock) |
| Market velocity / volatility / divergence | **Live** | `market_views` history |
| ESPN's own win projection (second opinion) | **Live** | ESPN `/summary?event=` predictor block |
| Critical injuries per team | **Live** | ESPN `/summary` injuries block (Out / Day-to-Day / Doubtful) |
| Foul trouble (players w/ ≥4 PF) | **Live** | ESPN `/summary` boxscore.players |
| Live FG% / FT% / 3P% / AST / REB / TO gaps | **Live** | ESPN `/summary` boxscore.teams.statistics |
| Possession-adjusted pace | TODO | `nba_api` live play-by-play |
| Shooting % vs xFG | TODO | `nba_api` shot chart + xFG training set |
| Lineup combinations on floor | TODO | `nba_api` play-by-play (sub events) |
| Rest/fatigue (minutes last N games) | TODO | season schedule + box scores |
| Clutch-time historical performance | TODO | aggregate from historical play-by-play |
| Implied pace vs pre-game pace | TODO | per-game pre-game pace baseline + live diff |
| Per-player minutes restriction | TODO | injury report + minutes feed |
| Bench vs starter efficiency | TODO | `nba_api` advanced box |

### Tennis (and table-tennis)

| Feature | Source needed |
| --- | --- |
| 1st/2nd serve %, ace rate, double faults | the bot's live feed exposes these — extend the watchlist row schema to write them out, then the in-game model can read them |
| breakpoint conversion / save % | live point-by-point feed |
| rally length trends | point-by-point feed with rally length |
| medical timeouts | live event feed (some providers expose; ATP/WTA APIs typically not) |
| serve velocity decline | radar gun data; only on broadcast-paired feeds |
| historical comeback probability from score state | offline analysis of ATP/WTA point-by-point database |
| body language / tilt | not measurable from a data feed; would need video + CV |

### Darts

| Feature | Source needed |
| --- | --- |
| checkout % | live leg-by-leg feed |
| point streak persistence | running leg stats; can compute from per-tick watchlist if it logs them |
| spin/style matchup | requires historical match outcomes by player style |

### Cross-sport (not yet implemented)

| Feature | Path |
| --- | --- |
| social / news / chatter signal | Twitter API + filter on team / player names; sentiment model. Heavy lift. |
| injury mentions | parse a news API (ESPN news endpoint, the-odds-api, NewsAPI) for player names in our open positions |
| crowd / home pressure | proxy via home/away from ESPN; weight by venue |
| historical choke probability | per-player aggregate over historical games; needs trained model |

## Architecture choices worth remembering

1. **The pre-game model is never modified.** `predict()` only
   reads from positions / market_views / watchlist.json. The bot's
   own pre-game classifier and its sim.db remain untouched.

2. **Errors are swallowed.** Any exception inside the in-game
   model returns `None`, falling back to threshold logic. A bad
   live data feed must never break the hedge monitor.

3. **Soft-clipped probabilities.** No in-game model ever claims
   100% or 0%. We clip to [0.02, 0.98] so a single fluky feature
   can't override the hedge gate.

4. **Confidence-gated overrides.** The hedge monitor only consults
   `recommended_action` when `confidence ≥ 0.5`. Below that, the
   model is purely advisory — surfaces on the UI for the user, but
   doesn't change automated behavior.

5. **Caching at the edges.** ESPN responses cached 30s; the
   tennis watchlist file is read with a 15s TTL. The 30s hedge
   tick will therefore touch each external dependency at most
   once per pass.

6. **Predictions are logged for self-evaluation.** Every confident
   action *transition* (model's recommended action flipping for a
   ticker) gets appended to `data/in_game_predictions.jsonl` via
   `in_game/logger.py`. The Models > In-game view's "Recent
   predictions" panel reads the tail and joins each row against
   the closed-bet ledger to surface WON / LOST / OPEN. Append-only;
   never rewritten.

7. **Feature snapshots are logged densely for training.** Every
   prediction (regardless of confidence or action) writes one JSON
   line to `data/in_game_features.jsonl` via
   `in_game/feature_log.py`. The trainer joins these snapshots
   against the closed-bet ledger to build labeled training pairs.

## Package inventory

| Module | Purpose |
| --- | --- |
| `base.py` | `LivePrediction` dataclass + action constants |
| `features.py` | Cross-sport market features (velocity / volatility / divergence) |
| `market_state.py` | Per-ticker history reader (market_views) for NBA |
| `news_signals.py` | ESPN /news scanner with injury-keyword matching |
| `nba.py` | NBA in-game scorer (ESPN + CDN + market + news) |
| `nba_cdn.py` | NBA.com CDN adapter (pace, FT rate, foul trouble, +/-) |
| `tennis.py` | Tennis-shape scorer (bot's live_prob + market overlay) — kept because `darts.py` and the table-tennis bot delegate to `predict(sport=…)`; the tennis bot itself no longer uses the in-game layer (removed 2026-07-08 alongside the in-match adjustment) |
| `darts.py` | Darts (thin wrapper on tennis.py) |
| `logger.py` | Transition audit log (sparse) |
| `feature_log.py` | Dense per-tick feature snapshot log (training data) |
| `train_classifier.py` | Stdlib SGD logistic regression trainer (runnable as script) |
| `model_loader.py` | Loads + applies a trained model when one exists |

## How to train a real model

The data substrate is **already in place**. Every prediction the
in-game model issues now writes a dense feature snapshot to
`data/in_game_features.jsonl` via `in_game/feature_log.py`. Once
the dashboard accumulates a few weeks of game time across the
sport bots, run the trainer:

```
python -m trading_dashboard.in_game.train_classifier \
    --bot nba --db-path /root/nba/data/sim.db
```

The script (stdlib only — no sklearn / numpy) does the full
pipeline:
1. Loads every feature snapshot for the bot from
   `data/in_game_features.jsonl`.
2. Joins each row against the bot's closed-position ledger to
   assign a `won` label (1 if realized P&L > 0, else 0).
3. Splits 80/20 train/holdout *grouped by position ticker* (so a
   position's snapshots don't leak across folds).
4. Per-feature z-score standardization on the training set;
   applies same transform to holdout.
5. Mini-batch SGD logistic regression with L2 regularization.
6. Reports holdout accuracy / precision / recall / F1 / ROC AUC.
7. Writes weights + standardization params to
   `data/in_game_models/<bot>_logreg.json`.

The next time the dashboard starts, `in_game/model_loader.py`
picks up the file and `nba.predict()` blends the trained
classifier's output at 40% weight against the heuristic's 60%.
**Heuristic remains as the high-fidelity fallback** — if a single
feature breaks or the classifier returns NaN, the live prediction
still works.

Same shape works for tennis once the tennis bot's watchlist
schema is extended with per-point features. Run the same command
with `--bot tennis` once the bot is publishing the per-point
stats and the snapshot log has accumulated.
