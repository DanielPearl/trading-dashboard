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
| NBA | ESPN scoreboard JSON | Logistic on lead / √(seconds remaining), overlaid with market velocity / volatility / divergence |
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

### NBA — full feature set

| Feature | Why we don't have it | What's needed |
| --- | --- | --- |
| possession-adjusted pace | not in ESPN scoreboard summary | scrape ESPN gamecast or use `nba_api` for live play-by-play |
| shooting % vs expected | needs shot location + xFG model | `nba_api` shot chart endpoint; train xFG on historical |
| foul trouble | per-player foul counts | `nba_api` box score endpoint |
| lineup combinations on floor | substitution data | `nba_api` play-by-play; map subs to current lineup |
| rest/fatigue | minutes per player in last N games | season schedule + box scores |
| turnover / rebound / FT rate | not in scoreboard summary | `nba_api` team stats endpoint |
| clutch-time historical performance | per-team clutch ledger | aggregate from historical play-by-play |
| live win-prob shifts | track our own series over time | already have; compute deltas from `_in_game.features["state_prob"]` history |
| implied pace vs pregame pace | needs pace estimate at game start | publish per-game pre-game pace, compare to live pace from box score |
| injury / minutes restriction | injury report feed | ESPN injuries endpoint or APIs like `the-odds-api` |
| bench vs starter efficiency | per-player +/- in real time | `nba_api` advanced box |

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

## How to train a real model when you're ready

For one sport (NBA) the rough pipeline:

1. Pull historical schedules + final scores from `nba_api` or
   basketball-reference. Filter to last 3-5 seasons.
2. For each game, walk the play-by-play stream. At fixed sample
   points (every 30 game-seconds), snapshot the feature vector
   you'd have had at that moment.
3. Label each sample with the eventual winner. The feature vector
   is your X; the binary "did this team win" is your y.
4. Train a gradient-boosted tree or simple logistic on the
   feature/label pairs. Hold out the last season for validation.
5. Drop the trained model into `nba.py`, replacing
   `_win_prob_from_state` with a call to `model.predict_proba`.
6. Keep the existing heuristic as the fallback for when the model
   says "low confidence" or the live feature isn't available.

Same shape works for tennis with the ATP/WTA point-by-point
database, table tennis with ITTF feeds, and darts with the PDC
data partners. Each is a multi-week pipeline of its own.
