"""Train a logistic-regression in-game classifier from accumulated
data. Stdlib only — no sklearn, no numpy. Runnable as a script:

    python -m trading_dashboard.in_game.train_classifier --bot nba

How it works
------------
1. Loads every feature-snapshot from
   ``data/in_game_features.jsonl`` for the given bot.
2. Loads the bot's closed-position ledger (sim.db for NBA; the
   tennis adapter for tennis-shape bots).
3. Joins each snapshot to its eventual outcome on
   ``(bot_key, ticker)`` — a snapshot taken during a position
   that later resolved at >0 P&L gets label 1, else 0.
4. Trains a logistic regression with mini-batch SGD + L2 reg.
   Splits 80/20 train/holdout by *position*, not by row, so
   the holdout doesn't leak (multiple snapshots per position
   would otherwise smear across folds).
5. Saves the trained weights to
   ``data/in_game_models/<bot>_logreg.json``.
6. Reports holdout accuracy / precision / recall / F1 / AUC.

If insufficient data (default: <500 labeled snapshots), prints a
status line and exits cleanly. The in-game model loader checks
for the file's existence; absence means "no trained model yet,
keep using heuristic".

Once a trained file exists, ``in_game.nba.predict`` (and the
parallel tennis path, when wired) will pick it up and use the
classifier's output as an additional feature in its existing
blend. Heuristic stays as the high-fidelity fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODELS_DIR = (Path(__file__).resolve().parents[3]
              / "data" / "in_game_models")


# The features we'll train on. Must match the keys the in-game
# model emits in its `features` dict. New features added later
# need a retrain to take effect.
DEFAULT_FEATURE_KEYS_NBA: Tuple[str, ...] = (
    "state_prob",
    "lead",
    "seconds_remaining",
    "market_velocity",
    "volatility",
    "divergence",
    "reversion_pull",
    "espn_win_proj_our",
    "vegas_win_prob_our",
    "our_critical_injuries",
    "opp_critical_injuries",
    "foul_trouble_count",
    "team_fg_pct_gap",
    "team_ft_pct_gap",
    "team_3pt_pct_gap",
    "team_to_gap",
    "team_reb_gap",
    "team_ast_gap",
    "our_recent_wins",
    "opp_recent_wins",
    "our_home_away",
    "news_injury_ours",
    "news_injury_opp",
    "cdn_live_pace",
    "cdn_fg_pct_gap",
    "cdn_3pt_pct_gap",
    "cdn_ft_rate_gap",
    "cdn_to_gap",
    "cdn_reb_gap",
    "cdn_fouls_out_ours",
    "cdn_foul_trouble_ours",
    "cdn_starter_plusminus_avg",
    "cdn_bench_plusminus_avg",
    "cdn_bench_minutes_share",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_closed_outcomes_nba(db_path: str) -> Dict[str, int]:
    """Map ticker -> 1/0 outcome for closed NBA bets."""
    out: Dict[str, int] = {}
    if not Path(db_path).exists():
        return out
    try:
        with closing(sqlite3.connect(db_path)) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT ticker, realized_pnl_cents "
                "FROM positions WHERE status='closed' "
                "  AND realized_pnl_cents IS NOT NULL"
            ).fetchall()
            for r in rows:
                t = r["ticker"]
                if not t:
                    continue
                try:
                    pnl = int(r["realized_pnl_cents"])
                except (TypeError, ValueError):
                    continue
                out[t] = 1 if pnl > 0 else 0
    except sqlite3.DatabaseError:
        pass
    return out


def _build_dataset(bot_key: str, db_path: str,
                     feature_keys: Tuple[str, ...]
                     ) -> Tuple[List[List[float]], List[int],
                                  List[str]]:
    """Load feature snapshots, join with closed-bet outcomes.
    Returns (X, y, ticker_groups) where ticker_groups is the per-row
    ticker so the train/holdout split can group by it.
    """
    from . import feature_log
    outcomes = _load_closed_outcomes_nba(db_path)
    X: List[List[float]] = []
    y: List[int] = []
    tickers: List[str] = []
    for entry in feature_log.iter_snapshots(bot_key=bot_key):
        t = entry.get("ticker")
        if not t or t not in outcomes:
            continue
        feats = entry.get("features") or {}
        row: List[float] = []
        all_present = True
        for k in feature_keys:
            v = feats.get(k)
            if v is None:
                # Missing features get 0 — common pattern for
                # logistic regression with sparse features.
                row.append(0.0)
            else:
                try:
                    row.append(float(v))
                except (TypeError, ValueError):
                    row.append(0.0)
        if all_present:
            pass  # placeholder; we keep all rows
        X.append(row)
        y.append(outcomes[t])
        tickers.append(t)
    return X, y, tickers


def _standardize(X: List[List[float]]
                   ) -> Tuple[List[List[float]], List[float], List[float]]:
    """Per-feature z-score normalization. Returns (X_normed, means,
    stds). Stds clamped to >= 1e-6 to avoid divide-by-zero on
    constant features.
    """
    if not X:
        return X, [], []
    n_features = len(X[0])
    n_rows = len(X)
    means = [0.0] * n_features
    for row in X:
        for j in range(n_features):
            means[j] += row[j]
    means = [m / n_rows for m in means]
    stds = [0.0] * n_features
    for row in X:
        for j in range(n_features):
            stds[j] += (row[j] - means[j]) ** 2
    stds = [max(1e-6, math.sqrt(s / n_rows)) for s in stds]
    X_n = [[(row[j] - means[j]) / stds[j] for j in range(n_features)]
            for row in X]
    return X_n, means, stds


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _train_logreg(X: List[List[float]], y: List[int],
                    *, epochs: int = 30, lr: float = 0.05,
                    l2: float = 0.001, batch: int = 64,
                    seed: int = 42
                    ) -> Tuple[List[float], float]:
    """Mini-batch SGD logistic regression. Returns (weights, bias).
    Pure Python, no numpy."""
    rng = random.Random(seed)
    n_rows = len(X)
    if n_rows == 0:
        return [], 0.0
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
    idx = list(range(n_rows))
    for epoch in range(epochs):
        rng.shuffle(idx)
        for start in range(0, n_rows, batch):
            mini = idx[start:start + batch]
            # Accumulate gradients across the mini-batch.
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for i in mini:
                z = b
                for j in range(n_features):
                    z += w[j] * X[i][j]
                p = _sigmoid(z)
                err = p - y[i]
                grad_b += err
                for j in range(n_features):
                    grad_w[j] += err * X[i][j]
            scale = lr / max(1, len(mini))
            for j in range(n_features):
                # Gradient + L2 shrinkage.
                w[j] -= scale * (grad_w[j] + l2 * w[j])
            b -= scale * grad_b
    return w, b


def _predict(w: List[float], bias: float, x: List[float]) -> float:
    z = bias
    for j in range(len(w)):
        z += w[j] * x[j]
    return _sigmoid(z)


def _evaluate(w: List[float], bias: float, X: List[List[float]],
                y: List[int]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    pos_scores: List[float] = []
    neg_scores: List[float] = []
    for i in range(len(X)):
        p = _predict(w, bias, X[i])
        pred = 1 if p >= 0.5 else 0
        actual = y[i]
        if pred == 1 and actual == 1: tp += 1
        elif pred == 1 and actual == 0: fp += 1
        elif pred == 0 and actual == 0: tn += 1
        else: fn += 1
        (pos_scores if actual == 1 else neg_scores).append(p)
    n = max(1, len(X))
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    pairs = 0
    correct = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            pairs += 1
            if ps > ns: correct += 1.0
            elif ps == ns: correct += 0.5
    auc = correct / pairs if pairs > 0 else 0.5
    return {
        "n": float(n), "tp": float(tp), "fp": float(fp),
        "tn": float(tn), "fn": float(fn),
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "roc_auc": auc,
    }


def train_bot(bot_key: str, db_path: str,
                feature_keys: Tuple[str, ...] = DEFAULT_FEATURE_KEYS_NBA,
                min_rows: int = 500,
                ) -> Optional[Dict[str, Any]]:
    """Run the full pipeline for one bot. Returns the saved model
    dict on success; ``None`` when there isn't enough data yet."""
    X_raw, y, tickers = _build_dataset(bot_key, db_path, feature_keys)
    print(f"loaded {len(X_raw)} labeled feature snapshots for "
          f"bot={bot_key}")
    if len(X_raw) < min_rows:
        print(f"insufficient data — need ≥ {min_rows} labeled rows. "
               f"Snapshot log will accumulate as the bot trades; "
               f"come back when the dashboard's been running a "
               f"few weeks.")
        return None
    # Group-aware train/holdout split: holdout = 20% of unique
    # tickers (not 20% of rows), so a position's snapshots all
    # land on the same side.
    unique_tickers = sorted(set(tickers))
    random.Random(42).shuffle(unique_tickers)
    holdout_n = max(1, len(unique_tickers) // 5)
    holdout_set = set(unique_tickers[:holdout_n])
    train_X, train_y = [], []
    hold_X, hold_y = [], []
    for i, t in enumerate(tickers):
        if t in holdout_set:
            hold_X.append(X_raw[i]); hold_y.append(y[i])
        else:
            train_X.append(X_raw[i]); train_y.append(y[i])
    print(f"split: {len(train_X)} train rows / {len(hold_X)} "
           f"holdout rows (across {holdout_n} holdout tickers)")
    # Standardize on training set; apply same transform to holdout.
    train_Xn, means, stds = _standardize(train_X)
    n_feat = len(means)
    hold_Xn = [[(row[j] - means[j]) / stds[j] for j in range(n_feat)]
                for row in hold_X]
    print("training logistic regression…")
    w, bias = _train_logreg(train_Xn, train_y)
    train_metrics = _evaluate(w, bias, train_Xn, train_y)
    hold_metrics = _evaluate(w, bias, hold_Xn, hold_y) if hold_X else {}
    print(f"  train metrics: {train_metrics}")
    print(f"  holdout metrics: {hold_metrics}")
    # Persist.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{bot_key}_logreg.json"
    payload = {
        "bot_key": bot_key,
        "feature_keys": list(feature_keys),
        "weights": w,
        "bias": bias,
        "means": means,
        "stds": stds,
        "trained_at": _now_iso(),
        "n_train": len(train_X),
        "n_holdout": len(hold_X),
        "train_metrics": train_metrics,
        "holdout_metrics": hold_metrics,
    }
    with model_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved model → {model_path}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bot", required=True,
                     help="Bot key, e.g. 'nba'")
    ap.add_argument("--db-path", required=True,
                     help="Path to the bot's sim.db (NBA path: "
                          "/root/nba/data/sim.db)")
    ap.add_argument("--min-rows", type=int, default=500,
                     help="Min labeled rows required before training")
    args = ap.parse_args()
    train_bot(args.bot, args.db_path, min_rows=args.min_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
