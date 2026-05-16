"""Cross-sport news scanner.

Polls ESPN's free per-sport news endpoint and surfaces recent
headlines that match injury / questionable-status keywords. The
in-game models read this to soft-flag positions where a fresh news
story could move the market in the next few minutes.

Public API
----------
``recent_injury_signals(sport_path, keywords, max_age_hours=24)``
    Returns a list of ``{"headline": str, "published": iso str,
    "description": str, "matched_keywords": [str]}``. Empty when
    ESPN is unreachable or no matches.

``injury_signal_count(sport_path, keywords, max_age_hours=12)``
    Convenience — just the count, for use as a numeric feature.

Caching
-------
News endpoint is cached 5 minutes per sport path; the in-game
model is called every 30s but ESPN updates news at most a few
times an hour, so the TTL is well-tuned.

What this is and isn't
----------------------
This is keyword matching, not NLP sentiment. It flags "X is OUT",
"Y is questionable", "Z suffers injury". It does NOT do tone
analysis. Good enough as a signal until a proper NLP layer lands.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("dashboard.in_game.news")


# Per-sport ESPN news endpoint paths. Keys mirror what the in-game
# modules pass in.
NEWS_PATHS: Dict[str, str] = {
    "nba": "basketball/nba",
    "tennis": "tennis",
    # ESPN doesn't carry standalone table-tennis or darts news
    # feeds — those bots get an empty signal until we plug into
    # a different source.
}


# Cache keyed by sport path. Value is (timestamp_fetched, articles).
_NEWS_CACHE: Dict[str, Tuple[float, List[dict]]] = {}
_NEWS_TTL_SECONDS = 300.0  # 5 minutes


# Keywords that flag a headline as injury / availability related.
# Lowercased before match.
DEFAULT_INJURY_KEYWORDS: Tuple[str, ...] = (
    "injur", "out for", "ruled out", "questionable", "doubtful",
    "day-to-day", "day to day", "limping", "limited", "minutes",
    "rest", "sprain", "strain", "tear", "fracture", "concussion",
    "knee", "ankle", "hamstring", "shoulder", "back",
    "medical timeout", "withdraw", "retire", "covid",
)


def _fetch_news(sport_path: str) -> List[dict]:
    """Pull and cache the ESPN news feed for a sport. 5-minute TTL.
    Returns ``[]`` on any failure so callers can chain freely.
    """
    if not sport_path:
        return []
    now = time.time()
    cached = _NEWS_CACHE.get(sport_path)
    if cached and (now - cached[0]) < _NEWS_TTL_SECONDS:
        return cached[1]
    url = (f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{sport_path}/news?limit=40")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "kalshi-dashboard/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.load(r)
    except Exception as exc:  # noqa: BLE001
        log.debug("ESPN news fetch failed (%s): %s", sport_path, exc)
        _NEWS_CACHE[sport_path] = (now, [])
        return []
    articles = data.get("articles") or []
    _NEWS_CACHE[sport_path] = (now, articles)
    return articles


def _parse_published(iso: str | None) -> Optional[datetime]:
    if not iso:
        return None
    try:
        # ESPN uses ISO 8601 like "2026-05-16T18:00Z" or with offset.
        s = iso.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _article_text(article: dict) -> str:
    parts = [
        article.get("headline") or "",
        article.get("description") or "",
    ]
    return " ".join(parts).lower()


def recent_injury_signals(sport_path: str,
                            keywords: List[str],
                            injury_keywords: Tuple[str, ...] = DEFAULT_INJURY_KEYWORDS,
                            max_age_hours: int = 24,
                            ) -> List[dict]:
    """Match recent ESPN news articles where one of ``keywords``
    (team code, player name, etc.) AND one of ``injury_keywords``
    both appear in the headline/description.

    Each result is ``{"headline", "published", "description",
    "matched_keywords": [...]}``.
    """
    if not keywords:
        return []
    norm_keywords = [k.lower() for k in keywords if k]
    if not norm_keywords:
        return []
    norm_inj = tuple(k.lower() for k in injury_keywords)
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - max_age_hours * 3600
    out: List[dict] = []
    for article in _fetch_news(sport_path):
        published = _parse_published(article.get("published"))
        if published is None or published.timestamp() < cutoff:
            continue
        text = _article_text(article)
        if not any(inj in text for inj in norm_inj):
            continue
        matched: List[str] = []
        for kw in norm_keywords:
            if kw in text:
                matched.append(kw)
        if not matched:
            continue
        out.append({
            "headline": article.get("headline", ""),
            "description": article.get("description", ""),
            "published": (published.isoformat()
                            if published else ""),
            "matched_keywords": matched,
        })
    return out


def injury_signal_count(sport_path: str, keywords: List[str],
                          max_age_hours: int = 12) -> int:
    """Count of recent injury-related articles matching any keyword.
    Used as a 0-N integer feature in the in-game models.
    """
    return len(recent_injury_signals(sport_path, keywords,
                                       max_age_hours=max_age_hours))
