"""Country flags for player / team names on the History table.

Resolution order for ``flag_for(name, ticker)``:

  1. League inference from the Kalshi ticker — NPB teams are Japanese,
     KBO Korean, MLB/NBA/WNBA American (Toronto franchises Canadian).
  2. World Cup tickers — the team name IS the country.
  3. Tennis tickers — player name → IOC code, lazily built from the
     tennis repo's Sackmann match CSVs (winner_name/winner_ioc +
     loser_name/loser_ioc, recent seasons only).

Returns the emoji + a space, or "" when the nationality is unknown
(darts / table-tennis players carry no country data anywhere we
ingest — better no flag than a wrong one).
"""
from __future__ import annotations

import csv
import glob
import logging
from pathlib import Path

log = logging.getLogger("dashboard.flags")

_IOC_TO_ISO2 = {
    "USA": "US", "ESP": "ES", "FRA": "FR", "GBR": "GB", "GER": "DE",
    "ITA": "IT", "SUI": "CH", "AUS": "AU", "ARG": "AR", "SRB": "RS",
    "RUS": "RU", "NED": "NL", "DEN": "DK", "GRE": "GR", "CRO": "HR",
    "POR": "PT", "CHI": "CL", "RSA": "ZA", "TPE": "TW", "KOR": "KR",
    "JPN": "JP", "CHN": "CN", "CAN": "CA", "BRA": "BR", "BEL": "BE",
    "AUT": "AT", "POL": "PL", "CZE": "CZ", "SVK": "SK", "UKR": "UA",
    "KAZ": "KZ", "BUL": "BG", "ROU": "RO", "HUN": "HU", "NOR": "NO",
    "SWE": "SE", "FIN": "FI", "IND": "IN", "MEX": "MX", "COL": "CO",
    "PER": "PE", "URU": "UY", "ECU": "EC", "VEN": "VE", "TUN": "TN",
    "MAR": "MA", "EGY": "EG", "ISR": "IL", "TUR": "TR", "GEO": "GE",
    "ARM": "AM", "AZE": "AZ", "BLR": "BY", "LAT": "LV", "LTU": "LT",
    "EST": "EE", "SLO": "SI", "BIH": "BA", "MKD": "MK", "MNE": "ME",
    "ALB": "AL", "CYP": "CY", "MDA": "MD", "IRL": "IE", "NZL": "NZ",
    "THA": "TH", "INA": "ID", "MAS": "MY", "PHI": "PH", "VIE": "VN",
    "HKG": "HK", "SGP": "SG", "UZB": "UZ", "PAK": "PK", "SRI": "LK",
    "DOM": "DO", "PUR": "PR", "CRC": "CR", "PAR": "PY", "BOL": "BO",
    "GUA": "GT", "ESA": "SV", "HON": "HN", "PAN": "PA", "JAM": "JM",
    "NGR": "NG", "GHA": "GH", "CIV": "CI", "SEN": "SN", "ALG": "DZ",
    "KEN": "KE", "ZIM": "ZW", "IRI": "IR", "IRQ": "IQ", "KSA": "SA",
    "UAE": "AE", "QAT": "QA", "KUW": "KW", "JOR": "JO", "LIB": "LB",
    "SYR": "SY", "LUX": "LU", "MON": "MC", "LIE": "LI", "AND": "AD",
    "ISL": "IS", "MLT": "MT",
}

_COUNTRY_NAME_TO_ISO2 = {
    "spain": "ES", "france": "FR", "germany": "DE", "italy": "IT",
    "portugal": "PT", "netherlands": "NL", "belgium": "BE",
    "croatia": "HR", "serbia": "RS", "switzerland": "CH",
    "austria": "AT", "poland": "PL", "denmark": "DK", "sweden": "SE",
    "norway": "NO", "argentina": "AR", "brazil": "BR", "uruguay": "UY",
    "colombia": "CO", "ecuador": "EC", "peru": "PE", "chile": "CL",
    "mexico": "MX", "canada": "CA", "united states": "US", "usa": "US",
    "japan": "JP", "south korea": "KR", "korea": "KR",
    "australia": "AU", "morocco": "MA", "senegal": "SN",
    "ghana": "GH", "nigeria": "NG", "cameroon": "CM", "tunisia": "TN",
    "egypt": "EG", "algeria": "DZ", "iran": "IR", "saudi arabia": "SA",
    "qatar": "QA", "uzbekistan": "UZ", "jordan": "JO", "ukraine": "UA",
    "turkey": "TR", "greece": "GR", "russia": "RU", "paraguay": "PY",
    "panama": "PA", "costa rica": "CR", "honduras": "HN",
    "new zealand": "NZ", "ivory coast": "CI", "cote d'ivoire": "CI",
}

# England / Scotland / Wales use flag tag sequences, not ISO pairs.
_SPECIAL_FLAGS = {
    "england": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E"
               "\U000E0067\U000E007F",
    "scotland": "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063"
                "\U000E0074\U000E007F",
    "wales": "\U0001F3F4\U000E0067\U000E0062\U000E0077\U000E006C"
             "\U000E0073\U000E007F",
}

# Lazily-built tennis player name → ISO2 (from Sackmann CSVs).
_PLAYER_ISO2: dict | None = None
# Companion index: last name → ISO2, or None when several players
# share the last name (ambiguous). Built once with the map above —
# the old code re-scanned the whole ~40k-name map per lookup, which
# py-spy showed as the single hottest frame of a page render (every
# history row does up to five flag lookups).
_PLAYER_LAST_ISO2: dict | None = None
# Result memo — flags are stable for the life of the process, and the
# History table asks for the same few thousand (name, ticker, context)
# triples on every render. Only NON-EMPTY results are memoized: an
# empty result can be transient (the Wikidata budget gate), and the
# "" paths are all O(1) dict probes now anyway.
_FLAG_MEMO: dict = {}

_TENNIS_DATA_GLOBS = (
    "/root/tennis-forecast/data/raw/{tour}/{tour}_matches_202[2-9].csv",
    str(Path(__file__).resolve().parents[3] / "Tennis Forecast" / "data"
        / "raw" / "{tour}" / "{tour}_matches_202[2-9].csv"),
)


def _iso2_flag(iso2: str) -> str:
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso2.upper())


def _load_player_map() -> dict:
    global _PLAYER_ISO2
    if _PLAYER_ISO2 is not None:
        return _PLAYER_ISO2
    out: dict = {}
    for tour in ("atp", "wta"):
        paths: list[str] = []
        for pat in _TENNIS_DATA_GLOBS:
            paths.extend(glob.glob(pat.format(tour=tour)))
        for path in sorted(set(paths)):
            try:
                with open(path, newline="", encoding="utf-8",
                          errors="replace") as fh:
                    for row in csv.DictReader(fh):
                        for nk, ck in (("winner_name", "winner_ioc"),
                                        ("loser_name", "loser_ioc")):
                            name = (row.get(nk) or "").strip().lower()
                            iso = _IOC_TO_ISO2.get(
                                (row.get(ck) or "").strip().upper())
                            if name and iso:
                                out.setdefault(name, iso)
            except OSError:
                continue
    _PLAYER_ISO2 = out
    # Build the last-name index in the same pass: unique last name →
    # its ISO2; last names shared across countries → None (ambiguous,
    # same semantics as the old per-lookup set comprehension).
    global _PLAYER_LAST_ISO2
    last_map: dict = {}
    for full, iso in out.items():
        last = full.rsplit(" ", 1)[-1]
        if last not in last_map:
            last_map[last] = iso
        elif last_map[last] != iso:
            last_map[last] = None
    _PLAYER_LAST_ISO2 = last_map
    if out:
        log.info("flags: player map loaded (%d names)", len(out))
    return out


def _load_last_map() -> dict:
    if _PLAYER_LAST_ISO2 is None:
        _load_player_map()
    return _PLAYER_LAST_ISO2 or {}


_WIKI_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" \
    / "player_countries.json"
_WIKI_CACHE: dict | None = None
_WIKI_RECENT: list = []          # timestamps of live lookups (budget)
_WIKI_BUDGET_PER_MIN = 2         # never let flag lookups slow a render


def _wiki_cache() -> dict:
    global _WIKI_CACHE
    if _WIKI_CACHE is None:
        try:
            import json
            _WIKI_CACHE = json.loads(_WIKI_CACHE_PATH.read_text())
        except (OSError, ValueError):
            _WIKI_CACHE = {}
    return _WIKI_CACHE


def _wiki_cache_put(name: str, iso2: str | None) -> None:
    cache = _wiki_cache()
    cache[name] = iso2
    try:
        import json
        _WIKI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _WIKI_CACHE_PATH.write_text(json.dumps(cache, indent=0,
                                                sort_keys=True))
    except OSError:
        pass


def _wikidata_iso2(full_name: str, sport_hint: str) -> str | None:
    """Country of citizenship via Wikidata (free, no key). Cached
    forever (misses too); at most _WIKI_BUDGET_PER_MIN live lookups a
    minute so a page render can never stall on the network."""
    import time as _time
    key = full_name.lower()
    cache = _wiki_cache()
    if key in cache:
        return cache[key]
    now = _time.time()
    while _WIKI_RECENT and now - _WIKI_RECENT[0] > 60:
        _WIKI_RECENT.pop(0)
    if len(_WIKI_RECENT) >= _WIKI_BUDGET_PER_MIN:
        return None            # over budget — try again a later render
    _WIKI_RECENT.append(now)
    try:
        import requests
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbsearchentities", "search": full_name,
                    "language": "en", "type": "item", "limit": 5,
                    "format": "json"},
            timeout=4, headers={"User-Agent": "kalshi-dashboard/1.0"})
        hits = (r.json().get("search") or [])
        qid = None
        for h in hits:
            desc = (h.get("description") or "").lower()
            if sport_hint in desc and "player" in desc:
                qid = h.get("id")
                break
        if qid is None:
            _wiki_cache_put(key, None)
            return None
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbgetclaims", "entity": qid,
                    "property": "P27", "format": "json"},
            timeout=4, headers={"User-Agent": "kalshi-dashboard/1.0"})
        claims = (r.json().get("claims") or {}).get("P27") or []
        country_qid = (claims[0]["mainsnak"]["datavalue"]["value"]["id"]
                        if claims else None)
        if not country_qid:
            _wiki_cache_put(key, None)
            return None
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbgetclaims", "entity": country_qid,
                    "property": "P297", "format": "json"},
            timeout=4, headers={"User-Agent": "kalshi-dashboard/1.0"})
        p297 = (r.json().get("claims") or {}).get("P297") or []
        iso2 = (p297[0]["mainsnak"]["datavalue"]["value"]
                 if p297 else None)
        _wiki_cache_put(key, iso2)
        return iso2
    except Exception:  # noqa: BLE001 — flags are cosmetic
        return None


def _expand_from_context(name: str, context: str | None) -> str:
    """Titles carry LAST names ("Samson") while position records carry
    full names ("Laura Samson vs Maya Joint") — expand through the
    matchup context so the nationality lookup sees the full name."""
    if not context:
        return name
    low = name.lower()
    for part in context.split(" vs "):
        part = part.strip()
        pl = part.lower()
        if pl == low:
            return part
        if pl.endswith(" " + low) or low in pl.split():
            return part
    return name


def flag_for(name: str | None, ticker: str | None,
             context: str | None = None) -> str:
    """Flag emoji + trailing space for this name, or ''."""
    name = (name or "").strip()
    t = (ticker or "").upper()
    if not name:
        return ""
    memo_key = (name, t[:14], context or "")
    hit = _FLAG_MEMO.get(memo_key)
    if hit is not None:
        return hit
    flag = _flag_for_uncached(name, t, context)
    if flag:
        _FLAG_MEMO[memo_key] = flag
    return flag


def _flag_for_uncached(name: str, t: str, context: str | None) -> str:
    low = name.lower()
    if low in _SPECIAL_FLAGS:
        return _SPECIAL_FLAGS[low] + " "
    # League-country tickers
    if t.startswith("KXNPBGAME"):
        return _iso2_flag("JP") + " "
    if t.startswith("KXKBOGAME"):
        return _iso2_flag("KR") + " "
    if t.startswith(("KXMLBGAME", "KXNBA", "KXWNBAGAME")):
        return _iso2_flag("CA" if "toronto" in low else "US") + " "
    # World Cup: the team name is a country
    if t.startswith(("KXWCGAME", "KXWCADVANCE")):
        iso = _COUNTRY_NAME_TO_ISO2.get(low)
        return (_iso2_flag(iso) + " ") if iso else ""
    # Tennis: expand last names through the matchup context, then
    # Sackmann lookup, then unique-last-name, then Wikidata fallback.
    if t.startswith(("KXATPMATCH", "KXWTAMATCH", "KXITFMATCH")):
        full = _expand_from_context(name, context)
        low = full.lower()
        pm = _load_player_map()
        iso = pm.get(low)
        if iso is None and " " in low:
            iso = _load_last_map().get(low.rsplit(" ", 1)[-1])
        if iso is None and " " in low:
            iso = _wikidata_iso2(full, "tennis")
        return (_iso2_flag(iso) + " ") if iso else ""
    # Darts / table tennis: no nationality in our data — Wikidata only.
    if t.startswith(("KXDARTS", "KXPDC", "KXPREMDARTS")):
        full = _expand_from_context(name, context)
        if " " in full:
            iso = _wikidata_iso2(full, "darts")
            return (_iso2_flag(iso) + " ") if iso else ""
        return ""
    return ""


def flag_matchup(title: str | None, ticker: str | None,
                 context: str | None = None) -> str:
    """Prepend flags to both sides of an 'A vs B' matchup title."""
    title = title or ""
    if " vs " not in title:
        return title
    a, _, b = title.partition(" vs ")
    fa = flag_for(a, ticker, context)
    fb = flag_for(b, ticker, context)
    return f"{fa}{a} vs {fb}{b}"
