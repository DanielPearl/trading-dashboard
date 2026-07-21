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
    if out:
        log.info("flags: player map loaded (%d names)", len(out))
    return out


def flag_for(name: str | None, ticker: str | None) -> str:
    """Flag emoji + trailing space for this name, or ''."""
    name = (name or "").strip()
    t = (ticker or "").upper()
    if not name:
        return ""
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
    # Tennis: Sackmann name lookup (full name, then last-name match)
    if t.startswith(("KXATPMATCH", "KXWTAMATCH", "KXITFMATCH")):
        pm = _load_player_map()
        iso = pm.get(low)
        if iso is None and " " in low:
            last = low.rsplit(" ", 1)[-1]
            hits = {v for k, v in pm.items()
                    if k.rsplit(" ", 1)[-1] == last}
            iso = hits.pop() if len(hits) == 1 else None
        return (_iso2_flag(iso) + " ") if iso else ""
    return ""


def flag_matchup(title: str | None, ticker: str | None) -> str:
    """Prepend flags to both sides of an 'A vs B' matchup title."""
    title = title or ""
    if " vs " not in title:
        return title
    a, _, b = title.partition(" vs ")
    fa, fb = flag_for(a, ticker), flag_for(b, ticker)
    return f"{fa}{a} vs {fb}{b}"
