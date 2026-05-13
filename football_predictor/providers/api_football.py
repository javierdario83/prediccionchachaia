"""Cliente opcional para API-Football/API-Sports.

La API key NO se guarda en el repositorio. Este adaptador queda disponible
para una integracion futura y debe recibir la key explicitamente al crear el cliente.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://v3.football.api-sports.io"


@dataclass(frozen=True)
class APIFootballContext:
    fixture_id: int | None
    fixture_status: str | None
    home_injuries: int
    away_injuries: int
    home_suspensions: int
    away_suspensions: int
    lineups_available: bool
    home_formation: str | None
    away_formation: str | None
    home_xg: float | None
    away_xg: float | None
    note: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class APIFootballClient:
    """Cliente liviano para endpoints utiles en prediccion."""

    def __init__(self, api_key: str, base_url: str = BASE_URL, session: requests.Session | None = None):
        if not api_key:
            raise ValueError("API-Football requiere una API key.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def get(self, endpoint: str, params: dict[str, object] | None = None) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            headers={"x-apisports-key": self.api_key},
            params={key: value for key, value in (params or {}).items() if value is not None},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def fixtures_by_date(self, date: str, league: int | None = None, season: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, object] = {"date": date}
        if league:
            params["league"] = league
        if season:
            params["season"] = season
        return self.get("fixtures", params).get("response", [])

    def injuries(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.get("injuries", {"fixture": fixture_id}).get("response", [])

    def lineups(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.get("fixtures/lineups", {"fixture": fixture_id}).get("response", [])

    def statistics(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.get("fixtures/statistics", {"fixture": fixture_id}).get("response", [])

    def find_fixture(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        league: int | None = None,
        season: int | None = None,
    ) -> dict[str, Any] | None:
        fixtures = self.fixtures_by_date(match_date[:10], league=league, season=season)
        best_fixture: dict[str, Any] | None = None
        best_score = 0.0
        for fixture in fixtures:
            teams = fixture.get("teams", {})
            api_home = teams.get("home", {}).get("name", "")
            api_away = teams.get("away", {}).get("name", "")
            score = (team_similarity(home_team, api_home) + team_similarity(away_team, api_away)) / 2
            if score > best_score:
                best_score = score
                best_fixture = fixture
        return best_fixture if best_score >= 0.62 else None

    def match_context(
        self,
        home_team: str,
        away_team: str,
        match_date: str | None,
        league: int | None = None,
        season: int | None = None,
    ) -> APIFootballContext | None:
        if not match_date:
            return None
        fixture = self.find_fixture(home_team, away_team, match_date, league=league, season=season)
        if not fixture:
            return None
        fixture_id = fixture.get("fixture", {}).get("id")
        if not fixture_id:
            return None

        injuries = self.injuries(int(fixture_id))
        lineups = self.lineups(int(fixture_id))
        statistics = self.statistics(int(fixture_id))
        context = build_context_from_api_payloads(
            fixture=fixture,
            injuries=injuries,
            lineups=lineups,
            statistics=statistics,
            home_team=home_team,
            away_team=away_team,
        )
        return context


def build_context_from_api_payloads(
    fixture: dict[str, Any],
    injuries: list[dict[str, Any]],
    lineups: list[dict[str, Any]],
    statistics: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> APIFootballContext:
    fixture_id = fixture.get("fixture", {}).get("id")
    status = fixture.get("fixture", {}).get("status", {}).get("short")
    teams = fixture.get("teams", {})
    api_home = teams.get("home", {}).get("name", home_team)
    api_away = teams.get("away", {}).get("name", away_team)

    home_injuries, home_suspensions = count_absences(injuries, api_home)
    away_injuries, away_suspensions = count_absences(injuries, api_away)
    home_formation, away_formation = parse_formations(lineups, api_home, api_away)
    home_xg, away_xg = parse_expected_goals(statistics, api_home, api_away)
    notes = []
    if home_injuries or away_injuries or home_suspensions or away_suspensions:
        notes.append(
            f"Bajas API-Football: local {home_injuries} lesiones/{home_suspensions} susp., "
            f"visita {away_injuries} lesiones/{away_suspensions} susp."
        )
    if home_formation or away_formation:
        notes.append(f"Alineaciones/formaciones disponibles: {home_formation or 'N/D'} vs {away_formation or 'N/D'}")
    if home_xg is not None or away_xg is not None:
        notes.append(f"xG API-Football: local {home_xg if home_xg is not None else 'N/D'}, visita {away_xg if away_xg is not None else 'N/D'}")

    return APIFootballContext(
        fixture_id=int(fixture_id) if fixture_id else None,
        fixture_status=status,
        home_injuries=home_injuries,
        away_injuries=away_injuries,
        home_suspensions=home_suspensions,
        away_suspensions=away_suspensions,
        lineups_available=bool(lineups),
        home_formation=home_formation,
        away_formation=away_formation,
        home_xg=home_xg,
        away_xg=away_xg,
        note=" | ".join(notes) if notes else "Sin contexto adicional disponible en API-Football.",
    )


def count_absences(injuries: list[dict[str, Any]], team_name: str) -> tuple[int, int]:
    injuries_count = 0
    suspensions_count = 0
    for item in injuries:
        api_team = item.get("team", {}).get("name", "")
        if team_similarity(team_name, api_team) < 0.62:
            continue
        absence_type = str(item.get("player", {}).get("type") or item.get("type") or "").lower()
        reason = str(item.get("player", {}).get("reason") or item.get("reason") or "").lower()
        if "susp" in absence_type or "susp" in reason:
            suspensions_count += 1
        else:
            injuries_count += 1
    return injuries_count, suspensions_count


def parse_formations(lineups: list[dict[str, Any]], home_team: str, away_team: str) -> tuple[str | None, str | None]:
    home_formation = None
    away_formation = None
    for lineup in lineups:
        team = lineup.get("team", {}).get("name", "")
        formation = lineup.get("formation")
        if team_similarity(home_team, team) >= 0.62:
            home_formation = formation
        elif team_similarity(away_team, team) >= 0.62:
            away_formation = formation
    return home_formation, away_formation


def parse_expected_goals(statistics: list[dict[str, Any]], home_team: str, away_team: str) -> tuple[float | None, float | None]:
    home_xg = None
    away_xg = None
    for item in statistics:
        team = item.get("team", {}).get("name", "")
        value = find_xg_value(item.get("statistics", []))
        if value is None:
            continue
        if team_similarity(home_team, team) >= 0.62:
            home_xg = value
        elif team_similarity(away_team, team) >= 0.62:
            away_xg = value
    return home_xg, away_xg


def find_xg_value(statistics: list[dict[str, Any]]) -> float | None:
    for stat in statistics:
        stat_type = str(stat.get("type", "")).lower()
        if "expected" in stat_type or stat_type in {"xg", "expected goals"}:
            value = stat.get("value")
            try:
                return None if pd.isna(value) else float(str(value).replace("%", ""))
            except (TypeError, ValueError):
                return None
    return None


def team_similarity(a: str, b: str) -> float:
    a_norm = normalise_team_name(a)
    b_norm = normalise_team_name(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm in b_norm or b_norm in a_norm:
        return 0.9
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def normalise_team_name(name: str) -> str:
    replacements = {
        "fc": "",
        "afc": "",
        "cf": "",
        "club": "",
        "the": "",
        ".": " ",
        "-": " ",
    }
    normalised = name.lower()
    for old, new in replacements.items():
        normalised = normalised.replace(old, new)
    return " ".join(normalised.split())
