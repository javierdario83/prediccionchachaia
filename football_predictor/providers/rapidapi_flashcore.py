"""Cliente configurable para APIs tipo Flashscore/Flashcore vía RapidAPI.

RapidAPI requiere enviar ``X-RapidAPI-Key`` y ``X-RapidAPI-Host``. La key debe
entrar por variable de entorno o UI; nunca se guarda en el repositorio.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
import requests

DEFAULT_RAPIDAPI_HOST = "flashscore4.p.rapidapi.com"


@dataclass(frozen=True)
class RapidAPIContext:
    provider: str
    match_id: str | None
    home_name: str | None
    away_name: str | None
    lineups_available: bool
    injuries_count: int
    home_xg: float | None
    away_xg: float | None
    note: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class RapidAPIFlashcoreClient:
    """Cliente genérico para proveedores RapidAPI de Flashscore/Flashcore.

    Como cada listing de RapidAPI puede tener endpoints distintos, el path de
    partidos se deja configurable. El cliente intenta normalizar respuestas JSON
    comunes de partidos, participantes, lineups, injuries y xG.
    """

    def __init__(
        self,
        api_key: str,
        host: str = DEFAULT_RAPIDAPI_HOST,
        base_url: str | None = None,
        session: requests.Session | None = None,
    ):
        if not api_key:
            raise ValueError("RapidAPI requiere una API key.")
        if not host:
            raise ValueError("RapidAPI requiere X-RapidAPI-Host.")
        self.api_key = api_key
        self.host = host.strip()
        self.base_url = (base_url or f"https://{self.host}").rstrip("/")
        self.session = session or requests.Session()

    def get(self, path: str, params: dict[str, object] | None = None) -> dict[str, Any] | list[Any]:
        response = self.session.get(
            f"{self.base_url}/{path.lstrip('/')}",
            headers={"X-RapidAPI-Key": self.api_key, "X-RapidAPI-Host": self.host},
            params={key: value for key, value in (params or {}).items() if value is not None},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def matches_by_date(self, date: str, matches_path: str, date_param: str = "date") -> list[dict[str, Any]]:
        payload = self.get(matches_path, {date_param: date[:10]})
        return find_match_like_items(payload)

    def match_context(
        self,
        home_team: str,
        away_team: str,
        match_date: str | None,
        matches_path: str,
        date_param: str = "date",
    ) -> RapidAPIContext | None:
        if not match_date or not matches_path:
            return None
        matches = self.matches_by_date(match_date[:10], matches_path=matches_path, date_param=date_param)
        match = find_best_match(matches, home_team, away_team)
        if match is None:
            return None
        return build_context_from_match(match, home_team, away_team, provider=self.host)


def find_match_like_items(payload: Any) -> list[dict[str, Any]]:
    """Busca objetos que parecen partidos dentro de una respuesta JSON."""

    items: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if extract_team_names(value) != (None, None):
                items.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    # Quita duplicados por identidad textual simple.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item)[:500]
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def find_best_match(matches: list[dict[str, Any]], home_team: str, away_team: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = 0.0
    for item in matches:
        api_home, api_away = extract_team_names(item)
        if not api_home or not api_away:
            continue
        score = (team_similarity(home_team, api_home) + team_similarity(away_team, api_away)) / 2
        if score > best_score:
            best = item
            best_score = score
    return best if best_score >= 0.58 else None


def build_context_from_match(match: dict[str, Any], home_team: str, away_team: str, provider: str) -> RapidAPIContext:
    api_home, api_away = extract_team_names(match)
    home_xg, away_xg = extract_xg(match, api_home or home_team, api_away or away_team)
    injuries_count = count_injury_like_entries(match)
    lineups_available = has_lineup_like_data(match)
    match_id = extract_first_value(match, ["id", "matchId", "eventId", "fixtureId"])
    notes = []
    if lineups_available:
        notes.append("lineups/detalles de jugadores detectados")
    if injuries_count:
        notes.append(f"{injuries_count} registros tipo lesión/baja detectados")
    if home_xg is not None or away_xg is not None:
        notes.append(f"xG detectado: local {home_xg if home_xg is not None else 'N/D'}, visita {away_xg if away_xg is not None else 'N/D'}")
    return RapidAPIContext(
        provider=provider,
        match_id=str(match_id) if match_id is not None else None,
        home_name=api_home,
        away_name=api_away,
        lineups_available=lineups_available,
        injuries_count=injuries_count,
        home_xg=home_xg,
        away_xg=away_xg,
        note=" | ".join(notes) if notes else "RapidAPI conectado, sin contexto extra normalizable.",
    )


def extract_team_names(item: dict[str, Any]) -> tuple[str | None, str | None]:
    teams = item.get("teams") or item.get("participants") or item.get("competitors")
    if isinstance(teams, dict):
        home = teams.get("home") or teams.get("homeTeam") or teams.get("local")
        away = teams.get("away") or teams.get("awayTeam") or teams.get("visitor")
        home_name = extract_name(home)
        away_name = extract_name(away)
        if home_name and away_name:
            return home_name, away_name
    if isinstance(teams, list) and len(teams) >= 2:
        home_candidates = [team for team in teams if str(team.get("type", team.get("side", ""))).lower() in {"home", "local"}]
        away_candidates = [team for team in teams if str(team.get("type", team.get("side", ""))).lower() in {"away", "visitor"}]
        if home_candidates and away_candidates:
            return extract_name(home_candidates[0]), extract_name(away_candidates[0])
        return extract_name(teams[0]), extract_name(teams[1])

    home = item.get("homeTeam") or item.get("home") or item.get("localTeam")
    away = item.get("awayTeam") or item.get("away") or item.get("awayTeam") or item.get("visitorTeam")
    return extract_name(home), extract_name(away)


def extract_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ["name", "shortName", "participantName", "teamName", "title"]:
            if value.get(key):
                return str(value[key])
    return None


def extract_xg(match: dict[str, Any], home_name: str, away_name: str) -> tuple[float | None, float | None]:
    home_xg = None
    away_xg = None

    def walk(value: Any, side_hint: str | None = None) -> None:
        nonlocal home_xg, away_xg
        if isinstance(value, dict):
            key_text = " ".join(str(key).lower() for key in value.keys())
            if "xg" in key_text or "expected" in key_text:
                number = first_numeric(value)
                if number is not None:
                    side = side_hint or infer_side(value, home_name, away_name)
                    if side == "home" and home_xg is None:
                        home_xg = number
                    elif side == "away" and away_xg is None:
                        away_xg = number
            next_side = side_hint or infer_side(value, home_name, away_name)
            for child in value.values():
                walk(child, next_side)
        elif isinstance(value, list):
            for child in value:
                walk(child, side_hint)

    walk(match)
    return home_xg, away_xg


def infer_side(value: dict[str, Any], home_name: str, away_name: str) -> str | None:
    text = str(value).lower()
    if normalise_team_name(home_name) in normalise_team_name(text):
        return "home"
    if normalise_team_name(away_name) in normalise_team_name(text):
        return "away"
    side = str(value.get("side", value.get("type", ""))).lower()
    if side in {"home", "local"}:
        return "home"
    if side in {"away", "visitor"}:
        return "away"
    return None


def count_injury_like_entries(match: dict[str, Any]) -> int:
    count = 0

    def walk(value: Any) -> None:
        nonlocal count
        if isinstance(value, dict):
            text = " ".join(str(part).lower() for part in list(value.keys()) + list(value.values())[:5])
            if any(token in text for token in ["injury", "injured", "suspended", "missing", "doubtful"]):
                count += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(match)
    return count


def has_lineup_like_data(match: dict[str, Any]) -> bool:
    text = str(match).lower()
    return any(token in text for token in ["lineup", "formation", "starting", "substitutes", "bench"])


def extract_first_value(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    for value in item.values():
        if isinstance(value, dict):
            found = extract_first_value(value, keys)
            if found is not None:
                return found
    return None


def first_numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace("%", ""))
        except ValueError:
            return None
    if isinstance(value, dict):
        for child in value.values():
            number = first_numeric(child)
            if number is not None:
                return number
    if isinstance(value, list):
        for child in value:
            number = first_numeric(child)
            if number is not None:
                return number
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
    normalised = name.lower()
    for token in ["fc", "afc", "cf", "club", ".", "-", "_"]:
        normalised = normalised.replace(token, " ")
    return " ".join(normalised.split())
