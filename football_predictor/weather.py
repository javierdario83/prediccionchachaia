"""Integracion opcional con Open-Meteo para contexto climatico del partido.

Open-Meteo no requiere API key para uso no comercial. Se usa solo como senal
contextual: el modelo base sigue funcionando aunque el clima no este disponible.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Mapeo practico para las ligas iniciales del MVP. Cuando un equipo no esta en
# esta lista se intenta geocodificar el nombre del equipo directamente.
TEAM_CITY_OVERRIDES: dict[str, str] = {
    "Arsenal": "London",
    "Chelsea": "London",
    "Tottenham": "London",
    "West Ham": "London",
    "Crystal Palace": "London",
    "Fulham": "London",
    "Man City": "Manchester",
    "Man United": "Manchester",
    "Everton": "Liverpool",
    "Liverpool": "Liverpool",
    "Aston Villa": "Birmingham",
    "Wolves": "Wolverhampton",
    "Newcastle": "Newcastle upon Tyne",
    "Brighton": "Brighton",
    "Bournemouth": "Bournemouth",
    "Leeds": "Leeds",
    "Burnley": "Burnley",
    "Brentford": "London",
    "Nott'm Forest": "Nottingham",
    "Real Madrid": "Madrid",
    "Ath Madrid": "Madrid",
    "Barcelona": "Barcelona",
    "Espanol": "Barcelona",
    "Sevilla": "Seville",
    "Betis": "Seville",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
    "Sociedad": "San Sebastian",
    "Ath Bilbao": "Bilbao",
    "Celta": "Vigo",
    "Getafe": "Getafe",
    "Osasuna": "Pamplona",
    "Mallorca": "Palma",
    "Girona": "Girona",
    "Milan": "Milan",
    "Inter": "Milan",
    "Juventus": "Turin",
    "Torino": "Turin",
    "Roma": "Rome",
    "Lazio": "Rome",
    "Napoli": "Naples",
    "Atalanta": "Bergamo",
    "Fiorentina": "Florence",
    "Bologna": "Bologna",
    "Genoa": "Genoa",
    "Bayern Munich": "Munich",
    "Dortmund": "Dortmund",
    "Leverkusen": "Leverkusen",
    "Ein Frankfurt": "Frankfurt am Main",
    "Stuttgart": "Stuttgart",
    "Wolfsburg": "Wolfsburg",
    "Freiburg": "Freiburg im Breisgau",
    "Mainz": "Mainz",
    "Paris SG": "Paris",
    "Marseille": "Marseille",
    "Lyon": "Lyon",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Rennes": "Rennes",
    "Nice": "Nice",
    "Nantes": "Nantes",
    "Lens": "Lens",
}


@dataclass(frozen=True)
class WeatherContext:
    city: str
    temperature_c: float | None
    precipitation_probability: float | None
    wind_speed_kmh: float | None
    weather_risk: str
    note: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def get_match_weather(home_team: str, match_date: object | None = None) -> WeatherContext | None:
    """Obtiene clima estimado para la ciudad del equipo local.

    Devuelve ``None`` si Open-Meteo no responde, si no hay coordenadas o si la
    fecha no esta disponible en el rango de forecast. Esto evita que una falla de
    red rompa las predicciones.
    """

    city = TEAM_CITY_OVERRIDES.get(home_team, home_team)
    coordinates = geocode_city(city)
    if coordinates is None:
        return None

    target_date = _normalise_date(match_date) or date.today()
    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": coordinates[0],
                "longitude": coordinates[1],
                "daily": "temperature_2m_max,precipitation_probability_max,wind_speed_10m_max",
                "timezone": "auto",
                "forecast_days": 16,
            },
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    daily = response.json().get("daily", {})
    dates = daily.get("time", [])
    target = target_date.isoformat()
    if target not in dates:
        return None

    index = dates.index(target)
    temperature = _value_at(daily.get("temperature_2m_max"), index)
    precipitation = _value_at(daily.get("precipitation_probability_max"), index)
    wind = _value_at(daily.get("wind_speed_10m_max"), index)
    risk, note = classify_weather_risk(precipitation, wind)
    return WeatherContext(
        city=city,
        temperature_c=temperature,
        precipitation_probability=precipitation,
        wind_speed_kmh=wind,
        weather_risk=risk,
        note=note,
    )


@lru_cache(maxsize=256)
def geocode_city(city: str) -> tuple[float, float] | None:
    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    results = response.json().get("results") or []
    if not results:
        return None
    first = results[0]
    return float(first["latitude"]), float(first["longitude"])


def classify_weather_risk(precipitation_probability: float | None, wind_speed_kmh: float | None) -> tuple[str, str]:
    precipitation = precipitation_probability or 0.0
    wind = wind_speed_kmh or 0.0
    if precipitation >= 70 or wind >= 45:
        return "Alto", "Clima adverso: lluvia/viento pueden bajar ritmo y aumentar incertidumbre."
    if precipitation >= 40 or wind >= 30:
        return "Medio", "Clima moderado: conviene bajar ligeramente la confianza del pronostico."
    return "Bajo", "Clima sin alerta fuerte para el modelo."


def weather_confidence_penalty(weather: WeatherContext | None) -> float:
    if weather is None:
        return 0.0
    if weather.weather_risk == "Alto":
        return 0.08
    if weather.weather_risk == "Medio":
        return 0.04
    return 0.0


def _value_at(values: list[object] | None, index: int) -> float | None:
    if not values or index >= len(values):
        return None
    value = values[index]
    return None if value is None else float(value)


def _normalise_date(value: object | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None
