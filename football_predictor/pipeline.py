"""Orquestacion del flujo autonomo: descargar, limpiar, guardar y predecir."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cleaner import clean_fixtures, clean_matches
from .backtesting import BacktestConfig, run_backtest
from .config import DEFAULT_SEASONS, LEAGUES, DATABASE_PATH, DownloadTarget
from .database import load_matches, save_matches, save_predictions
from .downloader import download_fixtures, download_many
from .poisson_model import PoissonFootballModel
from .weather import get_match_weather


@dataclass(frozen=True)
class UpdateSummary:
    downloaded_rows: int
    clean_rows: int
    inserted_rows: int


def build_targets(seasons: list[str] | None = None, leagues: list[str] | None = None) -> list[DownloadTarget]:
    seasons = seasons or DEFAULT_SEASONS
    leagues = leagues or list(LEAGUES.keys())
    return [DownloadTarget(season=season, league=league) for season in seasons for league in leagues]


def update_historical_data(
    seasons: list[str] | None = None,
    leagues: list[str] | None = None,
    db_path: Path = DATABASE_PATH,
) -> UpdateSummary:
    """Descarga historicos, los limpia y los guarda localmente."""

    raw = download_many(build_targets(seasons, leagues))
    clean = clean_matches(raw)
    inserted = save_matches(clean, db_path=db_path)
    return UpdateSummary(downloaded_rows=len(raw), clean_rows=len(clean), inserted_rows=inserted)


def train_model_from_database(db_path: Path = DATABASE_PATH, league_code: str | None = None) -> PoissonFootballModel:
    matches = load_matches(db_path=db_path)
    if league_code and "league_code" in matches.columns:
        matches = matches[matches["league_code"] == league_code]
    return PoissonFootballModel().fit(matches)


def predict_upcoming_matches(
    db_path: Path = DATABASE_PATH,
    league_codes: list[str] | None = None,
    limit: int = 50,
    save: bool = True,
    include_weather: bool = False,
) -> pd.DataFrame:
    """Descarga fixtures actuales y genera probabilidades para partidos futuros."""

    fixtures = clean_fixtures(download_fixtures())
    if league_codes:
        fixtures = fixtures[fixtures["LeagueCode"].isin(league_codes)]

    if limit:
        fixtures = fixtures.head(limit)

    predictions: list[pd.DataFrame] = []
    for league_code, league_fixtures in fixtures.groupby("LeagueCode", dropna=False):
        try:
            model = train_model_from_database(db_path=db_path, league_code=league_code)
        except ValueError:
            model = train_model_from_database(db_path=db_path, league_code=None)
        predictions.append(model.predict_dataframe(league_fixtures))

    result = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if include_weather and not result.empty:
        result = enrich_predictions_with_weather(result)
    result = rank_predictions(result)
    if save and not result.empty:
        save_predictions(result, db_path=db_path)
    return result


def predict_manual_match(
    home_team: str,
    away_team: str,
    league_code: str | None = None,
    db_path: Path = DATABASE_PATH,
    include_weather: bool = False,
) -> pd.DataFrame:
    model = train_model_from_database(db_path=db_path, league_code=league_code)
    result = pd.DataFrame([model.predict_match(home_team, away_team, league_code=league_code).as_dict()])
    if include_weather:
        result = enrich_predictions_with_weather(result)
    return rank_predictions(result)


def rank_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Ordena por confianza y agrega una columna de pick para el cliente."""

    if predictions.empty:
        return predictions
    ranked = predictions.copy()
    ranked["pick"] = ranked["recommended_market"]
    ranked["pick_probability"] = ranked["recommended_probability"]
    if "confidence_score" in ranked.columns:
        ranked = ranked.sort_values(["confidence_score", "pick_probability"], ascending=False)
    return ranked.reset_index(drop=True)


def enrich_predictions_with_weather(predictions: pd.DataFrame) -> pd.DataFrame:
    """Agrega clima Open-Meteo y reduce confianza si hay clima adverso."""

    if predictions.empty:
        return predictions
    enriched = predictions.copy()
    weather_rows = []
    for _, row in enriched.iterrows():
        weather = get_match_weather(row["home_team"], row.get("match_date"))
        weather_rows.append(weather.as_dict() if weather else {})
    weather_df = pd.DataFrame(weather_rows)
    for column in ["city", "temperature_c", "precipitation_probability", "wind_speed_kmh", "weather_risk", "note"]:
        if column not in weather_df.columns:
            weather_df[column] = None
    weather_df = weather_df.rename(
        columns={
            "city": "weather_city",
            "temperature_c": "weather_temperature_c",
            "precipitation_probability": "weather_precipitation_probability",
            "wind_speed_kmh": "weather_wind_speed_kmh",
            "weather_risk": "weather_risk",
            "note": "weather_note",
        }
    )
    enriched = pd.concat([enriched.reset_index(drop=True), weather_df.reset_index(drop=True)], axis=1)
    penalties = []
    for _, row in enriched.iterrows():
        risk = row.get("weather_risk")
        if risk == "Alto":
            penalties.append(0.08)
        elif risk == "Medio":
            penalties.append(0.04)
        else:
            penalties.append(0.0)
    enriched["confidence_score"] = (enriched["confidence_score"] - pd.Series(penalties)).clip(lower=0)
    enriched["confidence"] = enriched["confidence_score"].apply(_confidence_from_score)
    return enriched


def backtest_model(db_path: Path = DATABASE_PATH, league_code: str | None = None, max_test_matches: int | None = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = load_matches(db_path=db_path)
    if league_code and "league_code" in matches.columns:
        matches = matches[matches["league_code"] == league_code]
    return run_backtest(matches, BacktestConfig(max_test_matches=max_test_matches))


def _confidence_from_score(score: float) -> str:
    if score >= 0.72:
        return "Alta"
    if score >= 0.62:
        return "Media-alta"
    if score >= 0.52:
        return "Media"
    return "Baja"
