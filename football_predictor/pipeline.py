"""Orquestacion del flujo autonomo: descargar, limpiar, guardar y predecir."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cleaner import clean_fixtures, clean_matches
from .config import DEFAULT_SEASONS, LEAGUES, DATABASE_PATH, DownloadTarget
from .database import load_matches, save_matches, save_predictions
from .downloader import download_fixtures, download_many
from .poisson_model import PoissonFootballModel


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
    if save and not result.empty:
        save_predictions(result, db_path=db_path)
    return result


def predict_manual_match(
    home_team: str,
    away_team: str,
    league_code: str | None = None,
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    model = train_model_from_database(db_path=db_path, league_code=league_code)
    return pd.DataFrame([model.predict_match(home_team, away_team, league_code=league_code).as_dict()])
