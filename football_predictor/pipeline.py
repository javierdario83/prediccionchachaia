"""Orquestacion del flujo autonomo: descargar, limpiar, guardar y predecir."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

from .cleaner import clean_fixtures, clean_matches
from .backtesting import BacktestConfig, run_backtest
from .calibration import apply_calibration, build_calibration_table, summarise_markets
from .config import DEFAULT_SEASONS, LEAGUES, DATABASE_PATH, DownloadTarget
from .database import load_matches, save_matches, save_predictions
from .downloader import download_fixtures, download_many
from .features import confidence_from_score, enrich_with_reliability_signals
from .odds import add_market_probabilities
from .poisson_model import PoissonFootballModel
from .providers.api_football import APIFootballClient
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


def neutral_model() -> PoissonFootballModel:
    """Crea un modelo base para que la app pueda predecir antes de descargar historicos."""

    model = PoissonFootballModel()
    model.is_fitted = True
    return model


def train_model_or_neutral(db_path: Path = DATABASE_PATH, league_code: str | None = None) -> PoissonFootballModel:
    """Entrena con SQLite y usa medias neutrales si todavía no hay datos locales."""

    try:
        return train_model_from_database(db_path=db_path, league_code=league_code)
    except ValueError:
        if league_code is not None:
            try:
                return train_model_from_database(db_path=db_path, league_code=None)
            except ValueError:
                pass
        return neutral_model()


def predict_upcoming_matches(
    db_path: Path = DATABASE_PATH,
    league_codes: list[str] | None = None,
    limit: int = 50,
    save: bool = True,
    include_weather: bool = False,
    calibrate: bool = False,
    api_football_key: str | None = None,
) -> pd.DataFrame:
    """Descarga fixtures actuales y genera probabilidades para partidos futuros."""

    fixtures = clean_fixtures(download_fixtures())
    if league_codes:
        fixtures = fixtures[fixtures["LeagueCode"].isin(league_codes)]

    if limit:
        fixtures = fixtures.head(limit)

    predictions: list[pd.DataFrame] = []
    for league_code, league_fixtures in fixtures.groupby("LeagueCode", dropna=False):
        model = train_model_or_neutral(db_path=db_path, league_code=league_code)
        predicted = model.predict_dataframe(league_fixtures)
        predictions.append(_attach_fixture_odds(predicted, league_fixtures))

    result = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    result = enrich_predictions_with_reliability(result, db_path=db_path)
    if include_weather and not result.empty:
        result = enrich_predictions_with_weather(result)
    if api_football_key and not result.empty:
        result = enrich_predictions_with_api_football(result, api_key=api_football_key)
    result = add_market_probabilities(result)
    if calibrate and not result.empty:
        result = calibrate_predictions(result, db_path=db_path)
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
    calibrate: bool = False,
    api_football_key: str | None = None,
) -> pd.DataFrame:
    model = train_model_or_neutral(db_path=db_path, league_code=league_code)
    result = pd.DataFrame([model.predict_match(home_team, away_team, league_code=league_code).as_dict()])
    result = enrich_predictions_with_reliability(result, db_path=db_path, league_code=league_code)
    if include_weather:
        result = enrich_predictions_with_weather(result)
    if api_football_key:
        result = enrich_predictions_with_api_football(result, api_key=api_football_key)
    if calibrate:
        result = calibrate_predictions(result, db_path=db_path, league_code=league_code)
    return rank_predictions(result)


def rank_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Ordena por confianza y agrega una columna de pick para el cliente."""

    if predictions.empty:
        return predictions
    ranked = predictions.copy()
    ranked["pick"] = ranked["recommended_market"]
    ranked["pick_probability"] = ranked["recommended_probability"]
    if "action" not in ranked.columns:
        ranked["action"] = ranked.apply(_fallback_action, axis=1)
    ranked = _apply_calibration_to_action(ranked)
    ranked = _apply_value_gap_to_action(ranked)
    if "confidence_score" in ranked.columns:
        ranked["action_rank"] = ranked["action"].map({"Recomendado": 0, "Informativo": 1, "Evitar": 2}).fillna(3)
        ranked = ranked.sort_values(["action_rank", "confidence_score", "pick_probability"], ascending=[True, False, False])
        ranked = ranked.drop(columns=["action_rank"])
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
    enriched["confidence"] = enriched["confidence_score"].apply(confidence_from_score)
    if "action" in enriched.columns:
        enriched.loc[enriched["weather_risk"] == "Alto", "action"] = "Informativo"
    return enriched


def enrich_predictions_with_api_football(predictions: pd.DataFrame, api_key: str | None = None) -> pd.DataFrame:
    """Agrega lesiones/alineaciones/xG desde API-Football y ajusta confianza."""

    if predictions.empty:
        return predictions
    key = api_key or os.getenv("API_FOOTBALL_KEY")
    if not key:
        return predictions
    client = APIFootballClient(key)
    enriched = predictions.copy()
    context_rows = []
    for _, row in enriched.iterrows():
        try:
            context = client.match_context(
                home_team=row["home_team"],
                away_team=row["away_team"],
                match_date=row.get("match_date"),
            )
        except Exception as exc:
            context = None
            context_rows.append({"api_context_note": f"API-Football no disponible: {exc}"})
            continue
        if context is None:
            context_rows.append({"api_context_note": "API-Football sin fixture compatible para este partido."})
        else:
            context_rows.append(_api_context_to_prediction_row(context.as_dict()))

    context_df = pd.DataFrame(context_rows)
    enriched = pd.concat([enriched.reset_index(drop=True), context_df.reset_index(drop=True)], axis=1)
    return _apply_api_football_adjustments(enriched)


def _api_context_to_prediction_row(context: dict[str, object]) -> dict[str, object]:
    return {
        "api_football_fixture_id": context.get("fixture_id"),
        "api_football_status": context.get("fixture_status"),
        "api_home_injuries": context.get("home_injuries"),
        "api_away_injuries": context.get("away_injuries"),
        "api_home_suspensions": context.get("home_suspensions"),
        "api_away_suspensions": context.get("away_suspensions"),
        "api_lineups_available": int(bool(context.get("lineups_available"))),
        "api_home_formation": context.get("home_formation"),
        "api_away_formation": context.get("away_formation"),
        "api_home_xg": context.get("home_xg"),
        "api_away_xg": context.get("away_xg"),
        "api_context_note": context.get("note"),
    }


def _apply_api_football_adjustments(predictions: pd.DataFrame) -> pd.DataFrame:
    adjusted = predictions.copy()
    for index, row in adjusted.iterrows():
        confidence = float(row.get("confidence_score", 0.0) or 0.0)
        home_absences = int(row.get("api_home_injuries", 0) or 0) + int(row.get("api_home_suspensions", 0) or 0)
        away_absences = int(row.get("api_away_injuries", 0) or 0) + int(row.get("api_away_suspensions", 0) or 0)
        market = str(row.get("recommended_market", ""))
        if home_absences >= 3 and "Gana local" in market:
            confidence -= 0.06
            adjusted.at[index, "action"] = "Informativo"
            adjusted.at[index, "action_reason"] = "API-Football reporta varias bajas del local"
        if away_absences >= 3 and "Gana visitante" in market:
            confidence -= 0.06
            adjusted.at[index, "action"] = "Informativo"
            adjusted.at[index, "action_reason"] = "API-Football reporta varias bajas del visitante"

        home_xg = row.get("api_home_xg")
        away_xg = row.get("api_away_xg")
        if not pd.isna(home_xg) and not pd.isna(away_xg):
            xg_total = float(home_xg) + float(away_xg)
            if "Over" in market and xg_total >= 2.8:
                confidence += 0.03
            elif "Over" in market and xg_total <= 2.0:
                confidence -= 0.04
            elif "Under" in market and xg_total <= 2.2:
                confidence += 0.03
            elif "Under" in market and xg_total >= 3.0:
                confidence -= 0.04

        adjusted.at[index, "confidence_score"] = max(0.0, min(1.0, confidence))
    adjusted["confidence"] = adjusted["confidence_score"].apply(confidence_from_score)
    return adjusted


def enrich_predictions_with_reliability(
    predictions: pd.DataFrame,
    db_path: Path = DATABASE_PATH,
    league_code: str | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    matches = load_matches(db_path=db_path)
    if league_code and "league_code" in matches.columns:
        matches = matches[matches["league_code"] == league_code]
    return enrich_with_reliability_signals(predictions, matches)


def _attach_fixture_odds(predictions: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    odds_columns = ["B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]
    available = [column for column in odds_columns if column in fixtures.columns]
    if not available:
        return predictions
    odds = fixtures[available].reset_index(drop=True).rename(
        columns={
            "B365H": "b365_home",
            "B365D": "b365_draw",
            "B365A": "b365_away",
            "B365>2.5": "b365_over25",
            "B365<2.5": "b365_under25",
        }
    )
    return pd.concat([predictions.reset_index(drop=True), odds], axis=1)


def _apply_value_gap_to_action(predictions: pd.DataFrame) -> pd.DataFrame:
    if "value_gap" not in predictions.columns:
        return predictions
    adjusted = predictions.copy()
    for index, row in adjusted.iterrows():
        value_gap = row.get("value_gap")
        if pd.isna(value_gap):
            continue
        if float(value_gap) >= 0.05 and row.get("action") == "Informativo" and float(row.get("confidence_score", 0) or 0) >= 0.60:
            adjusted.at[index, "action"] = "Recomendado"
            adjusted.at[index, "action_reason"] = "valor positivo frente a cuotas y confianza suficiente"
        elif float(value_gap) <= -0.08 and row.get("action") == "Recomendado":
            adjusted.at[index, "action"] = "Informativo"
            adjusted.at[index, "action_reason"] = "el mercado no respalda suficiente ventaja del modelo"
    return adjusted


def _fallback_action(row: pd.Series) -> str:
    score = float(row.get("confidence_score", 0.0) or 0.0)
    probability = float(row.get("recommended_probability", 0.0) or 0.0)
    if score >= 0.64 and probability >= 0.64:
        return "Recomendado"
    if score >= 0.52 and probability >= 0.56:
        return "Informativo"
    return "Evitar"


def calibrate_predictions(
    predictions: pd.DataFrame,
    db_path: Path = DATABASE_PATH,
    league_code: str | None = None,
    max_test_matches: int | None = 500,
) -> pd.DataFrame:
    details, _summary = backtest_model(db_path=db_path, league_code=league_code, max_test_matches=max_test_matches)
    calibration_table = build_calibration_table(details)
    return apply_calibration(predictions, calibration_table)


def _apply_calibration_to_action(predictions: pd.DataFrame) -> pd.DataFrame:
    if "calibrated_pick_probability" not in predictions.columns:
        return predictions
    adjusted = predictions.copy()
    for index, row in adjusted.iterrows():
        calibrated = row.get("calibrated_pick_probability")
        if pd.isna(calibrated):
            continue
        adjusted.at[index, "pick_probability"] = float(calibrated)
        if float(calibrated) < 0.52:
            adjusted.at[index, "action"] = "Evitar"
            adjusted.at[index, "action_reason"] = "probabilidad calibrada por backtesting demasiado baja"
        elif float(calibrated) >= 0.64 and row.get("action") == "Informativo":
            adjusted.at[index, "action"] = "Recomendado"
            adjusted.at[index, "action_reason"] = "probabilidad calibrada respalda el pick"
    return adjusted


def backtest_model(db_path: Path = DATABASE_PATH, league_code: str | None = None, max_test_matches: int | None = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = load_matches(db_path=db_path)
    if league_code and "league_code" in matches.columns:
        matches = matches[matches["league_code"] == league_code]
    return run_backtest(matches, BacktestConfig(max_test_matches=max_test_matches))


def backtest_diagnostics(
    db_path: Path = DATABASE_PATH,
    league_code: str | None = None,
    max_test_matches: int | None = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    details, summary = backtest_model(db_path=db_path, league_code=league_code, max_test_matches=max_test_matches)
    calibration_table = build_calibration_table(details)
    market_summary = summarise_markets(details)
    return details, summary, calibration_table, market_summary
