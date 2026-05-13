"""Backtesting walk-forward para medir la fiabilidad real del modelo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .poisson_model import PoissonFootballModel


@dataclass(frozen=True)
class BacktestConfig:
    min_train_matches: int = 120
    step: int = 1
    max_test_matches: int | None = 300


def run_backtest(matches: pd.DataFrame, config: BacktestConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta backtesting cronologico y devuelve predicciones + resumen.

    Cada partido se predice entrenando solo con partidos anteriores. Esto evita
    leakage de datos futuros y permite medir rendimiento de forma realista.
    """

    config = config or BacktestConfig()
    if matches.empty or len(matches) <= config.min_train_matches:
        return pd.DataFrame(), pd.DataFrame()

    df = matches.sort_values("match_date").reset_index(drop=True).copy()
    start = config.min_train_matches
    end = len(df) if config.max_test_matches is None else min(len(df), start + config.max_test_matches)
    records: list[dict[str, object]] = []

    for index in range(start, end, config.step):
        train = df.iloc[:index]
        row = df.iloc[index]
        try:
            model = PoissonFootballModel().fit(train)
            prediction = model.predict_match(
                row["home_team"],
                row["away_team"],
                league_code=row.get("league_code"),
                match_date=str(row.get("match_date"))[:10],
            )
        except ValueError:
            continue

        actual_result = _actual_result(row)
        predicted_result = _predicted_result(prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob)
        records.append(
            {
                **prediction.as_dict(),
                "actual_result": actual_result,
                "actual_total_goals": int(row["total_goals"]),
                "actual_btts": int(row["btts"]),
                "hit_1x2": int(predicted_result == actual_result),
                "hit_over25": int((prediction.over25_prob >= 0.5) == bool(row["over25"])),
                "hit_btts": int((prediction.btts_yes_prob >= 0.5) == bool(row["btts"])),
                "brier_1x2": _brier_1x2(prediction, actual_result),
                "brier_over25": _brier_binary(prediction.over25_prob, int(row["over25"])),
                "brier_btts": _brier_binary(prediction.btts_yes_prob, int(row["btts"])),
            }
        )

    details = pd.DataFrame(records)
    return details, summarise_backtest(details)


def summarise_backtest(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()

    group_columns = ["league_code"] if "league_code" in details.columns else []
    rows = [_summary_row(details, "Todas")]
    if group_columns:
        for league_code, group in details.groupby("league_code", dropna=False):
            rows.append(_summary_row(group, str(league_code)))
    return pd.DataFrame(rows)


def _summary_row(df: pd.DataFrame, league: str) -> dict[str, object]:
    return {
        "Liga": league,
        "Partidos evaluados": len(df),
        "Acierto 1X2": round(float(df["hit_1x2"].mean()), 4),
        "Acierto Over 2.5": round(float(df["hit_over25"].mean()), 4),
        "Acierto BTTS": round(float(df["hit_btts"].mean()), 4),
        "Brier 1X2": round(float(df["brier_1x2"].mean()), 4),
        "Brier Over 2.5": round(float(df["brier_over25"].mean()), 4),
        "Brier BTTS": round(float(df["brier_btts"].mean()), 4),
        "Confianza media": round(float(df["confidence_score"].mean()), 4),
    }


def _actual_result(row: pd.Series) -> str:
    if int(row["home_goals"]) > int(row["away_goals"]):
        return "H"
    if int(row["home_goals"]) < int(row["away_goals"]):
        return "A"
    return "D"


def _predicted_result(home: float, draw: float, away: float) -> str:
    labels = ["H", "D", "A"]
    return labels[int(np.argmax([home, draw, away]))]


def _brier_1x2(prediction, actual: str) -> float:
    targets = np.array([actual == "H", actual == "D", actual == "A"], dtype=float)
    probs = np.array([prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob], dtype=float)
    return float(np.mean((probs - targets) ** 2))


def _brier_binary(probability: float, actual: int) -> float:
    return float((probability - actual) ** 2)
