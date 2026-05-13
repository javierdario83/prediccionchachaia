"""Utilidades para convertir cuotas en probabilidades de mercado."""
from __future__ import annotations

import pandas as pd


def add_market_probabilities(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    df = predictions.copy()
    df[["market_home_prob", "market_draw_prob", "market_away_prob"]] = df.apply(
        lambda row: pd.Series(_normalise_three(row.get("b365_home", row.get("B365H")), row.get("b365_draw", row.get("B365D")), row.get("b365_away", row.get("B365A")))), axis=1
    )
    df[["market_over25_prob", "market_under25_prob"]] = df.apply(
        lambda row: pd.Series(_normalise_two(row.get("b365_over25", row.get("B365>2.5")), row.get("b365_under25", row.get("B365<2.5")))), axis=1
    )
    df["value_gap"] = df.apply(_value_gap_for_pick, axis=1)
    return df


def _normalise_three(home: object, draw: object, away: object) -> tuple[float | None, float | None, float | None]:
    odds = [_as_float(home), _as_float(draw), _as_float(away)]
    if any(value is None or value <= 1 for value in odds):
        return None, None, None
    implied = [1 / value for value in odds if value is not None]
    total = sum(implied)
    return tuple(round(value / total, 4) for value in implied)  # type: ignore[return-value]


def _normalise_two(over: object, under: object) -> tuple[float | None, float | None]:
    odds = [_as_float(over), _as_float(under)]
    if any(value is None or value <= 1 for value in odds):
        return None, None
    implied = [1 / value for value in odds if value is not None]
    total = sum(implied)
    return round(implied[0] / total, 4), round(implied[1] / total, 4)


def _value_gap_for_pick(row: pd.Series) -> float | None:
    pick = str(row.get("recommended_market", ""))
    mapping = {
        "Gana local": ("home_win_prob", "market_home_prob"),
        "Empate": ("draw_prob", "market_draw_prob"),
        "Gana visitante": ("away_win_prob", "market_away_prob"),
        "Over 2.5": ("over25_prob", "market_over25_prob"),
        "Under 2.5": ("under25_prob", "market_under25_prob"),
    }
    if pick not in mapping:
        return None
    model_column, market_column = mapping[pick]
    model_probability = row.get(model_column)
    market_probability = row.get(market_column)
    if pd.isna(model_probability) or pd.isna(market_probability):
        return None
    return round(float(model_probability) - float(market_probability), 4)


def _as_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
