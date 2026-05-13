"""Rating Elo simple para medir fuerza relativa de equipos."""
from __future__ import annotations

import math

import pandas as pd

DEFAULT_ELO = 1500.0
HOME_ADVANTAGE = 60.0
K_FACTOR = 24.0


def compute_elo_ratings(matches: pd.DataFrame) -> dict[str, float]:
    """Calcula Elo actual por equipo usando partidos en orden cronologico."""

    ratings: dict[str, float] = {}
    if matches.empty:
        return ratings

    date_column = "match_date" if "match_date" in matches.columns else "MatchDate"
    df = matches.sort_values(date_column).copy()
    for _, row in df.iterrows():
        home_team = row.get("home_team", row.get("HomeTeam"))
        away_team = row.get("away_team", row.get("AwayTeam"))
        if not home_team or not away_team:
            continue

        home_goals = row.get("home_goals", row.get("FTHG"))
        away_goals = row.get("away_goals", row.get("FTAG"))
        if pd.isna(home_goals) or pd.isna(away_goals):
            continue

        home_rating = ratings.get(home_team, DEFAULT_ELO)
        away_rating = ratings.get(away_team, DEFAULT_ELO)
        expected_home = expected_score(home_rating + HOME_ADVANTAGE, away_rating)
        actual_home = actual_score(int(home_goals), int(away_goals))
        multiplier = margin_multiplier(int(home_goals), int(away_goals))
        change = K_FACTOR * multiplier * (actual_home - expected_home)
        ratings[home_team] = round(home_rating + change, 3)
        ratings[away_team] = round(away_rating - change, 3)
    return ratings


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))


def actual_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def margin_multiplier(home_goals: int, away_goals: int) -> float:
    goal_diff = abs(home_goals - away_goals)
    if goal_diff <= 1:
        return 1.0
    return min(1.75, 1.0 + (goal_diff - 1) * 0.18)
