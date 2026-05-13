"""Modelo Poisson interpretable para marcador, 1X2, over/under y BTTS."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TeamStrength:
    attack_home: float
    defence_home: float
    attack_away: float
    defence_away: float


@dataclass(frozen=True)
class MatchPrediction:
    league_code: str | None
    match_date: str | None
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    double_chance_1x_prob: float
    double_chance_x2_prob: float
    double_chance_12_prob: float
    over15_prob: float
    under15_prob: float
    over25_prob: float
    under25_prob: float
    over35_prob: float
    under35_prob: float
    btts_yes_prob: float
    btts_no_prob: float
    expected_home_goals: float
    expected_away_goals: float
    predicted_score: str
    recommended_market: str
    recommended_probability: float
    confidence: str
    confidence_score: float
    confidence_note: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class PoissonFootballModel:
    """Calcula probabilidades con fortalezas ofensivas/defensivas por equipo.

    El modelo usa medias de goles de la liga, ventaja local y fuerza relativa de
    cada equipo. Es ideal para el MVP porque no necesita entrenamiento pesado,
    es explicable y permite generar marcador exacto desde una matriz de goles.
    """

    def __init__(self, max_goals: int = 8, min_lambda: float = 0.15, recent_matches: int | None = 60):
        self.max_goals = max_goals
        self.min_lambda = min_lambda
        self.recent_matches = recent_matches
        self.league_home_avg = 1.35
        self.league_away_avg = 1.10
        self.teams: dict[str, TeamStrength] = {}
        self.is_fitted = False

    def fit(self, matches: pd.DataFrame) -> "PoissonFootballModel":
        if matches.empty:
            raise ValueError("No hay partidos historicos para entrenar el modelo.")

        df = matches.copy()
        if "match_date" in df.columns:
            df = df.sort_values("match_date")
        elif "MatchDate" in df.columns:
            df = df.sort_values("MatchDate")

        if self.recent_matches:
            team_rows = []
            for team in sorted(set(df["home_team"]).union(set(df["away_team"]))):
                team_df = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(self.recent_matches)
                team_rows.append(team_df)
            if team_rows:
                df = pd.concat(team_rows).drop_duplicates().sort_index()

        home_goals = _column(df, "home_goals", "FTHG")
        away_goals = _column(df, "away_goals", "FTAG")
        home_team = _column(df, "home_team", "HomeTeam")
        away_team = _column(df, "away_team", "AwayTeam")

        self.league_home_avg = max(float(home_goals.mean()), self.min_lambda)
        self.league_away_avg = max(float(away_goals.mean()), self.min_lambda)

        teams = sorted(set(home_team).union(set(away_team)))
        strengths: dict[str, TeamStrength] = {}
        for team in teams:
            home_mask = home_team == team
            away_mask = away_team == team

            home_scored = home_goals[home_mask].mean() if home_mask.any() else self.league_home_avg
            home_conceded = away_goals[home_mask].mean() if home_mask.any() else self.league_away_avg
            away_scored = away_goals[away_mask].mean() if away_mask.any() else self.league_away_avg
            away_conceded = home_goals[away_mask].mean() if away_mask.any() else self.league_home_avg

            strengths[team] = TeamStrength(
                attack_home=_safe_ratio(home_scored, self.league_home_avg),
                defence_home=_safe_ratio(home_conceded, self.league_away_avg),
                attack_away=_safe_ratio(away_scored, self.league_away_avg),
                defence_away=_safe_ratio(away_conceded, self.league_home_avg),
            )

        self.teams = strengths
        self.is_fitted = True
        return self

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        league_code: str | None = None,
        match_date: str | None = None,
    ) -> MatchPrediction:
        if not self.is_fitted:
            raise RuntimeError("El modelo debe entrenarse antes de predecir.")

        home_strength = self.teams.get(home_team, _neutral_strength())
        away_strength = self.teams.get(away_team, _neutral_strength())

        expected_home = max(self.league_home_avg * home_strength.attack_home * away_strength.defence_away, self.min_lambda)
        expected_away = max(self.league_away_avg * away_strength.attack_away * home_strength.defence_home, self.min_lambda)

        matrix = score_probability_matrix(expected_home, expected_away, self.max_goals)
        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())
        total = home_win + draw + away_win
        home_win, draw, away_win = home_win / total, draw / total, away_win / total

        over15 = _over_probability(matrix, 1.5)
        over25 = _over_probability(matrix, 2.5)
        over35 = _over_probability(matrix, 3.5)
        btts = float(matrix[1:, 1:].sum())
        probabilities = {
            "Gana local": home_win,
            "Empate": draw,
            "Gana visitante": away_win,
            "Doble oportunidad 1X": home_win + draw,
            "Doble oportunidad X2": draw + away_win,
            "Doble oportunidad 12": home_win + away_win,
            "Over 1.5": over15,
            "Under 1.5": 1 - over15,
            "Over 2.5": over25,
            "Under 2.5": 1 - over25,
            "Over 3.5": over35,
            "Under 3.5": 1 - over35,
            "Ambos anotan: Sí": btts,
            "Ambos anotan: No": 1 - btts,
        }
        recommended_market, recommended_probability = max(probabilities.items(), key=lambda item: item[1])

        score_index = np.unravel_index(np.argmax(matrix), matrix.shape)
        predicted_score = f"{score_index[0]}-{score_index[1]}"
        confidence_score, confidence, confidence_note = _confidence_details(
            [home_win, draw, away_win], recommended_probability
        )

        return MatchPrediction(
            league_code=league_code,
            match_date=match_date,
            home_team=home_team,
            away_team=away_team,
            home_win_prob=round(home_win, 4),
            draw_prob=round(draw, 4),
            away_win_prob=round(away_win, 4),
            double_chance_1x_prob=round(home_win + draw, 4),
            double_chance_x2_prob=round(draw + away_win, 4),
            double_chance_12_prob=round(home_win + away_win, 4),
            over15_prob=round(over15, 4),
            under15_prob=round(1 - over15, 4),
            over25_prob=round(over25, 4),
            under25_prob=round(1 - over25, 4),
            over35_prob=round(over35, 4),
            under35_prob=round(1 - over35, 4),
            btts_yes_prob=round(btts, 4),
            btts_no_prob=round(1 - btts, 4),
            expected_home_goals=round(expected_home, 3),
            expected_away_goals=round(expected_away, 3),
            predicted_score=predicted_score,
            recommended_market=recommended_market,
            recommended_probability=round(recommended_probability, 4),
            confidence=confidence,
            confidence_score=round(confidence_score, 4),
            confidence_note=confidence_note,
        )

    def predict_dataframe(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        predictions = []
        for _, row in fixtures.iterrows():
            predictions.append(
                self.predict_match(
                    home_team=row.get("HomeTeam", row.get("home_team")),
                    away_team=row.get("AwayTeam", row.get("away_team")),
                    league_code=row.get("LeagueCode", row.get("Div", row.get("league_code"))),
                    match_date=_format_date(row.get("MatchDate", row.get("match_date"))),
                ).as_dict()
            )
        return pd.DataFrame(predictions)


def score_probability_matrix(home_lambda: float, away_lambda: float, max_goals: int = 8) -> np.ndarray:
    home_probs = np.array([poisson_probability(goals, home_lambda) for goals in range(max_goals + 1)])
    away_probs = np.array([poisson_probability(goals, away_lambda) for goals in range(max_goals + 1)])
    matrix = np.outer(home_probs, away_probs)
    return matrix / matrix.sum()


def poisson_probability(goals: int, lambda_value: float) -> float:
    return (lambda_value**goals * exp(-lambda_value)) / factorial(goals)


def _over_probability(matrix: np.ndarray, line: float) -> float:
    threshold = int(line)
    probability = 0.0
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            if home_goals + away_goals > threshold:
                probability += matrix[home_goals, away_goals]
    return float(probability)


def _safe_ratio(value: float, denominator: float) -> float:
    if pd.isna(value) or denominator == 0:
        return 1.0
    return max(float(value) / float(denominator), 0.2)


def _neutral_strength() -> TeamStrength:
    return TeamStrength(1.0, 1.0, 1.0, 1.0)


def _column(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    return df[primary] if primary in df.columns else df[fallback]


def _confidence_details(outcome_probabilities: list[float], recommended_probability: float) -> tuple[float, str, str]:
    ordered = sorted(outcome_probabilities, reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    # Combina claridad del 1X2 con fuerza del mercado recomendado.
    score = (0.65 * recommended_probability) + (0.35 * min(1.0, margin * 2.5))
    if score >= 0.72:
        return score, "Alta", "Mercado recomendado fuerte y ventaja clara frente a alternativas."
    if score >= 0.62:
        return score, "Media-alta", "Buena probabilidad, aunque conviene revisar contexto del partido."
    if score >= 0.52:
        return score, "Media", "Pronostico util, pero el partido no esta completamente desequilibrado."
    return score, "Baja", "Partido parejo o mercado sin ventaja suficiente; mejor usar cautela."


def _format_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)
