"""Features recientes y senales de fiabilidad derivadas del historico."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .elo import DEFAULT_ELO, compute_elo_ratings


@dataclass(frozen=True)
class TeamProfile:
    matches: int
    points_per_match_5: float | None
    goals_for_5: float | None
    goals_against_5: float | None
    over25_rate_5: float | None
    btts_rate_5: float | None
    shots_for_5: float | None
    shots_on_target_for_5: float | None
    corners_for_5: float | None
    cards_for_5: float | None

    @property
    def data_quality(self) -> float:
        return round(min(1.0, self.matches / 10), 3)


def enrich_with_reliability_signals(predictions: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Agrega Elo, forma reciente, accion sugerida y explicacion a predicciones."""

    if predictions.empty:
        return predictions

    enriched = predictions.copy()
    ratings = compute_elo_ratings(matches)
    context_rows = []
    for _, row in enriched.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_profile = build_team_profile(matches, home_team)
        away_profile = build_team_profile(matches, away_team)
        home_elo = ratings.get(home_team, DEFAULT_ELO)
        away_elo = ratings.get(away_team, DEFAULT_ELO)
        elo_diff = home_elo - away_elo
        adjusted_score, adjustment_note = adjust_confidence(row, home_profile, away_profile, elo_diff)
        action, action_reason = classify_action(row, adjusted_score, home_profile, away_profile)
        context_rows.append(
            {
                "home_elo": round(home_elo, 1),
                "away_elo": round(away_elo, 1),
                "elo_diff": round(elo_diff, 1),
                "home_form_points_5": home_profile.points_per_match_5,
                "away_form_points_5": away_profile.points_per_match_5,
                "home_goals_for_5": home_profile.goals_for_5,
                "away_goals_for_5": away_profile.goals_for_5,
                "home_goals_against_5": home_profile.goals_against_5,
                "away_goals_against_5": away_profile.goals_against_5,
                "home_over25_rate_5": home_profile.over25_rate_5,
                "away_over25_rate_5": away_profile.over25_rate_5,
                "home_btts_rate_5": home_profile.btts_rate_5,
                "away_btts_rate_5": away_profile.btts_rate_5,
                "home_shots_on_target_for_5": home_profile.shots_on_target_for_5,
                "away_shots_on_target_for_5": away_profile.shots_on_target_for_5,
                "home_data_quality": home_profile.data_quality,
                "away_data_quality": away_profile.data_quality,
                "confidence_score": round(adjusted_score, 4),
                "confidence": confidence_from_score(adjusted_score),
                "reliability_note": adjustment_note,
                "action": action,
                "action_reason": action_reason,
                "explanation": build_explanation(row, home_profile, away_profile, elo_diff, action_reason),
            }
        )

    context = pd.DataFrame(context_rows)
    for column in ["confidence_score", "confidence"]:
        if column in enriched.columns:
            enriched = enriched.drop(columns=[column])
    return pd.concat([enriched.reset_index(drop=True), context.reset_index(drop=True)], axis=1)


def build_team_profile(matches: pd.DataFrame, team: str, window: int = 5) -> TeamProfile:
    if matches.empty:
        return empty_profile()

    df = matches.copy()
    date_column = "match_date" if "match_date" in df.columns else "MatchDate"
    home_col = "home_team" if "home_team" in df.columns else "HomeTeam"
    away_col = "away_team" if "away_team" in df.columns else "AwayTeam"
    df = df[(df[home_col] == team) | (df[away_col] == team)].sort_values(date_column).tail(window)
    if df.empty:
        return empty_profile()

    rows = []
    for _, row in df.iterrows():
        is_home = row[home_col] == team
        gf = _value(row, "home_goals", "FTHG") if is_home else _value(row, "away_goals", "FTAG")
        ga = _value(row, "away_goals", "FTAG") if is_home else _value(row, "home_goals", "FTHG")
        points = 3 if gf > ga else 1 if gf == ga else 0
        shots_for = _side_value(row, is_home, "home_shots", "away_shots")
        shots_on_target_for = _side_value(row, is_home, "home_shots_on_target", "away_shots_on_target")
        corners_for = _side_value(row, is_home, "home_corners", "away_corners")
        yellow_for = _side_value(row, is_home, "home_yellow_cards", "away_yellow_cards")
        red_for = _side_value(row, is_home, "home_red_cards", "away_red_cards")
        rows.append(
            {
                "gf": gf,
                "ga": ga,
                "points": points,
                "over25": int((gf + ga) > 2.5),
                "btts": int(gf > 0 and ga > 0),
                "shots_for": shots_for,
                "shots_on_target_for": shots_on_target_for,
                "corners_for": corners_for,
                "cards_for": _sum_optional(yellow_for, red_for),
            }
        )
    recent = pd.DataFrame(rows)
    return TeamProfile(
        matches=len(recent),
        points_per_match_5=_mean(recent["points"]),
        goals_for_5=_mean(recent["gf"]),
        goals_against_5=_mean(recent["ga"]),
        over25_rate_5=_mean(recent["over25"]),
        btts_rate_5=_mean(recent["btts"]),
        shots_for_5=_mean(recent["shots_for"]),
        shots_on_target_for_5=_mean(recent["shots_on_target_for"]),
        corners_for_5=_mean(recent["corners_for"]),
        cards_for_5=_mean(recent["cards_for"]),
    )


def adjust_confidence(row: pd.Series, home: TeamProfile, away: TeamProfile, elo_diff: float) -> tuple[float, str]:
    score = float(row.get("confidence_score", 0.0) or 0.0)
    notes = []
    min_quality = min(home.data_quality, away.data_quality)
    if min_quality < 0.5:
        score -= 0.08
        notes.append("pocos datos recientes")
    elif min_quality >= 0.9:
        score += 0.02
        notes.append("muestra reciente suficiente")

    market = str(row.get("recommended_market", ""))
    if "Gana local" in market and elo_diff > 90:
        score += 0.03
        notes.append("Elo favorece al local")
    elif "Gana visitante" in market and elo_diff < -90:
        score += 0.03
        notes.append("Elo favorece al visitante")
    elif "Gana" in market and abs(elo_diff) < 35:
        score -= 0.03
        notes.append("Elo muy parejo")

    home_ppm = home.points_per_match_5
    away_ppm = away.points_per_match_5
    if home_ppm is not None and away_ppm is not None:
        form_diff = home_ppm - away_ppm
        if "Gana local" in market and form_diff > 0.6:
            score += 0.025
            notes.append("forma reciente favorece al local")
        elif "Gana visitante" in market and form_diff < -0.6:
            score += 0.025
            notes.append("forma reciente favorece al visitante")
        elif "Gana" in market and abs(form_diff) < 0.25:
            score -= 0.02
            notes.append("forma reciente equilibrada")

    return max(0.0, min(1.0, score)), ", ".join(notes) or "sin ajustes fuertes de fiabilidad"


def classify_action(row: pd.Series, score: float, home: TeamProfile, away: TeamProfile) -> tuple[str, str]:
    probability = float(row.get("recommended_probability", 0.0) or 0.0)
    min_quality = min(home.data_quality, away.data_quality)
    market = str(row.get("recommended_market", ""))
    if min_quality < 0.4:
        return "Evitar", "histórico reciente insuficiente para ambos equipos"
    if score < 0.52 or probability < 0.56:
        return "Evitar", "probabilidad/confianza insuficiente"
    if "Empate" in market:
        return "Informativo", "el empate es volátil; usar solo como referencia"
    if score >= 0.64 and probability >= 0.64:
        return "Recomendado", "probabilidad y confianza superan el umbral"
    return "Informativo", "hay señal útil, pero no alcanza nivel recomendado"


def build_explanation(row: pd.Series, home: TeamProfile, away: TeamProfile, elo_diff: float, action_reason: str) -> str:
    return (
        f"Pick: {row.get('recommended_market')} ({float(row.get('recommended_probability', 0.0)):.1%}). "
        f"Elo diff local-visita: {elo_diff:.0f}. "
        f"Forma últimos 5: local {home.points_per_match_5 if home.points_per_match_5 is not None else 'N/D'} pts/partido, "
        f"visita {away.points_per_match_5 if away.points_per_match_5 is not None else 'N/D'} pts/partido. "
        f"Acción: {action_reason}."
    )


def confidence_from_score(score: float) -> str:
    if score >= 0.72:
        return "Alta"
    if score >= 0.62:
        return "Media-alta"
    if score >= 0.52:
        return "Media"
    return "Baja"


def empty_profile() -> TeamProfile:
    return TeamProfile(0, None, None, None, None, None, None, None, None, None)


def _value(row: pd.Series, primary: str, fallback: str) -> float:
    value = row.get(primary, row.get(fallback, 0))
    return 0.0 if pd.isna(value) else float(value)


def _side_value(row: pd.Series, is_home: bool, home_column: str, away_column: str) -> float | None:
    column = home_column if is_home else away_column
    value = row.get(column)
    return None if value is None or pd.isna(value) else float(value)


def _sum_optional(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    return (a or 0.0) + (b or 0.0)


def _mean(series: pd.Series) -> float | None:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None
    return round(float(cleaned.mean()), 3)
