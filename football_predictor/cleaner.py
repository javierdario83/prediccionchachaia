"""Limpieza y normalizacion de datos historicos de Football-Data."""
from __future__ import annotations

import pandas as pd

CORE_COLUMNS = [
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
    "B365H",
    "B365D",
    "B365A",
    "B365>2.5",
    "B365<2.5",
    "Season",
    "LeagueCode",
]


def parse_match_date(series: pd.Series) -> pd.Series:
    """Convierte fechas de Football-Data, tolerando formatos historicos mixtos."""

    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza partidos historicos finalizados y crea variables objetivo."""

    if df.empty:
        return df.copy()

    available_columns = [column for column in CORE_COLUMNS if column in df.columns]
    cleaned = df[available_columns].copy()
    cleaned = cleaned.dropna(subset=["Date", "HomeTeam", "AwayTeam"], how="any")
    cleaned["MatchDate"] = parse_match_date(cleaned["Date"])
    cleaned = cleaned.dropna(subset=["MatchDate"])

    for column in ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Conservamos solo partidos con marcador final para entrenar y evaluar.
    cleaned = cleaned.dropna(subset=["FTHG", "FTAG"])
    cleaned["FTHG"] = cleaned["FTHG"].astype(int)
    cleaned["FTAG"] = cleaned["FTAG"].astype(int)

    cleaned["TotalGoals"] = cleaned["FTHG"] + cleaned["FTAG"]
    cleaned["Over15"] = (cleaned["TotalGoals"] > 1.5).astype(int)
    cleaned["Over25"] = (cleaned["TotalGoals"] > 2.5).astype(int)
    cleaned["Over35"] = (cleaned["TotalGoals"] > 3.5).astype(int)
    cleaned["BTTS"] = ((cleaned["FTHG"] > 0) & (cleaned["FTAG"] > 0)).astype(int)
    cleaned["HomeWin"] = (cleaned["FTHG"] > cleaned["FTAG"]).astype(int)
    cleaned["Draw"] = (cleaned["FTHG"] == cleaned["FTAG"]).astype(int)
    cleaned["AwayWin"] = (cleaned["FTHG"] < cleaned["FTAG"]).astype(int)

    if "FTR" not in cleaned.columns:
        cleaned["FTR"] = cleaned.apply(_result_code, axis=1)

    cleaned = cleaned.sort_values(["MatchDate", "LeagueCode", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return cleaned


def clean_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el archivo de proximos partidos."""

    if df.empty:
        return df.copy()

    columns = [column for column in ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"] if column in df.columns]
    fixtures = df[columns].copy()
    fixtures = fixtures.dropna(subset=["Date", "HomeTeam", "AwayTeam"], how="any")
    fixtures["MatchDate"] = parse_match_date(fixtures["Date"])
    fixtures = fixtures.dropna(subset=["MatchDate"])
    for column in ["B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]:
        if column in fixtures.columns:
            fixtures[column] = pd.to_numeric(fixtures[column], errors="coerce")
    if "Div" in fixtures.columns:
        fixtures["LeagueCode"] = fixtures["Div"]
    return fixtures.sort_values(["MatchDate", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def _result_code(row: pd.Series) -> str:
    if row["FTHG"] > row["FTAG"]:
        return "H"
    if row["FTHG"] < row["FTAG"]:
        return "A"
    return "D"
