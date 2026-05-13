"""Persistencia local en SQLite para que el MVP funcione sin servidor externo."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import DATABASE_PATH


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def initialize_database(db_path: Path = DATABASE_PATH) -> None:
    """Crea las tablas minimas del MVP."""

    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season TEXT,
                league_code TEXT,
                match_date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_goals INTEGER NOT NULL,
                away_goals INTEGER NOT NULL,
                result TEXT NOT NULL,
                total_goals INTEGER NOT NULL,
                over15 INTEGER NOT NULL,
                over25 INTEGER NOT NULL,
                over35 INTEGER NOT NULL,
                btts INTEGER NOT NULL,
                UNIQUE(season, league_code, match_date, home_team, away_team)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                league_code TEXT,
                match_date TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_win_prob REAL NOT NULL,
                draw_prob REAL NOT NULL,
                away_win_prob REAL NOT NULL,
                over15_prob REAL NOT NULL,
                over25_prob REAL NOT NULL,
                over35_prob REAL NOT NULL,
                btts_yes_prob REAL NOT NULL,
                expected_home_goals REAL NOT NULL,
                expected_away_goals REAL NOT NULL,
                predicted_score TEXT NOT NULL,
                confidence TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_matches(df: pd.DataFrame, db_path: Path = DATABASE_PATH) -> int:
    """Guarda partidos limpios en SQLite y evita duplicados logicos."""

    initialize_database(db_path)
    if df.empty:
        return 0

    records = pd.DataFrame(
        {
            "season": df.get("Season"),
            "league_code": df.get("LeagueCode", df.get("Div")),
            "match_date": df["MatchDate"].dt.strftime("%Y-%m-%d"),
            "home_team": df["HomeTeam"],
            "away_team": df["AwayTeam"],
            "home_goals": df["FTHG"],
            "away_goals": df["FTAG"],
            "result": df["FTR"],
            "total_goals": df["TotalGoals"],
            "over15": df["Over15"],
            "over25": df["Over25"],
            "over35": df["Over35"],
            "btts": df["BTTS"],
        }
    )

    with get_connection(db_path) as connection:
        before = connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        records.to_sql("matches_staging", connection, if_exists="replace", index=False)
        connection.execute(
            """
            INSERT OR IGNORE INTO matches (
                season, league_code, match_date, home_team, away_team,
                home_goals, away_goals, result, total_goals, over15, over25, over35, btts
            )
            SELECT season, league_code, match_date, home_team, away_team,
                   home_goals, away_goals, result, total_goals, over15, over25, over35, btts
            FROM matches_staging
            """
        )
        connection.execute("DROP TABLE matches_staging")
        after = connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    return after - before


def load_matches(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        df = pd.read_sql_query("SELECT * FROM matches ORDER BY match_date", connection, parse_dates=["match_date"])
    return df


def save_predictions(df: pd.DataFrame, db_path: Path = DATABASE_PATH) -> int:
    initialize_database(db_path)
    if df.empty:
        return 0
    with get_connection(db_path) as connection:
        df.to_sql("predictions", connection, if_exists="append", index=False)
    return len(df)
