"""Persistencia local en SQLite para que el MVP funcione sin servidor externo."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import DATABASE_PATH

MATCH_OPTIONAL_COLUMNS: dict[str, str] = {
    "home_shots": "REAL",
    "away_shots": "REAL",
    "home_shots_on_target": "REAL",
    "away_shots_on_target": "REAL",
    "home_corners": "REAL",
    "away_corners": "REAL",
    "home_fouls": "REAL",
    "away_fouls": "REAL",
    "home_yellow_cards": "REAL",
    "away_yellow_cards": "REAL",
    "home_red_cards": "REAL",
    "away_red_cards": "REAL",
    "b365_home": "REAL",
    "b365_draw": "REAL",
    "b365_away": "REAL",
    "b365_over25": "REAL",
    "b365_under25": "REAL",
}

PREDICTION_OPTIONAL_COLUMNS: dict[str, str] = {
    "double_chance_1x_prob": "REAL",
    "double_chance_x2_prob": "REAL",
    "double_chance_12_prob": "REAL",
    "under15_prob": "REAL",
    "under25_prob": "REAL",
    "under35_prob": "REAL",
    "btts_no_prob": "REAL",
    "recommended_market": "TEXT",
    "recommended_probability": "REAL",
    "pick": "TEXT",
    "pick_probability": "REAL",
    "confidence_score": "REAL",
    "confidence_note": "TEXT",
    "reliability_note": "TEXT",
    "action": "TEXT",
    "action_reason": "TEXT",
    "explanation": "TEXT",
    "home_elo": "REAL",
    "away_elo": "REAL",
    "elo_diff": "REAL",
    "home_form_points_5": "REAL",
    "away_form_points_5": "REAL",
    "home_goals_for_5": "REAL",
    "away_goals_for_5": "REAL",
    "home_goals_against_5": "REAL",
    "away_goals_against_5": "REAL",
    "home_over25_rate_5": "REAL",
    "away_over25_rate_5": "REAL",
    "home_btts_rate_5": "REAL",
    "away_btts_rate_5": "REAL",
    "home_shots_on_target_for_5": "REAL",
    "away_shots_on_target_for_5": "REAL",
    "home_data_quality": "REAL",
    "away_data_quality": "REAL",
    "b365_home": "REAL",
    "b365_draw": "REAL",
    "b365_away": "REAL",
    "b365_over25": "REAL",
    "b365_under25": "REAL",
    "market_home_prob": "REAL",
    "market_draw_prob": "REAL",
    "market_away_prob": "REAL",
    "market_over25_prob": "REAL",
    "market_under25_prob": "REAL",
    "value_gap": "REAL",
    "weather_city": "TEXT",
    "weather_temperature_c": "REAL",
    "weather_precipitation_probability": "REAL",
    "weather_wind_speed_kmh": "REAL",
    "weather_risk": "TEXT",
    "weather_note": "TEXT",
}


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
        _ensure_columns(connection, "matches", MATCH_OPTIONAL_COLUMNS)
        _ensure_columns(connection, "predictions", PREDICTION_OPTIONAL_COLUMNS)


def _ensure_columns(connection: sqlite3.Connection, table: str, migrations: dict[str, str]) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    for column, column_type in migrations.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


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
            "home_shots": df.get("HS"),
            "away_shots": df.get("AS"),
            "home_shots_on_target": df.get("HST"),
            "away_shots_on_target": df.get("AST"),
            "home_corners": df.get("HC"),
            "away_corners": df.get("AC"),
            "home_fouls": df.get("HF"),
            "away_fouls": df.get("AF"),
            "home_yellow_cards": df.get("HY"),
            "away_yellow_cards": df.get("AY"),
            "home_red_cards": df.get("HR"),
            "away_red_cards": df.get("AR"),
            "b365_home": df.get("B365H"),
            "b365_draw": df.get("B365D"),
            "b365_away": df.get("B365A"),
            "b365_over25": df.get("B365>2.5"),
            "b365_under25": df.get("B365<2.5"),
        }
    )

    columns = list(records.columns)
    placeholders = ", ".join(columns)
    with get_connection(db_path) as connection:
        before = connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        records.to_sql("matches_staging", connection, if_exists="replace", index=False)
        connection.execute(
            f"""
            INSERT OR IGNORE INTO matches ({placeholders})
            SELECT {placeholders}
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
