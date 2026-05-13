"""Configuracion central del MVP de prediccion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
EXTRA_LEAGUE_URL = "https://www.football-data.co.uk/new/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATABASE_PATH = DATA_DIR / "futbol_predicciones.db"

# Ligas principales recomendadas para el primer MVP.
LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "MEX": "Liga MX",
    "USA": "MLS",
    "ARG": "Liga Profesional Argentina",
}

# Football-Data publica estas ligas extra como CSV acumulado en /new/{league}.csv,
# no como archivo separado por temporada.
EXTRA_LEAGUES = {"MEX", "USA", "ARG"}
EXTRA_LEAGUE_SEASON = "all"

# Temporadas iniciales. El formato 2526 representa 2025/2026.
DEFAULT_SEASONS = ["2122", "2223", "2324", "2425", "2526"]


@dataclass(frozen=True)
class DownloadTarget:
    """Representa una liga/temporada descargable desde Football-Data."""

    season: str
    league: str

    @property
    def url(self) -> str:
        if self.league in EXTRA_LEAGUES:
            return EXTRA_LEAGUE_URL.format(league=self.league)
        return BASE_URL.format(season=self.season, league=self.league)

    @property
    def filename(self) -> str:
        return f"{self.season}_{self.league}.csv"
