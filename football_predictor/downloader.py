"""Descarga automatica de CSV publicos de football-data.co.uk."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import FIXTURES_URL, RAW_DIR, DownloadTarget


class FootballDataDownloadError(RuntimeError):
    """Error al descargar datos desde Football-Data."""


def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_csv(url: str, destination: Path | None = None, timeout: int = 30) -> pd.DataFrame:
    """Descarga un CSV remoto y lo devuelve como DataFrame.

    Si se entrega ``destination``, guarda una copia local para auditoria y cache.
    """

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise FootballDataDownloadError(f"No se pudo descargar {url}: {exc}") from exc

    content = response.content
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    try:
        return pd.read_csv(BytesIO(content))
    except Exception as exc:  # pandas puede lanzar varios errores de parseo
        raise FootballDataDownloadError(f"El CSV descargado desde {url} no se pudo leer: {exc}") from exc


def download_league_season(target: DownloadTarget, save_raw: bool = True) -> pd.DataFrame:
    """Descarga una liga y temporada historica."""

    ensure_data_dirs()
    destination = RAW_DIR / target.filename if save_raw else None
    df = download_csv(target.url, destination=destination)
    df["Season"] = target.season
    df["LeagueCode"] = target.league
    return df


def download_many(targets: Iterable[DownloadTarget], save_raw: bool = True) -> pd.DataFrame:
    """Descarga varias ligas/temporadas y concatena los resultados validos."""

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for target in targets:
        try:
            frames.append(download_league_season(target, save_raw=save_raw))
        except FootballDataDownloadError as exc:
            errors.append(str(exc))

    if not frames:
        joined = "\n".join(errors) if errors else "No se recibieron objetivos de descarga."
        raise FootballDataDownloadError(joined)

    return pd.concat(frames, ignore_index=True, sort=False)


def download_fixtures(save_raw: bool = True) -> pd.DataFrame:
    """Descarga el archivo publico de proximos partidos."""

    ensure_data_dirs()
    destination = RAW_DIR / "fixtures.csv" if save_raw else None
    return download_csv(FIXTURES_URL, destination=destination)
