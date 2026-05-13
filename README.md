# Predicción Chacha IA

MVP autónomo para generar predicciones de fútbol usando datos públicos de [football-data.co.uk](https://www.football-data.co.uk/). La plataforma no expone una API JSON tradicional: publica archivos CSV descargables por URL. Este proyecto los consume automáticamente, los limpia, los guarda en SQLite y alimenta un modelo Poisson para generar probabilidades.

## Qué predice el MVP

- Ganador probable: local / empate / visitante.
- Over 1.5, Over 2.5 y Over 3.5 goles.
- Ambos equipos anotan.
- Goles esperados por equipo.
- Marcador exacto más probable.
- Nivel de confianza.
- Exportación de resultados a CSV desde la interfaz.

> Importante: el sistema entrega probabilidades, no garantías. El fútbol conserva incertidumbre por lesiones, expulsiones, clima, alineaciones y decisiones arbitrales.

## Fuente de datos

Los históricos se descargan con el patrón:

```text
https://www.football-data.co.uk/mmz4281/{temporada}/{liga}.csv
```

Ejemplos:

```text
https://www.football-data.co.uk/mmz4281/2526/E0.csv
https://www.football-data.co.uk/mmz4281/2425/SP1.csv
```

Los próximos partidos se descargan desde:

```text
https://www.football-data.co.uk/fixtures.csv
```

## Ligas iniciales

| Código | Liga |
| --- | --- |
| E0 | Premier League |
| SP1 | La Liga |
| I1 | Serie A |
| D1 | Bundesliga |
| F1 | Ligue 1 |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso con interfaz gráfica

```bash
streamlit run app.py
```

Flujo recomendado para el cliente:

1. Seleccionar ligas y temporadas en la barra lateral.
2. Presionar **Actualizar datos**.
3. Presionar **Generar predicciones** para próximos partidos, o usar **Partido manual**.
4. Revisar la tabla y exportar CSV si hace falta.

## Uso desde Python

```python
from football_predictor.pipeline import update_historical_data, predict_upcoming_matches

update_historical_data(seasons=["2425", "2526"], leagues=["E0", "SP1"])
predictions = predict_upcoming_matches(league_codes=["E0", "SP1"], limit=20)
print(predictions.head())
```

## Arquitectura

```text
football-data.co.uk CSV
        ↓
football_predictor.downloader
        ↓
football_predictor.cleaner
        ↓
SQLite local
        ↓
football_predictor.poisson_model
        ↓
Streamlit / CSV exportable
```

## Limitaciones actuales

- No predice goleadores porque Football-Data no entrega datos detallados por jugador.
- No usa xG porque esta fuente no lo incluye de forma general.
- Las cuotas dependen de las columnas disponibles por temporada.
- El modelo inicial es estadístico e interpretable; más adelante se puede agregar XGBoost/LightGBM, calibración, backtesting y otras APIs.
