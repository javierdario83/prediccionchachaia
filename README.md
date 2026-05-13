# Predicción Chacha IA

MVP autónomo para generar predicciones de fútbol usando datos públicos de [football-data.co.uk](https://www.football-data.co.uk/). La plataforma no expone una API JSON tradicional: publica archivos CSV descargables por URL. Este proyecto los consume automáticamente, los limpia, los guarda en SQLite y alimenta un modelo Poisson para generar probabilidades.

## Qué predice el MVP

- Ganador probable: local / empate / visitante.
- Doble oportunidad: 1X, X2 y 12.
- Over y Under 1.5, 2.5 y 3.5 goles.
- Ambos equipos anotan: sí / no.
- Goles esperados por equipo.
- Marcador exacto más probable.
- Pick o mercado recomendado.
- Nivel y score de confianza.
- Contexto climático opcional con Open-Meteo.
- Backtesting histórico por liga y mercado.
- Exportación de resultados a CSV desde la interfaz.

> Importante: el sistema entrega probabilidades, no garantías. El fútbol conserva incertidumbre por lesiones, expulsiones, clima, alineaciones y decisiones arbitrales.

## Fuente de datos deportivos

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

## API gratuita de clima

El sistema puede enriquecer las predicciones con clima usando **Open-Meteo**:

```text
https://api.open-meteo.com/v1/forecast
https://geocoding-api.open-meteo.com/v1/search
```

Open-Meteo no requiere API key para uso no comercial. El clima se usa como contexto de riesgo: si hay alta probabilidad de lluvia o viento fuerte, la app reduce la confianza del pick porque el partido puede volverse más incierto.

## Ligas iniciales

| Código | Liga |
| --- | --- |
| E0 | Premier League |
| SP1 | La Liga |
| I1 | Serie A |
| D1 | Bundesliga |
| F1 | Ligue 1 |

## Instalación desde cero en Windows

### 1. Instalar Python

1. Entra a `https://www.python.org/downloads/`.
2. Descarga Python 3.10 o superior.
3. Durante la instalación marca **Add Python to PATH**.
4. Cierra y vuelve a abrir la terminal.
5. Verifica la instalación:

```cmd
python --version
```

### 2. Abrir la carpeta del proyecto

Si el proyecto está en el escritorio, por ejemplo:

```text
C:\Users\TU_USUARIO\Desktop\prediccionchachaia
```

Abre esa carpeta, escribe `cmd` en la barra de ruta del explorador de Windows y presiona Enter.

También puedes entrar desde CMD:

```cmd
cd C:\Users\TU_USUARIO\Desktop\prediccionchachaia
```

### 3. Crear entorno virtual

```cmd
python -m venv .venv
```

### 4. Activar entorno virtual

En CMD:

```cmd
.venv\Scripts\activate
```

En PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación, usa CMD para evitar permisos de ejecución.

### 5. Instalar dependencias

```cmd
pip install -r requirements.txt
```

### 6. Abrir el programa

```cmd
streamlit run app.py
```

Si `streamlit` no se reconoce, usa:

```cmd
python -m streamlit run app.py
```

La app abrirá una página local en el navegador. Si no abre automáticamente, entra a:

```text
http://localhost:8501
```

## Instalación desde cero en Mac o Linux

```bash
cd /ruta/a/prediccionchachaia
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Flujo de uso en la interfaz

1. Selecciona ligas y temporadas en la barra lateral.
2. Opcionalmente activa **Agregar clima Open-Meteo**.
3. Presiona **Actualizar datos**.
4. En la pestaña **Predicciones**, elige:
   - **Próximos partidos automáticos**, o
   - **Partido manual**.
5. Presiona **Generar predicciones** o **Predecir partido manual**.
6. Revisa probabilidades, pick recomendado, confianza y clima.
7. Exporta CSV si hace falta.
8. En la pestaña **Rendimiento histórico**, ejecuta backtesting para medir fiabilidad.

## Crear un acceso rápido sin empaquetar .exe

Para una entrega sencilla al cliente, crea un archivo llamado `abrir_programa.bat` en la raíz del proyecto:

```bat
@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run app.py
pause
```

Después de instalar dependencias una vez, el cliente puede abrir el programa con doble clic en ese `.bat`.

## Sobre empaquetar como .exe

Sí se puede empaquetar, pero Streamlit funciona como una app web local. Lo recomendable es validar primero con:

```cmd
streamlit run app.py
```

Luego, si el cliente exige `.exe`, conviene empaquetar un lanzador que ejecute internamente `python -m streamlit run app.py`, no convertir directamente la app sin ajustes.

## Uso desde Python

```python
from football_predictor.pipeline import update_historical_data, predict_upcoming_matches

update_historical_data(seasons=["2425", "2526"], leagues=["E0", "SP1"])
predictions = predict_upcoming_matches(league_codes=["E0", "SP1"], limit=20, include_weather=True)
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
Open-Meteo opcional para clima
        ↓
Streamlit / CSV exportable / backtesting
```

## Mejoras implementadas para fiabilidad

- Backtesting walk-forward: predice partidos históricos entrenando solo con datos anteriores.
- Métricas por liga: 1X2, Over 2.5, BTTS y Brier score.
- Doble oportunidad y Under explícitos.
- Pick recomendado por mayor probabilidad.
- Ranking de picks por confianza.
- Score de confianza más explicable.
- Penalización de confianza por clima adverso cuando Open-Meteo está activo.

## Limitaciones actuales

- No predice goleadores porque Football-Data no entrega datos detallados por jugador.
- No usa xG porque esta fuente no lo incluye de forma general.
- No conoce lesiones ni alineaciones probables sin integrar otra API.
- Las cuotas dependen de las columnas disponibles por temporada.
- El modelo sigue siendo estadístico e interpretable; más adelante se puede agregar scikit-learn, XGBoost/LightGBM y calibración avanzada.
