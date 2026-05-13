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
- Acción sugerida: Recomendado / Informativo / Evitar.
- Forma reciente de últimos 5 partidos.
- Elo por equipo y diferencia de fuerza.
- Comparación opcional contra cuotas Bet365 cuando están disponibles.
- Contexto climático opcional con Open-Meteo.
- Calibración automática por backtesting cuando hay histórico suficiente.
- Backtesting histórico por liga y mercado.
- Backtesting histórico por liga y mercado.
- Over 1.5, Over 2.5 y Over 3.5 goles.
- Ambos equipos anotan.
- Goles esperados por equipo.
- Marcador exacto más probable.
- Nivel de confianza.
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
3. La calibración por backtesting se aplica automáticamente cuando hay histórico suficiente; no tienes que activar nada.
4. Presiona **Actualizar datos**.
5. En la pestaña **Predicciones**, elige:
   - **Próximos partidos automáticos**, o
   - **Partido manual**.
6. Presiona **Generar predicciones** o **Predecir partido manual**.
7. Revisa probabilidades, pick recomendado, probabilidad calibrada, confianza y clima si está activo.
8. Exporta CSV si hace falta.
9. En la pestaña **Rendimiento histórico**, ejecuta backtesting para medir fiabilidad.

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


## Si GitHub no te deja descargar el proyecto

Si el botón de descarga de GitHub falla por conflictos del PR o por la interfaz web, usa una de estas opciones:

### Opción A: clonar por consola

```cmd
git clone URL_DEL_REPOSITORIO
cd prediccionchachaia
```

Luego sigue la instalación normal con `python -m venv .venv`, `pip install -r requirements.txt` y `streamlit run app.py`.

### Opción B: crear un ZIP desde la rama actual

Si ya tienes el repo en tu máquina, ejecuta:

```cmd
python scripts/create_release_zip.py
```

El archivo quedará en:

```text
dist/prediccionchachaia.zip
```

### Opción C: verificar que el proyecto esté sano antes de entregarlo

```cmd
python scripts/verify_project.py
```

Este comando revisa compilación básica de Python y genera un ZIP de verificación sin necesitar pandas, streamlit ni internet.


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
- Calibración por buckets basada en backtesting para ajustar la probabilidad del pick recomendado.
- Diagnóstico por mercado recomendado para saber qué mercados son más fiables.
- Métricas por liga: 1X2, Over 2.5, BTTS y Brier score.
- Doble oportunidad y Under explícitos.
- Pick recomendado por mayor probabilidad.
- Ranking de picks por confianza.
- Score de confianza más explicable.
- Elo rating para medir fuerza relativa y dificultad del rival.
- Forma reciente de últimos 5 partidos: puntos, goles, Over 2.5, BTTS y estadísticas disponibles.
- Sistema de acción: **Recomendado**, **Informativo** o **Evitar** para no forzar picks débiles.
- Comparación contra cuotas Bet365 cuando existen, calculando probabilidad implícita y ventaja vs mercado.
- Penalización de confianza por clima adverso cuando Open-Meteo está activo.
- Explicación automática del pick combinando probabilidad, Elo, forma reciente y motivo de acción.


## APIs externas investigadas para lesiones, alineaciones y xG

Para mejorar el modelo con datos que `football-data.co.uk` no trae completo, se revisaron estas opciones:

- **Sportmonks**: mejor candidato premium para lesiones/suspendidos, alineaciones, expected lineups y xG.
- **API-Football / API-Sports**: integración opcional disponible en la app para injuries, lineups, fixture/player stats y xG cuando el endpoint lo entregue.
- **TheStatsAPI**: opción de pago/trial con xG, match stats, player stats y datos históricos.
- **foot.io**: opción interesante para prototipo o investigación con lineups y shot-level xG; public reads con rate limit.
- **football-data.org**: API JSON útil como complemento, pero no es la fuente ideal para xG profundo o lesiones.

Más detalle técnico y endpoints revisados: [`docs/external_apis.md`](docs/external_apis.md).

## Limitaciones actuales

- No predice goleadores porque Football-Data no entrega datos detallados por jugador.
- No usa xG porque esta fuente no lo incluye de forma general.
- No conoce lesiones ni alineaciones probables dentro de la interfaz actual; eso queda para una integración externa futura.
- Las cuotas dependen de las columnas disponibles por temporada y no siempre existen para todos los partidos.
- El modelo sigue siendo estadístico e interpretable; más adelante se puede agregar scikit-learn, XGBoost/LightGBM y calibración avanzada.
- La acción **Recomendado** no garantiza acierto: solo indica que el pick superó umbrales internos de probabilidad, confianza, datos disponibles y/o valor frente al mercado.
