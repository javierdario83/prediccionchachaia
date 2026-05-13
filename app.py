"""Interfaz grafica sencilla para el MVP de prediccion de futbol."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from football_predictor.config import DEFAULT_SEASONS, LEAGUES
from football_predictor.database import load_matches
from football_predictor.pipeline import backtest_diagnostics, predict_manual_match, predict_upcoming_matches, update_historical_data
from football_predictor.pipeline import backtest_model, predict_manual_match, predict_upcoming_matches, update_historical_data


PERCENT_COLUMNS = [
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "double_chance_1x_prob",
    "double_chance_x2_prob",
    "double_chance_12_prob",
    "over15_prob",
    "under15_prob",
    "over25_prob",
    "under25_prob",
    "over35_prob",
    "under35_prob",
    "btts_yes_prob",
    "btts_no_prob",
    "recommended_probability",
    "pick_probability",
    "confidence_score",
    "home_over25_rate_5",
    "away_over25_rate_5",
    "home_btts_rate_5",
    "away_btts_rate_5",
    "home_data_quality",
    "away_data_quality",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "market_over25_prob",
    "market_under25_prob",
    "value_gap",
    "calibrated_pick_probability",
    "calibration_hit_rate",
]


def render_predictions(predictions: pd.DataFrame) -> None:
    """Muestra resultados en formato amigable para el cliente."""

    if predictions.empty:
        st.warning("No hay predicciones para mostrar.")
        return

    display = predictions.copy()
    for column in PERCENT_COLUMNS:
        if column in display.columns:
            display[column] = (display[column] * 100).round(1).astype(str) + "%"

    display = display.rename(
        columns={
            "league_code": "Liga",
            "match_date": "Fecha",
            "home_team": "Local",
            "away_team": "Visitante",
            "home_win_prob": "Gana local",
            "draw_prob": "Empate",
            "away_win_prob": "Gana visita",
            "double_chance_1x_prob": "1X",
            "double_chance_x2_prob": "X2",
            "double_chance_12_prob": "12",
            "over15_prob": "Over 1.5",
            "under15_prob": "Under 1.5",
            "over25_prob": "Over 2.5",
            "under25_prob": "Under 2.5",
            "over35_prob": "Over 3.5",
            "under35_prob": "Under 3.5",
            "btts_yes_prob": "Ambos anotan Sí",
            "btts_no_prob": "Ambos anotan No",
            "expected_home_goals": "Goles esp. local",
            "expected_away_goals": "Goles esp. visita",
            "predicted_score": "Marcador probable",
            "recommended_market": "Mercado recomendado",
            "recommended_probability": "Prob. recomendada",
            "pick": "Pick",
            "pick_probability": "Prob. pick",
            "confidence": "Confianza",
            "confidence_score": "Score confianza",
            "confidence_note": "Nota confianza",
            "action": "Acción",
            "action_reason": "Motivo acción",
            "reliability_note": "Nota fiabilidad",
            "explanation": "Explicación",
            "home_elo": "Elo local",
            "away_elo": "Elo visita",
            "elo_diff": "Dif. Elo",
            "home_form_points_5": "Forma local 5",
            "away_form_points_5": "Forma visita 5",
            "home_goals_for_5": "GF local 5",
            "away_goals_for_5": "GF visita 5",
            "home_goals_against_5": "GC local 5",
            "away_goals_against_5": "GC visita 5",
            "home_over25_rate_5": "Over2.5 local 5",
            "away_over25_rate_5": "Over2.5 visita 5",
            "home_btts_rate_5": "BTTS local 5",
            "away_btts_rate_5": "BTTS visita 5",
            "home_shots_on_target_for_5": "Tiros arco local 5",
            "away_shots_on_target_for_5": "Tiros arco visita 5",
            "home_data_quality": "Calidad datos local",
            "away_data_quality": "Calidad datos visita",
            "b365_home": "Cuota local",
            "b365_draw": "Cuota empate",
            "b365_away": "Cuota visita",
            "b365_over25": "Cuota Over2.5",
            "b365_under25": "Cuota Under2.5",
            "market_home_prob": "Mercado local",
            "market_draw_prob": "Mercado empate",
            "market_away_prob": "Mercado visita",
            "market_over25_prob": "Mercado Over2.5",
            "market_under25_prob": "Mercado Under2.5",
            "value_gap": "Ventaja vs mercado",
            "calibrated_pick_probability": "Prob. calibrada",
            "calibration_bucket": "Bucket calibración",
            "calibration_samples": "Muestras calibración",
            "calibration_hit_rate": "Acierto bucket",
            "api_football_fixture_id": "Fixture API-Football",
            "api_football_status": "Estado API-Football",
            "api_home_injuries": "Lesiones local",
            "api_away_injuries": "Lesiones visita",
            "api_home_suspensions": "Susp. local",
            "api_away_suspensions": "Susp. visita",
            "api_lineups_available": "Lineups API",
            "api_home_formation": "Formación local",
            "api_away_formation": "Formación visita",
            "api_home_xg": "xG local API",
            "api_away_xg": "xG visita API",
            "api_context_note": "Nota API-Football",
            "weather_city": "Ciudad clima",
            "weather_temperature_c": "Temp. °C",
            "weather_precipitation_probability": "Prob. lluvia %",
            "weather_wind_speed_kmh": "Viento km/h",
            "weather_risk": "Riesgo clima",
            "weather_note": "Nota clima",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Exportar CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_futbol.csv",
        mime="text/csv",
    )


def render_backtest(summary: pd.DataFrame, details: pd.DataFrame) -> None:
    if summary.empty:
        st.warning("No hay suficientes partidos históricos para ejecutar backtesting.")
        return

    summary_display = summary.copy()
    for column in ["Acierto 1X2", "Acierto Over 2.5", "Acierto BTTS", "Confianza media"]:
        summary_display[column] = (summary_display[column] * 100).round(1).astype(str) + "%"
    st.dataframe(summary_display, use_container_width=True, hide_index=True)

    csv = details.to_csv(index=False).encode("utf-8")
    st.download_button("Exportar detalle de backtesting", data=csv, file_name="backtesting_detalle.csv", mime="text/csv")


st.set_page_config(page_title="Prediccion Chacha IA", page_icon="⚽", layout="wide")

st.title("⚽ Predicción Chacha IA")
st.caption("MVP autónomo conectado a CSV públicos de football-data.co.uk")

with st.sidebar:
    st.header("Configuración")
    selected_leagues = st.multiselect(
        "Ligas",
        options=list(LEAGUES.keys()),
        default=["E0", "SP1", "I1"],
        format_func=lambda code: f"{code} - {LEAGUES.get(code, code)}",
    )
    selected_seasons = st.multiselect("Temporadas históricas", options=DEFAULT_SEASONS, default=DEFAULT_SEASONS[-3:])
    include_weather = st.checkbox("Agregar clima Open-Meteo", value=False)
    calibrate_probabilities = st.checkbox("Calibrar con backtesting", value=False)
    use_api_football = st.checkbox("Usar API-Football", value=False)
    api_football_key = st.text_input("API-Football key", type="password") if use_api_football else None
    st.info("Football-Data publica archivos CSV. La app los descarga, limpia y guarda localmente para entrenar el modelo.")

st.subheader("1. Actualizar base histórica")
col_update, col_status = st.columns([1, 2])
with col_update:
    if st.button("Actualizar datos", type="primary"):
        with st.spinner("Descargando y limpiando datos..."):
            try:
                summary = update_historical_data(seasons=selected_seasons, leagues=selected_leagues)
                st.success(
                    f"Listo: {summary.downloaded_rows} filas descargadas, "
                    f"{summary.clean_rows} partidos limpios, {summary.inserted_rows} nuevos registros."
                )
            except Exception as exc:  # mensaje visual para el usuario final
                st.error(f"No se pudieron actualizar los datos: {exc}")
with col_status:
    try:
        matches = load_matches()
        st.metric("Partidos en base local", len(matches))
        if not matches.empty:
            st.caption(f"Rango: {matches['match_date'].min().date()} → {matches['match_date'].max().date()}")
    except Exception:
        st.metric("Partidos en base local", 0)

predictions_tab, performance_tab = st.tabs(["Predicciones", "Rendimiento histórico"])

with predictions_tab:
    st.subheader("2. Generar predicciones")
    mode = st.radio("Modo", ["Próximos partidos automáticos", "Partido manual"], horizontal=True)

    if mode == "Próximos partidos automáticos":
        limit = st.slider("Cantidad máxima de partidos", min_value=5, max_value=100, value=25, step=5)
        if st.button("Generar predicciones"):
            with st.spinner("Descargando fixtures y calculando probabilidades..."):
                try:
                    predictions = predict_upcoming_matches(
                        league_codes=selected_leagues,
                        limit=limit,
                        include_weather=include_weather,
                        calibrate=calibrate_probabilities,
                        api_football_key=api_football_key,
                    )
                    render_predictions(predictions)
                except Exception as exc:
                    st.error(f"No se pudieron generar predicciones: {exc}")
    else:
        try:
            teams = sorted(set(load_matches()["home_team"]).union(set(load_matches()["away_team"])))
        except Exception:
            teams = []

        col_home, col_away, col_league = st.columns(3)
        with col_home:
            home_team = st.selectbox("Equipo local", options=teams) if teams else st.text_input("Equipo local")
        with col_away:
            away_team = st.selectbox("Equipo visitante", options=teams) if teams else st.text_input("Equipo visitante")
        with col_league:
            league_code = st.selectbox(
                "Liga",
                options=[None] + selected_leagues,
                format_func=lambda code: "Todas" if code is None else LEAGUES.get(code, code),
            )

        if st.button("Predecir partido manual"):
            if not home_team or not away_team:
                st.warning("Selecciona o escribe ambos equipos.")
            else:
                try:
                    predictions = predict_manual_match(
                        home_team,
                        away_team,
                        league_code=league_code,
                        include_weather=include_weather,
                        calibrate=calibrate_probabilities,
                        api_football_key=api_football_key,
                    )
                    render_predictions(predictions)
                except Exception as exc:
                    st.error(f"No se pudo predecir el partido: {exc}")

with performance_tab:
    st.subheader("Backtesting del modelo")
    st.caption("Evalúa partidos históricos entrenando solo con datos anteriores a cada partido.")
    col_l, col_n = st.columns(2)
    with col_l:
        backtest_league = st.selectbox(
            "Liga para evaluar",
            options=[None] + selected_leagues,
            format_func=lambda code: "Todas" if code is None else LEAGUES.get(code, code),
        )
    with col_n:
        max_matches = st.slider("Partidos máximos a evaluar", min_value=50, max_value=1000, value=300, step=50)
    if st.button("Ejecutar backtesting"):
        with st.spinner("Ejecutando backtesting histórico..."):
            try:
                details, summary, calibration_table, market_summary = backtest_diagnostics(
                    league_code=backtest_league,
                    max_test_matches=max_matches,
                )
                render_backtest(summary, details)
                if not market_summary.empty:
                    st.subheader("Rendimiento por mercado")
                    market_display = market_summary.copy()
                    for column in ["Acierto", "Confianza", "Probabilidad"]:
                        market_display[column] = (market_display[column] * 100).round(1).astype(str) + "%"
                    st.dataframe(market_display, use_container_width=True, hide_index=True)
                if not calibration_table.empty:
                    st.subheader("Calibración por buckets")
                    calibration_display = calibration_table.copy()
                    for column in ["avg_probability", "hit_rate"]:
                        calibration_display[column] = (calibration_display[column] * 100).round(1).astype(str) + "%"
                    st.dataframe(calibration_display, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"No se pudo ejecutar backtesting: {exc}")
