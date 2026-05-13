"""Interfaz grafica sencilla para el MVP de prediccion de futbol."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from football_predictor.config import DEFAULT_SEASONS, LEAGUES
from football_predictor.database import load_matches
from football_predictor.pipeline import predict_manual_match, predict_upcoming_matches, update_historical_data


def render_predictions(predictions: pd.DataFrame) -> None:
    """Muestra resultados en formato amigable para el cliente."""

    if predictions.empty:
        st.warning("No hay predicciones para mostrar.")
        return

    display = predictions.copy()
    percent_columns = [
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "over15_prob",
        "over25_prob",
        "over35_prob",
        "btts_yes_prob",
    ]
    for column in percent_columns:
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
            "over15_prob": "Over 1.5",
            "over25_prob": "Over 2.5",
            "over35_prob": "Over 3.5",
            "btts_yes_prob": "Ambos anotan",
            "expected_home_goals": "Goles esp. local",
            "expected_away_goals": "Goles esp. visita",
            "predicted_score": "Marcador probable",
            "confidence": "Confianza",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Exportar CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_futbol.csv",
        mime="text/csv",
    )


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

st.divider()
st.subheader("2. Generar predicciones")
mode = st.radio("Modo", ["Próximos partidos automáticos", "Partido manual"], horizontal=True)

if mode == "Próximos partidos automáticos":
    limit = st.slider("Cantidad máxima de partidos", min_value=5, max_value=100, value=25, step=5)
    if st.button("Generar predicciones"):
        with st.spinner("Descargando fixtures y calculando probabilidades..."):
            try:
                predictions = predict_upcoming_matches(league_codes=selected_leagues, limit=limit)
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
        league_code = st.selectbox("Liga", options=[None] + selected_leagues, format_func=lambda code: "Todas" if code is None else LEAGUES.get(code, code))

    if st.button("Predecir partido manual"):
        if not home_team or not away_team:
            st.warning("Selecciona o escribe ambos equipos.")
        else:
            try:
                predictions = predict_manual_match(home_team, away_team, league_code=league_code)
                render_predictions(predictions)
            except Exception as exc:
                st.error(f"No se pudo predecir el partido: {exc}")
