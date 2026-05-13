from datetime import date

import pandas as pd

from football_predictor.pipeline import enrich_predictions_with_weather
from football_predictor.weather import WeatherContext, classify_weather_risk, resolve_city_coordinates


def test_classify_weather_risk():
    assert classify_weather_risk(80, 10)[0] == "Alto"
    assert classify_weather_risk(20, 35)[0] == "Medio"
    assert classify_weather_risk(10, 5)[0] == "Bajo"


def test_resolve_city_coordinates_uses_local_open_meteo_cache():
    assert resolve_city_coordinates("London") == (51.5072, -0.1276)


def test_enrich_predictions_with_weather_marks_unavailable(monkeypatch):
    import football_predictor.pipeline as pipeline

    monkeypatch.setattr(pipeline, "get_match_weather", lambda *_args, **_kwargs: None)
    predictions = pd.DataFrame(
        [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "match_date": date.today(),
                "confidence_score": 0.7,
                "confidence": "Alta",
                "action": "Recomendado",
            }
        ]
    )

    enriched = enrich_predictions_with_weather(predictions)

    assert enriched.loc[0, "weather_risk"] == "No disponible"
    assert "Open-Meteo" in enriched.loc[0, "weather_note"]
    assert enriched.loc[0, "confidence_score"] == 0.7


def test_enrich_predictions_with_weather_penalizes_medium_risk(monkeypatch):
    import football_predictor.pipeline as pipeline

    context = WeatherContext(
        city="London",
        temperature_c=14,
        precipitation_probability=45,
        wind_speed_kmh=12,
        weather_risk="Medio",
        note="Clima moderado",
    )
    monkeypatch.setattr(pipeline, "get_match_weather", lambda *_args, **_kwargs: context)
    predictions = pd.DataFrame(
        [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "match_date": date.today(),
                "confidence_score": 0.7,
                "confidence": "Alta",
                "action": "Recomendado",
            }
        ]
    )

    enriched = enrich_predictions_with_weather(predictions)

    assert enriched.loc[0, "weather_city"] == "London"
    assert enriched.loc[0, "weather_risk"] == "Medio"
    assert round(enriched.loc[0, "confidence_score"], 2) == 0.66
