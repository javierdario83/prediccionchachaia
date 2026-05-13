from football_predictor.weather import classify_weather_risk


def test_classify_weather_risk():
    assert classify_weather_risk(80, 10)[0] == "Alto"
    assert classify_weather_risk(20, 35)[0] == "Medio"
    assert classify_weather_risk(10, 5)[0] == "Bajo"
