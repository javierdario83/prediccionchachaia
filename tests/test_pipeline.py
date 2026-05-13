from football_predictor.pipeline import predict_manual_match, train_model_or_neutral


def test_predict_manual_match_works_without_local_database(tmp_path):
    db_path = tmp_path / "empty.sqlite"

    predictions = predict_manual_match("Arsenal", "Chelsea", db_path=db_path)

    assert len(predictions) == 1
    row = predictions.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Chelsea"
    assert 0 <= row["home_win_prob"] <= 1
    assert row["action"] == "Evitar"
    assert "histórico" in row["action_reason"]


def test_train_model_or_neutral_returns_predictable_model_when_league_has_no_data(tmp_path):
    model = train_model_or_neutral(db_path=tmp_path / "empty.sqlite", league_code="E0")

    prediction = model.predict_match("Equipo A", "Equipo B")

    assert prediction.home_team == "Equipo A"
    assert prediction.away_team == "Equipo B"
    assert prediction.recommended_probability > 0


def test_predict_manual_match_adds_automatic_calibration_columns(tmp_path):
    predictions = predict_manual_match("Arsenal", "Chelsea", db_path=tmp_path / "empty.sqlite")

    assert "calibrated_pick_probability" in predictions.columns
    assert predictions.loc[0, "calibrated_pick_probability"] == predictions.loc[0, "recommended_probability"]
    assert predictions.loc[0, "calibration_samples"] == 0
