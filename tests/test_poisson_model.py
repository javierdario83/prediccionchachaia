import pandas as pd

from football_predictor.poisson_model import PoissonFootballModel, score_probability_matrix


def sample_matches():
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"]),
            "home_team": ["Arsenal", "Chelsea", "Arsenal", "Liverpool"],
            "away_team": ["Chelsea", "Arsenal", "Liverpool", "Chelsea"],
            "home_goals": [2, 1, 3, 0],
            "away_goals": [1, 1, 1, 2],
        }
    )


def test_score_matrix_is_normalized():
    matrix = score_probability_matrix(1.5, 1.1, max_goals=8)
    assert matrix.shape == (9, 9)
    assert abs(matrix.sum() - 1.0) < 1e-9


def test_model_predicts_probabilities():
    model = PoissonFootballModel().fit(sample_matches())
    prediction = model.predict_match("Arsenal", "Chelsea", league_code="E0")

    total_1x2 = prediction.home_win_prob + prediction.draw_prob + prediction.away_win_prob
    assert abs(total_1x2 - 1.0) < 0.01
    assert 0 <= prediction.over25_prob <= 1
    assert 0 <= prediction.btts_yes_prob <= 1
    assert "-" in prediction.predicted_score
