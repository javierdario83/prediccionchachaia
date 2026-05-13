import pandas as pd

from football_predictor.features import build_team_profile, enrich_with_reliability_signals


def sample_matches():
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]),
            "home_team": ["Arsenal", "Chelsea", "Arsenal", "Liverpool", "Arsenal"],
            "away_team": ["Chelsea", "Arsenal", "Liverpool", "Arsenal", "Everton"],
            "home_goals": [2, 0, 3, 1, 2],
            "away_goals": [1, 1, 1, 2, 0],
            "total_goals": [3, 1, 4, 3, 2],
            "over25": [1, 0, 1, 1, 0],
            "btts": [1, 0, 1, 1, 0],
        }
    )


def test_build_team_profile_has_form():
    profile = build_team_profile(sample_matches(), "Arsenal")

    assert profile.matches == 5
    assert profile.points_per_match_5 is not None
    assert profile.data_quality == 0.5


def test_enrich_with_reliability_adds_action():
    predictions = pd.DataFrame(
        [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "recommended_market": "Gana local",
                "recommended_probability": 0.66,
                "confidence_score": 0.64,
            }
        ]
    )

    enriched = enrich_with_reliability_signals(predictions, sample_matches())

    assert "action" in enriched.columns
    assert "elo_diff" in enriched.columns
