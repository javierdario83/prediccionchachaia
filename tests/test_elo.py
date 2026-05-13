import pandas as pd

from football_predictor.elo import compute_elo_ratings


def test_compute_elo_ratings_rewards_winner():
    matches = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-01-01"]),
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_goals": [3],
            "away_goals": [0],
        }
    )

    ratings = compute_elo_ratings(matches)

    assert ratings["Arsenal"] > ratings["Chelsea"]
