import pandas as pd

from football_predictor.backtesting import BacktestConfig, run_backtest


def test_run_backtest_returns_summary():
    rows = []
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    for index in range(36):
        rows.append(
            {
                "match_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=index),
                "league_code": "E0",
                "home_team": teams[index % 4],
                "away_team": teams[(index + 1) % 4],
                "home_goals": index % 3,
                "away_goals": (index + 1) % 2,
                "total_goals": (index % 3) + ((index + 1) % 2),
                "over25": int(((index % 3) + ((index + 1) % 2)) > 2.5),
                "btts": int((index % 3) > 0 and ((index + 1) % 2) > 0),
            }
        )
    details, summary = run_backtest(pd.DataFrame(rows), BacktestConfig(min_train_matches=20, max_test_matches=5))

    assert len(details) == 5
    assert not summary.empty
    assert "Acierto 1X2" in summary.columns
