import pandas as pd

from football_predictor.cleaner import clean_matches


def test_clean_matches_creates_targets():
    raw = pd.DataFrame(
        {
            "Div": ["E0"],
            "Date": ["12/08/2025"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1],
            "FTR": ["H"],
            "Season": ["2526"],
            "LeagueCode": ["E0"],
        }
    )

    cleaned = clean_matches(raw)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "TotalGoals"] == 3
    assert cleaned.loc[0, "Over25"] == 1
    assert cleaned.loc[0, "BTTS"] == 1
    assert cleaned.loc[0, "HomeWin"] == 1
