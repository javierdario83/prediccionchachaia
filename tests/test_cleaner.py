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


def test_clean_matches_accepts_rolling_new_file_aliases():
    raw = pd.DataFrame(
        {
            "Country": ["Mexico"],
            "League": ["Liga MX"],
            "Date": ["2025-08-12"],
            "Home": ["America"],
            "Away": ["Pumas"],
            "HG": ["2"],
            "AG": ["2"],
            "Res": ["D"],
            "Season": ["all"],
            "LeagueCode": ["MEX"],
        }
    )

    cleaned = clean_matches(raw)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "HomeTeam"] == "America"
    assert cleaned.loc[0, "AwayTeam"] == "Pumas"
    assert cleaned.loc[0, "FTHG"] == 2
    assert cleaned.loc[0, "FTAG"] == 2
    assert cleaned.loc[0, "FTR"] == "D"
    assert cleaned.loc[0, "BTTS"] == 1


def test_clean_matches_returns_empty_when_required_columns_are_missing():
    raw = pd.DataFrame({"Date": ["2025-08-12"], "Team": ["America"]})

    cleaned = clean_matches(raw)

    assert cleaned.empty
