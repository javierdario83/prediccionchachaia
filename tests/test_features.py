import pandas as pd

from football_predictor.features import (
    assess_data_freshness,
    assess_prediction_balance,
    build_team_profile,
    enrich_with_reliability_signals,
)


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


def test_assess_data_freshness_reports_days_and_note():
    days, note = assess_data_freshness(sample_matches(), as_of=pd.Timestamp("2025-01-20"))

    assert days == 15
    assert "actualizado" in note


def test_enrich_with_reliability_downgrades_very_stale_data():
    predictions = pd.DataFrame(
        [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "recommended_market": "Gana local",
                "recommended_probability": 0.72,
                "confidence_score": 0.74,
            }
        ]
    )

    enriched = enrich_with_reliability_signals(
        predictions,
        sample_matches(),
        as_of=pd.Timestamp("2026-01-20"),
    )

    assert enriched.loc[0, "action"] == "Evitar"
    assert "desactualizado" in enriched.loc[0, "data_freshness_note"]
    assert "Frescura datos" in enriched.loc[0, "explanation"]


def test_enrich_with_reliability_flags_unknown_team_coverage():
    predictions = pd.DataFrame(
        [
            {
                "home_team": "Arsenal",
                "away_team": "Unknown FC",
                "recommended_market": "Gana local",
                "recommended_probability": 0.72,
                "confidence_score": 0.74,
            }
        ]
    )

    enriched = enrich_with_reliability_signals(
        predictions,
        sample_matches(),
        as_of=pd.Timestamp("2025-01-20"),
    )

    assert enriched.loc[0, "home_team_seen"]
    assert not enriched.loc[0, "away_team_seen"]
    assert enriched.loc[0, "action"] == "Evitar"
    assert "Unknown FC" in enriched.loc[0, "team_coverage_note"]
    assert "Cobertura equipos" in enriched.loc[0, "explanation"]


def test_assess_prediction_balance_reports_tight_1x2_margin():
    margin, note = assess_prediction_balance(
        pd.Series({"home_win_prob": 0.36, "draw_prob": 0.34, "away_win_prob": 0.30})
    )

    assert margin == 0.02
    assert "parejo" in note


def test_enrich_with_reliability_downgrades_tight_result_pick():
    matches = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22", "2025-01-29"]),
            "home_team": ["Arsenal", "Chelsea", "Arsenal", "Chelsea", "Arsenal"],
            "away_team": ["Chelsea", "Arsenal", "Chelsea", "Arsenal", "Chelsea"],
            "home_goals": [1, 1, 2, 0, 1],
            "away_goals": [1, 0, 1, 1, 0],
            "total_goals": [2, 1, 3, 1, 1],
            "over25": [0, 0, 1, 0, 0],
            "btts": [1, 0, 1, 0, 0],
        }
    )
    predictions = pd.DataFrame(
        [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "recommended_market": "Gana local",
                "recommended_probability": 0.66,
                "confidence_score": 0.70,
                "home_win_prob": 0.36,
                "draw_prob": 0.34,
                "away_win_prob": 0.30,
            }
        ]
    )

    enriched = enrich_with_reliability_signals(predictions, matches, as_of=pd.Timestamp("2025-02-05"))

    assert enriched.loc[0, "action"] == "Informativo"
    assert enriched.loc[0, "one_x_two_margin"] == 0.02
    assert "parejo" in enriched.loc[0, "match_balance_note"]
    assert "Balance 1X2" in enriched.loc[0, "explanation"]
