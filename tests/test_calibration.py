import pandas as pd

from football_predictor.calibration import apply_calibration, build_calibration_table, summarise_markets


def test_build_calibration_table_and_apply():
    details = pd.DataFrame(
        {
            "recommended_market": ["Over 2.5"] * 10,
            "recommended_probability": [0.62] * 10,
            "actual_total_goals": [3, 4, 1, 3, 2, 4, 3, 1, 5, 3],
            "actual_result": ["H"] * 10,
            "actual_btts": [1] * 10,
            "hit_1x2": [1] * 10,
            "over25_prob": [0.62] * 10,
            "hit_over25": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            "btts_yes_prob": [0.55] * 10,
            "hit_btts": [1] * 10,
        }
    )
    table = build_calibration_table(details)
    predictions = pd.DataFrame([{"recommended_probability": 0.62}])

    calibrated = apply_calibration(predictions, table, min_samples=2)

    assert not table.empty
    assert "calibrated_pick_probability" in calibrated.columns
    assert calibrated.loc[0, "calibration_samples"] >= 2


def test_summarise_markets():
    details = pd.DataFrame(
        {
            "recommended_market": ["Gana local", "Gana local"],
            "actual_result": ["H", "A"],
            "actual_total_goals": [2, 1],
            "actual_btts": [0, 0],
            "confidence_score": [0.7, 0.6],
            "recommended_probability": [0.65, 0.62],
        }
    )

    summary = summarise_markets(details)

    assert summary.loc[0, "Partidos"] == 2
    assert "Acierto" in summary.columns
