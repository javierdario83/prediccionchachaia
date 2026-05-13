import pandas as pd

from football_predictor.odds import add_market_probabilities


def test_add_market_probabilities_computes_value_gap():
    predictions = pd.DataFrame(
        [
            {
                "recommended_market": "Gana local",
                "home_win_prob": 0.60,
                "b365_home": 2.2,
                "b365_draw": 3.4,
                "b365_away": 3.5,
            }
        ]
    )

    enriched = add_market_probabilities(predictions)

    assert enriched.loc[0, "market_home_prob"] is not None
    assert enriched.loc[0, "value_gap"] is not None
