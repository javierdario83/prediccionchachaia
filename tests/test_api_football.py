from football_predictor.providers.api_football import (
    build_context_from_api_payloads,
    parse_expected_goals,
    team_similarity,
)


def test_team_similarity_handles_common_variants():
    assert team_similarity("Man United", "Manchester United") > 0.62


def test_build_context_from_api_payloads_counts_absences_and_xg():
    fixture = {
        "fixture": {"id": 123, "status": {"short": "NS"}},
        "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}},
    }
    injuries = [
        {"team": {"name": "Arsenal"}, "player": {"type": "Injury", "reason": "Knee"}},
        {"team": {"name": "Chelsea"}, "player": {"type": "Suspended", "reason": "Red card"}},
    ]
    lineups = [
        {"team": {"name": "Arsenal"}, "formation": "4-3-3"},
        {"team": {"name": "Chelsea"}, "formation": "3-4-3"},
    ]
    statistics = [
        {"team": {"name": "Arsenal"}, "statistics": [{"type": "Expected Goals", "value": "1.8"}]},
        {"team": {"name": "Chelsea"}, "statistics": [{"type": "Expected Goals", "value": "0.9"}]},
    ]

    context = build_context_from_api_payloads(fixture, injuries, lineups, statistics, "Arsenal", "Chelsea")

    assert context.fixture_id == 123
    assert context.home_injuries == 1
    assert context.away_suspensions == 1
    assert context.home_formation == "4-3-3"
    assert context.home_xg == 1.8


def test_parse_expected_goals_missing_returns_none():
    assert parse_expected_goals([], "A", "B") == (None, None)
