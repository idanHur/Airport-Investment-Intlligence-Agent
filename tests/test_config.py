from airport_agent import config


def test_long_haul_threshold_is_3000_miles():
    assert config.LONG_HAUL_MILES == 3000


def test_baseline_year_is_in_years():
    assert config.BASELINE_YEAR in config.YEARS
