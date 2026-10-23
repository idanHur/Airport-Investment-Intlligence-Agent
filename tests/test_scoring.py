import pandas as pd
import pytest

from airport_agent.scoring import common
from airport_agent.scoring.congestion import congestion_index
from airport_agent.scoring.expansion import expansion_scores
from airport_agent.scoring.unmet_demand import unmet_demand_proxy


def _metrics():
    """Four airports in 2024: A is best on every expansion metric, C worst, D too small."""
    return pd.DataFrame({
        "airport": ["A", "B", "C", "D"], "year": [2024] * 4, "name": list("abcd"), "state": ["MA"] * 4,
        "departures": [20000, 20000, 20000, 100], "passengers": [1, 1, 1, 1],
        "load_factor": [0.90, 0.80, 0.70, 0.95],
        "passengers_vs_2019": [1.3, 1.1, 0.9, 2.0], "passengers_yoy": [0.10, 0.05, 0.00, 0.5],
        "seats_yoy": [0.08, 0.04, 0.0, 0.5],
        "long_haul_share": [0.1] * 4, "intl_share": [0.1] * 4,
        "delay_rate": [0.30, 0.20, 0.10, 0.5], "avg_taxi_out_min": [20.0, 15.0, 10.0, 30.0],
        "nas_delay_share": [0.5, 0.3, 0.1, 0.9], "months_sampled": [4] * 4,
        "cancel_rate": [0.01] * 4, "avg_dep_delay_min": [10.0] * 4,
    })


def test_percentile_best_is_100_and_direction_can_flip():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert common.percentile(values).tolist() == [25, 50, 75, 100]
    assert common.percentile(values, higher_is_better=False).tolist() == [100, 75, 50, 25]


def test_small_airports_are_not_ranked():
    result = expansion_scores(_metrics(), 2024)
    assert "D" not in result.index
    assert result.iloc[0]["eligible_airports"] == 3


def test_expansion_ranking_and_exact_score():
    result = expansion_scores(_metrics(), 2024)
    assert result.index.tolist() == ["A", "B", "C"]
    assert result.loc["A", "score"] == pytest.approx(100.0)   # best on all three components
    # C is worst on all: percentile of 1 of 3 = 33.33 on every component
    assert result.loc["C", "score"] == pytest.approx(100 / 3)
    assert result.loc["A", "rank"] == 1


def test_weights_change_the_score():
    only_load_factor = expansion_scores(_metrics(), 2024, weights={"load_factor": 1, "momentum": 0, "seat_growth": 0})
    assert only_load_factor.loc["B", "score"] == pytest.approx(200 / 3)


def test_missing_component_rescales_weights_and_is_flagged():
    metrics = _metrics()
    metrics.loc[metrics.airport == "B", "seats_yoy"] = None
    result = expansion_scores(metrics, 2024)
    assert result.loc["B", "components_missing"] == 1
    # B: load factor and momentum are both 2nd of 3 = 66.67, weights re-scaled over those two
    assert result.loc["B", "score"] == pytest.approx(200 / 3)


def test_congestion_ranks_most_strained_first():
    result = congestion_index(_metrics(), 2024)
    assert result.index.tolist() == ["A", "B", "C"]
    assert result.loc["A", "congestion_index"] == pytest.approx(100.0)


def test_unmet_demand_proxy_levels_and_notes():
    high = unmet_demand_proxy(_metrics(), "A", 2024)
    low = unmet_demand_proxy(_metrics(), "C", 2024)
    assert high["level"] == "high" and high["proxy"] == pytest.approx(100.0)
    assert low["proxy"] == pytest.approx(100 / 3, abs=0.1)  # worst of 3 airports = 33.3rd percentile (proxy is rounded)
    assert low["level"] == "medium"                # sits exactly on the 33.3 boundary; with ~100+ airports worst is ~1
    assert any("Proxy only" in note for note in high["notes"])


def test_unmet_demand_rejects_airport_below_size_floor():
    with pytest.raises(KeyError):
        unmet_demand_proxy(_metrics(), "D", 2024)
