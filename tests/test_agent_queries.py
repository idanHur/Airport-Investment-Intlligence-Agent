import json

import numpy as np
import pandas as pd
import pytest

from airport_agent.agent import queries
from tests.test_scoring import _metrics


def _ref():
    return pd.DataFrame({
        "airport": ["A", "B", "C", "SNA"], "state": ["MA", "MA", "CA", "CA"],
        "city": ["Boston", "Worcester", "Santa Ana", "Santa Ana"], "name": ["Alpha Intl", "Beta Field", "Gamma", "John Wayne"],
    })


def test_clean_makes_results_json_serialisable():
    out = queries.clean({"a": np.float64(1.23456789), "b": np.int64(3), "c": float("nan"), "d": [np.float64(2.0)]})
    assert out == {"a": 1.2346, "b": 3, "c": None, "d": [2.0]}
    json.dumps(out)


def test_find_airports_by_city_name_and_code():
    assert queries.find_airports(_ref(), "santa ana")[0]["city"] == "Santa Ana"
    assert queries.find_airports(_ref(), "john wayne")[0]["airport"] == "SNA"
    assert queries.find_airports(_ref(), "sna")[0]["airport"] == "SNA"


def test_region_codes_accepts_state_national_list_and_rejects_unknown():
    assert queries.region_codes("ma", _ref()) == ["A", "B"]
    assert queries.region_codes("national", _ref()) == ["A", "B", "C", "SNA"]
    assert queries.region_codes("a, sna", _ref()) == ["A", "SNA"]
    with pytest.raises(KeyError):
        queries.region_codes("narnia", _ref())


def test_rank_region_returns_candidates_assumptions_and_json():
    metrics = _metrics()
    metrics["passengers"] = 5_000_000
    ref = pd.DataFrame({"airport": ["A", "B", "C", "D"], "state": ["MA"] * 4, "city": list("abcd"), "name": list("abcd")})
    result = queries.rank_region(metrics, ref, "MA", year=2024, top_n=2)
    assert [c["airport"] for c in result["top_candidates"]] == ["A", "B"]
    assert result["too_small_to_rank"] == ["D"]          # 100 departures: under the size floor
    assert result["national_field"] == {"scored": 3, "ranked": 3}
    assert any("departures" in a for a in result["assumptions"])
    json.dumps(result)


def test_compare_reports_missing_airports_in_notes():
    result = queries.compare(_metrics(), ["a", "zzz"], year=2024)
    assert [a["airport"] for a in result["airports"]] == ["A"]
    assert result["airports"][0]["congestion_rank"] == 1
    assert any("ZZZ" in n for n in result["notes"])


def test_long_haul_share_and_sensitivity():
    t100 = pd.DataFrame({
        "ORIGIN": ["ANC"] * 3, "DEST": ["SEA", "ORD", "NRT"], "YEAR": [2024] * 3, "MONTH": [1] * 3, "CLASS": ["F"] * 3,
        "DEPARTURES_PERFORMED": [80.0, 10.0, 10.0], "SEATS": [1.0] * 3, "PASSENGERS": [1.0] * 3,
        "DISTANCE": [1448.0, 2800.0, 3400.0], "ORIGIN_COUNTRY": ["US"] * 3, "DEST_COUNTRY": ["US", "US", "JP"],
    })
    result = queries.long_haul(t100, "anc", 2024)
    assert result["long_haul_share_of_flights"] == pytest.approx(0.10)
    at_2500 = result["sensitivity_by_threshold"][">= 2500 miles"]
    assert at_2500["share_of_flights"] == pytest.approx(0.20)   # Chicago joins
    assert at_2500["long_haul_departures"] == 20
    assert "10.0%" in result["summary"] and "20.0% (20 flights)" in result["summary"]
    assert any("assumption" in n for n in result["notes"])
    with pytest.raises(KeyError):
        queries.long_haul(t100, "XXX", 2024)
