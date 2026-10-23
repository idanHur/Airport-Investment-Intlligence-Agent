import pandas as pd
import pytest

from airport_agent import metrics


def _segment(origin, dest, year, deps, seats, pax, miles, cls="F", origin_country="US", dest_country="US"):
    return {
        "ORIGIN": origin, "DEST": dest, "YEAR": year, "MONTH": 1, "CLASS": cls,
        "DEPARTURES_PERFORMED": deps, "SEATS": seats, "PASSENGERS": pax, "DISTANCE": miles,
        "ORIGIN_COUNTRY": origin_country, "DEST_COUNTRY": dest_country,
    }


def test_load_factor_stage_length_and_long_haul_share():
    t100 = pd.DataFrame([
        _segment("ANC", "SEA", 2024, 10, 1000, 800, 1448),
        _segment("ANC", "ORD", 2024, 10, 1000, 900, 2800),
        _segment("ANC", "NRT", 2024, 10, 1000, 700, 3400, dest_country="JP"),
        _segment("ANC", "FAI", 2024, 70, 7000, 5000, 261),
    ])
    row = metrics.t100_airport_year(t100).iloc[0]
    assert row["departures"] == 100
    assert row["load_factor"] == pytest.approx(7400 / 10000)
    assert row["long_haul_share"] == pytest.approx(10 / 100)  # only NRT >= 3,000 mi
    assert row["intl_share"] == pytest.approx(10 / 100)
    assert row["avg_stage_miles"] == pytest.approx((10 * 1448 + 10 * 2800 + 10 * 3400 + 70 * 261) / 100)


def test_cargo_and_foreign_origins_and_empty_rows_are_excluded():
    t100 = pd.DataFrame([
        _segment("BOS", "JFK", 2024, 10, 1000, 900, 187),
        _segment("BOS", "ORD", 2024, 5, 0, 0, 867, cls="G"),                         # all-cargo
        _segment("YYZ", "BOS", 2024, 9, 900, 800, 430, origin_country="CA"),          # foreign origin
        _segment("BOS", "BOS", 2024, 0, 0, 0, 0),                                     # no departures
    ])
    result = metrics.t100_airport_year(t100)
    assert result["airport"].tolist() == ["BOS"]
    assert result.iloc[0]["departures"] == 10


def test_growth_is_only_computed_for_consecutive_years():
    base = pd.DataFrame({
        "airport": ["BOS"] * 3, "year": [2019, 2023, 2024],
        "passengers": [100.0, 80.0, 100.0], "seats": [120.0, 100.0, 110.0], "departures": [10.0, 8.0, 9.0],
    })
    result = metrics.add_growth(base).set_index("year")
    assert pd.isna(result.loc[2023, "passengers_yoy"])           # 2022 missing
    assert result.loc[2024, "passengers_yoy"] == pytest.approx(0.25)
    assert result.loc[2024, "passengers_vs_2019"] == pytest.approx(1.0)
    assert result.loc[2023, "passengers_vs_2019"] == pytest.approx(0.8)


def test_ontime_rates():
    monthly = pd.DataFrame({
        "airport": ["BOS", "BOS"], "year": [2024, 2024], "month": [1, 7],
        "scheduled_departures": [100, 100], "cancelled": [10, 0], "delayed_15": [18, 20],
        "dep_delay_minutes": [900, 1100], "flown": [90, 100],
        "taxi_out_minutes": [1800, 2000], "taxi_out_flights": [90, 100],
        "weather_delay": [100, 100], "nas_delay": [300, 300], "carrier_delay": [100, 100],
        "late_aircraft_delay": [100, 100], "security_delay": [0, 0],
    })
    row = metrics.ontime_airport_year(monthly).iloc[0]
    assert row["months_sampled"] == 2
    assert row["delay_rate"] == pytest.approx(38 / 190)
    assert row["cancel_rate"] == pytest.approx(10 / 200)
    assert row["avg_dep_delay_min"] == pytest.approx(2000 / 190)
    assert row["nas_delay_share"] == pytest.approx(600 / 1200)  # 600 NAS of 1,200 total cause minutes
