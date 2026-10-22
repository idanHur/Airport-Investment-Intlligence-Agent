import pandas as pd

from airport_agent.data import bts_ontime


def test_aggregate_counts_delays_and_cancellations():
    flights = pd.DataFrame({
        "Year": [2024] * 3, "Month": [1] * 3, "Origin": ["BOS", "BOS", "ANC"],
        "Cancelled": [0.0, 1.0, 0.0],
        "DepDelayMinutes": [30.0, None, 0.0],
        "DepDel15": [1.0, None, 0.0],
        "TaxiOut": [20.0, None, 10.0],
        "CarrierDelay": [30.0, None, None], "WeatherDelay": [None] * 3,
        "NASDelay": [None] * 3, "SecurityDelay": [None] * 3, "LateAircraftDelay": [None] * 3,
    })
    result = bts_ontime.aggregate(flights).set_index("airport")
    assert result.loc["BOS", "scheduled_departures"] == 2
    assert result.loc["BOS", "cancelled"] == 1
    assert result.loc["BOS", "flown"] == 1
    assert result.loc["BOS", "delayed_15"] == 1
    assert result.loc["BOS", "carrier_delay"] == 30
    assert result.loc["ANC", "delayed_15"] == 0
