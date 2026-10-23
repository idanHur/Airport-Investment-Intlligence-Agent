"""Congestion index: how strained is the airport operationally?

Equal-weight mean of three 0-100 percentiles, where 100 = most congested:
  delay_rate        : share of flown departures delayed 15+ min
  avg_taxi_out_min  : average minutes from gate to takeoff (queueing for the runway)
  nas_delay_share   : share of delay minutes caused by airport/airspace system limits

Built from the sampled On-Time months, so it is a seasonal estimate.
"""
import pandas as pd

from airport_agent.config import MIN_DEPARTURES
from airport_agent.scoring.common import eligible, percentile

COMPONENTS = ["delay_rate", "avg_taxi_out_min", "nas_delay_share"]


def congestion_index(metrics: pd.DataFrame, year: int, min_departures: int = MIN_DEPARTURES) -> pd.DataFrame:
    """Congestion index for every eligible airport that has delay data; most congested first."""
    pool = eligible(metrics, year, min_departures).dropna(subset=COMPONENTS)
    percentiles = pd.DataFrame({c: percentile(pool[c]) for c in COMPONENTS})
    result = percentiles.add_prefix("pct_").join(pool[["name", "state", "months_sampled", "cancel_rate", "avg_dep_delay_min"] + COMPONENTS])
    result["congestion_index"] = percentiles.mean(axis=1)
    result = result.sort_values("congestion_index", ascending=False)
    result["rank"] = range(1, len(result) + 1)
    result["airports_compared"] = len(result)
    return result
