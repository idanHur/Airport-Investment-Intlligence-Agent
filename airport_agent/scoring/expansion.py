"""Expansion-candidate score: where is demand outgrowing the airport?

score = weighted mean of three 0-100 percentiles (weights in config.EXPANSION_WEIGHTS)
  load_factor : passengers / seats          (planes are full)
  momentum    : mean of percentiles of passengers vs 2019 and passengers YoY
  seat_growth : seats YoY                   (airlines are adding capacity)
"""
import pandas as pd

from airport_agent.config import EXPANSION_WEIGHTS, MIN_DEPARTURES
from airport_agent.scoring.common import eligible, percentile, weighted_average

REPORTED = ["name", "state", "passengers", "load_factor", "passengers_vs_2019", "passengers_yoy",
            "seats_yoy", "long_haul_share", "intl_share"]


def expansion_scores(metrics: pd.DataFrame, year: int, weights: dict = EXPANSION_WEIGHTS,
                     min_departures: int = MIN_DEPARTURES) -> pd.DataFrame:
    """Score every eligible airport in `year`; sorted best first, with a `rank` column.
    Percentiles are computed against ALL eligible airports, so a region's score
    is relative to the national field, not just the region."""
    pool = eligible(metrics, year, min_departures)
    components = pd.DataFrame({
        "load_factor": percentile(pool["load_factor"]),
        "momentum": pd.concat([percentile(pool["passengers_vs_2019"]), percentile(pool["passengers_yoy"])], axis=1).mean(axis=1),
        "seat_growth": percentile(pool["seats_yoy"]),
    })
    score, missing = weighted_average(components, weights)
    result = components.add_prefix("pct_").join(pool[[c for c in REPORTED if c in pool.columns]])
    result["score"] = score
    result["components_missing"] = missing
    result = result.dropna(subset=["score"]).sort_values("score", ascending=False)
    result["rank"] = range(1, len(result) + 1)
    result["eligible_airports"] = len(pool)
    return result
