"""Expansion-candidate score: where is demand outgrowing the airport?

score = weighted mean of three 0-100 percentiles (weights in config.EXPANSION_WEIGHTS)
  load_factor : passengers / seats          (planes are full)
  momentum    : mean of percentiles of passengers vs 2019 and passengers YoY
  seat_growth : seats YoY                   (airlines are adding capacity)

Gate: airports under config.MIN_LOAD_FACTOR still get a score, but no rank, because
growth on mostly-empty planes is not evidence the terminal is too small.
Flags: plain-language warnings that travel with each row so the agent can repeat them.
"""
import pandas as pd

from airport_agent.config import (
    EXPANSION_WEIGHTS, EXTREME_GROWTH_VS_BASELINE, EXTREME_SEAT_GROWTH, MIN_DEPARTURES,
    MIN_LOAD_FACTOR, SMALL_AIRPORT_PASSENGERS,
)
from airport_agent.scoring.common import eligible, percentile, weighted_average

REPORTED = ["name", "state", "passengers", "load_factor", "passengers_vs_2019", "passengers_yoy",
            "seats_yoy", "long_haul_share", "intl_share", "months_sampled"]


def flags_for(row: pd.Series, components_missing: int) -> list[str]:
    """Warnings about how much to trust one airport's score."""
    flags = []
    if row["load_factor"] < MIN_LOAD_FACTOR:
        flags.append(f"Load factor {row['load_factor']:.0%} is below the {MIN_LOAD_FACTOR:.0%} gate: not ranked as a candidate")
    if row["passengers"] < SMALL_AIRPORT_PASSENGERS:
        flags.append(f"Small airport ({row['passengers']:,.0f} passengers): ratios are noisy")
    if pd.notna(row["passengers_vs_2019"]) and row["passengers_vs_2019"] > EXTREME_GROWTH_VS_BASELINE:
        flags.append(f"Traffic is {row['passengers_vs_2019']:.1f}x its 2019 level: likely new service, not a trend")
    if pd.notna(row["seats_yoy"]) and row["seats_yoy"] > EXTREME_SEAT_GROWTH:
        flags.append(f"Seats grew {row['seats_yoy']:.0%} in one year: a step change, not a trend")
    if components_missing:
        flags.append(f"{components_missing} score component(s) unavailable; weights re-scaled")
    return flags


def expansion_scores(metrics: pd.DataFrame, year: int, weights: dict = EXPANSION_WEIGHTS,
                     min_departures: int = MIN_DEPARTURES) -> pd.DataFrame:
    """Score every eligible airport in `year`. Ranked (gate-passing) airports come first,
    best first; gated airports follow with rank NaN.
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
    result = result.dropna(subset=["score"])
    result["gated"] = result["load_factor"] < MIN_LOAD_FACTOR
    result["flags"] = [flags_for(row, row["components_missing"]) for _, row in result.iterrows()]
    result = result.sort_values(["gated", "score"], ascending=[True, False])
    result["rank"] = pd.Series(range(1, len(result) + 1), index=result.index).where(~result["gated"])
    result["eligible_airports"] = len(pool)
    result["ranked_airports"] = int((~result["gated"]).sum())
    return result
