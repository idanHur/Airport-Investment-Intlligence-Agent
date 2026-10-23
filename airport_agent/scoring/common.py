"""Shared helpers for all scores: who is eligible, and how values become percentiles."""
import pandas as pd

from airport_agent.config import MIN_DEPARTURES


def eligible(metrics: pd.DataFrame, year: int, min_departures: int = MIN_DEPARTURES) -> pd.DataFrame:
    """Airports big enough to rank in `year`, one row each, indexed by IATA code."""
    rows = metrics[(metrics["year"] == year) & (metrics["departures"] >= min_departures)]
    return rows.set_index("airport")


def percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """0-100 rank among the given airports (100 = best). Missing values stay missing.
    Ranking instead of scaling means one huge outlier cannot squash everyone else."""
    ranks = values.rank(pct=True, ascending=higher_is_better) * 100
    return ranks


def weighted_average(components: pd.DataFrame, weights: dict) -> tuple[pd.Series, pd.Series]:
    """Weighted mean of component columns. If an airport lacks a component, the weights
    of the components it does have are re-scaled, and it is reported as partial.
    Returns (score, number of components missing)."""
    weight = pd.Series(weights, dtype=float)[components.columns]
    present = components.notna()
    used_weight = present.mul(weight, axis=1).sum(axis=1)
    score = components.fillna(0).mul(weight, axis=1).sum(axis=1) / used_weight.where(used_weight > 0)
    return score, (~present).sum(axis=1)
