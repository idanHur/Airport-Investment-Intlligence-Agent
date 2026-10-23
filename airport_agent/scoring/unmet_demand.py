"""Unmet-demand proxy. There is no direct data on turned-away passengers, so this
looks for the *symptoms*: full planes, growing demand, and operational strain.

proxy = mean of three 0-100 percentiles: load_factor, passengers_vs_2019, congestion index.
It is a screening signal, not a measurement; the notes returned with it say so.
"""
import pandas as pd

from airport_agent.config import MIN_DEPARTURES
from airport_agent.scoring.common import eligible, percentile
from airport_agent.scoring.congestion import congestion_index

LEVELS = [(66.7, "high"), (33.3, "medium"), (0, "low")]


def _level(value: float) -> str:
    return next(label for floor, label in LEVELS if value >= floor)


def unmet_demand_proxy(metrics: pd.DataFrame, airport: str, year: int, min_departures: int = MIN_DEPARTURES) -> dict:
    """Proxy for one airport. Percentiles are relative to all eligible US airports."""
    pool = eligible(metrics, year, min_departures)
    if airport not in pool.index:
        raise KeyError(f"{airport} is not ranked in {year} (missing or under {min_departures:,} departures)")
    congestion = congestion_index(metrics, year, min_departures)
    parts = {
        "load_factor": percentile(pool["load_factor"])[airport],
        "growth_vs_2019": percentile(pool["passengers_vs_2019"])[airport],
        "congestion": congestion["congestion_index"].get(airport, float("nan")),
    }
    available = {k: v for k, v in parts.items() if pd.notna(v)}
    value = sum(available.values()) / len(available)
    row = pool.loc[airport]
    notes = ["Proxy only: no data on passengers turned away, gate counts or terminal size."]
    if len(available) < len(parts):
        notes.append("No delay data for this airport, so the proxy uses fewer components.")
    if row.get("months_sampled") == row.get("months_sampled") and row.get("months_sampled", 4) < 4:
        notes.append("Delay figures use fewer than 4 sampled months.")
    return {
        "airport": airport, "year": year, "proxy": round(float(value), 1), "level": _level(value),
        "components": {k: (None if pd.isna(v) else round(float(v), 1)) for k, v in parts.items()},
        "raw": {"load_factor": float(row["load_factor"]), "passengers_vs_2019": float(row["passengers_vs_2019"])},
        "airports_compared": len(pool), "notes": notes,
    }
