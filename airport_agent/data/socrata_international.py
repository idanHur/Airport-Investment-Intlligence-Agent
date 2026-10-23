"""International passengers per US airport from the BTS Socrata API.

This is the one real REST API in the project. T-100 already contains international
segments, so this is used as a cross-check: do the two sources agree?
"""
import pandas as pd
import requests

from airport_agent.config import CACHE_DIR

URL = "https://datahub.transportation.gov/resource/xgub-n9bw.json"


def cache_path(year: int):
    return CACHE_DIR / f"socrata_intl_{year}.parquet"


def load_year(year: int, refresh: bool = False) -> pd.DataFrame:
    """Total international passengers (scheduled + charter) by US airport for a year."""
    path = cache_path(year)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    # Aggregation happens server-side (SoQL), so we download ~400 rows, not 800k.
    params = {
        "$select": "usg_apt as airport, sum(total) as intl_passengers",
        "$where": f"type='Passengers' AND year={year}",
        "$group": "usg_apt",
        "$limit": 5000,
    }
    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    result = pd.DataFrame(response.json())
    if result.empty:
        raise RuntimeError(f"Socrata returned no international passengers for {year}")
    result["intl_passengers"] = result["intl_passengers"].astype(float)
    result["year"] = year
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return result
