"""Airport reference data (OurAirports) and the named regions the agent understands."""
import pandas as pd

from airport_agent.config import CACHE_DIR

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
CACHE_FILE = CACHE_DIR / "airports.parquet"

# Regions are a judgement call, so they live in code where they can be read and argued with.
REGIONS = {
    "new england": {"states": ["CT", "ME", "MA", "NH", "RI", "VT"]},
    # Metro area airports, not a state: LA basin including Santa Ana (SNA).
    "los angeles area": {"airports": ["LAX", "BUR", "LGB", "ONT", "SNA"]},
}


def load(refresh: bool = False) -> pd.DataFrame:
    """US airports with scheduled airline service: iata, name, city, state, lat, lon."""
    if CACHE_FILE.exists() and not refresh:
        return pd.read_parquet(CACHE_FILE)
    raw = pd.read_csv(URL, low_memory=False)
    us = raw[(raw["iso_country"] == "US") & (raw["scheduled_service"] == "yes") & raw["iata_code"].notna()]
    us = us[us["type"].isin(["large_airport", "medium_airport", "small_airport"])]
    result = pd.DataFrame({
        "airport": us["iata_code"],
        "name": us["name"],
        "city": us["municipality"],
        "state": us["iso_region"].str.replace("US-", "", regex=False),
        "lat": us["latitude_deg"],
        "lon": us["longitude_deg"],
    }).drop_duplicates("airport").reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(CACHE_FILE, index=False)
    return result


def airports_in_region(region: str, airports: pd.DataFrame) -> list[str]:
    """IATA codes in a named region (case-insensitive)."""
    spec = REGIONS.get(region.strip().lower())
    if spec is None:
        raise KeyError(f"Unknown region '{region}'. Known: {sorted(REGIONS)}")
    if "states" in spec:
        return sorted(airports.loc[airports["state"].isin(spec["states"]), "airport"])
    return sorted(set(spec["airports"]) & set(airports["airport"]))
