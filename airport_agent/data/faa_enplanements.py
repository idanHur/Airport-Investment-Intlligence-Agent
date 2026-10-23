"""FAA enplanements at commercial service airports, one spreadsheet per year.

Used as an independent cross-check on T-100 passenger counts and for the FAA hub
size class (L/M/S/N). The FAA moves its files around, so the URLs are listed
explicitly instead of being guessed from a pattern.
"""
import io

import pandas as pd
import requests

from airport_agent.config import CACHE_DIR

_OLD = "https://www.faa.gov/sites/faa.gov/files/airports/planning_capacity/passenger_allcargo_stats/passenger"
URLS = {
    2019: f"{_OLD}/cy19-commercial-service-enplanements.xlsx",
    2022: "https://www.faa.gov/sites/faa.gov/files/2023-09/cy22-commercial-service-enplanements.xlsx",
    2023: "https://www.faa.gov/sites/faa.gov/files/2024-10/cy23-commercial-service-enplanements.xlsx",
    2024: "https://www.faa.gov/sites/faa.gov/files/arp-cy2024-commercial-service-enplanements.xlsx",
    2025: "https://www.faa.gov/sites/faa.gov/files/arp-cy2025-commercial-service-enplanements.xlsx",
}


def cache_path(year: int):
    return CACHE_DIR / f"faa_enplanements_{year}.parquet"


def parse(content: bytes, year: int) -> pd.DataFrame:
    """Spreadsheet bytes -> airport, state, hub class, enplanements for `year`."""
    sheet = pd.read_excel(io.BytesIO(content))
    wanted = f"CY {year % 100} Enplanements"
    column = next((c for c in sheet.columns if str(c).strip().upper() == wanted.upper()), None)
    if column is None:
        raise RuntimeError(f"FAA {year}: column '{wanted}' not found in {list(sheet.columns)}")
    result = sheet.rename(columns={
        "Locid": "airport", "ST": "state", "City": "city", "Airport Name": "name", "Hub": "hub_class", column: "enplanements",
    })[["airport", "state", "city", "name", "hub_class", "enplanements"]]
    result["enplanements"] = pd.to_numeric(result["enplanements"], errors="coerce")  # blanks/footnotes -> NaN
    result = result.dropna(subset=["airport", "enplanements"])
    text_columns = ["airport", "state", "city", "name", "hub_class"]
    result[text_columns] = result[text_columns].astype(str)  # some cells are numeric in old files
    result["year"] = year
    return result.reset_index(drop=True)


def load_year(year: int, refresh: bool = False) -> pd.DataFrame:
    path = cache_path(year)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    response = requests.get(URLS[year], timeout=60)
    response.raise_for_status()
    result = parse(response.content, year)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return result
