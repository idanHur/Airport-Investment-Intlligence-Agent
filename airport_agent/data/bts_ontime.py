"""BTS On-Time Performance: one row per flight, published as one big zip per month.

Again no API, just static files on a slow server. Each month is ~30 MB, so we
  1. download with resume (HTTP Range) so a dropped connection doesn't restart,
  2. immediately shrink it to one row per origin airport per month,
  3. cache only that small aggregate and delete the big zip.
"""
import time
import zipfile

import pandas as pd
import requests

from airport_agent.config import CACHE_DIR

URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
RAW_DIR = CACHE_DIR / "raw"
USECOLS = [
    "Year", "Month", "Origin", "Cancelled", "DepDelayMinutes", "DepDel15", "TaxiOut",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]


def cache_path(year: int, month: int):
    return CACHE_DIR / f"ontime_{year}_{month:02d}.parquet"


def _download(year: int, month: int, retries: int = 8, chunk: int = 1 << 20) -> "Path":
    """Download the monthly zip, resuming from where we stopped after any failure."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    part = RAW_DIR / f"ontime_{year}_{month}.zip.part"
    url = URL.format(year=year, month=month)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
                if response.status_code == 416:  # already have every byte
                    break
                response.raise_for_status()
                if have and response.status_code != 206:  # server ignored Range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as handle:
                    for block in response.iter_content(chunk):
                        handle.write(block)
            break
        except requests.RequestException as error:
            print(f"On-Time {year}-{month:02d}: attempt {attempt} failed ({error}); resuming")
            time.sleep(3 * attempt)
    else:
        raise RuntimeError(f"On-Time {year}-{month:02d}: gave up after {retries} attempts")
    try:
        zipfile.ZipFile(part).testzip()
    except zipfile.BadZipFile as error:
        part.unlink()
        raise RuntimeError(f"On-Time {year}-{month:02d}: corrupt download, deleted") from error
    return part


def aggregate(flights: pd.DataFrame) -> pd.DataFrame:
    """Flights -> one row per origin airport per month."""
    flights = flights.copy()
    flights["Flown"] = 1 - flights["Cancelled"]
    grouped = flights.groupby(["Year", "Month", "Origin"])
    return pd.DataFrame({
        "scheduled_departures": grouped.size(),
        "cancelled": grouped["Cancelled"].sum(),
        "delayed_15": grouped["DepDel15"].sum(),
        "dep_delay_minutes": grouped["DepDelayMinutes"].sum(),
        "flown": grouped["Flown"].sum(),
        "taxi_out_minutes": grouped["TaxiOut"].sum(),
        "taxi_out_flights": grouped["TaxiOut"].count(),
        "carrier_delay": grouped["CarrierDelay"].sum(),
        "weather_delay": grouped["WeatherDelay"].sum(),
        "nas_delay": grouped["NASDelay"].sum(),
        "security_delay": grouped["SecurityDelay"].sum(),
        "late_aircraft_delay": grouped["LateAircraftDelay"].sum(),
    }).reset_index().rename(columns={"Year": "year", "Month": "month", "Origin": "airport"})


def load_month(year: int, month: int, refresh: bool = False) -> pd.DataFrame:
    path = cache_path(year, month)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    zip_path = _download(year, month)
    with zipfile.ZipFile(zip_path) as archive:
        csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        flights = pd.read_csv(archive.open(csv_name), usecols=USECOLS, low_memory=False)
    result = aggregate(flights)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    zip_path.unlink()
    return result
