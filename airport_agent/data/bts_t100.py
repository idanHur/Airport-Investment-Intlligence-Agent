"""BTS T-100 Segment (All Carriers): domestic + international, one row per
origin-destination-carrier-month.

BTS publishes no API for this table. The only way to get it is the TranStats
download form, so we replay what the browser does: GET the form (to collect the
hidden ASP.NET VIEWSTATE fields), then POST it with the columns and year we want.
The result is cached as parquet, so the network is only touched once per year.
"""
import io
import re
import time
import zipfile

import pandas as pd
import requests

from airport_agent.config import CACHE_DIR

FORM_URL = (
    "https://www.transtats.bts.gov/DL_SelectFields.aspx"
    "?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4%20Pn44vr45"
)
COLUMNS = [
    "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS", "DISTANCE",
    "UNIQUE_CARRIER", "CLASS",
    "ORIGIN", "ORIGIN_COUNTRY", "DEST", "DEST_COUNTRY",
    "YEAR", "MONTH",
]
HIDDEN_FIELDS = ["__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"]


def cache_path(year: int):
    return CACHE_DIR / f"t100_{year}.parquet"


def _hidden_fields(html: str) -> dict:
    """Pull the hidden ASP.NET state fields out of the form page."""
    fields = {}
    for name in HIDDEN_FIELDS:
        match = re.search(rf'id="{name}" value="([^"]*)"', html)
        if not match:
            raise RuntimeError(f"Form field {name} not found; BTS page layout may have changed")
        fields[name] = match.group(1)
    return fields


def _download_year(year: int, timeout: int = 300) -> pd.DataFrame:
    with requests.Session() as session:
        form = session.get(FORM_URL, timeout=60)
        form.raise_for_status()
        payload = {
            **_hidden_fields(form.text),
            "cboGeography": "All",
            "cboYear": str(year),
            "cboPeriod": "All",
            "chkDownloadZip": "on",
            "btnDownload": "Download",
            **{column: "on" for column in COLUMNS},
        }
        response = session.post(FORM_URL, data=payload, timeout=timeout)
        response.raise_for_status()
    if "zip" not in response.headers.get("Content-Type", ""):
        raise RuntimeError(f"BTS did not return a zip for {year} (got {response.headers.get('Content-Type')})")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv") and "documentation" not in n.lower())
        return pd.read_csv(archive.open(csv_name), low_memory=False)


def load_year(year: int, retries: int = 3, refresh: bool = False) -> pd.DataFrame:
    """Return T-100 segments for a year, downloading once and then reading the cache."""
    path = cache_path(year)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    for attempt in range(1, retries + 1):
        try:
            frame = _download_year(year)
            break
        except (requests.RequestException, RuntimeError) as error:
            if attempt == retries:
                raise
            print(f"T-100 {year}: attempt {attempt} failed ({error}); retrying")
            time.sleep(5 * attempt)
    if frame.empty:
        raise RuntimeError(f"BTS returned no rows for {year}; data may not be published yet")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame
