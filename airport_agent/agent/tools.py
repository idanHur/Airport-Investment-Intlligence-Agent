"""LangChain tool wrappers. Each is a thin shell: load cached data, call a pure query
function, return its dict. The LLM never sees raw data and never does arithmetic."""
from functools import lru_cache

from langchain_core.tools import tool

from airport_agent import metrics as metrics_module
from airport_agent.agent import queries
from airport_agent.data import airports as airports_module, bts_t100


@lru_cache(maxsize=1)
def _metrics():
    return metrics_module.build_metrics()


@lru_cache(maxsize=1)
def _airports():
    return airports_module.load()


@lru_cache(maxsize=8)
def _t100(year: int):
    return bts_t100.load_year(year)


def _safe(function, *args, **kwargs):
    """Turn expected lookup errors into a message the LLM can relay instead of crashing the graph."""
    try:
        return function(*args, **kwargs)
    except KeyError as error:
        return {"error": str(error).strip("'\"")}


@tool
def find_airport(query: str) -> list[dict]:
    """Look up US airports by IATA code, city or airport name (e.g. 'Santa Ana' or 'Anchorage').
    Use this to get the IATA code before calling other tools."""
    return queries.find_airports(_airports(), query)


@tool
def rank_airports(region: str, year: int = queries.LATEST_YEAR, top_n: int = 5) -> dict:
    """Rank airports in a region as terminal-expansion candidates using the deterministic expansion score.
    region: 'new england', 'los angeles area', a 2-letter US state code, 'national', or comma-separated IATA codes."""
    return _safe(queries.rank_region, _metrics(), _airports(), region, year, top_n)


@tool
def compare_airports(airport_codes: list[str], year: int = queries.LATEST_YEAR) -> dict:
    """Compare airports side by side on traffic, load factor, growth and congestion (delays, taxi-out time).
    Use for congestion comparisons, e.g. ['LAX', 'SNA']."""
    return _safe(queries.compare, _metrics(), airport_codes, year)


@tool
def long_haul_share(airport_code: str, year: int = queries.LATEST_YEAR, threshold_miles: float = 3000) -> dict:
    """Share of an airport's departing flights that are long haul (nonstop segment >= threshold_miles, default 3000).
    Also returns how the answer changes with other thresholds."""
    return _safe(queries.long_haul, _t100(year), airport_code, year, threshold_miles)


@tool
def unmet_demand(airport_code: str, year: int = queries.LATEST_YEAR) -> dict:
    """Proxy for unmet flight demand at one airport (full planes + growth + congestion). It is a screening
    signal, not a measurement; always state that to the user."""
    return _safe(queries.unmet_demand, _metrics(), airport_code, year)


@tool
def data_coverage() -> dict:
    """Describe the data years, definitions, thresholds, weights and known limitations. Use when the user asks
    about methodology, assumptions or data quality."""
    return queries.coverage()


TOOLS = [find_airport, rank_airports, compare_airports, long_haul_share, unmet_demand, data_coverage]
