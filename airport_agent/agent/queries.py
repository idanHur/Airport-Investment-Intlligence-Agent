"""Pure query functions the agent's tools call. They take DataFrames and return plain
JSON-friendly dicts, so they can be unit tested without an LLM or the network.

Every result carries an `assumptions` / `notes` list: the LLM is told to repeat these,
which is how scoping and uncertainty reach the user.
"""
import math

import pandas as pd

from airport_agent import config, metrics as metrics_module
from airport_agent.data import airports as airports_module
from airport_agent.scoring.common import eligible
from airport_agent.scoring.congestion import congestion_index
from airport_agent.scoring.expansion import expansion_scores
from airport_agent.scoring.unmet_demand import unmet_demand_proxy

LATEST_YEAR = max(config.YEARS)


def clean(value):
    """numpy/pandas values -> plain Python, NaN -> None, so results serialise to JSON."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    return value


def find_airports(ref: pd.DataFrame, query: str, limit: int = 8) -> list[dict]:
    """Look up airports by IATA code, city or name. 'Santa Ana' -> SNA."""
    q = query.strip().lower()
    exact = ref[ref["airport"].str.lower() == q]
    text = ref["city"].fillna("").str.lower().str.contains(q, regex=False) | ref["name"].str.lower().str.contains(q, regex=False)
    hits = pd.concat([exact, ref[text]]).drop_duplicates("airport").head(limit)
    return clean(hits[["airport", "name", "city", "state"]].to_dict("records"))


def region_codes(region: str, ref: pd.DataFrame) -> list[str]:
    """Named region, 2-letter state, 'national', or a comma-separated list of IATA codes."""
    text = region.strip()
    if text.lower() in ("us", "usa", "national", "united states"):
        return sorted(ref["airport"])
    if text.lower() in airports_module.REGIONS:
        return airports_module.airports_in_region(text, ref)
    if len(text) == 2 and text.upper() in set(ref["state"]):
        return sorted(ref.loc[ref["state"] == text.upper(), "airport"])
    codes = [c.strip().upper() for c in text.split(",")]
    if all(c in set(ref["airport"]) for c in codes):
        return codes
    raise KeyError(
        f"Unknown region '{region}'. Use one of {sorted(airports_module.REGIONS)}, a 2-letter state code, "
        "'national', or IATA codes separated by commas."
    )


def _assumptions(year: int) -> list[str]:
    return [
        f"Year analysed: {year}. Traffic is measured on departures (flights and passengers leaving the airport).",
        "Source: BTS T-100 segment data (US + foreign carriers, passenger service only); delays from BTS On-Time "
        f"Performance using only months {config.ONTIME_MONTHS}, so delay figures are seasonal estimates.",
        f"Only airports with at least {config.MIN_DEPARTURES:,} passenger departures a year are ranked; percentiles are relative to that national field.",
    ]


def rank_region(metrics: pd.DataFrame, ref: pd.DataFrame, region: str, year: int = LATEST_YEAR, top_n: int = 5) -> dict:
    codes = region_codes(region, ref)
    scores = expansion_scores(metrics, year)
    inside = scores[scores.index.isin(codes)]
    columns = ["rank", "score", "pct_load_factor", "pct_momentum", "pct_seat_growth", "name", "state", "passengers",
               "load_factor", "passengers_vs_2019", "passengers_yoy", "seats_yoy", "long_haul_share", "intl_share", "flags"]
    candidates = inside[~inside["gated"]].head(top_n)
    gated = inside[inside["gated"]]
    not_ranked = sorted(set(codes) - set(scores.index))
    return clean({
        "region": region, "year": year, "airports_in_region": len(codes),
        "ranked_in_region": int((~inside["gated"]).sum()),
        "top_candidates": [{"airport": a, **row[columns].to_dict()} for a, row in candidates.iterrows()],
        "gated_out": [{"airport": a, "score": row["score"], "load_factor": row["load_factor"], "flags": row["flags"]} for a, row in gated.iterrows()],
        "too_small_to_rank": not_ranked,
        "national_field": {"scored": int(scores.iloc[0]["eligible_airports"]), "ranked": int(scores.iloc[0]["ranked_airports"])},
        "method": {
            "score": "0-100: " + ", ".join(f"{k} {v}%" for k, v in config.EXPANSION_WEIGHTS.items())
                     + " (each a percentile among national peers; momentum = passengers vs 2019 and YoY)",
            "load_factor_gate": config.MIN_LOAD_FACTOR,
        },
        "assumptions": _assumptions(year) + [
            "Score is a screening signal for demand outgrowing capacity. It does not include gate counts, terminal condition, "
            "land, financing or returns, none of which are in free public data.",
        ],
    })


def compare(metrics: pd.DataFrame, codes: list[str], year: int = LATEST_YEAR) -> dict:
    codes = [c.upper() for c in codes]
    congestion = congestion_index(metrics, year)
    pool = metrics[metrics["year"] == year].set_index("airport")
    fields = ["name", "state", "departures", "passengers", "seats", "load_factor", "avg_stage_miles", "passengers_vs_2019",
              "passengers_yoy", "delay_rate", "cancel_rate", "avg_dep_delay_min", "avg_taxi_out_min", "nas_delay_share",
              "weather_delay_share", "months_sampled", "hub_class"]
    rows, notes = [], []
    for code in codes:
        if code not in pool.index:
            notes.append(f"{code}: no passenger service data for {year}.")
            continue
        row = {"airport": code, **pool.loc[code, [f for f in fields if f in pool.columns]].to_dict()}
        if code in congestion.index:
            row["congestion_index"] = congestion.loc[code, "congestion_index"]
            row["congestion_rank"] = congestion.loc[code, "rank"]
        else:
            notes.append(f"{code}: not in the congestion ranking (under {config.MIN_DEPARTURES:,} departures or no delay data).")
        rows.append(row)
    return clean({
        "year": year, "airports": rows, "congestion_airports_compared": len(congestion),
        "congestion_definition": "0-100, 100 = most congested: mean percentile of delay_rate, avg_taxi_out_min, nas_delay_share among national peers.",
        "notes": notes, "assumptions": _assumptions(year),
    })


def long_haul(t100_year: pd.DataFrame, airport: str, year: int, threshold_miles: float = config.LONG_HAUL_MILES) -> dict:
    airport = airport.upper()
    row = metrics_module.t100_airport_year(t100_year, threshold_miles)
    row = row[row["airport"] == airport]
    if row.empty:
        raise KeyError(f"No passenger departures for {airport} in {year}")
    r = row.iloc[0]
    sensitivity = {}
    for miles in (2000, 2500, 3000, 3500):
        alt = metrics_module.t100_airport_year(t100_year, miles)
        alt = alt[alt["airport"] == airport].iloc[0]
        sensitivity[f">= {miles} miles"] = {"share_of_flights": alt["long_haul_share"], "long_haul_departures": alt["long_haul_departures"]}
    # Pre-written so the LLM repeats the definition-dependence instead of computing anything itself.
    summary = (
        f"{r['long_haul_share']:.1%} of {r['departures']:,.0f} departing flights are long haul at the {threshold_miles:,.0f}-mile threshold "
        f"({r['long_haul_departures']:,.0f} flights). The answer depends on the definition: "
        + "; ".join(f"{label}: {v['share_of_flights']:.1%} ({v['long_haul_departures']:,.0f} flights)" for label, v in sensitivity.items())
        + "."
    )
    notes = [
        f"'Long haul' is defined as a nonstop segment of {threshold_miles:,.0f} miles or more. This is an assumption: there is no official cut-off.",
        "Share is of departing flights (T-100 departures performed). Share of departing passengers is given separately.",
        "Segments are nonstop legs, so connecting itineraries are counted by leg, not by trip.",
    ]
    return clean({
        "airport": airport, "year": year, "threshold_miles": threshold_miles, "summary": summary,
        "departures": r["departures"], "long_haul_departures": r["long_haul_departures"],
        "long_haul_share_of_flights": r["long_haul_share"], "long_haul_share_of_passengers": r["long_haul_pax_share"],
        "avg_stage_miles": r["avg_stage_miles"], "sensitivity_by_threshold": sensitivity,
        "notes": notes, "assumptions": _assumptions(year)[:2],
    })


def unmet_demand(metrics: pd.DataFrame, airport: str, year: int = LATEST_YEAR) -> dict:
    result = unmet_demand_proxy(metrics, airport.upper(), year)
    # Named *_percentile so the LLM cannot mistake a rank (0-100) for a real percentage.
    result["components_percentile"] = {f"{k}_percentile": v for k, v in result.pop("components").items()}
    result["raw"] = {"load_factor": result["raw"]["load_factor"], "passengers_vs_2019_ratio": result["raw"]["passengers_vs_2019"]}
    row =metrics[(metrics["airport"] == airport.upper()) & (metrics["year"] == year)].iloc[0]
    result["context"] = {k: row.get(k) for k in ["passengers", "load_factor", "passengers_vs_2019", "passengers_yoy",
                                                 "seats_yoy", "delay_rate", "avg_taxi_out_min", "nas_delay_share"]}
    result["definition"] = (
        "proxy = mean of three percentile ranks (0-100, 100 = highest among national peers): load factor, passenger growth "
        "vs 2019, and congestion index. Percentiles are ranks, NOT real values; real values are in `raw` and `context`. "
        "Levels: <33 low, 33-67 medium, >67 high. passengers_vs_2019_ratio below 1.0 means traffic is still below 2019."
    )
    result["assumptions"] = _assumptions(year)
    return clean(result)


def coverage() -> dict:
    return clean({
        "years_available": config.YEARS, "latest_year": LATEST_YEAR, "baseline_year": config.BASELINE_YEAR,
        "delay_months_sampled": config.ONTIME_MONTHS, "long_haul_miles": config.LONG_HAUL_MILES,
        "min_departures_to_rank": config.MIN_DEPARTURES, "min_load_factor_gate": config.MIN_LOAD_FACTOR,
        "expansion_weights": config.EXPANSION_WEIGHTS,
        "regions": sorted(airports_module.REGIONS) + ["any US state code", "national", "comma-separated IATA codes"],
        "limitations": [
            "No gate, runway, terminal-size, financing or land data: capacity strain is inferred, never measured.",
            "Unmet demand is a proxy built from load factor, growth and congestion, not a measurement of turned-away passengers.",
            "Delay metrics use only Jan/Apr/Jul/Oct of each year.",
            "Years 2020 and 2021 are not loaded (COVID); 2019 is the pre-COVID baseline.",
        ],
    })
