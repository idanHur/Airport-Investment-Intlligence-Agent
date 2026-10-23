"""Turn the raw cached tables into one row per airport per year.

Everything here is plain pandas with no network and no LLM, so each number can be
traced back to the source column and checked by hand.

Convention: an airport's traffic is measured on the *departure* side (flights and
passengers leaving it). That matches how the FAA counts enplanements.
"""
import pandas as pd

from airport_agent.config import BASELINE_YEAR, LONG_HAUL_MILES, YEARS
from airport_agent.data import airports, bts_ontime, bts_t100, faa_enplanements

# T-100 CLASS codes: F scheduled passenger/cargo, L non-scheduled passenger/cargo.
# G and P are all-cargo, so they are excluded from passenger metrics.
PASSENGER_CLASSES = ["F", "L"]


def t100_airport_year(t100: pd.DataFrame, long_haul_miles: float = LONG_HAUL_MILES) -> pd.DataFrame:
    """T-100 segments -> traffic and capacity metrics per US origin airport per year.
    `long_haul_miles` defaults to the project threshold; the agent can vary it to show sensitivity."""
    flights = t100[
        t100["CLASS"].isin(PASSENGER_CLASSES)
        & (t100["ORIGIN_COUNTRY"] == "US")
        & (t100["DEPARTURES_PERFORMED"] > 0)
    ].copy()
    flights["miles_flown"] = flights["DISTANCE"] * flights["DEPARTURES_PERFORMED"]
    flights["long_haul_departures"] = flights["DEPARTURES_PERFORMED"].where(flights["DISTANCE"] >= long_haul_miles, 0)
    flights["intl_departures"] = flights["DEPARTURES_PERFORMED"].where(flights["DEST_COUNTRY"] != "US", 0)
    flights["long_haul_passengers"] = flights["PASSENGERS"].where(flights["DISTANCE"] >= long_haul_miles, 0)

    totals = flights.groupby(["ORIGIN", "YEAR"]).agg(
        departures=("DEPARTURES_PERFORMED", "sum"),
        seats=("SEATS", "sum"),
        passengers=("PASSENGERS", "sum"),
        miles_flown=("miles_flown", "sum"),
        long_haul_departures=("long_haul_departures", "sum"),
        long_haul_passengers=("long_haul_passengers", "sum"),
        intl_departures=("intl_departures", "sum"),
        destinations=("DEST", "nunique"),
    ).reset_index().rename(columns={"ORIGIN": "airport", "YEAR": "year"})

    totals["load_factor"] = totals["passengers"] / totals["seats"]
    totals["avg_stage_miles"] = totals["miles_flown"] / totals["departures"]
    totals["long_haul_share"] = totals["long_haul_departures"] / totals["departures"]
    totals["long_haul_pax_share"] = totals["long_haul_passengers"] / totals["passengers"]
    totals["intl_share"] = totals["intl_departures"] / totals["departures"]
    return totals.drop(columns=["miles_flown", "long_haul_passengers"])


def add_growth(metrics: pd.DataFrame, baseline_year: int = BASELINE_YEAR) -> pd.DataFrame:
    """Add year-over-year growth (only when the previous calendar year exists) and
    the ratio to the pre-COVID baseline year."""
    metrics = metrics.sort_values(["airport", "year"]).copy()
    for column in ["passengers", "seats", "departures"]:
        previous = metrics[["airport", "year", column]].copy()
        previous["year"] += 1
        merged = metrics[["airport", "year"]].merge(previous, on=["airport", "year"], how="left")
        metrics[f"{column}_yoy"] = (metrics[column].to_numpy() / merged[column].to_numpy()) - 1
        base = metrics.loc[metrics["year"] == baseline_year, ["airport", column]].rename(columns={column: "base"})
        metrics = metrics.merge(base, on="airport", how="left")
        metrics[f"{column}_vs_{baseline_year}"] = metrics[column] / metrics["base"]
        metrics = metrics.drop(columns="base")
    return metrics


def ontime_airport_year(ontime: pd.DataFrame) -> pd.DataFrame:
    """Monthly On-Time aggregates -> delay metrics per origin airport per year.
    Only the sampled months are present, so these are seasonal estimates."""
    yearly = ontime.groupby(["airport", "year"]).agg(
        scheduled_departures=("scheduled_departures", "sum"),
        cancelled=("cancelled", "sum"),
        delayed_15=("delayed_15", "sum"),
        dep_delay_minutes=("dep_delay_minutes", "sum"),
        flown=("flown", "sum"),
        taxi_out_minutes=("taxi_out_minutes", "sum"),
        taxi_out_flights=("taxi_out_flights", "sum"),
        weather_delay=("weather_delay", "sum"),
        nas_delay=("nas_delay", "sum"),
        carrier_delay=("carrier_delay", "sum"),
        late_aircraft_delay=("late_aircraft_delay", "sum"),
        security_delay=("security_delay", "sum"),
        months_sampled=("month", "nunique"),
    ).reset_index()
    cause_minutes = yearly[["weather_delay", "nas_delay", "carrier_delay", "late_aircraft_delay", "security_delay"]].sum(axis=1)
    yearly["delay_rate"] = yearly["delayed_15"] / yearly["flown"]
    yearly["cancel_rate"] = yearly["cancelled"] / yearly["scheduled_departures"]
    yearly["avg_dep_delay_min"] = yearly["dep_delay_minutes"] / yearly["flown"]
    yearly["avg_taxi_out_min"] = yearly["taxi_out_minutes"] / yearly["taxi_out_flights"]
    yearly["nas_delay_share"] = yearly["nas_delay"] / cause_minutes  # congestion-type delay
    yearly["weather_delay_share"] = yearly["weather_delay"] / cause_minutes
    return yearly[[
        "airport", "year", "months_sampled", "delay_rate", "cancel_rate",
        "avg_dep_delay_min", "avg_taxi_out_min", "nas_delay_share", "weather_delay_share",
    ]]


def build_metrics() -> pd.DataFrame:
    """One row per airport-year with traffic, growth, delay and reference columns."""
    t100 = pd.concat([bts_t100.load_year(y) for y in YEARS], ignore_index=True)
    traffic = add_growth(t100_airport_year(t100))

    months = sorted({m for m in range(1, 13) if any(bts_ontime.cache_path(y, m).exists() for y in YEARS)})
    ontime = pd.concat(
        [bts_ontime.load_month(y, m) for y in YEARS for m in months if bts_ontime.cache_path(y, m).exists()],
        ignore_index=True,
    )
    delays = ontime_airport_year(ontime)

    faa = pd.concat([faa_enplanements.load_year(y) for y in YEARS], ignore_index=True)[["airport", "year", "hub_class", "enplanements"]]
    reference = airports.load()[["airport", "name", "city", "state"]]

    result = (
        traffic
        .merge(delays, on=["airport", "year"], how="left")
        .merge(faa, on=["airport", "year"], how="left")
        .merge(reference, on="airport", how="left")
    )
    return result
