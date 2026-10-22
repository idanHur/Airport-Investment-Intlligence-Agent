"""Download and cache all data. Run: .venv\\Scripts\\python -m airport_agent.data.build_cache"""
from airport_agent.config import ONTIME_MONTHS, YEARS
from airport_agent.data import bts_ontime, bts_t100


def build_t100():
    for year in YEARS:
        try:
            frame = bts_t100.load_year(year)
        except Exception as error:  # report the gap and keep going
            print(f"T-100 {year}: FAILED - {error}")
            continue
        months = sorted(frame["MONTH"].unique())
        print(f"T-100 {year}: {len(frame):,} rows, months {months[0]}-{months[-1]} ({len(months)} present)")


def build_ontime():
    for year in YEARS:
        for month in ONTIME_MONTHS:
            try:
                frame = bts_ontime.load_month(year, month)
            except Exception as error:
                print(f"On-Time {year}-{month:02d}: FAILED - {error}")
                continue
            print(f"On-Time {year}-{month:02d}: {len(frame)} airports")


def main():
    build_t100()
    build_ontime()


if __name__ == "__main__":
    main()
