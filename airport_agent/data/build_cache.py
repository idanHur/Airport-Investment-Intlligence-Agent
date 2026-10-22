"""Download and cache all data. Run: .venv\\Scripts\\python -m airport_agent.data.build_cache"""
from airport_agent.config import YEARS
from airport_agent.data import bts_t100


def main():
    for year in YEARS:
        try:
            frame = bts_t100.load_year(year)
        except Exception as error:  # report the gap and keep going
            print(f"T-100 {year}: FAILED - {error}")
            continue
        months = sorted(frame["MONTH"].unique())
        print(f"T-100 {year}: {len(frame):,} rows, months {months[0]}-{months[-1]} ({len(months)} present)")


if __name__ == "__main__":
    main()
