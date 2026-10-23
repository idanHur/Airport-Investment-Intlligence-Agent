"""Central settings. Anything that is a judgement call lives here, with a name."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# 2019 is the pre-COVID baseline; 2022 onward is the recovered period.
BASELINE_YEAR = 2019
YEARS = [2019, 2022, 2023, 2024, 2025]

# On-Time files are ~30 MB/month and take ~1 min each, so we sample one month per
# season. Delay rates are therefore estimates, not full-year figures (see README.md).
ONTIME_MONTHS = [1, 4, 7, 10]

# A segment this long or longer counts as "long haul" (documented in README.md).
LONG_HAUL_MILES = 3000

# Airports below this many passenger departures a year are not ranked: tiny airports
# have noisy ratios and would dominate the extremes of every percentile.
MIN_DEPARTURES = 5000

# Expansion-candidate score weights (sum to 100). Each component is a 0-100 percentile.
EXPANSION_WEIGHTS = {"load_factor": 40, "momentum": 40, "seat_growth": 20}

# Load-factor gate: below this, planes are mostly empty, so strong growth cannot make
# an airport an expansion candidate (it is scored but not ranked). See README.md.
MIN_LOAD_FACTOR = 0.60

# Warning-flag thresholds (flags explain results; they never change a score).
SMALL_AIRPORT_PASSENGERS = 250_000   # tiny base: ratios are noisy
EXTREME_GROWTH_VS_BASELINE = 3.0     # >3x 2019 traffic usually means new service, not trend
EXTREME_SEAT_GROWTH = 0.5            # +50% seats in a year is a step change, not a trend

OPENAI_MODEL = "gpt-4o-mini"
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"  # speech-to-text for voice input
