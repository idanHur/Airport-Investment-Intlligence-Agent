"""Central settings. Anything that is a judgement call lives here, with a name."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# 2019 is the pre-COVID baseline; 2022 onward is the recovered period.
BASELINE_YEAR = 2019
YEARS = [2019, 2022, 2023, 2024, 2025]

# A segment this long or longer counts as "long haul" (documented in docs/DESIGN.md).
LONG_HAUL_MILES = 3000

OPENAI_MODEL = "gpt-4o-mini"
