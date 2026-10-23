import io

import pandas as pd
import pytest

from airport_agent.data import airports, faa_enplanements


def _airports():
    return pd.DataFrame({
        "airport": ["BOS", "PVD", "JFK", "LAX", "SNA"],
        "state": ["MA", "RI", "NY", "CA", "CA"],
    })


def test_new_england_is_selected_by_state():
    assert airports.airports_in_region("New England", _airports()) == ["BOS", "PVD"]


def test_la_area_uses_airport_list_and_includes_santa_ana():
    assert airports.airports_in_region("los angeles area", _airports()) == ["LAX", "SNA"]


def test_unknown_region_raises():
    with pytest.raises(KeyError):
        airports.airports_in_region("mars", _airports())


def test_faa_parse_picks_the_right_year_column_and_drops_blanks():
    sheet = pd.DataFrame({
        "Locid": ["BOS", "XXX", None], "ST": ["MA", "ZZ", None], "City": ["Boston", "Nowhere", None],
        "Airport Name": ["Logan", "Tiny", None], "Hub": ["L", "N", None],
        "CY 24 Enplanements": [21090721, " ", None], "CY 23 Enplanements": [19962678, 5, None],
    })
    buffer = io.BytesIO()
    sheet.to_excel(buffer, index=False)
    result = faa_enplanements.parse(buffer.getvalue(), 2024)
    assert result["airport"].tolist() == ["BOS"]
    assert result["enplanements"].tolist() == [21090721]
