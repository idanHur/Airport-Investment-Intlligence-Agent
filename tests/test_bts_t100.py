import pandas as pd
import pytest

from airport_agent.data import bts_t100


def test_hidden_fields_are_extracted():
    html = (
        '<input type="hidden" name="__VIEWSTATE" id="__VIEWSTATE" value="abc+/=" />'
        '<input type="hidden" name="__VIEWSTATEGENERATOR" id="__VIEWSTATEGENERATOR" value="82CE8F70" />'
        '<input type="hidden" name="__EVENTVALIDATION" id="__EVENTVALIDATION" value="xyz" />'
    )
    assert bts_t100._hidden_fields(html) == {
        "__VIEWSTATE": "abc+/=",
        "__VIEWSTATEGENERATOR": "82CE8F70",
        "__EVENTVALIDATION": "xyz",
    }


def test_missing_hidden_field_fails_loudly():
    with pytest.raises(RuntimeError):
        bts_t100._hidden_fields("<html></html>")


def test_load_year_uses_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(bts_t100, "CACHE_DIR", tmp_path)
    cached = pd.DataFrame({"ORIGIN": ["BOS"], "SEATS": [100.0]})
    cached.to_parquet(tmp_path / "t100_2024.parquet")
    monkeypatch.setattr(bts_t100, "_download_year", lambda year: pytest.fail("should not download"))
    assert bts_t100.load_year(2024).equals(cached)
