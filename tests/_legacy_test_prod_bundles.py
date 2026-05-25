"""Tests for production-side bundle construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from firm_rp_garp.prod.bundles import INPUT_COLUMNS, build_firm_bundles


def _make_finstat(orgnr: str, years: list[int], values: list[list[float]]) -> pd.DataFrame:
    """Construct a minimal finstat-shaped DataFrame for testing."""
    rows = []
    for year, vals in zip(years, values):
        row = {"organisasjonsnummer": orgnr, "Regnskapsar": year}
        for col, val in zip(INPUT_COLUMNS, vals):
            row[col] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _make_prices(years: list[int]) -> pd.DataFrame:
    """Construct a price-index DataFrame with constant unit prices."""
    return pd.DataFrame(
        {col: [1.0] * len(years) for col in INPUT_COLUMNS},
        index=years,
    )


class TestBuildFirmBundles:
    """Tests for :func:`build_firm_bundles`."""

    def test_minimum_two_years_required(self) -> None:
        """A firm with only one observed year is excluded."""
        finstat = _make_finstat("123456789", [2020], [[100.0, 50.0, 25.0, 10.0]])
        prices = _make_prices([2020])
        result = build_firm_bundles(finstat, prices)
        assert result == {}

    def test_two_year_bundle_constructed(self) -> None:
        """A two-year firm produces a FirmBundle with shape (2, 4)."""
        finstat = _make_finstat(
            "123456789",
            [2020, 2021],
            [[100.0, 50.0, 25.0, 10.0], [110.0, 55.0, 27.0, 11.0]],
        )
        prices = _make_prices([2020, 2021])
        result = build_firm_bundles(finstat, prices)
        assert "123456789" in result
        bundle = result["123456789"]
        assert bundle.bundles.shape == (2, 4)
        assert bundle.prices.shape == (2, 4)
        assert bundle.years == [2020, 2021]

    def test_missing_expenditure_drops_year(self) -> None:
        """A firm-year with a NaN expenditure component is excluded."""
        finstat = _make_finstat(
            "123456789",
            [2020, 2021, 2022],
            [
                [100.0, 50.0, 25.0, 10.0],
                [110.0, float("nan"), 27.0, 11.0],
                [120.0, 60.0, 30.0, 12.0],
            ],
        )
        prices = _make_prices([2020, 2021, 2022])
        result = build_firm_bundles(finstat, prices)
        assert result["123456789"].years == [2020, 2022]

    def test_missing_price_year_excludes_firm(self) -> None:
        """A firm whose years are not in price_index_df is excluded."""
        finstat = _make_finstat(
            "123456789",
            [2020, 2021],
            [[100.0, 50.0, 25.0, 10.0], [110.0, 55.0, 27.0, 11.0]],
        )
        prices = _make_prices([2020])
        result = build_firm_bundles(finstat, prices)
        assert result == {}

    def test_negative_expenditure_excluded(self) -> None:
        """A firm-year with a negative expenditure is excluded."""
        finstat = _make_finstat(
            "123456789",
            [2020, 2021],
            [[100.0, 50.0, 25.0, 10.0], [110.0, -1.0, 27.0, 11.0]],
        )
        prices = _make_prices([2020, 2021])
        result = build_firm_bundles(finstat, prices)
        # Only one valid year remains, so the firm is dropped entirely
        assert result == {}

    def test_missing_required_column_raises(self) -> None:
        """A finstat slice missing a required column raises ValueError."""
        bad = pd.DataFrame({"organisasjonsnummer": ["123"], "Regnskapsar": [2020]})
        prices = _make_prices([2020])
        with pytest.raises(ValueError):
            build_firm_bundles(bad, prices)
