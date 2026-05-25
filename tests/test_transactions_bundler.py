"""Tests for the transactions bundler."""
from __future__ import annotations

from datetime import date

import pytest

from firm_rp_garp.transactions.bundler import (
    _is_contiguous,
    assemble_firm_panel,
    bundle_to_monthly,
    map_mcc_to_category,
)
from firm_rp_garp.transactions.prices import MacroSeries
from firm_rp_garp.transactions.schema import (
    Category,
    MonthlyBundle,
    N_CATEGORIES,
    Transaction,
)


def _make_macro(months: list[tuple[int, int]]) -> MacroSeries:
    return MacroSeries(
        nibor={m: 0.03 for m in months},
        sector_ppi={m: 100.0 for m in months},
        capital_goods_ppi={m: 100.0 for m in months},
        dividend_tax={m: 0.35 for m in months},
        risk_free={m: 0.02 for m in months},
    )


class TestBundleToMonthly:
    """Tests for :func:`bundle_to_monthly`."""

    def test_single_transaction(self) -> None:
        """One transaction yields one bundle."""
        txns = [
            Transaction(
                orgnr="123",
                txn_date=date(2020, 6, 10),
                amount_nok=500.0,
                category=Category.OPERATING,
                counterparty="x",
                mcc="OPS",
            )
        ]
        bundles = bundle_to_monthly(txns)
        assert "123" in bundles
        assert len(bundles["123"]) == 1
        b = bundles["123"][0]
        assert b.amounts_nok[0] == 500.0
        assert sum(b.amounts_nok[1:]) == 0.0

    def test_multiple_firms_separated(self) -> None:
        """Transactions from multiple firms produce separate bundle lists."""
        txns = [
            Transaction("A", date(2020, 6, 1), 100.0, Category.OPERATING, "x", "OPS"),
            Transaction("B", date(2020, 6, 1), 200.0, Category.OPERATING, "x", "OPS"),
        ]
        bundles = bundle_to_monthly(txns)
        assert set(bundles) == {"A", "B"}
        assert bundles["A"][0].amounts_nok[0] == 100.0
        assert bundles["B"][0].amounts_nok[0] == 200.0

    def test_categories_summed_independently(self) -> None:
        """Transactions across categories sum into separate vector slots."""
        txns = [
            Transaction("A", date(2020, 6, 1), 100.0, Category.OPERATING, "x", "OPS"),
            Transaction("A", date(2020, 6, 1), 50.0, Category.DEBT_SERVICE, "y", "DEB"),
        ]
        bundles = bundle_to_monthly(txns)
        b = bundles["A"][0]
        from firm_rp_garp.transactions.schema import CATEGORY_INDEX
        assert b.amounts_nok[CATEGORY_INDEX[Category.OPERATING]] == 100.0
        assert b.amounts_nok[CATEGORY_INDEX[Category.DEBT_SERVICE]] == 50.0

    def test_bundles_sorted_ascending(self) -> None:
        """Per-firm bundle lists are sorted by (year, month)."""
        txns = [
            Transaction("A", date(2020, 8, 1), 100.0, Category.OPERATING, "x", "OPS"),
            Transaction("A", date(2020, 6, 1), 50.0, Category.OPERATING, "x", "OPS"),
            Transaction("A", date(2020, 7, 1), 25.0, Category.OPERATING, "x", "OPS"),
        ]
        bundles = bundle_to_monthly(txns)
        months = [(b.year, b.month) for b in bundles["A"]]
        assert months == sorted(months)


class TestAssembleFirmPanel:
    """Tests for :func:`assemble_firm_panel`."""

    def test_aligned_panel(self) -> None:
        """A two-month bundle list produces a length-2 aligned panel."""
        bundles = [
            MonthlyBundle(
                "A",
                2020,
                6,
                tuple([100.0] * N_CATEGORIES),
                n_transactions=1,
            ),
            MonthlyBundle(
                "A",
                2020,
                7,
                tuple([200.0] * N_CATEGORIES),
                n_transactions=1,
            ),
        ]
        macro = _make_macro([(2020, 6), (2020, 7)])
        panel = assemble_firm_panel(bundles, macro)
        assert len(panel) == 2

    def test_empty_raises(self) -> None:
        """Empty bundle list raises ValueError."""
        macro = _make_macro([(2020, 6)])
        with pytest.raises(ValueError):
            assemble_firm_panel([], macro)

    def test_non_contiguous_raises(self) -> None:
        """Non-consecutive months raise ValueError."""
        bundles = [
            MonthlyBundle(
                "A",
                2020,
                6,
                tuple([100.0] * N_CATEGORIES),
                n_transactions=1,
            ),
            MonthlyBundle(
                "A",
                2020,
                9,
                tuple([200.0] * N_CATEGORIES),
                n_transactions=1,
            ),
        ]
        macro = _make_macro([(2020, 6), (2020, 9)])
        with pytest.raises(ValueError):
            assemble_firm_panel(bundles, macro)

    def test_multi_firm_raises(self) -> None:
        """Bundles from multiple firms raise ValueError."""
        bundles = [
            MonthlyBundle(
                "A",
                2020,
                6,
                tuple([100.0] * N_CATEGORIES),
                n_transactions=1,
            ),
            MonthlyBundle(
                "B",
                2020,
                7,
                tuple([200.0] * N_CATEGORIES),
                n_transactions=1,
            ),
        ]
        macro = _make_macro([(2020, 6), (2020, 7)])
        with pytest.raises(ValueError):
            assemble_firm_panel(bundles, macro)


class TestIsContiguous:
    """Tests for the contiguity helper."""

    def test_contiguous_within_year(self) -> None:
        """Consecutive months within a year are contiguous."""
        bs = [
            MonthlyBundle(
                "A", 2020, m, tuple([1.0] * N_CATEGORIES), n_transactions=1
            )
            for m in (1, 2, 3)
        ]
        assert _is_contiguous(bs)

    def test_contiguous_across_year(self) -> None:
        """December to January is contiguous."""
        bs = [
            MonthlyBundle("A", 2020, 12, tuple([1.0] * N_CATEGORIES), 1),
            MonthlyBundle("A", 2021, 1, tuple([1.0] * N_CATEGORIES), 1),
        ]
        assert _is_contiguous(bs)

    def test_gap_not_contiguous(self) -> None:
        """A skipped month breaks contiguity."""
        bs = [
            MonthlyBundle("A", 2020, 1, tuple([1.0] * N_CATEGORIES), 1),
            MonthlyBundle("A", 2020, 3, tuple([1.0] * N_CATEGORIES), 1),
        ]
        assert not _is_contiguous(bs)


class TestMccMap:
    """Tests for the MCC→Category mapping."""

    def test_known_codes(self) -> None:
        """All controlled vocabulary codes map back to their categories."""
        from firm_rp_garp.transactions.schema import Category

        assert map_mcc_to_category("OPS") == Category.OPERATING
        assert map_mcc_to_category("DEB") == Category.DEBT_SERVICE
        assert map_mcc_to_category("INV") == Category.INVESTMENT

    def test_unknown_code_is_none(self) -> None:
        """Unknown codes return None rather than raising."""
        assert map_mcc_to_category("9999") is None
