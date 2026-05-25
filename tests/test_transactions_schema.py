"""Tests for transaction-schema dataclasses."""
from __future__ import annotations

from datetime import date

import pytest

from firm_rp_garp.transactions.schema import (
    CATEGORIES,
    CATEGORY_INDEX,
    Category,
    FirmTransactionPanel,
    MonthlyBundle,
    N_CATEGORIES,
    PriceContext,
    Transaction,
)


class TestCategories:
    """Tests for the Category enum and its indexing."""

    def test_six_categories(self) -> None:
        """The canonical category list has six entries."""
        assert N_CATEGORIES == 6
        assert len(CATEGORIES) == 6

    def test_category_index_is_consistent(self) -> None:
        """Each category maps to its position in CATEGORIES."""
        for idx, cat in enumerate(CATEGORIES):
            assert CATEGORY_INDEX[cat] == idx

    def test_category_values_are_strings(self) -> None:
        """Each enum value is a stable string."""
        for cat in CATEGORIES:
            assert isinstance(cat.value, str)


class TestTransaction:
    """Tests for the Transaction dataclass."""

    def test_construct_minimal(self) -> None:
        """A minimal transaction can be constructed."""
        t = Transaction(
            orgnr="123456789",
            txn_date=date(2020, 6, 15),
            amount_nok=1000.0,
            category=Category.OPERATING,
            counterparty="supplier_1",
            mcc="OPS",
        )
        assert t.orgnr == "123456789"
        assert t.amount_nok == 1000.0


class TestMonthlyBundle:
    """Tests for the MonthlyBundle dataclass and its invariants."""

    def test_correct_length(self) -> None:
        """A bundle with N_CATEGORIES amounts constructs cleanly."""
        b = MonthlyBundle(
            orgnr="123",
            year=2020,
            month=6,
            amounts_nok=tuple([100.0] * N_CATEGORIES),
            n_transactions=5,
        )
        assert len(b.amounts_nok) == N_CATEGORIES

    def test_wrong_length_raises(self) -> None:
        """Length mismatch raises ValueError."""
        with pytest.raises(ValueError):
            MonthlyBundle(
                orgnr="123",
                year=2020,
                month=6,
                amounts_nok=(100.0, 200.0),
                n_transactions=2,
            )

    def test_negative_amount_raises(self) -> None:
        """Negative amount in the vector raises ValueError."""
        with pytest.raises(ValueError):
            MonthlyBundle(
                orgnr="123",
                year=2020,
                month=6,
                amounts_nok=tuple([-1.0] + [100.0] * (N_CATEGORIES - 1)),
                n_transactions=1,
            )


class TestPriceContext:
    """Tests for the PriceContext dataclass."""

    def test_construct_positive(self) -> None:
        """A positive price vector constructs cleanly."""
        p = PriceContext(year=2020, month=6, prices=tuple([1.0] * N_CATEGORIES))
        assert all(price > 0 for price in p.prices)

    def test_wrong_length_raises(self) -> None:
        """Wrong vector length raises ValueError."""
        with pytest.raises(ValueError):
            PriceContext(year=2020, month=6, prices=(1.0, 2.0))

    def test_zero_price_raises(self) -> None:
        """Zero price raises ValueError."""
        with pytest.raises(ValueError):
            PriceContext(
                year=2020,
                month=6,
                prices=tuple([0.0] + [1.0] * (N_CATEGORIES - 1)),
            )


class TestFirmTransactionPanel:
    """Tests for the FirmTransactionPanel alignment invariant."""

    def _make_bundle(self, year: int, month: int, orgnr: str = "123") -> MonthlyBundle:
        return MonthlyBundle(
            orgnr=orgnr,
            year=year,
            month=month,
            amounts_nok=tuple([100.0] * N_CATEGORIES),
            n_transactions=1,
        )

    def _make_price(self, year: int, month: int) -> PriceContext:
        return PriceContext(year=year, month=month, prices=tuple([1.0] * N_CATEGORIES))

    def test_aligned_panel(self) -> None:
        """A panel with aligned bundles and prices constructs cleanly."""
        bundles = (
            self._make_bundle(2020, 6),
            self._make_bundle(2020, 7),
        )
        prices = (
            self._make_price(2020, 6),
            self._make_price(2020, 7),
        )
        panel = FirmTransactionPanel(orgnr="123", bundles=bundles, prices=prices)
        assert len(panel) == 2

    def test_length_mismatch_raises(self) -> None:
        """Mismatched bundle/price counts raise ValueError."""
        bundles = (self._make_bundle(2020, 6),)
        prices = (self._make_price(2020, 6), self._make_price(2020, 7))
        with pytest.raises(ValueError):
            FirmTransactionPanel(orgnr="123", bundles=bundles, prices=prices)

    def test_month_misalignment_raises(self) -> None:
        """Misaligned (year, month) pairs raise ValueError."""
        bundles = (self._make_bundle(2020, 6),)
        prices = (self._make_price(2020, 7),)
        with pytest.raises(ValueError):
            FirmTransactionPanel(orgnr="123", bundles=bundles, prices=prices)
