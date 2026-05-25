"""Schema for transaction-level RP analysis.

The six categories of cash use are the granularity at which we test
revealed preference. Each transaction is classified into exactly one
category. Each firm-month gets one bundle vector summing transactions
within each category and one price vector of opportunity costs.

Categories
----------
- ``operating``        — supplier and payroll payments
- ``debt_service``     — interest payments and principal repayments
- ``investment``       — capex outlays
- ``dividend``         — owner distributions and dividends
- ``retained``         — increase in cash and money-market positions
- ``working_capital``  — inventory and receivables changes

These six are deliberately chosen to be jointly exhaustive of cash use
for an operating company. They can be refined later (e.g. splitting
``operating`` by NACE-specific input groups) without changing the
test machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Final


class Category(str, Enum):
    """Cash-use category for a transaction.

    Membership is jointly exhaustive: every transaction belongs to
    exactly one category. The enum values are stable strings safe for
    serialization to parquet, CSV, and JSON.
    """

    OPERATING = "operating"
    DEBT_SERVICE = "debt_service"
    INVESTMENT = "investment"
    DIVIDEND = "dividend"
    RETAINED = "retained"
    WORKING_CAPITAL = "working_capital"


CATEGORIES: Final[tuple[Category, ...]] = tuple(Category)
"""Canonical ordering of categories, used for indexing bundle vectors."""

CATEGORY_INDEX: Final[dict[Category, int]] = {
    cat: idx for idx, cat in enumerate(CATEGORIES)
}
"""Map from :class:`Category` to its index in a bundle vector."""

N_CATEGORIES: Final[int] = len(CATEGORIES)
"""Number of categories in a bundle vector."""


@dataclass(frozen=True, slots=True)
class Transaction:
    """A single dated cash-use transaction for one firm.

    Parameters
    ----------
    orgnr : str
        Norwegian organisation number (9 digits, zero-padded as string).
    txn_date : datetime.date
        Date of the transaction (settlement date).
    amount_nok : float
        Amount in NOK. Positive = outflow from firm's accounts.
    category : Category
        Which of the six cash-use categories this transaction falls into.
    counterparty : str
        Anonymized counterparty identifier. For synthetic data this is
        a generated tag like ``"supplier_3"``; for real data this is the
        masked counterparty from the bank feed.
    mcc : str
        Merchant Category Code. For bank-transaction data this is the
        4-digit ISO 18245 code; for synthetic data we use a small
        controlled vocabulary.

    Notes
    -----
    This dataclass is frozen and slotted for memory efficiency when
    generating millions of synthetic transactions. The ``orgnr`` is
    stored as a string to preserve leading zeros.
    """

    orgnr: str
    txn_date: date
    amount_nok: float
    category: Category
    counterparty: str
    mcc: str


@dataclass(frozen=True, slots=True)
class MonthlyBundle:
    """A firm's monthly cash-allocation bundle.

    Parameters
    ----------
    orgnr : str
        Norwegian organisation number.
    year : int
        Calendar year.
    month : int
        Calendar month, 1-12.
    amounts_nok : tuple[float, ...]
        NOK amount allocated to each category. Indexed by
        :data:`CATEGORIES`. Length equals :data:`N_CATEGORIES`.
    n_transactions : int
        Number of underlying granular transactions that aggregated
        into this bundle. Kept for diagnostics; not used in the RP
        test.

    Notes
    -----
    A bundle's components are always non-negative. If a real-data
    feed contains net flows that would produce a negative allocation
    in some category (e.g. a refund of a supplier payment), the
    bundler must net them within the month before emitting a bundle.
    """

    orgnr: str
    year: int
    month: int
    amounts_nok: tuple[float, ...]
    n_transactions: int

    def __post_init__(self) -> None:
        """Validate bundle shape and non-negativity invariants.

        Raises
        ------
        ValueError
            If the amount vector length is wrong or contains
            negative values.
        """
        if len(self.amounts_nok) != N_CATEGORIES:
            raise ValueError(
                f"amounts_nok must have length {N_CATEGORIES}, "
                f"got {len(self.amounts_nok)}"
            )
        if any(amount < 0 for amount in self.amounts_nok):
            raise ValueError(
                f"amounts_nok must be non-negative, got {self.amounts_nok}"
            )


@dataclass(frozen=True, slots=True)
class PriceContext:
    """Opportunity-cost prices for the six categories in a given month.

    Parameters
    ----------
    year : int
        Calendar year.
    month : int
        Calendar month, 1-12.
    prices : tuple[float, ...]
        Opportunity-cost prices for each category. Indexed by
        :data:`CATEGORIES`. Length equals :data:`N_CATEGORIES`. All
        components must be strictly positive.

    Notes
    -----
    Price components are interpreted as relative opportunity costs:
    they are the marginal cost of a unit of cash deployed to each
    category in this month. Only relative levels matter for GARP
    (the test is invariant to common scaling).
    """

    year: int
    month: int
    prices: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate price-vector shape and positivity invariants.

        Raises
        ------
        ValueError
            If the price-vector length is wrong or contains
            non-positive values.
        """
        if len(self.prices) != N_CATEGORIES:
            raise ValueError(
                f"prices must have length {N_CATEGORIES}, "
                f"got {len(self.prices)}"
            )
        if any(price <= 0 for price in self.prices):
            raise ValueError(
                f"prices must be strictly positive, got {self.prices}"
            )


@dataclass(frozen=True, slots=True)
class FirmTransactionPanel:
    """A firm's full transaction record plus aligned price context.

    Parameters
    ----------
    orgnr : str
        Norwegian organisation number.
    bundles : tuple[MonthlyBundle, ...]
        Monthly bundles in ascending (year, month) order, contiguous.
    prices : tuple[PriceContext, ...]
        Price contexts aligned 1:1 with ``bundles``. Same length and
        same (year, month) sequence.

    Notes
    -----
    The pipeline assumes bundles and prices are pre-aligned. The
    :func:`firm_rp_garp.transactions.bundler.assemble_firm_panel`
    helper enforces this alignment.
    """

    orgnr: str
    bundles: tuple[MonthlyBundle, ...] = field(default_factory=tuple)
    prices: tuple[PriceContext, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate that bundles and prices are aligned.

        Raises
        ------
        ValueError
            If lengths differ or any (year, month) pair is misaligned.
        """
        if len(self.bundles) != len(self.prices):
            raise ValueError(
                f"bundles ({len(self.bundles)}) and prices "
                f"({len(self.prices)}) must have equal length"
            )
        for bundle, price in zip(self.bundles, self.prices):
            if bundle.year != price.year or bundle.month != price.month:
                raise ValueError(
                    f"bundle ({bundle.year}, {bundle.month}) misaligned "
                    f"with price ({price.year}, {price.month})"
                )

    def __len__(self) -> int:
        """Return the number of monthly observations."""
        return len(self.bundles)
