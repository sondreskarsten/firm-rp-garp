"""Aggregate granular transactions into monthly category bundles.

The bundler is the pipeline stage that converts a stream of
:class:`firm_rp_garp.transactions.schema.Transaction` records into one
:class:`firm_rp_garp.transactions.schema.MonthlyBundle` per firm-month.

In production the input comes from a bank feed (DNB transaction stream);
in the validation rig it comes from the synthetic generator. Both
emit identical :class:`Transaction` records, so this module is shared.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Iterable

from firm_rp_garp.transactions.prices import MacroSeries, assemble_price_panel
from firm_rp_garp.transactions.schema import (
    CATEGORIES,
    CATEGORY_INDEX,
    FirmTransactionPanel,
    MonthlyBundle,
    N_CATEGORIES,
    PriceContext,
    Transaction,
)


def bundle_to_monthly(
    transactions: Iterable[Transaction],
) -> dict[str, list[MonthlyBundle]]:
    """Aggregate a transaction stream into per-firm monthly bundles.

    Parameters
    ----------
    transactions : Iterable[Transaction]
        Stream of granular transactions. May contain multiple firms
        and any number of months per firm. Records can arrive in any
        order; the bundler sorts internally.

    Returns
    -------
    dict[str, list[MonthlyBundle]]
        Map ``orgnr -> [MonthlyBundle, ...]`` with bundles sorted in
        ascending (year, month) order. Each firm gets one bundle
        per month that contains at least one transaction.

    Notes
    -----
    Months with zero activity for a firm are *not* emitted. Downstream
    consumers must decide how to handle gaps — the
    :func:`assemble_firm_panel` helper drops firms with non-contiguous
    months.
    """
    accum: dict[tuple[str, int, int], list[float]] = defaultdict(
        lambda: [0.0] * N_CATEGORIES
    )
    counts: dict[tuple[str, int, int], int] = defaultdict(int)
    for txn in transactions:
        key = (txn.orgnr, txn.txn_date.year, txn.txn_date.month)
        accum[key][CATEGORY_INDEX[txn.category]] += txn.amount_nok
        counts[key] += 1

    by_firm: dict[str, list[MonthlyBundle]] = defaultdict(list)
    for (orgnr, year, month), amounts in accum.items():
        by_firm[orgnr].append(
            MonthlyBundle(
                orgnr=orgnr,
                year=year,
                month=month,
                amounts_nok=tuple(amounts),
                n_transactions=counts[(orgnr, year, month)],
            )
        )
    for orgnr in by_firm:
        by_firm[orgnr].sort(key=lambda b: (b.year, b.month))
    return dict(by_firm)


def assemble_firm_panel(
    bundles: list[MonthlyBundle], macro: MacroSeries
) -> FirmTransactionPanel:
    """Combine a firm's monthly bundles with aligned price contexts.

    Parameters
    ----------
    bundles : list[MonthlyBundle]
        Monthly bundles for a single firm, in ascending order. Must
        be contiguous: every consecutive pair must be adjacent months.
    macro : MacroSeries
        Macro time series covering all bundle months.

    Returns
    -------
    FirmTransactionPanel
        Aligned bundles and prices.

    Raises
    ------
    ValueError
        If ``bundles`` is empty, contains bundles from multiple firms,
        or has non-contiguous months.
    """
    if not bundles:
        raise ValueError("bundles must be non-empty")
    orgnrs = {b.orgnr for b in bundles}
    if len(orgnrs) > 1:
        raise ValueError(f"bundles must be from one firm, got {orgnrs}")
    if not _is_contiguous(bundles):
        raise ValueError(
            f"bundles must span contiguous months; "
            f"got {[(b.year, b.month) for b in bundles]}"
        )
    months = [(b.year, b.month) for b in bundles]
    prices = assemble_price_panel(months, macro)
    return FirmTransactionPanel(
        orgnr=bundles[0].orgnr,
        bundles=tuple(bundles),
        prices=prices,
    )


def _is_contiguous(bundles: list[MonthlyBundle]) -> bool:
    """Check that consecutive bundles are in consecutive months.

    Parameters
    ----------
    bundles : list[MonthlyBundle]
        Pre-sorted bundles for one firm.

    Returns
    -------
    bool
        True iff every consecutive pair differs by exactly one month.
    """
    for prev, curr in zip(bundles, bundles[1:]):
        next_year = prev.year if prev.month < 12 else prev.year + 1
        next_month = prev.month + 1 if prev.month < 12 else 1
        if (curr.year, curr.month) != (next_year, next_month):
            return False
    return True


def map_mcc_to_category(mcc: str) -> "Category | None":
    """Generic MCC-code to category mapping for transaction classification.

    Parameters
    ----------
    mcc : str
        Merchant Category Code (ISO 18245), 4-digit string. Synthetic
        data may use a controlled vocabulary like ``"OPS"``, ``"DEB"``.

    Returns
    -------
    Category or None
        The category if the code maps unambiguously, else ``None``.

    Notes
    -----
    This is a placeholder generic mapping. Real DNB integration will
    override it with the bank's chart-of-accounts mapping. Only used
    in the synthetic rig path where transactions are emitted with
    a controlled MCC vocabulary.
    """
    from firm_rp_garp.transactions.schema import Category

    mapping: dict[str, Category] = {
        "OPS": Category.OPERATING,
        "DEB": Category.DEBT_SERVICE,
        "INV": Category.INVESTMENT,
        "DIV": Category.DIVIDEND,
        "RET": Category.RETAINED,
        "WCC": Category.WORKING_CAPITAL,
    }
    return mapping.get(mcc)
