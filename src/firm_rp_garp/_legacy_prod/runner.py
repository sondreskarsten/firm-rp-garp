"""Apply core RP tests across a panel of firm bundles.

This module is the orchestration layer. It calls
:func:`firm_rp_garp.core.check_garp`, :func:`firm_rp_garp.core.afriat_ccei`,
and :func:`firm_rp_garp.core.money_pump_index` on each
:class:`firm_rp_garp.prod.bundles.FirmBundle` and returns a tidy
per-firm result frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from firm_rp_garp.core import afriat_ccei, check_garp, money_pump_index
from firm_rp_garp.prod.bundles import FirmBundle


@dataclass(frozen=True)
class FirmRPResult:
    """RP test results for a single firm.

    Parameters
    ----------
    orgnr : str
        Organisation number.
    n_years : int
        Number of years in the bundle.
    year_min : int
        Earliest year.
    year_max : int
        Latest year.
    satisfies_garp : bool
        True iff the firm's bundles satisfy GARP across all observed years.
    n_violations : int
        Number of GARP-violating pairs.
    ccei : float
        Critical Cost Efficiency Index in [0, 1].
    mpi_median : float
        Median money pump value across short cycles (NOK).
    mpi_max : float
        Maximum money pump value across short cycles (NOK).
    """

    orgnr: str
    n_years: int
    year_min: int
    year_max: int
    satisfies_garp: bool
    n_violations: int
    ccei: float
    mpi_median: float
    mpi_max: float


def run_firm(bundle: FirmBundle) -> FirmRPResult:
    """Run all three RP tests on a single firm bundle.

    Parameters
    ----------
    bundle : FirmBundle
        Multi-year (quantity, price, expenditure) data for one firm.

    Returns
    -------
    FirmRPResult
        Tidy summary of GARP / CCEI / MPI for this firm.
    """
    garp = check_garp(bundle.bundles, bundle.prices)
    ccei = afriat_ccei(bundle.bundles, bundle.prices)
    mpi_med = money_pump_index(
        bundle.expenditures, np.ones_like(bundle.expenditures), statistic="median"
    )
    mpi_max = money_pump_index(
        bundle.expenditures, np.ones_like(bundle.expenditures), statistic="max"
    )
    return FirmRPResult(
        orgnr=bundle.orgnr,
        n_years=len(bundle.years),
        year_min=min(bundle.years),
        year_max=max(bundle.years),
        satisfies_garp=garp.satisfies_garp,
        n_violations=len(garp.violations),
        ccei=ccei,
        mpi_median=mpi_med,
        mpi_max=mpi_max,
    )


def run_panel(firm_bundles: dict[str, FirmBundle]) -> pd.DataFrame:
    """Run RP tests on every firm in the panel.

    Parameters
    ----------
    firm_bundles : dict[str, FirmBundle]
        Output of :func:`firm_rp_garp.prod.bundles.build_firm_bundles`.

    Returns
    -------
    pandas.DataFrame
        One row per firm with columns: ``orgnr``, ``n_years``,
        ``year_min``, ``year_max``, ``satisfies_garp``, ``n_violations``,
        ``ccei``, ``mpi_median``, ``mpi_max``.
    """
    rows = [run_firm(bundle).__dict__ for bundle in firm_bundles.values()]
    return pd.DataFrame(rows)
