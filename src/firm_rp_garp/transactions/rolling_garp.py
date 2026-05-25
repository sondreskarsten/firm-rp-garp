"""Rolling-window revealed-preference tests on firm transaction panels.

This module applies the core GARP/CCEI/MPI algorithms from
:mod:`firm_rp_garp.core` to a firm's :class:`FirmTransactionPanel` in
a rolling window. The output is a time series of CCEI values indexed
by the window's end month — the natural signal for distress detection.

The window size defaults to 12 months, matching the IRL paper's
recommendation (Stage 1: "rolling 24-month window"; we choose 12 to
get earlier detection of regime shifts in the rig).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from firm_rp_garp.core import afriat_ccei, check_garp, money_pump_index
from firm_rp_garp.transactions.schema import (
    FirmTransactionPanel,
    MonthlyBundle,
    PriceContext,
)


@dataclass(frozen=True, slots=True)
class WindowResult:
    """RP test result for a single rolling window of one firm.

    Parameters
    ----------
    orgnr : str
        Norwegian organisation number.
    window_end_year : int
        Calendar year of the window's last month.
    window_end_month : int
        Calendar month, 1-12, of the window's last month.
    window_size : int
        Number of monthly observations in the window.
    satisfies_garp : bool
        True iff the window satisfies GARP.
    n_violations : int
        Number of GARP-violating pairs in the window.
    ccei : float
        Critical Cost Efficiency Index in [0, 1].
    mpi_median : float
        Median money pump value across short cycles (NOK).
    mpi_max : float
        Maximum money pump value across short cycles (NOK).
    """

    orgnr: str
    window_end_year: int
    window_end_month: int
    window_size: int
    satisfies_garp: bool
    n_violations: int
    ccei: float
    mpi_median: float
    mpi_max: float


def _bundles_to_arrays(
    bundles: tuple[MonthlyBundle, ...],
    prices: tuple[PriceContext, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dataclass sequences to numpy arrays for the core algorithms.

    Parameters
    ----------
    bundles : tuple[MonthlyBundle, ...]
        Aligned monthly bundles.
    prices : tuple[PriceContext, ...]
        Aligned price contexts.

    Returns
    -------
    bundle_array : np.ndarray
        Shape ``(T, N_CATEGORIES)``, dtype float64. NOK amounts.
    price_array : np.ndarray
        Shape ``(T, N_CATEGORIES)``, dtype float64. Opportunity-cost
        prices.
    """
    bundle_array = np.array(
        [list(b.amounts_nok) for b in bundles], dtype=np.float64
    )
    price_array = np.array(
        [list(p.prices) for p in prices], dtype=np.float64
    )
    return bundle_array, price_array


def evaluate_window(
    panel: FirmTransactionPanel, start: int, end: int
) -> WindowResult:
    """Run GARP/CCEI/MPI on a single window of one firm's panel.

    Parameters
    ----------
    panel : FirmTransactionPanel
        Firm's full transaction panel.
    start : int
        Inclusive start index into ``panel.bundles``.
    end : int
        Exclusive end index into ``panel.bundles``.

    Returns
    -------
    WindowResult
        Test results for the window.

    Raises
    ------
    ValueError
        If the window has fewer than 2 observations (GARP requires at
        least two bundles).
    """
    if end - start < 2:
        raise ValueError(
            f"window must contain at least 2 observations, got {end - start}"
        )
    bundles = panel.bundles[start:end]
    prices = panel.prices[start:end]
    bundle_array, price_array = _bundles_to_arrays(bundles, prices)
    garp = check_garp(bundle_array, price_array)
    ccei = afriat_ccei(bundle_array, price_array)
    mpi_med = money_pump_index(bundle_array, price_array, statistic="median")
    mpi_max = money_pump_index(bundle_array, price_array, statistic="max")
    last = bundles[-1]
    return WindowResult(
        orgnr=panel.orgnr,
        window_end_year=last.year,
        window_end_month=last.month,
        window_size=end - start,
        satisfies_garp=garp.satisfies_garp,
        n_violations=len(garp.violations),
        ccei=ccei,
        mpi_median=mpi_med,
        mpi_max=mpi_max,
    )


def rolling_window_ccei(
    panel: FirmTransactionPanel,
    window_size: int = 12,
    step: int = 1,
) -> list[WindowResult]:
    """Compute CCEI over a rolling window across the firm's panel.

    Parameters
    ----------
    panel : FirmTransactionPanel
        Firm's full transaction panel.
    window_size : int, default 12
        Window length in months.
    step : int, default 1
        Stride between consecutive window starts.

    Returns
    -------
    list[WindowResult]
        One result per window, ordered by window-end date. Empty if
        the panel is shorter than the window.

    Notes
    -----
    The window-end date is the date of the last observation *in* the
    window, so a CCEI drop at index t reflects evidence accumulated
    through month t.
    """
    if len(panel) < window_size:
        return []
    results: list[WindowResult] = []
    for start in range(0, len(panel) - window_size + 1, step):
        end = start + window_size
        results.append(evaluate_window(panel, start, end))
    return results


def detect_ccei_break(
    windows: list[WindowResult],
    drop_threshold: float = 0.01,
    statistic: Literal["level", "running_max"] = "running_max",
    min_consecutive: int = 1,
) -> int | None:
    """Detect the first month where rolling CCEI signals a regime break.

    Parameters
    ----------
    windows : list[WindowResult]
        Rolling-window results from :func:`rolling_window_ccei`,
        ordered by window-end date.
    drop_threshold : float, default 0.01
        Minimum CCEI drop (in level units) that qualifies as a break.
    statistic : {'level', 'running_max'}, default 'running_max'
        Detection rule. ``'level'`` flags when CCEI falls below
        ``1 - drop_threshold``. ``'running_max'`` flags when CCEI
        drops by more than ``drop_threshold`` from any prior maximum.
    min_consecutive : int, default 1
        Number of consecutive windows that must satisfy the drop
        condition before firing. ``1`` fires on first qualifying
        window. Higher values reject transient single-window noise.

    Returns
    -------
    int or None
        Index into ``windows`` of the first detected break, or
        ``None`` if no break is detected. The returned index is the
        first qualifying window — not the last of the confirming run.

    Notes
    -----
    For real bank-data distress detection the rolling-CCEI dip is
    sustained for the length of the rolling window (12 months by
    default) because the window straddles the regime shift. A
    ``min_consecutive`` of 2-3 reliably distinguishes regime shifts
    from single-window numerical noise without sacrificing detection
    timing materially.
    """
    if not windows:
        return None
    if statistic == "level":
        threshold = 1.0 - drop_threshold
        run_start: int | None = None
        for idx, w in enumerate(windows):
            if w.ccei < threshold:
                if run_start is None:
                    run_start = idx
                if idx - run_start + 1 >= min_consecutive:
                    return run_start
            else:
                run_start = None
        return None
    if statistic == "running_max":
        running_max = windows[0].ccei
        run_start = None
        for idx, w in enumerate(windows):
            running_max = max(running_max, w.ccei)
            if running_max - w.ccei > drop_threshold:
                if run_start is None:
                    run_start = idx
                if idx - run_start + 1 >= min_consecutive:
                    return run_start
            else:
                run_start = None
        return None
    raise ValueError(f"statistic must be level/running_max, got {statistic!r}")
