"""Core revealed-preference algorithms.

This module implements the data-source-independent algorithms that operate
on bundle-price pairs ``(x_t, p_t)`` for ``t = 1, ..., T``. Each bundle
``x_t`` is a vector of quantities chosen at prices ``p_t``.

Functions
---------
direct_revealed_preference : compute the direct revealed-preference relation.
transitive_closure : Warshall closure of a binary relation.
check_garp : test whether observations satisfy GARP.
afriat_ccei : Afriat's Critical Cost Efficiency Index via bisection.
money_pump_index : Echenique-Lee-Shum money pump value over short cycles.

Notes
-----
All algorithms operate on numpy arrays. No data-source assumptions.
For multi-firm batched application, wrap these in higher-level helpers
in :mod:`firm_rp_garp.prod` or :mod:`firm_rp_garp.events`.

References
----------
.. [1] Afriat, S. (1967). "The construction of utility functions from
       expenditure data." *International Economic Review* 8(1): 67–77.
.. [2] Varian, H. (1982). "The nonparametric approach to demand analysis."
       *Econometrica* 50(4): 945–973.
.. [3] Varian, H. (1984). "The nonparametric approach to production
       analysis." *Econometrica* 52(3): 579–597.
.. [4] Afriat, S. (1973). "On a system of inequalities in demand
       analysis: an extension of the classical method." *International
       Economic Review* 14(2): 460–472.
.. [5] Echenique, F., Lee, S. & Shum, M. (2011). "The money pump as a
       measure of revealed preference violations." *Journal of Political
       Economy* 119(6): 1201–1223.
"""
from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class GarpResult(NamedTuple):
    """Result of a GARP consistency test.

    Parameters
    ----------
    satisfies_garp : bool
        True iff no violation was found.
    direct_rp : BoolArray
        ``(T, T)`` matrix where ``direct_rp[i, j]`` is True if bundle
        ``i`` was directly revealed preferred to bundle ``j``.
    strict_direct_rp : BoolArray
        ``(T, T)`` matrix of strict direct revealed preference.
    closure : BoolArray
        Transitive closure of ``direct_rp``.
    violations : list[tuple[int, int]]
        Pairs ``(i, j)`` violating GARP. ``i`` is revealed preferred to
        ``j`` (via closure) but ``j`` is strictly directly revealed
        preferred to ``i``.
    """

    satisfies_garp: bool
    direct_rp: BoolArray
    strict_direct_rp: BoolArray
    closure: BoolArray
    violations: list[tuple[int, int]]


def _validate_bundles_prices(
    bundles: FloatArray, prices: FloatArray
) -> None:
    """Validate that bundles and prices have matching shapes and are finite.

    Parameters
    ----------
    bundles : FloatArray
        Shape ``(T, K)``. Row ``t`` is the bundle chosen at time ``t``.
    prices : FloatArray
        Shape ``(T, K)``. Row ``t`` is the price vector at time ``t``.

    Raises
    ------
    ValueError
        If shapes do not match or values are not all finite.
    """
    if bundles.shape != prices.shape:
        raise ValueError(
            f"bundles shape {bundles.shape} != prices shape {prices.shape}"
        )
    if bundles.ndim != 2:
        raise ValueError(f"bundles must be 2D, got {bundles.ndim}D")
    if not np.isfinite(bundles).all():
        raise ValueError("bundles contains non-finite values")
    if not np.isfinite(prices).all():
        raise ValueError("prices contains non-finite values")


def direct_revealed_preference(
    bundles: FloatArray, prices: FloatArray, *, strict_tolerance: float = 0.0
) -> tuple[BoolArray, BoolArray]:
    """Compute the direct revealed-preference relation.

    Bundle ``x_i`` is directly revealed preferred to ``x_j`` iff
    ``p_i · x_i >= p_i · x_j`` — that is, ``x_j`` was affordable when
    ``x_i`` was chosen. The relation is *strict* when the inequality is
    strict: ``p_i · x_i > p_i · x_j + tol``.

    Parameters
    ----------
    bundles : FloatArray
        Shape ``(T, K)``.
    prices : FloatArray
        Shape ``(T, K)``.
    strict_tolerance : float, default 0.0
        Numerical slack added to the strict comparison.

    Returns
    -------
    direct_rp : BoolArray
        ``(T, T)`` boolean matrix. ``direct_rp[i, j]`` is True iff
        ``x_j`` was affordable at prices ``p_i`` given the budget
        ``p_i · x_i``. Diagonal is True by construction.
    strict_direct_rp : BoolArray
        ``(T, T)`` boolean matrix. ``strict_direct_rp[i, j]`` is True
        iff ``p_i · x_i > p_i · x_j + strict_tolerance``.

    Notes
    -----
    The cost at own prices is ``c_i = p_i · x_i``. The cost of bundle
    ``j`` at prices ``p_i`` is ``p_i · x_j``. ``direct_rp[i, j] = True``
    when ``c_i >= p_i · x_j``.
    """
    _validate_bundles_prices(bundles, prices)
    expenditure = (prices * bundles).sum(axis=1)
    cross_expenditure = prices @ bundles.T  # shape (T, T)
    direct_rp = cross_expenditure <= expenditure[:, None]
    strict_direct_rp = cross_expenditure < (expenditure[:, None] - strict_tolerance)
    return direct_rp, strict_direct_rp


def transitive_closure(relation: BoolArray) -> BoolArray:
    """Compute the transitive closure of a binary relation.

    Implements Warshall's algorithm. For ``T <= 1000`` this is
    ``O(T^3)`` in time but trivial in memory.

    Parameters
    ----------
    relation : BoolArray
        ``(T, T)`` binary relation.

    Returns
    -------
    BoolArray
        ``(T, T)`` transitive closure. ``closure[i, j]`` is True iff
        there exists a chain ``i -> k_1 -> ... -> k_m -> j`` in the
        original relation.
    """
    closure = relation.copy()
    n = closure.shape[0]
    for k in range(n):
        closure |= closure[:, k:k + 1] & closure[k:k + 1, :]
    return closure


def check_garp(
    bundles: FloatArray, prices: FloatArray, *, strict_tolerance: float = 0.0
) -> GarpResult:
    """Test the Generalized Axiom of Revealed Preference.

    GARP holds iff for all ``i, j``: if ``i`` is revealed preferred to
    ``j`` (transitively), then ``j`` is *not* strictly directly revealed
    preferred to ``i``.

    Parameters
    ----------
    bundles : FloatArray
        Shape ``(T, K)``. Row ``t`` is the bundle chosen at time ``t``.
    prices : FloatArray
        Shape ``(T, K)``. Row ``t`` is the price vector at time ``t``.
    strict_tolerance : float, default 0.0
        Tolerance for strict comparison in
        :func:`direct_revealed_preference`.

    Returns
    -------
    GarpResult
        Named tuple with the GARP verdict, the underlying relations,
        the transitive closure, and the list of violating pairs.

    Examples
    --------
    >>> bundles = np.array([[1.0, 2.0], [2.0, 1.0]])
    >>> prices = np.array([[1.0, 1.0], [1.0, 1.0]])
    >>> result = check_garp(bundles, prices)
    >>> result.satisfies_garp
    True
    """
    direct_rp, strict_direct_rp = direct_revealed_preference(
        bundles, prices, strict_tolerance=strict_tolerance
    )
    closure = transitive_closure(direct_rp)
    violation_mask = closure & strict_direct_rp.T
    np.fill_diagonal(violation_mask, False)
    violations = [
        (int(i), int(j)) for i, j in zip(*np.where(violation_mask))
    ]
    return GarpResult(
        satisfies_garp=len(violations) == 0,
        direct_rp=direct_rp,
        strict_direct_rp=strict_direct_rp,
        closure=closure,
        violations=violations,
    )


def afriat_ccei(
    bundles: FloatArray,
    prices: FloatArray,
    *,
    tolerance: float = 1e-4,
    max_iterations: int = 30,
) -> float:
    """Compute the Critical Cost Efficiency Index via bisection.

    The CCEI is ``sup{e in (0, 1] : the data are e-rationalizable}``.
    Data are ``e``-rationalizable if scaling each budget by ``e``
    eliminates all GARP violations. CCEI = 1 means the data perfectly
    satisfy GARP. CCEI < 1 quantifies the fractional budget waste
    required to make the data consistent.

    Parameters
    ----------
    bundles : FloatArray
        Shape ``(T, K)``.
    prices : FloatArray
        Shape ``(T, K)``.
    tolerance : float, default 1e-4
        Bisection halts when the interval is narrower than ``tolerance``.
    max_iterations : int, default 30
        Hard cap on bisection iterations. With default tolerance, 14
        iterations always suffice.

    Returns
    -------
    float
        CCEI in ``[0, 1]``. Equal to 1 iff GARP holds.

    Notes
    -----
    The bisection scales the *budget* by ``e``: at each candidate ``e``,
    bundle ``i`` is revealed preferred to bundle ``j`` iff
    ``e * p_i * x_i >= p_i * x_j``. This is the standard Afriat (1973)
    construction.

    References
    ----------
    .. [1] Afriat, S. (1973). "On a system of inequalities in demand
           analysis." *International Economic Review* 14(2): 460–472.
    .. [2] Polisson, M. & Quah, J. (2024). "Cost rationalizability and
           the Critical Cost Efficiency Index." arXiv:2406.10136.
    """
    _validate_bundles_prices(bundles, prices)
    if check_garp(bundles, prices).satisfies_garp:
        return 1.0
    expenditure = (prices * bundles).sum(axis=1)
    cross_expenditure = prices @ bundles.T

    def is_e_rationalizable(e: float) -> bool:
        """Test whether scaling budgets by ``e`` removes all GARP violations.

        Parameters
        ----------
        e : float
            Budget scaling factor in (0, 1].

        Returns
        -------
        bool
            True iff the (e-scaled) data satisfy GARP.
        """
        scaled_budget = e * expenditure
        direct_rp = cross_expenditure <= scaled_budget[:, None]
        strict_direct_rp = cross_expenditure < scaled_budget[:, None]
        closure = transitive_closure(direct_rp)
        violation_mask = closure & strict_direct_rp.T
        np.fill_diagonal(violation_mask, False)
        return not violation_mask.any()

    low, high = 0.0, 1.0
    for _ in range(max_iterations):
        if high - low < tolerance:
            break
        mid = 0.5 * (low + high)
        if is_e_rationalizable(mid):
            low = mid
        else:
            high = mid
    return low


def money_pump_index(
    bundles: FloatArray,
    prices: FloatArray,
    *,
    max_cycle_length: int = 4,
    statistic: str = "median",
) -> float:
    """Compute the Echenique-Lee-Shum money pump index.

    The MPI is the dollar value extractable by exploiting a
    GARP-violating cycle. For each cycle ``i_1 -> i_2 -> ... -> i_m ->
    i_1`` in the revealed-preference relation, the pump value is
    ``sum_t (p_{i_t} . x_{i_t} - p_{i_t} . x_{i_{t+1}})``. The MPI
    aggregates these over all short cycles via a chosen statistic.

    Parameters
    ----------
    bundles : FloatArray
        Shape ``(T, K)``.
    prices : FloatArray
        Shape ``(T, K)``.
    max_cycle_length : int, default 4
        Enumerate cycles up to this length. The original paper restricts
        to short cycles (2-4) for tractability.
    statistic : {'mean', 'median', 'max'}, default 'median'
        How to aggregate per-cycle pump values into a single score.

    Returns
    -------
    float
        MPI in the units of expenditure (NOK). Zero if no violating
        cycles exist.

    Notes
    -----
    Only cycles of length ``>= 2`` and ``<= max_cycle_length`` are
    enumerated. The aggregation defaults to ``median`` because the
    original paper notes that ``mean`` is sensitive to extreme cycles.

    References
    ----------
    .. [1] Echenique, F., Lee, S. & Shum, M. (2011). "The money pump
           as a measure of revealed preference violations." *Journal
           of Political Economy* 119(6): 1201–1223.
    """
    if statistic not in {"mean", "median", "max"}:
        raise ValueError(f"statistic must be mean/median/max, got {statistic!r}")
    _validate_bundles_prices(bundles, prices)
    direct_rp, _ = direct_revealed_preference(bundles, prices)
    n = bundles.shape[0]
    expenditure = (prices * bundles).sum(axis=1)
    cross_expenditure = prices @ bundles.T
    cycle_values: list[float] = []
    for length in range(2, max_cycle_length + 1):
        for cycle in itertools.permutations(range(n), length):
            valid = all(direct_rp[cycle[t], cycle[(t + 1) % length]]
                        for t in range(length))
            if not valid:
                continue
            value = sum(
                expenditure[cycle[t]] - cross_expenditure[cycle[t], cycle[(t + 1) % length]]
                for t in range(length)
            )
            if value > 0:
                cycle_values.append(float(value))
    if not cycle_values:
        return 0.0
    if statistic == "mean":
        return float(np.mean(cycle_values))
    if statistic == "median":
        return float(np.median(cycle_values))
    return float(np.max(cycle_values))
