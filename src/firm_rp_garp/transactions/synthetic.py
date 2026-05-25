"""Synthetic transaction generators with known ground-truth archetypes.

This module is the validation rig. It generates pseudo-transactions for
firms whose decision rules are known by construction, so the production
pipeline (bundler → rolling GARP → break detection) can be tested
against ground truth before being wired into real bank data.

Three archetypes
----------------
- :func:`generate_rational_firm` — Cobb-Douglas optimizer with shares
  ``alpha_k``. Each month allocates ``x_k = alpha_k * W / p_k`` where W
  is total cash available. Should satisfy GARP with CCEI ≈ 1.
- :func:`generate_satisficer_firm` — fixed proportional rule
  ``x_k = beta_k * W`` ignoring prices. Should fail GARP whenever
  relative prices move; CCEI drop should scale with price variance.
- :func:`generate_distressed_firm` — rational until month ``distress_start``,
  then forced reallocation toward ``DEBT_SERVICE`` capped by liquidity.
  Rolling-window CCEI should drop sharply within 3 months of
  ``distress_start``.

Macro generator
---------------
:func:`generate_synthetic_macro` produces an AR(1)-anchored NIBOR path,
a multivariate-correlated sector and capital-goods PPI, and stepwise
tax-wedge changes. Designed to mimic the actual Norwegian 2017-2024
trajectory while being fully reproducible from a seed.
"""
from __future__ import annotations

import calendar
import math
import random
from dataclasses import dataclass
from datetime import date

from firm_rp_garp.transactions.prices import MacroSeries
from firm_rp_garp.transactions.schema import (
    CATEGORIES,
    Category,
    N_CATEGORIES,
    Transaction,
)


_MCC_BY_CATEGORY: dict[Category, str] = {
    Category.OPERATING: "OPS",
    Category.DEBT_SERVICE: "DEB",
    Category.INVESTMENT: "INV",
    Category.DIVIDEND: "DIV",
    Category.RETAINED: "RET",
    Category.WORKING_CAPITAL: "WCC",
}


def month_iter(
    start_year: int, start_month: int, n_months: int
) -> list[tuple[int, int]]:
    """Generate a list of (year, month) tuples spanning a date range.

    Parameters
    ----------
    start_year : int
        Year of the first month.
    start_month : int
        Month (1-12) of the first month.
    n_months : int
        Number of months to generate.

    Returns
    -------
    list[tuple[int, int]]
        Ascending (year, month) pairs.
    """
    months: list[tuple[int, int]] = []
    y, m = start_year, start_month
    for _ in range(n_months):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def generate_synthetic_macro(
    months: list[tuple[int, int]],
    seed: int = 42,
) -> MacroSeries:
    """Generate a reproducible synthetic macro panel covering given months.

    Parameters
    ----------
    months : list[tuple[int, int]]
        Months to populate, ascending and contiguous.
    seed : int, default 42
        RNG seed.

    Returns
    -------
    MacroSeries
        Macro panel with NIBOR, sector PPI, capital-goods PPI, dividend
        tax, and risk-free rate keyed by (year, month).

    Notes
    -----
    The generator is calibrated to roughly match Norwegian 2017-2024
    moves. NIBOR follows a slow AR(1) with mean reversion and drift
    upward 0.5%→4.5%. Sector PPI is a random walk with annualised
    drift +5% and monthly volatility 1.5%. Capital-goods PPI is
    correlated 0.5 with sector PPI. Dividend tax is stepwise
    (Norwegian fiscal years). Risk-free rate tracks NIBOR minus 50bp.
    """
    rng = random.Random(seed)
    nibor: dict[tuple[int, int], float] = {}
    sector_ppi: dict[tuple[int, int], float] = {}
    capital_goods_ppi: dict[tuple[int, int], float] = {}
    dividend_tax: dict[tuple[int, int], float] = {}
    risk_free: dict[tuple[int, int], float] = {}

    nibor_level = 0.005
    target_nibor = 0.045
    ppi_level = 100.0
    cap_level = 100.0
    correlation = 0.5

    for year, month in months:
        nibor_drift = (target_nibor - nibor_level) * 0.04
        nibor_shock = rng.gauss(0.0, 0.0015)
        nibor_level = max(0.0001, nibor_level + nibor_drift + nibor_shock)
        nibor[(year, month)] = nibor_level
        risk_free[(year, month)] = max(0.0001, nibor_level - 0.005)

        ppi_shock = rng.gauss(0.05 / 12.0, 0.015)
        cap_independent = rng.gauss(0.04 / 12.0, 0.012)
        cap_shock = correlation * ppi_shock + math.sqrt(
            1.0 - correlation**2
        ) * cap_independent
        ppi_level *= math.exp(ppi_shock)
        cap_level *= math.exp(cap_shock)
        sector_ppi[(year, month)] = ppi_level
        capital_goods_ppi[(year, month)] = cap_level

        if year < 2022:
            tax = 0.3168
        elif year < 2023:
            tax = 0.354
        else:
            tax = 0.378
        dividend_tax[(year, month)] = tax

    return MacroSeries(
        nibor=nibor,
        sector_ppi=sector_ppi,
        capital_goods_ppi=capital_goods_ppi,
        dividend_tax=dividend_tax,
        risk_free=risk_free,
    )


def _opportunity_prices_for_month(
    year: int, month: int, macro: MacroSeries
) -> tuple[float, ...]:
    """Return the opportunity-cost price vector for one month.

    Parameters
    ----------
    year : int
        Calendar year.
    month : int
        Calendar month, 1-12.
    macro : MacroSeries
        Macro series containing the requested month.

    Returns
    -------
    tuple[float, ...]
        Six-element tuple aligned to :data:`CATEGORIES`.
    """
    from firm_rp_garp.transactions.prices import assemble_price_context

    ctx = assemble_price_context(year, month, macro)
    return ctx.prices


def _emit_monthly_transactions(
    orgnr: str,
    year: int,
    month: int,
    amounts_by_category: dict[Category, float],
    rng: random.Random,
    n_per_category: int = 3,
) -> list[Transaction]:
    """Split a monthly category allocation into granular transactions.

    Parameters
    ----------
    orgnr : str
        Norwegian organisation number.
    year : int
        Calendar year.
    month : int
        Calendar month, 1-12.
    amounts_by_category : dict[Category, float]
        NOK amounts per category for this month.
    rng : random.Random
        RNG for transaction-date jitter and split proportions.
    n_per_category : int, default 3
        Number of granular transactions to emit per category. Larger
        values produce more realistic granular streams; the test
        result is invariant to this value because the bundler
        re-aggregates them exactly.

    Returns
    -------
    list[Transaction]
        Granular transactions for this month. The bundler will
        re-aggregate them exactly to the input amounts.
    """
    last_day = calendar.monthrange(year, month)[1]
    transactions: list[Transaction] = []
    for cat, total in amounts_by_category.items():
        if total <= 0.0:
            continue
        if n_per_category <= 1:
            splits = [total]
        else:
            weights = [rng.uniform(0.5, 1.5) for _ in range(n_per_category)]
            weight_sum = sum(weights)
            splits = [total * w / weight_sum for w in weights]
        mcc = _MCC_BY_CATEGORY[cat]
        for idx, amount in enumerate(splits):
            day = rng.randint(1, last_day)
            transactions.append(
                Transaction(
                    orgnr=orgnr,
                    txn_date=date(year, month, day),
                    amount_nok=amount,
                    category=cat,
                    counterparty=f"{cat.value}_cp_{idx}",
                    mcc=mcc,
                )
            )
    return transactions


def _normalize_shares(shares: dict[Category, float]) -> dict[Category, float]:
    """Normalize a share dict to sum to 1, preserving relative weights.

    Parameters
    ----------
    shares : dict[Category, float]
        Non-negative weights keyed by category.

    Returns
    -------
    dict[Category, float]
        Same keys with values summing to 1.0.

    Raises
    ------
    ValueError
        If all weights are zero.
    """
    total = sum(shares.values())
    if total <= 0.0:
        raise ValueError(f"shares must have a positive sum, got {shares}")
    return {cat: weight / total for cat, weight in shares.items()}


def generate_rational_firm(
    orgnr: str,
    months: list[tuple[int, int]],
    macro: MacroSeries,
    alpha: dict[Category, float] | None = None,
    monthly_cash_mean: float = 1_000_000.0,
    monthly_cash_volatility: float = 0.05,
    n_per_category: int = 3,
    seed: int = 0,
) -> list[Transaction]:
    """Generate transactions for a Cobb-Douglas-rational firm.

    Parameters
    ----------
    orgnr : str
        Norwegian organisation number.
    months : list[tuple[int, int]]
        Months to populate.
    macro : MacroSeries
        Macro panel covering ``months``.
    alpha : dict[Category, float] or None
        Cobb-Douglas weights (will be normalized). Defaults to a
        balanced allocation across categories.
    monthly_cash_mean : float, default 1_000_000
        Mean monthly cash available (NOK).
    monthly_cash_volatility : float, default 0.05
        Lognormal volatility of monthly cash.
    n_per_category : int, default 3
        Number of granular transactions per category-month.
    seed : int, default 0
        RNG seed.

    Returns
    -------
    list[Transaction]
        Granular transactions for the firm across ``months``.

    Notes
    -----
    The firm's optimal allocation is ``x_k = alpha_k * W / p_k``,
    chosen to satisfy ``sum(p_k * x_k) = W``. A pure rational firm
    will pass GARP with CCEI ≈ 1.
    """
    rng = random.Random(seed)
    if alpha is None:
        alpha = {cat: 1.0 for cat in CATEGORIES}
    alpha_norm = _normalize_shares(alpha)
    transactions: list[Transaction] = []
    for year, month in months:
        prices = _opportunity_prices_for_month(year, month, macro)
        cash = monthly_cash_mean * math.exp(
            rng.gauss(-(monthly_cash_volatility**2) / 2.0, monthly_cash_volatility)
        )
        amounts = {
            cat: alpha_norm[cat] * cash / prices[i]
            for i, cat in enumerate(CATEGORIES)
        }
        transactions.extend(
            _emit_monthly_transactions(
                orgnr, year, month, amounts, rng, n_per_category
            )
        )
    return transactions


def generate_satisficer_firm(
    orgnr: str,
    months: list[tuple[int, int]],
    macro: MacroSeries,
    beta: dict[Category, float] | None = None,
    share_noise: float = 0.5,
    monthly_cash_mean: float = 1_000_000.0,
    monthly_cash_volatility: float = 0.05,
    n_per_category: int = 3,
    seed: int = 0,
) -> list[Transaction]:
    """Generate transactions for a noisy random-allocator satisficer.

    Parameters
    ----------
    orgnr : str
        Norwegian organisation number.
    months : list[tuple[int, int]]
        Months to populate.
    macro : MacroSeries
        Macro panel. Used only to convert expenditure shares to
        quantities at current prices; the satisficer's expenditure
        shares are drawn independently of prices.
    beta : dict[Category, float] or None
        Mean expenditure-share weights (will be normalized into a
        Dirichlet mean). Defaults to balanced.
    share_noise : float, default 0.5
        Variance scale for the Dirichlet draws. Larger values produce
        more random month-to-month share variation. ``0.0`` would
        reproduce a fixed-share allocator (which GARP cannot detect);
        ``0.5`` produces noticeable random reallocation.
    monthly_cash_mean : float, default 1_000_000
        Mean monthly cash (NOK).
    monthly_cash_volatility : float, default 0.05
        Lognormal volatility of monthly cash.
    n_per_category : int, default 3
        Number of granular transactions per category-month.
    seed : int, default 0
        RNG seed.

    Returns
    -------
    list[Transaction]
        Granular transactions across ``months``.

    Notes
    -----
    Each month the firm draws expenditure shares from a Dirichlet
    distribution centered on ``beta``. The shares are independent of
    the current price vector — this is the satisficing pattern that
    GARP can detect. A pure fixed-share allocator (``share_noise=0``)
    would actually pass GARP because constant choices are
    rationalizable by Leontief preferences; the random reallocation
    breaks that rationalization.
    """
    rng = random.Random(seed)
    if beta is None:
        beta = {cat: 1.0 for cat in CATEGORIES}
    beta_norm = _normalize_shares(beta)
    dirichlet_alpha = {
        cat: max(beta_norm[cat] / max(share_noise**2, 1e-6), 0.1)
        for cat in CATEGORIES
    }
    transactions: list[Transaction] = []
    for year, month in months:
        prices = _opportunity_prices_for_month(year, month, macro)
        raw_shares = {
            cat: rng.gammavariate(dirichlet_alpha[cat], 1.0) for cat in CATEGORIES
        }
        share_total = sum(raw_shares.values())
        expenditure_shares = {
            cat: raw_shares[cat] / share_total for cat in CATEGORIES
        }
        cash = monthly_cash_mean * math.exp(
            rng.gauss(-(monthly_cash_volatility**2) / 2.0, monthly_cash_volatility)
        )
        amounts = {
            cat: expenditure_shares[cat] * cash / prices[i]
            for i, cat in enumerate(CATEGORIES)
        }
        transactions.extend(
            _emit_monthly_transactions(
                orgnr, year, month, amounts, rng, n_per_category
            )
        )
    return transactions


def generate_distressed_firm(
    orgnr: str,
    months: list[tuple[int, int]],
    macro: MacroSeries,
    distress_start: tuple[int, int],
    alpha: dict[Category, float] | None = None,
    distress_debt_share: float = 0.85,
    monthly_cash_mean: float = 1_000_000.0,
    distress_cash_decay: float = 0.0,
    monthly_cash_volatility: float = 0.05,
    n_per_category: int = 3,
    seed: int = 0,
) -> list[Transaction]:
    """Generate a firm that is rational pre-distress and forced post-distress.

    Parameters
    ----------
    orgnr : str
        Norwegian organisation number.
    months : list[tuple[int, int]]
        Months to populate.
    macro : MacroSeries
        Macro panel covering ``months``.
    distress_start : tuple[int, int]
        (year, month) when the regime shift occurs. Must lie within
        ``months``.
    alpha : dict[Category, float] or None
        Pre-distress Cobb-Douglas weights. Defaults to balanced.
    distress_debt_share : float, default 0.65
        Share of cash forcibly diverted to ``DEBT_SERVICE`` after
        ``distress_start``. The remaining ``1 - distress_debt_share``
        is distributed across non-debt categories proportional to
        their ``alpha`` weights.
    monthly_cash_mean : float, default 1_000_000
        Mean monthly cash pre-distress.
    distress_cash_decay : float, default 0.10
        Per-month proportional decline in available cash after
        distress (compounding). Captures revenue deterioration that
        accompanies the desperation signature.
    monthly_cash_volatility : float, default 0.05
        Lognormal cash volatility (applies both pre- and post-distress).
    n_per_category : int, default 3
        Number of granular transactions per category-month.
    seed : int, default 0
        RNG seed.

    Returns
    -------
    list[Transaction]
        Granular transactions across ``months``.

    Notes
    -----
    This generator implements the IRL paper's "desperation signature":
    the firm sacrifices investment, dividends, and retained liquidity
    to keep debt service current. The rolling CCEI should drop within
    3 months of ``distress_start`` because the post-distress
    allocation is GARP-inconsistent with the pre-distress observations.
    """
    rng = random.Random(seed)
    if alpha is None:
        alpha = {cat: 1.0 for cat in CATEGORIES}
    alpha_norm = _normalize_shares(alpha)
    non_debt = {
        cat: w for cat, w in alpha_norm.items() if cat != Category.DEBT_SERVICE
    }
    non_debt_norm = _normalize_shares(non_debt)
    transactions: list[Transaction] = []
    months_since_distress = 0
    for year, month in months:
        prices = _opportunity_prices_for_month(year, month, macro)
        is_distressed = (year, month) >= distress_start
        if not is_distressed:
            cash = monthly_cash_mean * math.exp(
                rng.gauss(-(monthly_cash_volatility**2) / 2.0, monthly_cash_volatility)
            )
            amounts = {
                cat: alpha_norm[cat] * cash / prices[i]
                for i, cat in enumerate(CATEGORIES)
            }
        else:
            decay = (1.0 - distress_cash_decay) ** months_since_distress
            cash = monthly_cash_mean * decay * math.exp(
                rng.gauss(-(monthly_cash_volatility**2) / 2.0, monthly_cash_volatility)
            )
            debt_idx = CATEGORIES.index(Category.DEBT_SERVICE)
            debt_quantity = distress_debt_share * cash / prices[debt_idx]
            non_debt_remaining = (1.0 - distress_debt_share) * cash
            amounts = {Category.DEBT_SERVICE: debt_quantity}
            for i, cat in enumerate(CATEGORIES):
                if cat == Category.DEBT_SERVICE:
                    continue
                amounts[cat] = non_debt_remaining * non_debt_norm[cat] / prices[i]
            months_since_distress += 1
        transactions.extend(
            _emit_monthly_transactions(
                orgnr, year, month, amounts, rng, n_per_category
            )
        )
    return transactions


def add_lognormal_noise(
    transactions: list[Transaction],
    sigma: float = 0.1,
    seed: int = 0,
) -> list[Transaction]:
    """Apply multiplicative lognormal noise to transaction amounts.

    Parameters
    ----------
    transactions : list[Transaction]
        Input transactions.
    sigma : float, default 0.1
        Lognormal noise scale (in log-space). Sigma of 0.1 is roughly
        10% multiplicative noise.
    seed : int, default 0
        RNG seed.

    Returns
    -------
    list[Transaction]
        New transactions with perturbed amounts. Identities, dates,
        and categories are preserved.

    Notes
    -----
    The noise multiplier is ``exp(N(-sigma^2/2, sigma))`` to keep the
    expected amount unchanged. Used to test the GARP test's tolerance
    of measurement-error-style noise in an otherwise rational firm.
    """
    rng = random.Random(seed)
    return [
        Transaction(
            orgnr=t.orgnr,
            txn_date=t.txn_date,
            amount_nok=t.amount_nok * math.exp(
                rng.gauss(-(sigma**2) / 2.0, sigma)
            ),
            category=t.category,
            counterparty=t.counterparty,
            mcc=t.mcc,
        )
        for t in transactions
    ]
