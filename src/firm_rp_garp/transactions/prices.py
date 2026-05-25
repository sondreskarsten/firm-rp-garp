"""Opportunity-cost price assembly for the six cash-use categories.

Each category has a monthly opportunity-cost price. These are not
prices the firm pays; they are the marginal cost of dedicating cash
to that use rather than the alternatives. The categories and their
price proxies are:

==================  ================================================
Category            Opportunity-cost proxy
==================  ================================================
operating           sector PPI (firm's NACE input PPI)
debt_service        NIBOR + funding margin (cost of *not* repaying)
investment          capital-goods PPI (cost of the asset bought)
dividend            1 / (1 - dividend-tax wedge)
retained            1 + risk-free rate (foregone return on cash)
working_capital     sector PPI × inventory holding-cost markup
==================  ================================================

The :func:`assemble_price_context` function takes the underlying
macro time series and produces one :class:`PriceContext` per
firm-month. The synthetic rig in
:mod:`firm_rp_garp.transactions.synthetic` provides synthetic versions
of all four input series; the production version would query SSB,
Norges Bank, and tax rules.
"""
from __future__ import annotations

from dataclasses import dataclass

from firm_rp_garp.transactions.schema import (
    CATEGORIES,
    Category,
    PriceContext,
)


@dataclass(frozen=True, slots=True)
class MacroSeries:
    """Container for the macro time series used to build price contexts.

    Parameters
    ----------
    nibor : dict[tuple[int, int], float]
        Three-month NIBOR rate (decimal, e.g. 0.045 for 4.5%) keyed by
        (year, month).
    sector_ppi : dict[tuple[int, int], float]
        Sector-specific producer price index keyed by (year, month).
        Any normalization; only relative levels matter.
    capital_goods_ppi : dict[tuple[int, int], float]
        Capital-goods producer price index keyed by (year, month).
    dividend_tax : dict[tuple[int, int], float]
        Effective dividend tax rate (decimal, e.g. 0.378 for 37.8%)
        keyed by (year, month). Constant within fiscal years for
        Norwegian tax law.
    risk_free : dict[tuple[int, int], float]
        Risk-free rate proxy (decimal) keyed by (year, month). For
        Norway this tracks the Norges Bank deposit rate.
    funding_margin : float
        Spread over NIBOR charged on SME debt. Typically 200-400 bps
        for unrated AS firms. Defaults to ``0.025``.
    inventory_markup : float
        Multiplier on sector PPI representing inventory holding
        cost (storage, shrinkage, opportunity). Defaults to ``1.05``.
    """

    nibor: dict[tuple[int, int], float]
    sector_ppi: dict[tuple[int, int], float]
    capital_goods_ppi: dict[tuple[int, int], float]
    dividend_tax: dict[tuple[int, int], float]
    risk_free: dict[tuple[int, int], float]
    funding_margin: float = 0.025
    inventory_markup: float = 1.05


def _scale_to_strictly_positive(value: float) -> float:
    """Ensure a price is strictly positive by clipping at a tiny floor.

    Parameters
    ----------
    value : float
        Candidate price level.

    Returns
    -------
    float
        ``max(value, 1e-6)``. Used to defend against pathological
        macro inputs that might drive a derived price to zero or
        negative.
    """
    return max(value, 1e-6)


def assemble_price_context(
    year: int, month: int, macro: MacroSeries
) -> PriceContext:
    """Build a single-month opportunity-cost price vector.

    Parameters
    ----------
    year : int
        Calendar year.
    month : int
        Calendar month, 1-12.
    macro : MacroSeries
        Macro time series for the period containing ``(year, month)``.

    Returns
    -------
    PriceContext
        Six-element price vector aligned to :data:`CATEGORIES`.

    Notes
    -----
    The price computations are:

    - ``operating``        = sector PPI (level)
    - ``debt_service``     = sector PPI × (NIBOR + funding margin)
    - ``investment``       = capital-goods PPI
    - ``dividend``         = sector PPI × 1 / (1 - dividend tax)
    - ``retained``         = sector PPI × (1 + risk-free rate)
    - ``working_capital``  = sector PPI × inventory markup

    Each category's price is anchored to the sector PPI so the prices
    are commensurable. The relative variation across categories comes
    from NIBOR, the tax wedge, and the risk-free rate moving
    differently from goods inflation.
    """
    key = (year, month)
    ppi = macro.sector_ppi[key]
    cap = macro.capital_goods_ppi[key]
    nibor = macro.nibor[key]
    tax = macro.dividend_tax[key]
    rf = macro.risk_free[key]

    by_cat: dict[Category, float] = {
        Category.OPERATING: ppi,
        Category.DEBT_SERVICE: ppi * 20.0 * (nibor + macro.funding_margin),
        Category.INVESTMENT: cap,
        Category.DIVIDEND: ppi / max(1.0 - tax, 1e-6),
        Category.RETAINED: ppi * (1.0 + rf),
        Category.WORKING_CAPITAL: ppi * macro.inventory_markup,
    }
    prices = tuple(_scale_to_strictly_positive(by_cat[cat]) for cat in CATEGORIES)
    return PriceContext(year=year, month=month, prices=prices)


def assemble_price_panel(
    months: list[tuple[int, int]], macro: MacroSeries
) -> tuple[PriceContext, ...]:
    """Build a sequence of monthly price contexts.

    Parameters
    ----------
    months : list[tuple[int, int]]
        List of (year, month) pairs in ascending order.
    macro : MacroSeries
        Macro time series covering at least all the requested months.

    Returns
    -------
    tuple[PriceContext, ...]
        One price context per requested month, in input order.
    """
    return tuple(assemble_price_context(y, m, macro) for y, m in months)
