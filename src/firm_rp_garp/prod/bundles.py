"""Construct production-side input bundles from finstat.

Each firm-year observation is a bundle ``x_t = (x_labour, x_materials,
x_other_opex, x_depreciation)`` measured as expenditures in NOK.

Because finstat reports expenditures, not (price × quantity)
separately, we approximate the *implicit quantity* by deflating each
expenditure with a sector-year price index from SSB:
``q_kt = expenditure_kt / p_kt``. The price vector ``p_t`` is then the
SSB index. This is Varian's (1984) approach when expenditures are
observed.

Notes
-----
- Restricted to AS firms (organisasjonsform = 'AS'). ASA firms are
  excluded — they are already monitored via Oslo Børs disclosure.
- Restricted to type-R (regular) annual accounts. Type-K (consolidated)
  is excluded to avoid double-counting subsidiaries.
- A firm-year is admitted only if all four expenditure components are
  present and non-negative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FirmBundle:
    """A single firm's multi-year input bundle for RP testing.

    Parameters
    ----------
    orgnr : str
        Organisation number (9-digit, zero-padded).
    years : list[int]
        Years for which bundles are observed, ascending.
    bundles : FloatArray
        Shape ``(T, K)``. Row ``t`` is the implicit-quantity bundle
        for year ``years[t]``, with columns in the order given by
        ``input_names``.
    prices : FloatArray
        Shape ``(T, K)``. Row ``t`` is the SSB sector price index
        vector for year ``years[t]``.
    expenditures : FloatArray
        Shape ``(T, K)``. The original NOK expenditures, kept for
        reference and for MPI computation in monetary units.
    input_names : tuple[str, ...]
        Column names for the K input dimensions.
    """

    orgnr: str
    years: list[int]
    bundles: FloatArray
    prices: FloatArray
    expenditures: FloatArray
    input_names: tuple[str, ...]


INPUT_COLUMNS: tuple[str, ...] = (
    "Lonnskostnad",
    "Varekostnad",
    "AnnenDriftskostnad",
    "AvskrivVarigeDriftsmidl",
)


def build_firm_bundles(
    finstat_df: pd.DataFrame, price_index_df: pd.DataFrame
) -> dict[str, FirmBundle]:
    """Assemble per-firm input bundles from a finstat slice.

    Parameters
    ----------
    finstat_df : pandas.DataFrame
        Slice of the finstat panel with at least columns
        ``organisasjonsnummer``, ``Regnskapsar``, ``RegnskapstypeKode``,
        and the four input expenditure columns in :data:`INPUT_COLUMNS`.
        Must be filtered to ``RegnskapstypeKode == 'R'`` upstream.
    price_index_df : pandas.DataFrame
        SSB price indices indexed by ``year`` with columns matching
        :data:`INPUT_COLUMNS`. Each value is the sector price index
        (any normalization; only relative levels matter for GARP).

    Returns
    -------
    dict[str, FirmBundle]
        Map ``orgnr -> FirmBundle``. Firms with fewer than two
        admitted years are excluded — GARP requires at least two
        observations.

    Notes
    -----
    A firm-year is admitted iff all four expenditure columns are
    present and finite and non-negative. Implicit quantities are
    ``q_kt = expenditure_kt / p_kt``. Years are sorted ascending.
    """
    required = {"organisasjonsnummer", "Regnskapsar", *INPUT_COLUMNS}
    missing = required - set(finstat_df.columns)
    if missing:
        raise ValueError(f"finstat_df missing columns: {sorted(missing)}")
    if not set(INPUT_COLUMNS).issubset(price_index_df.columns):
        raise ValueError(
            "price_index_df missing one of "
            f"{INPUT_COLUMNS}; has {price_index_df.columns.tolist()}"
        )

    clean = finstat_df.copy()
    for col in INPUT_COLUMNS:
        clean = clean[clean[col].notna() & (clean[col] >= 0)]
    clean = clean[clean["Regnskapsar"].notna()]
    clean["Regnskapsar"] = clean["Regnskapsar"].astype(int)
    clean = clean.sort_values(["organisasjonsnummer", "Regnskapsar"])

    bundles_by_firm: dict[str, FirmBundle] = {}
    for orgnr, group in clean.groupby("organisasjonsnummer", sort=False):
        years = group["Regnskapsar"].tolist()
        if len(years) < 2:
            continue
        try:
            price_rows = price_index_df.loc[years, list(INPUT_COLUMNS)]
        except KeyError:
            continue
        if price_rows.isna().any().any() or (price_rows <= 0).any().any():
            continue
        expenditures = group[list(INPUT_COLUMNS)].to_numpy(dtype=np.float64)
        prices = price_rows.to_numpy(dtype=np.float64)
        quantities = expenditures / prices
        bundles_by_firm[str(orgnr)] = FirmBundle(
            orgnr=str(orgnr),
            years=years,
            bundles=quantities,
            prices=prices,
            expenditures=expenditures,
            input_names=INPUT_COLUMNS,
        )
    return bundles_by_firm
