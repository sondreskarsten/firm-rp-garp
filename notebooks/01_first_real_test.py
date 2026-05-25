"""First end-to-end test of production-side GARP/CCEI on real Norwegian firms.

This script reproduces the first empirical run of the production-side
pipeline. It pulls 500 AS firms with full 2017-2023 history from
finstat, fetches SSB price indices (CPI, PPI, wages, capital goods PPI),
and runs the GARP / CCEI / MPI battery from
:mod:`firm_rp_garp.prod.runner`.

Key empirical finding
---------------------
With economy-wide SSB price indices applied to 500 random AS firms over
2017-2023:

- 95.0% of firms satisfy GARP (0 violations)
- Mean CCEI = 0.9998 (median 1.0)
- Min CCEI = 0.9824

This is **far above** the Choi et al. (2014, AER 104:1518) lab benchmark
of 22.8% zero-violation rate and mean CCEI 0.881.

Why so clean? Three honest explanations to investigate:

1. **Price variation too smooth.** Wages +26%, materials +66%, opex
   (CPI) +23%, capital +29% over 2017-2023. The four indices move in
   the same direction — limited relative-price variation reduces GARP
   power.

2. **Implicit-quantity construction degenerate.** When q = expenditure
   / price and all prices rise together, q just scales proportionally
   — no inconsistent re-allocation pattern can emerge.

3. **Yearly granularity too coarse.** Within-year price spikes are
   smoothed out by annual aggregation.

4. **Bundle too aggregated.** Four macro categories
   (Lonnskostnad, Varekostnad, AnnenDriftskostnad, AvskrivVarigeDriftsmidl)
   may bundle items whose individual prices moved very differently.

Next investigation: pull *sector-specific* PPI (table 12463 has 60
industries) and merge each firm to its NACE-specific input price.
The economy-wide PPI may be hiding sector-level relative-price moves.
"""
from __future__ import annotations

import os
import sys

import pandas as pd


# Bootstrap the GCS DuckDB skill in __main__ scope
exec(open("/mnt/skills/user/arrow-duckdb-gcs/scripts/bootstrap.py").read())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from firm_rp_garp.prod.bundles import build_firm_bundles  # noqa: E402
from firm_rp_garp.prod.runner import run_panel  # noqa: E402
from firm_rp_garp.prod.ssb import build_price_index_panel  # noqa: E402


def pull_firm_panel(n_firms: int = 500, year_min: int = 2017, year_max: int = 2023) -> pd.DataFrame:
    """Pull a finstat panel for AS firms with at least 5 admitted years.

    Parameters
    ----------
    n_firms : int, default 500
        Number of firms to include.
    year_min : int, default 2017
        Earliest year (inclusive).
    year_max : int, default 2023
        Latest year (inclusive).

    Returns
    -------
    pandas.DataFrame
        Finstat slice with columns ``organisasjonsnummer``, ``Regnskapsar``,
        ``RegnskapstypeKode``, and the four input expenditure columns.
    """
    candidates = q(f"""  # noqa: F821
        SELECT organisasjonsnummer, COUNT(*) AS n_years
        FROM finstat
        WHERE RegnskapstypeKode = 'R'
          AND Regnskapsar BETWEEN {year_min} AND {year_max}
          AND Lonnskostnad > 0 AND Varekostnad > 0
          AND AnnenDriftskostnad IS NOT NULL
          AND AvskrivVarigeDriftsmidl IS NOT NULL
        GROUP BY organisasjonsnummer
        HAVING COUNT(*) >= 5
        LIMIT {n_firms}
    """)
    orgnr_list = candidates["organisasjonsnummer"].tolist()
    return q(f"""  # noqa: F821
        SELECT organisasjonsnummer, Regnskapsar, RegnskapstypeKode,
               Lonnskostnad, Varekostnad, AnnenDriftskostnad,
               AvskrivVarigeDriftsmidl
        FROM finstat
        WHERE organisasjonsnummer IN ({fmt(orgnr_list)})  # noqa: F821
          AND RegnskapstypeKode = 'R'
          AND Regnskapsar BETWEEN {year_min} AND {year_max}
    """)


if __name__ == "__main__":
    panel = pull_firm_panel(n_firms=500, year_min=2017, year_max=2023)
    print(f"Firm-years: {len(panel)}")
    prices = build_price_index_panel(list(range(2017, 2024)))
    print(f"Price panel: {prices.shape}")
    bundles = build_firm_bundles(panel, prices)
    print(f"Built {len(bundles)} firm bundles")
    results = run_panel(bundles)
    print()
    print(f"=== Results, {len(results)} firms, 2017-2023 ===")
    print(f"  satisfies_garp: {results['satisfies_garp'].sum()}/{len(results)} ({results['satisfies_garp'].mean():.1%})")
    print(f"  CCEI mean:      {results['ccei'].mean():.4f}")
    print(f"  CCEI min:       {results['ccei'].min():.4f}")
    print(f"  CCEI < 1.0:     {(results['ccei'] < 0.9999).sum()} firms")
