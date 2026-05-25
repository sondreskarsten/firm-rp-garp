"""Revealed-preference diagnostics on Norwegian AS firm data.

This package implements the GARP (Generalized Axiom of Revealed Preference)
test of Afriat (1967), Varian's WACM/WAPM extensions for firm production
data (Varian 1984), the Critical Cost Efficiency Index (Afriat 1973), and
the Money Pump Index (Echenique, Lee & Shum 2011).

Two complementary tests:
- :mod:`firm_rp_garp.prod` — production-side WACM/WAPM on finstat annual
  bundles.
- :mod:`firm_rp_garp.events` — event-driven cash-allocation GARP using
  aksjeeierbok, kunngjoring, and enheter event streams.

The core revealed-preference algorithms live in :mod:`firm_rp_garp.core`
and are independent of the data source.
"""

__version__ = "0.1.0"
