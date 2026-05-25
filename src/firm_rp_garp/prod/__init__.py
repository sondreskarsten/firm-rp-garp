"""Production-side revealed-preference tests on Norwegian firm regnskap data.

This subpackage builds (input-expenditure, sector-price-index) bundles
per AS firm-year from the finstat panel and applies the core
:mod:`firm_rp_garp.core` algorithms.

The bundle construction follows Varian (1984): for each year, the firm
chooses an input mix to produce a given output, and WACM requires
cost minimization at observed input prices.

Modules
-------
:mod:`firm_rp_garp.prod.bundles` : construct (x, p) bundles from finstat.
:mod:`firm_rp_garp.prod.runner`  : iterate the core RP tests over firms.
"""
