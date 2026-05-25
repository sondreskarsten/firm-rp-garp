"""Transaction-level revealed-preference diagnostics for SME firms.

This subpackage implements the IRL paper's actual recommendation:
GARP on monthly cash-allocation bundles aggregated from granular
transaction streams, not on yearly accounting aggregates.

Data flow
---------
1. Granular transactions (from a bank feed or the synthetic rig)
   are emitted as :class:`firm_rp_garp.transactions.schema.Transaction`
   records: dated, counterparty-tagged, MCC-classified.
2. :func:`firm_rp_garp.transactions.bundler.bundle_to_monthly` aggregates
   them into six category bundles per firm-month.
3. :func:`firm_rp_garp.transactions.prices.assemble_price_context`
   assembles the per-month opportunity-cost price vector (NIBOR,
   sector PPI, capital goods PPI, tax wedges).
4. :func:`firm_rp_garp.transactions.rolling_garp.rolling_window_ccei`
   runs the core RP tests over a rolling window per firm.

The synthetic rig at :mod:`firm_rp_garp.transactions.synthetic` generates
ground-truth firms (rational / satisficer / distressed) for validation
before real bank data is wired in.
"""
