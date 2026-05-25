"""Validation rig for the transaction-level RP pipeline.

The rig at :mod:`firm_rp_garp.validate.rig` instantiates synthetic
firms with known behavior, runs the production pipeline on them, and
checks five validation contracts before the pipeline is declared
fit for real bank data.
"""
