"""Reproduce the validation rig and report all five contract outcomes.

This script runs the full transaction-level RP pipeline against the
synthetic ground-truth archetypes from
:mod:`firm_rp_garp.transactions.synthetic` and prints the
contract-by-contract pass/fail summary.

Expected output at the parameters below (seed=42):

    C1 rationality ≥95%:        pass=100%  OK
    C2 |t|≥2.58:                t=11.48    OK
    C3 detect ≥80% AUC ≥0.75:   detect=100% AUC=1.000  OK
    C4 noisy CCEI≥0.99 ≥70%:    pass=100%  OK
    C5 min ≥75% AND mean ≥80%:  curve ranges 87-100%  OK
    ALL CONTRACTS PASS: True

Runtime: ~90s on a typical laptop. Scaling: O(n_firms × T^3) for the
core algorithms inside each rolling window.
"""
from __future__ import annotations

import time

from firm_rp_garp.validate.rig import run_validation_contract


def main() -> None:
    """Run the rig and print the contract report."""
    t0 = time.time()
    report = run_validation_contract(
        n_firms_per_archetype=25,
        panel_months=60,
        window_size=12,
        seed=42,
        power_panel_lengths=(18, 30, 42, 60),
        power_firms_per_length=15,
    )
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s\n")

    def ok(b: bool) -> str:
        """Format a contract pass/fail label."""
        return "OK" if b else "FAIL"

    print(
        f"C1 rationality ≥95%:        "
        f"pass={report.rationality_pass_rate:.0%}  "
        f"{ok(report.rationality_passes)}"
    )
    print(
        f"C2 |t|≥2.58:                "
        f"t={report.satisficer_t_statistic:.2f}  "
        f"{ok(report.satisficer_passes)}"
    )
    print(
        f"C3 detect ≥80% AUC ≥0.75:   "
        f"detect={report.distress_detection_rate:.0%} "
        f"AUC={report.distress_auc:.3f}  "
        f"{ok(report.distress_passes)}"
    )
    print(
        f"C4 noisy CCEI≥0.99 ≥70%:    "
        f"pass={report.noise_pass_rate:.0%}  "
        f"{ok(report.noise_passes)}"
    )
    print(
        f"C5 min ≥75% AND mean ≥80%:  "
        f"curve={[(t, f'{r:.0%}') for t, r in report.power_curve]}  "
        f"{ok(report.power_monotonic)}"
    )
    print()
    print(f"ALL CONTRACTS PASS: {report.all_contracts_pass}")


if __name__ == "__main__":
    main()
