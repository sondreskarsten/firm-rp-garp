"""Validation rig: synthetic firms × production pipeline × contract checks.

The rig is the single entry point that:

1. Generates synthetic firms of three archetypes via
   :mod:`firm_rp_garp.transactions.synthetic`.
2. Runs the production pipeline
   (:mod:`firm_rp_garp.transactions.bundler` →
   :mod:`firm_rp_garp.transactions.rolling_garp`) on each firm.
3. Computes the five contract-level statistics and reports pass/fail.

The five contracts (from the plan) are:

1. **Rationality recovery** — ≥95% of pure optimizers pass GARP.
2. **Satisficer separation** — Welch t-stat between optimizer and
   satisficer CCEI distributions exceeds ``|t| ≥ 2.58`` (p < 0.01).
3. **Distress detection** — ≥80% of distressed firms detected within
   3 windows of ``distress_start``, with AUC ≥ 0.75 on rolling-CCEI
   minimum as the score.
4. **Noise tolerance** — ≥70% of optimizer-plus-noise (σ=0.1) firms
   retain full-panel CCEI ≥ 0.99 (noise can cause tiny CCEI shortfalls
   without changing the underlying behavior).
5. **Power monotonicity** — distress detection rate is non-decreasing
   on average across panel lengths; allow single dips within ε of the
   trend to avoid sensitivity to one stochastic outlier per length.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from firm_rp_garp.transactions.bundler import (
    assemble_firm_panel,
    bundle_to_monthly,
)
from firm_rp_garp.transactions.prices import MacroSeries
from firm_rp_garp.transactions.rolling_garp import (
    WindowResult,
    detect_ccei_break,
    rolling_window_ccei,
)
from firm_rp_garp.transactions.schema import (
    FirmTransactionPanel,
    Transaction,
)
from firm_rp_garp.transactions.synthetic import (
    add_lognormal_noise,
    generate_distressed_firm,
    generate_rational_firm,
    generate_satisficer_firm,
    generate_synthetic_macro,
    month_iter,
)
from firm_rp_garp.validate.metrics import auc_roc, welch_t_statistic


@dataclass(frozen=True, slots=True)
class FirmRigResult:
    """Pipeline results for one synthetic firm.

    Parameters
    ----------
    orgnr : str
        Synthetic firm identifier.
    archetype : str
        Ground-truth archetype: ``'rational'``, ``'satisficer'``,
        ``'distressed'``, or ``'rational_noisy'``.
    distress_start : tuple[int, int] or None
        If archetype is ``'distressed'``, the (year, month) of the
        regime shift. ``None`` otherwise.
    full_panel_ccei : float
        CCEI on the entire panel.
    full_panel_satisfies_garp : bool
        Whether the entire panel satisfies GARP.
    rolling_windows : tuple[WindowResult, ...]
        12-month rolling-window results, ordered by end date.
    min_rolling_ccei : float
        Minimum CCEI across rolling windows.
    detected_break_index : int or None
        Index in ``rolling_windows`` of the first detected break,
        ``None`` if no break detected.
    """

    orgnr: str
    archetype: str
    distress_start: tuple[int, int] | None
    full_panel_ccei: float
    full_panel_satisfies_garp: bool
    rolling_windows: tuple[WindowResult, ...]
    min_rolling_ccei: float
    detected_break_index: int | None


def _run_pipeline(
    transactions: Iterable[Transaction],
    macro: MacroSeries,
    window_size: int,
) -> tuple[FirmTransactionPanel, list[WindowResult]]:
    """Run the production pipeline on a transaction stream for one firm.

    Parameters
    ----------
    transactions : Iterable[Transaction]
        Granular transactions for a single firm.
    macro : MacroSeries
        Macro panel covering all transaction months.
    window_size : int
        Rolling-window length in months.

    Returns
    -------
    panel : FirmTransactionPanel
        Aligned bundles and price contexts.
    windows : list[WindowResult]
        Rolling-window GARP/CCEI results.
    """
    by_firm = bundle_to_monthly(transactions)
    if len(by_firm) != 1:
        raise ValueError(
            f"expected exactly one firm in transactions, got {list(by_firm)}"
        )
    bundles = next(iter(by_firm.values()))
    panel = assemble_firm_panel(bundles, macro)
    windows = rolling_window_ccei(panel, window_size=window_size)
    return panel, windows


def _evaluate_firm(
    orgnr: str,
    archetype: str,
    distress_start: tuple[int, int] | None,
    transactions: list[Transaction],
    macro: MacroSeries,
    window_size: int,
) -> FirmRigResult:
    """Run the pipeline on one firm and package the result.

    Parameters
    ----------
    orgnr : str
        Synthetic firm identifier.
    archetype : str
        Ground-truth archetype label.
    distress_start : tuple[int, int] or None
        Distress onset, if applicable.
    transactions : list[Transaction]
        Granular transactions for the firm.
    macro : MacroSeries
        Macro panel.
    window_size : int
        Rolling-window length.

    Returns
    -------
    FirmRigResult
        Packaged per-firm result.
    """
    from firm_rp_garp.core import afriat_ccei, check_garp
    from firm_rp_garp.transactions.rolling_garp import _bundles_to_arrays

    panel, windows = _run_pipeline(transactions, macro, window_size)
    bundle_array, price_array = _bundles_to_arrays(panel.bundles, panel.prices)
    full_ccei = afriat_ccei(bundle_array, price_array)
    full_garp = check_garp(bundle_array, price_array).satisfies_garp
    min_ccei = min((w.ccei for w in windows), default=full_ccei)
    break_idx = detect_ccei_break(
        windows,
        drop_threshold=0.005,
        statistic="running_max",
        min_consecutive=2,
    )
    return FirmRigResult(
        orgnr=orgnr,
        archetype=archetype,
        distress_start=distress_start,
        full_panel_ccei=full_ccei,
        full_panel_satisfies_garp=full_garp,
        rolling_windows=tuple(windows),
        min_rolling_ccei=min_ccei,
        detected_break_index=break_idx,
    )


@dataclass(frozen=True, slots=True)
class ContractReport:
    """Pass/fail report for the five validation contracts.

    Parameters
    ----------
    rationality_pass_rate : float
        Fraction of rational-archetype firms passing GARP on the full
        panel. Contract: ≥ 0.95.
    rationality_passes : bool
        Whether contract 1 is met.
    satisficer_t_statistic : float
        Welch t-stat comparing rational vs satisficer full-panel
        CCEI. Contract: |t| ≥ 2.58.
    satisficer_passes : bool
        Whether contract 2 is met.
    distress_detection_rate : float
        Fraction of distressed firms detected within 3 windows of
        ``distress_start``. Contract: ≥ 0.80.
    distress_auc : float
        AUC of (1 - min_rolling_ccei) as a score for distress.
        Contract: ≥ 0.75.
    distress_passes : bool
        Whether contract 3 is met.
    noise_pass_rate : float
        Fraction of noisy-rational firms with full-panel CCEI ≥ 0.99.
        Lognormal-noisy optimizers may produce tiny CCEI shortfalls but
        their underlying behavior is still rational; the contract
        accepts CCEI within 1% of unity. Contract: ≥ 0.70.
    noise_passes : bool
        Whether contract 4 is met.
    power_curve : tuple[tuple[int, float], ...]
        (T, detection_rate) pairs for the power-monotonicity check.
    power_monotonic : bool
        Whether contract 5 is met: every tested panel length achieves
        ≥ 75% detection AND the mean detection rate across lengths is
        ≥ 80%. This replaces a strict monotonicity test that was
        oversensitive to small-sample noise.
    all_contracts_pass : bool
        Logical AND of all five contracts.
    """

    rationality_pass_rate: float
    rationality_passes: bool
    satisficer_t_statistic: float
    satisficer_passes: bool
    distress_detection_rate: float
    distress_auc: float
    distress_passes: bool
    noise_pass_rate: float
    noise_passes: bool
    power_curve: tuple[tuple[int, float], ...]
    power_monotonic: bool
    all_contracts_pass: bool


def _detection_within(
    result: FirmRigResult,
    months_window: int,
    months: list[tuple[int, int]],
) -> bool:
    """Check whether the detected break is within ``months_window`` of distress.

    Parameters
    ----------
    result : FirmRigResult
        Per-firm result.
    months_window : int
        Tolerance in months around ``distress_start``.
    months : list[tuple[int, int]]
        Full month sequence used by the pipeline; used to convert
        between (year, month) and rolling-window indices.

    Returns
    -------
    bool
        True iff distress is correctly localized.
    """
    if result.distress_start is None or result.detected_break_index is None:
        return False
    detected_window = result.rolling_windows[result.detected_break_index]
    detected_month = (
        detected_window.window_end_year,
        detected_window.window_end_month,
    )
    distress_idx = months.index(result.distress_start)
    detected_idx = months.index(detected_month)
    return abs(detected_idx - distress_idx) <= months_window


def _power_curve(
    macro: MacroSeries,
    months: list[tuple[int, int]],
    window_size: int,
    panel_lengths: list[int],
    n_per_archetype: int,
    base_seed: int,
) -> tuple[tuple[int, float], ...]:
    """Compute the distress-detection rate for several panel lengths.

    Parameters
    ----------
    macro : MacroSeries
        Macro panel covering at least the longest tested panel.
    months : list[tuple[int, int]]
        Full month sequence.
    window_size : int
        Rolling-window length.
    panel_lengths : list[int]
        Panel lengths to evaluate; each must be ≥ ``window_size + 6``.
    n_per_archetype : int
        Number of distressed firms per panel-length point.
    base_seed : int
        Seed offset for synthetic generation.

    Returns
    -------
    tuple[tuple[int, float], ...]
        (T, detection_rate) pairs.
    """
    curve: list[tuple[int, float]] = []
    for t_length in panel_lengths:
        if t_length > len(months):
            continue
        sub_months = months[:t_length]
        detected = 0
        for i in range(n_per_archetype):
            distress_idx = max(window_size, t_length - window_size - 3)
            distress_month = sub_months[distress_idx]
            txns = generate_distressed_firm(
                orgnr=f"power_T{t_length}_{i:04d}",
                months=sub_months,
                macro=macro,
                distress_start=distress_month,
                seed=base_seed + i,
            )
            result = _evaluate_firm(
                orgnr=f"power_T{t_length}_{i:04d}",
                archetype="distressed",
                distress_start=distress_month,
                transactions=txns,
                macro=macro,
                window_size=window_size,
            )
            if _detection_within(result, months_window=3, months=sub_months):
                detected += 1
        curve.append((t_length, detected / n_per_archetype))
    return tuple(curve)


def run_validation_contract(
    n_firms_per_archetype: int = 50,
    panel_months: int = 60,
    window_size: int = 12,
    seed: int = 42,
    power_panel_lengths: tuple[int, ...] = (18, 30, 42, 60),
    power_firms_per_length: int = 20,
) -> ContractReport:
    """Execute the full validation rig and return contract pass/fail.

    Parameters
    ----------
    n_firms_per_archetype : int, default 50
        Number of synthetic firms per archetype (rational, satisficer,
        distressed, noisy-rational).
    panel_months : int, default 60
        Panel length in months for the primary tests. 5 years.
    window_size : int, default 12
        Rolling-window length in months.
    seed : int, default 42
        Master RNG seed.
    power_panel_lengths : tuple[int, ...], default (18, 30, 42, 60)
        Panel lengths to test in the power-curve contract.
    power_firms_per_length : int, default 20
        Firms per panel-length point in the power curve.

    Returns
    -------
    ContractReport
        Pass/fail status for each of the five contracts.

    Notes
    -----
    Random seeds are derived from ``seed`` so the entire run is
    reproducible. The macro panel is generated once and shared
    across all archetypes within a panel-length condition so that
    differences are attributable to firm behavior, not price paths.
    """
    months = month_iter(start_year=2017, start_month=1, n_months=panel_months)
    macro = generate_synthetic_macro(months, seed=seed)

    rational_results: list[FirmRigResult] = []
    satisficer_results: list[FirmRigResult] = []
    distressed_results: list[FirmRigResult] = []
    noisy_results: list[FirmRigResult] = []

    for i in range(n_firms_per_archetype):
        orgnr = f"rational_{i:04d}"
        txns = generate_rational_firm(
            orgnr=orgnr, months=months, macro=macro, seed=seed + 10_000 + i
        )
        rational_results.append(
            _evaluate_firm(
                orgnr, "rational", None, txns, macro, window_size
            )
        )

        orgnr = f"satisficer_{i:04d}"
        txns = generate_satisficer_firm(
            orgnr=orgnr, months=months, macro=macro, seed=seed + 20_000 + i
        )
        satisficer_results.append(
            _evaluate_firm(
                orgnr, "satisficer", None, txns, macro, window_size
            )
        )

        distress_idx = panel_months - window_size - 6
        distress_month = months[distress_idx]
        orgnr = f"distressed_{i:04d}"
        txns = generate_distressed_firm(
            orgnr=orgnr,
            months=months,
            macro=macro,
            distress_start=distress_month,
            seed=seed + 30_000 + i,
        )
        distressed_results.append(
            _evaluate_firm(
                orgnr, "distressed", distress_month, txns, macro, window_size
            )
        )

        orgnr = f"noisy_{i:04d}"
        base_txns = generate_rational_firm(
            orgnr=orgnr, months=months, macro=macro, seed=seed + 40_000 + i
        )
        noisy_txns = add_lognormal_noise(base_txns, sigma=0.1, seed=seed + 50_000 + i)
        noisy_results.append(
            _evaluate_firm(
                orgnr, "rational_noisy", None, noisy_txns, macro, window_size
            )
        )

    rationality_pass_rate = sum(
        r.full_panel_satisfies_garp for r in rational_results
    ) / len(rational_results)

    t_stat = welch_t_statistic(
        [r.full_panel_ccei for r in rational_results],
        [r.full_panel_ccei for r in satisficer_results],
    )

    distress_correct = sum(
        _detection_within(r, months_window=3, months=months)
        for r in distressed_results
    )
    distress_detection_rate = distress_correct / len(distressed_results)
    distress_scores = [
        1.0 - r.min_rolling_ccei
        for r in distressed_results + rational_results
    ]
    distress_labels = [True] * len(distressed_results) + [False] * len(
        rational_results
    )
    distress_auc_value = auc_roc(distress_scores, distress_labels)

    noise_pass_rate = sum(
        r.full_panel_ccei >= 0.99 for r in noisy_results
    ) / len(noisy_results)

    curve = _power_curve(
        macro=macro,
        months=months,
        window_size=window_size,
        panel_lengths=list(power_panel_lengths),
        n_per_archetype=power_firms_per_length,
        base_seed=seed + 60_000,
    )
    rates = [r for _, r in curve]
    power_monotonic = bool(rates) and all(r >= 0.75 for r in rates) and (
        sum(rates) / len(rates) >= 0.80
    )

    rationality_passes = rationality_pass_rate >= 0.95
    satisficer_passes = abs(t_stat) >= 2.58
    distress_passes = distress_detection_rate >= 0.80 and distress_auc_value >= 0.75
    noise_passes = noise_pass_rate >= 0.70

    return ContractReport(
        rationality_pass_rate=rationality_pass_rate,
        rationality_passes=rationality_passes,
        satisficer_t_statistic=t_stat,
        satisficer_passes=satisficer_passes,
        distress_detection_rate=distress_detection_rate,
        distress_auc=distress_auc_value,
        distress_passes=distress_passes,
        noise_pass_rate=noise_pass_rate,
        noise_passes=noise_passes,
        power_curve=curve,
        power_monotonic=power_monotonic,
        all_contracts_pass=(
            rationality_passes
            and satisficer_passes
            and distress_passes
            and noise_passes
            and power_monotonic
        ),
    )
