"""Tests for the rolling-window GARP module."""
from __future__ import annotations

import pytest

from firm_rp_garp.transactions.bundler import assemble_firm_panel
from firm_rp_garp.transactions.prices import MacroSeries
from firm_rp_garp.transactions.rolling_garp import (
    detect_ccei_break,
    evaluate_window,
    rolling_window_ccei,
)
from firm_rp_garp.transactions.schema import (
    MonthlyBundle,
    N_CATEGORIES,
)


def _macro(months: list[tuple[int, int]], ppi_path: list[float]) -> MacroSeries:
    """Build a macro with a custom PPI path; other series held flat."""
    return MacroSeries(
        nibor={m: 0.03 for m in months},
        sector_ppi=dict(zip(months, ppi_path)),
        capital_goods_ppi=dict(zip(months, ppi_path)),
        dividend_tax={m: 0.35 for m in months},
        risk_free={m: 0.02 for m in months},
    )


def _flat_bundles(
    months: list[tuple[int, int]], orgnr: str = "A"
) -> list[MonthlyBundle]:
    """Generate identical bundles for each month — perfectly rational."""
    return [
        MonthlyBundle(
            orgnr=orgnr,
            year=y,
            month=m,
            amounts_nok=tuple([100.0] * N_CATEGORIES),
            n_transactions=1,
        )
        for y, m in months
    ]


class TestEvaluateWindow:
    """Tests for the single-window evaluator."""

    def test_two_observation_window_satisfies_garp(self) -> None:
        """Two identical bundles at identical prices trivially satisfy GARP."""
        months = [(2020, 1), (2020, 2)]
        macro = _macro(months, [100.0, 100.0])
        panel = assemble_firm_panel(_flat_bundles(months), macro)
        result = evaluate_window(panel, 0, 2)
        assert result.satisfies_garp
        assert result.ccei == pytest.approx(1.0)

    def test_window_too_short_raises(self) -> None:
        """A length-1 window raises ValueError."""
        months = [(2020, 1), (2020, 2)]
        macro = _macro(months, [100.0, 105.0])
        panel = assemble_firm_panel(_flat_bundles(months), macro)
        with pytest.raises(ValueError):
            evaluate_window(panel, 0, 1)


class TestRollingWindowCCEI:
    """Tests for the rolling-window driver."""

    def test_panel_shorter_than_window_is_empty(self) -> None:
        """Returns empty list when panel is shorter than window."""
        months = [(2020, 1), (2020, 2)]
        macro = _macro(months, [100.0, 105.0])
        panel = assemble_firm_panel(_flat_bundles(months), macro)
        assert rolling_window_ccei(panel, window_size=3) == []

    def test_rolling_returns_correct_count(self) -> None:
        """A panel of length T yields T - window + 1 results at step 1."""
        from firm_rp_garp.transactions.synthetic import month_iter

        months = month_iter(2020, 1, 24)
        macro = _macro(months, [100.0 + i for i in range(len(months))])
        panel = assemble_firm_panel(_flat_bundles(months), macro)
        results = rolling_window_ccei(panel, window_size=12)
        assert len(results) == 24 - 12 + 1


class TestDetectCCEIBreak:
    """Tests for the rolling-CCEI break detector."""

    def _make_window(self, end_month: int, ccei: float) -> "WindowResult":
        from firm_rp_garp.transactions.rolling_garp import WindowResult

        return WindowResult(
            orgnr="A",
            window_end_year=2020,
            window_end_month=end_month,
            window_size=12,
            satisfies_garp=ccei >= 1.0,
            n_violations=0 if ccei >= 1.0 else 1,
            ccei=ccei,
            mpi_median=0.0,
            mpi_max=0.0,
        )

    def test_no_break_returns_none(self) -> None:
        """Constant high CCEI gives no break."""
        windows = [self._make_window(m, 1.0) for m in range(1, 13)]
        assert detect_ccei_break(windows) is None

    def test_running_max_drop_detected(self) -> None:
        """A CCEI drop is detected at the right index under running_max."""
        ccei_path = [1.0, 1.0, 1.0, 0.9, 0.85, 0.8]
        windows = [self._make_window(m, c) for m, c in zip(range(1, 7), ccei_path)]
        idx = detect_ccei_break(
            windows, drop_threshold=0.05, statistic="running_max"
        )
        assert idx == 3

    def test_min_consecutive_rejects_single_dip(self) -> None:
        """A single-window dip is rejected when min_consecutive >= 2."""
        ccei_path = [1.0, 1.0, 1.0, 0.95, 1.0, 1.0]
        windows = [self._make_window(m, c) for m, c in zip(range(1, 7), ccei_path)]
        idx = detect_ccei_break(
            windows,
            drop_threshold=0.04,
            statistic="running_max",
            min_consecutive=2,
        )
        assert idx is None

    def test_min_consecutive_accepts_sustained_dip(self) -> None:
        """A sustained dip fires when min_consecutive=2."""
        ccei_path = [1.0, 1.0, 1.0, 0.95, 0.94, 0.93]
        windows = [self._make_window(m, c) for m, c in zip(range(1, 7), ccei_path)]
        idx = detect_ccei_break(
            windows,
            drop_threshold=0.04,
            statistic="running_max",
            min_consecutive=2,
        )
        assert idx == 3

    def test_level_rule(self) -> None:
        """The level rule fires when CCEI falls below 1 - threshold."""
        ccei_path = [1.0, 1.0, 0.95, 1.0]
        windows = [self._make_window(m, c) for m, c in zip(range(1, 5), ccei_path)]
        idx = detect_ccei_break(
            windows, drop_threshold=0.04, statistic="level"
        )
        assert idx == 2

    def test_invalid_statistic_raises(self) -> None:
        """An unknown statistic raises ValueError."""
        windows = [self._make_window(1, 1.0)]
        with pytest.raises(ValueError):
            detect_ccei_break(windows, statistic="garbage")  # type: ignore[arg-type]
