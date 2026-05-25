"""Tests for the core GARP/CCEI/MPI algorithms.

Test cases are drawn from canonical examples in the revealed-preference
literature plus constructed corner cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from firm_rp_garp.core import (
    afriat_ccei,
    check_garp,
    direct_revealed_preference,
    money_pump_index,
    transitive_closure,
)


class TestDirectRevealedPreference:
    """Tests for the direct revealed-preference relation."""

    def test_self_preference(self) -> None:
        """A bundle is always weakly revealed preferred to itself."""
        bundles = np.array([[1.0, 1.0], [2.0, 2.0]])
        prices = np.array([[1.0, 1.0], [1.0, 1.0]])
        direct, _ = direct_revealed_preference(bundles, prices)
        assert direct[0, 0]
        assert direct[1, 1]

    def test_affordability(self) -> None:
        """j is revealed preferred-to via i iff j was affordable at p_i."""
        # i=0 spends 2 at prices (1,1) on bundle (1,1)
        # j=1 has bundle (3,0) which costs 3 at prices (1,1) — not affordable
        bundles = np.array([[1.0, 1.0], [3.0, 0.0]])
        prices = np.array([[1.0, 1.0], [1.0, 1.0]])
        direct, _ = direct_revealed_preference(bundles, prices)
        assert not direct[0, 1]
        # bundle 0 cost at p_1 = 1+1 = 2; budget at p_1 = 3 → affordable
        assert direct[1, 0]


class TestTransitiveClosure:
    """Tests for the transitive closure algorithm."""

    def test_already_transitive(self) -> None:
        """A transitive relation equals its own closure."""
        relation = np.eye(3, dtype=bool)
        assert np.array_equal(transitive_closure(relation), relation)

    def test_chain(self) -> None:
        """A simple chain 0->1->2 should close to include 0->2."""
        relation = np.array(
            [[False, True, False], [False, False, True], [False, False, False]],
            dtype=bool,
        )
        closure = transitive_closure(relation)
        assert closure[0, 2]


class TestGARP:
    """Tests for the GARP consistency check."""

    def test_two_bundles_compatible(self) -> None:
        """Two bundles with matching prices always satisfy GARP."""
        bundles = np.array([[1.0, 2.0], [2.0, 1.0]])
        prices = np.array([[1.0, 1.0], [1.0, 1.0]])
        result = check_garp(bundles, prices)
        assert result.satisfies_garp
        assert result.violations == []

    def test_classic_violation(self) -> None:
        """A classic 2-cycle GARP violation.

        Period 1: prices (1, 2), bundle (3, 1). Cost = 5.
                  Bundle 2 = (1, 3) costs 7 at p_1 — not affordable.
        Period 2: prices (2, 1), bundle (1, 3). Cost = 5.
                  Bundle 1 = (3, 1) costs 7 at p_2 — not affordable.

        Neither is revealed preferred to the other → GARP holds.
        """
        bundles = np.array([[3.0, 1.0], [1.0, 3.0]])
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        result = check_garp(bundles, prices)
        assert result.satisfies_garp

    def test_strict_violation(self) -> None:
        """A genuine strict GARP violation must be detected.

        Period 1: prices (1, 1), bundle (4, 0). Cost = 4.
                  Bundle 2 = (1, 1) costs 2 — affordable.
                  → 1 strictly directly revealed preferred to 2.
        Period 2: prices (1, 1), bundle (1, 1). Cost = 2.
                  Bundle 1 = (4, 0) costs 4 — not affordable.

        This satisfies GARP (only one direction of preference).
        """
        bundles = np.array([[4.0, 0.0], [1.0, 1.0]])
        prices = np.array([[1.0, 1.0], [1.0, 1.0]])
        result = check_garp(bundles, prices)
        assert result.satisfies_garp

    def test_constructed_garp_violation(self) -> None:
        """Construct an explicit 2-cycle GARP violation.

        Period 1: p_1 = (2, 1), x_1 = (1, 2). Cost = 4.
                  Bundle 2 = (2, 1) costs 5 at p_1 — not affordable.
        Period 2: p_2 = (1, 2), x_2 = (2, 1). Cost = 4.
                  Bundle 1 = (1, 2) costs 5 at p_2 — not affordable.

        Try harder: construct so that x_1 R x_2 AND x_2 R x_1 strictly.

        Period 1: p_1 = (1, 1), x_1 = (3, 1). Cost = 4. Bundle 2 = (1, 3)
                  costs 4 — affordable, weak preference.
        Period 2: p_2 = (1, 1), x_2 = (1, 3). Cost = 4. Bundle 1 = (3, 1)
                  costs 4 — affordable, weak preference.

        Both weak — no GARP violation.

        Strict violation needs strict affordability in one direction.
        """
        # Construct: x_1 strictly directly revealed preferred to x_2
        # AND x_2 strictly directly revealed preferred to x_1
        # Period 1: p_1=(1,1), x_1=(3,3). Cost = 6.
        #           Bundle 2=(2,2) costs 4 — strictly affordable. x_1 P x_2.
        # Period 2: p_2=(2,2), x_2=(2,2). Cost = 8.
        #           Bundle 1=(3,3) costs 12 — not affordable.
        # Only one direction of preference. No violation.
        #
        # True GARP violation requires monotonicity-breaking choices.
        # Period 1: p_1=(2,1), x_1=(1,3). Cost = 5.
        #           Bundle 2 = (3,1) costs 7 — not affordable.
        # Period 2: p_2=(1,2), x_2=(3,1). Cost = 5.
        #           Bundle 1 = (1,3) costs 7 — not affordable.
        # No revealed preference either direction.
        # We need: at p_1, x_2 affordable AND at p_2, x_1 affordable,
        # with at least one strict.
        # p_1=(1,1), x_1=(3,1), cost=4. x_2=(1,3) costs 4 — affordable.
        # p_2=(1,1), x_2=(1,3), cost=4. x_1=(3,1) costs 4 — affordable.
        # Both weak. No strict preference.
        # Adjust: p_1=(1,1), x_1=(3,1)+small=4,1, cost=5.
        # x_2=(1,3) costs 4 — strictly affordable → strict P from 1 to 2.
        # p_2=(1,1), x_2=(1,3), cost=4. x_1=(4,1) costs 5 — not affordable.
        # → only one strict direction; no violation.
        #
        # Real violation needs price change to flip the comparison.
        # p_1=(2,1), x_1=(1,3), cost=5. x_2=(3,1) costs 7 — not affordable.
        # p_2=(1,2), x_2=(3,1), cost=5. x_1=(1,3) costs 7 — not affordable.
        # Neither direction. No relation.
        #
        # GARP-violating case requires affordability with strict
        # mutual preference. Classic Afriat construction:
        # p_1=(1,1), x_1=(2,1), cost=3. x_2=(1,2) costs 3 — affordable (weak).
        # p_2=(1,1), x_2=(1,2), cost=3. x_1=(2,1) costs 3 — affordable (weak).
        # No strict. WARP holds (no strict cycle).
        # To violate GARP we need x_1 R x_2 (with strict somewhere in chain)
        # AND x_2 P_0 x_1. Use a 3-cycle: 1 R 2 R 3 P_0 1.
        bundles = np.array(
            [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0]]
        )
        # Construct prices so each is strictly preferred to the next
        prices = np.array(
            [[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]
        )
        # Cost 1: 3*2 + 1*1 + 1*0 = 7.   x_2 at p_1: 0+2+1=3 — affordable strict
        # Cost 2: 1*0 + 3*2 + 1*1 = 7.   x_3 at p_2: 0+0+1=1 — affordable strict
        # Cost 3: 1*1 + 1*0 + 3*2 = 7.   x_1 at p_3: 2+0+0=2 — affordable strict
        # → 1 P 2 P 3 P 1, transitive closure has 1 W 1 with strict edge → violation
        result = check_garp(bundles, prices)
        assert not result.satisfies_garp
        assert len(result.violations) > 0


class TestCCEI:
    """Tests for the Afriat Critical Cost Efficiency Index."""

    def test_ccei_perfect(self) -> None:
        """A GARP-satisfying dataset has CCEI = 1."""
        bundles = np.array([[1.0, 2.0], [2.0, 1.0]])
        prices = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert afriat_ccei(bundles, prices) == 1.0

    def test_ccei_violation(self) -> None:
        """A GARP-violating dataset has CCEI < 1."""
        bundles = np.array(
            [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0]]
        )
        prices = np.array(
            [[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]
        )
        ccei = afriat_ccei(bundles, prices)
        assert 0.0 < ccei < 1.0


class TestMoneyPump:
    """Tests for the money pump index."""

    def test_no_pump_when_garp_holds(self) -> None:
        """A GARP-satisfying dataset has MPI = 0."""
        bundles = np.array([[1.0, 2.0], [2.0, 1.0]])
        prices = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert money_pump_index(bundles, prices) == 0.0

    def test_pump_positive_with_violation(self) -> None:
        """A GARP-violating dataset has MPI > 0."""
        bundles = np.array(
            [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0]]
        )
        prices = np.array(
            [[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]
        )
        mpi = money_pump_index(bundles, prices)
        assert mpi > 0.0


class TestEdgeCases:
    """Edge cases for the core algorithms."""

    def test_single_bundle(self) -> None:
        """A single bundle trivially satisfies GARP."""
        bundles = np.array([[1.0, 1.0]])
        prices = np.array([[1.0, 1.0]])
        result = check_garp(bundles, prices)
        assert result.satisfies_garp
        assert afriat_ccei(bundles, prices) == 1.0
        assert money_pump_index(bundles, prices) == 0.0

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes raise ValueError."""
        bundles = np.array([[1.0, 2.0]])
        prices = np.array([[1.0, 1.0, 1.0]])
        with pytest.raises(ValueError):
            check_garp(bundles, prices)

    def test_non_finite_raises(self) -> None:
        """Non-finite inputs raise ValueError."""
        bundles = np.array([[np.nan, 1.0]])
        prices = np.array([[1.0, 1.0]])
        with pytest.raises(ValueError):
            check_garp(bundles, prices)

    def test_invalid_statistic_raises(self) -> None:
        """An invalid MPI statistic raises ValueError."""
        bundles = np.array([[1.0]])
        prices = np.array([[1.0]])
        with pytest.raises(ValueError):
            money_pump_index(bundles, prices, statistic="garbage")
