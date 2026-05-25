# Design: GARP on Norwegian firm regnskap data

## The problem with applying Varian directly

Varian's WACM requires quantities × prices observed separately. Finstat
provides expenditures (`Lonnskostnad`, `Varekostnad`, `AnnenDriftskostnad`)
which are p × x bundled, not p and x separately.

## Three approaches, in order of deployment difficulty

### Approach 1 (this module): Cash-allocation GARP

**Subject**: the firm's deployment of operating cash flow each year across
competing uses: debt service, dividends, capex, retained earnings.

**Choice bundle** at year t:
- y_t = "income" = `Salgsinntekt` + `AnnenDriftsinntekt` + financial income
- choice vector x_t over uses:
  - debt service (interest + principal repayment via `AvdragLangsGjeld`)
  - dividends (`Utbytte`)
  - capex proxy (Δ `SumVarigeDriftsmidl`)
  - retained earnings (Δ `SumEK` − new equity issuance)
  - working-capital change (Δ `SumOmlopsmidler` − Δ `KortsiktigGjeld`)
- "prices" = 1 for each use (cash is fungible at face value) — this is the
  CONSUMER-SIDE setup where the budget constraint is the cash flow and the
  utility is over allocation shares.

**Test**: GARP on the (share-vector, opportunity-cost) bundles where the
opportunity cost of each use varies year-to-year (e.g., when interest rates
are high, debt service is "expensive"; when growth opportunities are good,
capex is "cheap" relative to dividends).

**The price vector** uses external indices:
- p_debt-service = NIBOR + risk premium proxy (or just NIBOR)
- p_capex = capital goods price index (SSB)
- p_dividend = 1 (numeraire) or 1/(1−τ_div) for tax wedge
- p_retained = 1 + r_alt (foregone alternative return)
- p_wc = short-term funding cost

If a firm allocates more to "expensive" uses across periods, that's a
GARP violation = behavioral inconsistency = potential distress.

### Approach 2 (future module: firm-rp-prod): Production-side GARP

WACM / WAPM on the production technology: observed (Salgsinntekt,
input expenditures, capital stock) bundles should satisfy cost-minimization
at sector input price indices.

Needs SSB:
- Sector wage index
- Sector materials cost index (proxy via PPI by NACE)
- Sector capital user cost

This is closer to Varian (1984) but requires sector × year price merges.

### Approach 3 (future module: firm-rp-monthly): Within-year cash-allocation

The IRL paper's actual recommendation: monthly transaction-level bundles
instead of yearly accounting bundles. Needs bank transaction data, which
this repo does not have. Out of scope.

## Choice this module makes

**This module implements Approach 1 only.** Approach 2 will be a sibling
module (`firm-rp-prod`) once the cash-allocation pipeline is validated.

## Empirical claim to test

A firm whose CCEI is high and stable is allocating cash consistently with
some utility function. A firm whose CCEI is low or falling is either:
- changing its preferences (regime shift — growth → mature → distressed),
- constrained (cannot follow its actual preferences because of binding
  constraints), or
- not optimizing at all (lifestyle business; satisficer).

The credit-risk hypothesis: **falling CCEI in the 12-24 months before
konkurs.** The IRL paper's stop-rule applies: AUC improvement ≥0.03 over
regnskap-only baseline justifies productionizing. Below that, drop the
module.

## What this module is NOT

- Not a profit-maximization test (that's Approach 2)
- Not a transaction-level analysis (that's Approach 3)
- Not committed to any particular utility function over allocations
- Not assuming firms are utility maximizers (the CCEI score IS the
  diagnostic of how close they are)

## Anchors from the IRL paper

- Choi et al. (2014, AER 104:1518) — only 22.8% of human subjects show
  zero GARP violations; mean CCEI is 0.881 across 1,182 subjects. Expect
  the firm population to look similar or worse for lifestyle SMEs.
- Polisson & Quah (2024) — CCEI = e means "data is e-rationalizable" =
  firm could only be utility-maximizing if it was wasting (1−e) of its
  budget.
- Echenique-Lee-Shum (2011, JPE 119:1201) — Money Pump Index = dollar
  value extractable from a GARP-violator.
