# Finding 01: economy-wide SSB prices yield too few GARP violations

## Setup

- 500 random Norwegian AS firms with full 2017-2023 history in finstat
- 4-dimensional input bundle: Lonnskostnad, Varekostnad,
  AnnenDriftskostnad, AvskrivVarigeDriftsmidl
- Prices from SSB economy-wide indices: monthly earnings (11418),
  PPI total domestic (12463), CPI all-item (03013), PPI capital goods
  E2 (12463)

## Result

| Metric | Value |
|---|---|
| Satisfies GARP | 475/500 (95.0%) |
| Mean CCEI | 0.9998 |
| Median CCEI | 1.0000 |
| Min CCEI | 0.9824 |
| Firms with CCEI < 0.9 | 0 |
| Mean violations per firm | 0.1 |

## Interpretation: the result is suspicious

The Choi et al. (2014, AER 104:1518) lab benchmark, with controlled
prices and student subjects, found:
- 22.8% zero-violation rate
- Mean CCEI = 0.881

We are getting **95.0%** zero-violation and **CCEI ≈ 1.0** on real
Norwegian firms. Either Norwegian SMEs are dramatically more rational
than American students (implausible), or the GARP test has low power
in this setup.

## Four hypotheses for why the test has low power

### H1: economy-wide prices are too smooth

The four SSB indices all rose 25-66% over 2017-2023:

| Index | 2017 | 2023 | Change |
|---|---|---|---|
| Wage | 100.0 | 126.4 | +26% |
| PPI total | 100.0 | 165.6 | +66% |
| CPI | 100.0 | 122.8 | +23% |
| Capital goods PPI | 100.0 | 128.8 | +29% |

They are not independent — they all reflect the same macroeconomic
inflation environment. GARP power comes from *relative* price changes
between inputs; when relative prices are stable, the test has nothing
to detect.

### H2: implicit-quantity construction is degenerate

We compute ``q_kt = expenditure_kt / p_kt`` and feed (q, p) to GARP.
When prices p_t scale uniformly, q_t just rescales — the relative
shares of (q_1, q_2, q_3, q_4) are preserved, so the firm "appears"
to make identical real choices in every year. This is a mechanical
near-tautology, not an empirical regularity about firm rationality.

### H3: yearly aggregation smooths within-year adjustment

A firm responding to a Q2 commodity shock by Q4 looks consistent
in annual data even if the within-year path was wildly non-optimal.

### H4: 4-category aggregation hides relative-price moves

"Varekostnad" combines all material inputs into one number. Two firms
in different industries face very different "material prices" — a fish
processor and a software firm shouldn't be priced with the same PPI.

## Next steps to test which hypothesis matters most

1. **Use sector-specific PPI.** Table 12463 has 60 industry codes
   (NaringUtenriks); match each firm to its NACE 2-digit input PPI
   instead of total PPI. If sector PPI moves more independently than
   total PPI, GARP should fire more.

2. **Add NIBOR / cost of capital as a 5th input price.** Interest
   rates moved very differently from goods prices over 2017-2023
   (NIBOR was near zero 2017-2021, then 4%+ by 2023). Adding interest
   cost as a separate input dimension should expose intertemporal
   inconsistency in financing choices.

3. **Bootstrap test against synthetic random allocators.** Compare
   the firm CCEI distribution to a null where firms randomly allocate
   total expenditure across the four inputs each year. If our CCEI is
   not statistically distinguishable from random, the test is
   uninformative; if it is well above random, even high CCEIs may
   carry distress signal.

4. **Validate against konkurs labels.** Even if mean CCEI is high, the
   *bottom tail* (CCEI < 0.99) may predict default. The IRL paper's
   stop-rule applies: AUC ≥ 0.03 over a regnskap-only baseline
   justifies productionizing. Run that validation before tuning.

## Decision

Do not abandon the framework yet. The pipeline works end-to-end. The
empirical signal is weak with economy-wide prices, but the 25 firms
with non-trivial CCEI (0.98-0.9999 range) may already contain the
distress signal we want. Next step is the konkurs validation harness
to test that.
