# firm-rp-garp

GARP / CCEI / Money-Pump-Index diagnostics on Norwegian AS firms.
Per-firm revealed-preference consistency scores as a model-free distress
signal. ASA firms are excluded (already well-monitored via Oslo Børs
disclosure and analyst coverage — no marginal information).

Stage 1 of the `firm-rp-*` family. Sibling modules planned:
`firm-rp-panel` (behavioral feature panel), `firm-rp-ccp` (Hotz-Miller
CCP), `firm-rp-irl` (MaxEnt-IRL).

## Two complementary RP tests, both on AS population

### A. Production-side (`src/firm_rp_garp/prod/`)

Varian's WACM/WAPM applied to firm production technology. Tests whether
observed (output, input-expenditure, capital-stock) bundles satisfy
cost-minimization at sector input price indices.

Data: finstat annual panel + SSB sector price indices.
Cadence: yearly per firm.
Output: per-firm WACM/WAPM violation set, Afriat CCEI, time series.

### B. Event-driven cash-allocation (`src/firm_rp_garp/events/`)

GARP on the firm's cash-allocation choices observed via discrete corporate
events. Each event is a dated revealed choice about deployment of cash.

Data:
- `aksjeeierbok` — share issuance, share retirement, ownership transfers
- `kunngjoring` — kapitalforhøyelse, kapitalnedsettelse, fusjon, fisjon,
  utbytte, omdanning
- `enheter` — status changes, address changes, role changes
- `finstat` — annual balance for normalization

Cadence: event-driven per firm.
Output: per-firm event sequence, GARP violations on consecutive event
windows, CCEI trajectory.

## Why these two

Production-side covers the operational efficiency question
(is the firm running its production technology consistently?).
Event-driven covers the capital structure question (is the firm making
financing decisions consistently?). Distress can show up in either or
both.

## Stop-rules

Validate each test against historical konkurs labels:
- AUC improvement of ≥0.03 over a regnskap-only baseline → productionize
- Below 0.03 → drop the module

The IRL paper's stop-rule applies symmetrically to both tests.

## Conventions

- Python 3.13. 100% docstring coverage. Type hints throughout.
- All public functions have NumPy-style docstrings.
- Tests via pytest. Coverage target ≥80% on the algorithmic core
  (GARP solver, CCEI bisection, MPI cycle enumeration).
- No try/except unless the failure is a documented part of the contract.
- GCS reads via `arrow-duckdb-gcs` skill. No local data downloads.
- Every GCS write co-located with the generating script.

## References

- Afriat (1967, IER 8:67–77) — original GARP construction
- Varian (1982, ECTA 50:945–973; 1984, JE 30:445–458) — firm WACM/WAPM
- Echenique, Lee & Shum (2011, JPE 119:1201–1223) — money pump index
- Dean & Martin (2016, REStat 98:524–534) — minimum cost index
- Polisson & Quah (2024, arXiv:2406.10136) — CCEI cost-rationalizability
- Choi, Kariv, Müller, Silverman (2014, AER 104:1518–1550) — empirical CCEI
- Cao, Cohen & Szpruch (2021, NeurIPS) — IRL identifiability
