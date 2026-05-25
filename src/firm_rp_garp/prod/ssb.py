"""Fetch SSB price and wage indices for the four input-expenditure categories.

The four expenditure columns in
:data:`firm_rp_garp.prod.bundles.INPUT_COLUMNS` are mapped to SSB tables:

================================  ===============================================
Expenditure column                SSB index
================================  ===============================================
Lonnskostnad                      11418 — monthly earnings, all occupations,
                                  all sectors, both sexes, full-time
Varekostnad                       12463 — Producer Price Index, total
                                  (2021=100), domestic and export market
AnnenDriftskostnad                03013 — Consumer Price Index, all-item
                                  (proxy for services costs)
AvskrivVarigeDriftsmidl           12463 NaringUtenriks=E2 — PPI capital
                                  goods aggregate
================================  ===============================================

The CPI (table 03013) is monthly; we aggregate to yearly means. PPI table
12463 and the wage table 11418 are already yearly.

For RP testing only *relative* price levels matter, so each series is
rebased to 100 at the earliest year in the panel.

Notes
-----
This module pulls economy-wide indices. Sector-specific refinement
(matching a firm's NACE to its industry-specific PPI) is left to a
future module. The current mapping is a defensible economy-wide baseline.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

import pandas as pd


PXWEB_BASE = "https://data.ssb.no/api/v0/en/table"


SSB_TABLES: dict[str, dict[str, Any]] = {
    "Lonnskostnad": {
        "table_id": "11418",
        "description": "Monthly earnings, all occupations, all sectors, both sexes, full-time",
        "frequency": "yearly",
        "filters": {
            "MaaleMetode": ["02"],
            "Yrke": ["0-9"],
            "Sektor": ["ALLE"],
            "Kjonn": ["0"],
            "AvtaltVanlig": ["5"],
            "ContentsCode": ["Manedslonn"],
        },
    },
    "Varekostnad": {
        "table_id": "12463",
        "description": "Producer Price Index, total, domestic market (2021=100)",
        "frequency": "yearly",
        "filters": {
            "Marked": ["01"],
            "NaringUtenriks": ["SNN0"],
            "ContentsCode": ["Indeksnivo"],
        },
    },
    "AnnenDriftskostnad": {
        "table_id": "03013",
        "description": "Consumer Price Index, all-item (2015=100)",
        "frequency": "monthly",
        "filters": {
            "Konsumgrp": ["TOTAL"],
            "ContentsCode": ["KpiIndMnd"],
        },
    },
    "AvskrivVarigeDriftsmidl": {
        "table_id": "12463",
        "description": "PPI Capital goods, domestic market (2021=100)",
        "frequency": "yearly",
        "filters": {
            "Marked": ["01"],
            "NaringUtenriks": ["E2"],
            "ContentsCode": ["Indeksnivo"],
        },
    },
}


def _time_values(years: list[int], frequency: str) -> list[str]:
    """Generate SSB time codes for the given years and frequency.

    Parameters
    ----------
    years : list[int]
        Calendar years.
    frequency : {'monthly', 'yearly'}
        SSB frequency code.

    Returns
    -------
    list[str]
        Time codes, e.g. ``['2020', '2021']`` (yearly) or
        ``['2020M01', ..., '2020M12']`` (monthly).
    """
    if frequency == "yearly":
        return [str(year) for year in years]
    if frequency == "monthly":
        return [
            f"{year}M{month:02d}" for year in years for month in range(1, 13)
        ]
    raise ValueError(f"Unsupported frequency: {frequency!r}")


def _build_pxweb_query(spec: dict[str, Any], years: list[int]) -> dict[str, Any]:
    """Build a PxWebApi JSON query body for one SSB table.

    Parameters
    ----------
    spec : dict
        Entry from :data:`SSB_TABLES`.
    years : list[int]
        Years to retrieve.

    Returns
    -------
    dict
        JSON-serializable query body for PxWebApi.
    """
    queries = [
        {
            "code": variable,
            "selection": {"filter": "item", "values": values},
        }
        for variable, values in spec["filters"].items()
    ]
    queries.append(
        {
            "code": "Tid",
            "selection": {
                "filter": "item",
                "values": _time_values(years, spec["frequency"]),
            },
        }
    )
    return {"query": queries, "response": {"format": "json-stat2"}}


def fetch_ssb_index(spec: dict[str, Any], years: list[int]) -> pd.Series:
    """Fetch one SSB index series and aggregate to yearly means.

    Parameters
    ----------
    spec : dict
        Entry from :data:`SSB_TABLES`.
    years : list[int]
        Years to retrieve.

    Returns
    -------
    pandas.Series
        Indexed by year (int). For monthly series, values are the yearly
        mean. For yearly series, the raw yearly value. Years with no
        data are absent from the Series.

    Notes
    -----
    PxWebApi returns ``json-stat2`` format. We parse the ``value`` array
    and the ``Tid`` dimension to construct the series, then group monthly
    series by year.
    """
    query_body = _build_pxweb_query(spec, years)
    url = f"{PXWEB_BASE}/{spec['table_id']}"
    request = urllib.request.Request(
        url,
        data=json.dumps(query_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    time_index = payload["dimension"]["Tid"]["category"]["index"]
    values = payload["value"]
    time_keys_ordered = sorted(time_index.keys(), key=lambda t: time_index[t])
    series_values = [values[time_index[t]] for t in time_keys_ordered]
    if spec["frequency"] == "yearly":
        series = pd.Series(
            series_values,
            index=[int(t) for t in time_keys_ordered],
            dtype="float64",
        )
    else:
        monthly_index = pd.PeriodIndex(
            [t.replace("M", "-") for t in time_keys_ordered], freq="M"
        )
        monthly = pd.Series(series_values, index=monthly_index, dtype="float64")
        series = monthly.groupby(monthly.index.year).mean()
    series.index.name = "year"
    return series.dropna()


def build_price_index_panel(years: list[int]) -> pd.DataFrame:
    """Fetch SSB price indices for all four input categories.

    Parameters
    ----------
    years : list[int]
        Calendar years to retrieve.

    Returns
    -------
    pandas.DataFrame
        Indexed by year (int). One column per entry in
        :data:`firm_rp_garp.prod.bundles.INPUT_COLUMNS`. Each column is
        rebased to 100 at the earliest year for which all four indices
        are available.

    Examples
    --------
    >>> panel = build_price_index_panel(list(range(2017, 2024)))  # doctest: +SKIP
    >>> panel.columns.tolist()  # doctest: +SKIP
    ['Lonnskostnad', 'Varekostnad', 'AnnenDriftskostnad', 'AvskrivVarigeDriftsmidl']
    """
    from firm_rp_garp.prod.bundles import INPUT_COLUMNS

    series_by_input: dict[str, pd.Series] = {}
    for input_name in INPUT_COLUMNS:
        spec = SSB_TABLES[input_name]
        series_by_input[input_name] = fetch_ssb_index(spec, years)
    panel = pd.DataFrame(series_by_input).dropna(how="any")
    if panel.empty:
        return panel
    base = panel.iloc[0]
    return panel.divide(base) * 100.0
