"""Offline tests for the transform and load layers.

These use a stub extractor and therefore need no network access and no yfinance
installation -- the seam is :class:`truevalue.extract.YFinanceExtractor`, which
the pipeline accepts by injection.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pyarrow.parquet as pq
import pytest

from truevalue.extract import ExtractionError, RawTickerData
from truevalue.load import build_table
from truevalue.pipeline import normalise_tickers, run
from truevalue.schemas import load_all_schemas, load_schema, to_bool, to_float, to_int
from truevalue.transform import (
    COMPANY_ID_STRATEGY_TICKER,
    STATEMENT_MODE_PERIOD,
    STATEMENT_MODE_STATEMENT,
    RunContext,
    build_company_dimension_record,
    build_financial_statement_records,
    build_market_price_records,
)

PERIOD_END_2024 = pd.Timestamp("2024-09-28")
PERIOD_END_2023 = pd.Timestamp("2023-09-30")


def make_info() -> Dict[str, Any]:
    """A trimmed-down but realistic ``Ticker.info`` payload."""
    return {
        "shortName": "Apple Inc.",
        "longName": "Apple Inc.",
        "exchange": "NMS",
        "market": "us_market",
        "country": "United States",
        "currency": "USD",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "website": "https://www.apple.com",
        "fullTimeEmployees": 164000,
        "longBusinessSummary": "Apple Inc. designs, manufactures and markets ...",
        "firstTradeDateEpochUtc": 345479400,
        "currentPrice": 227.52,
        "marketCap": 3_450_000_000_000,
        "enterpriseValue": 3_500_000_000_000,
        "sharesOutstanding": 15_200_000_000,
        "trailingPE": 34.5,
        "priceToBook": 51.2,
        "priceToSalesTrailing12Months": 8.9,
        "enterpriseToEbitda": 26.1,
        "dividendYield": 0.44,
    }


def make_history() -> pd.DataFrame:
    """Three trading days with distinct raw and adjusted closes."""
    return pd.DataFrame(
        {
            "Open": [220.0, 223.5, 226.0],
            "High": [224.0, 227.0, 229.5],
            "Low": [219.0, 222.0, 225.0],
            "Close": [223.0, 226.5, 227.52],
            "Adj Close": [222.1, 225.6, 227.52],
            "Volume": [41_000_000, 38_500_000, 44_200_000],
        },
        index=pd.DatetimeIndex(
            ["2024-09-24", "2024-09-25", "2024-09-26"], name="Date"
        ),
    )


def make_income() -> pd.DataFrame:
    """Annual income statement, line items on the index like yfinance returns."""
    return pd.DataFrame(
        {
            PERIOD_END_2024: [391_035e6, 210_352e6, 180_683e6, 31_370e6, 123_216e6,
                              93_736e6, 6.11, 6.08, 15_400e6, 15_343e6],
            PERIOD_END_2023: [383_285e6, 214_137e6, 169_148e6, 29_915e6, 113_736e6,
                              96_995e6, 6.16, 6.13, 15_744e6, 15_813e6],
        },
        index=[
            "Total Revenue",
            "Cost Of Revenue",
            "Gross Profit",
            "Research And Development",
            "Operating Income",
            "Net Income",
            "Basic EPS",
            "Diluted EPS",
            "Basic Average Shares",
            "Diluted Average Shares",
        ],
    )


def make_balance() -> pd.DataFrame:
    """Annual balance sheet with the fields the derived metrics need."""
    return pd.DataFrame(
        {
            PERIOD_END_2024: [29_943e6, 35_228e6, 152_987e6, 176_392e6, 106_629e6,
                              364_980e6, 56_950e6],
            PERIOD_END_2023: [29_965e6, 31_590e6, 143_566e6, 145_308e6, 111_088e6,
                              352_583e6, 62_146e6],
        },
        index=[
            "Cash And Cash Equivalents",
            "Other Short Term Investments",
            "Current Assets",
            "Current Liabilities",
            "Total Debt",
            "Total Assets",
            "Stockholders Equity",
        ],
    )


def make_cashflow() -> pd.DataFrame:
    """Annual cash flow statement; Yahoo reports outflows as negative."""
    return pd.DataFrame(
        {
            PERIOD_END_2024: [118_254e6, -9_447e6, -125_820e6, -15_234e6],
            PERIOD_END_2023: [110_543e6, -10_959e6, -108_488e6, -15_025e6],
        },
        index=[
            "Operating Cash Flow",
            "Capital Expenditure",
            "Financing Cash Flow",
            "Cash Dividends Paid",
        ],
    )


def make_raw(ticker: str = "AAPL") -> RawTickerData:
    """Assemble a fully populated :class:`RawTickerData` stub."""
    return RawTickerData(
        ticker=ticker,
        info=make_info(),
        isin="US0378331005",
        history=make_history(),
        income_annual=make_income(),
        balance_annual=make_balance(),
        cashflow_annual=make_cashflow(),
    )


class StubExtractor:
    """Drop-in replacement for :class:`YFinanceExtractor` in tests."""

    def __init__(self, failing: tuple = ()) -> None:
        self.failing = set(failing)

    def fetch(self, ticker: str) -> RawTickerData:
        symbol = ticker.upper()
        if symbol in self.failing:
            raise ExtractionError("stub failure for {}".format(symbol))
        return make_raw(symbol)


@pytest.fixture()
def context() -> RunContext:
    return RunContext.create(company_id_strategy=COMPANY_ID_STRATEGY_TICKER)


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #


def test_schemas_load_with_expected_shape():
    schemas = load_all_schemas()
    assert set(schemas) == {
        "company_dimension",
        "financial_statement_fact",
        "market_price_fact",
    }
    assert schemas["company_dimension"].field_names[0] == "company_id"
    # Arrow field order must mirror the JSON Schema declaration order.
    arrow = schemas["market_price_fact"].arrow_schema()
    assert arrow.names == list(schemas["market_price_fact"].field_names)
    assert all(field.nullable for field in arrow)


def test_coerce_fills_missing_and_drops_unknown(caplog):
    schema = load_schema("company_dimension")
    coerced = schema.coerce({"ticker": "  aapl ", "not_a_field": 1})
    assert set(coerced) == set(schema.field_names)
    assert coerced["ticker"] == "aapl"
    assert coerced["company_id"] is None
    assert "not_a_field" not in coerced


@pytest.mark.parametrize(
    "value,expected",
    [(float("nan"), None), (pd.NaT, None), (None, None), (float("inf"), None), (3.5, 3.5)],
)
def test_to_float_handles_missing(value, expected):
    assert to_float(value) == expected


def test_to_int_and_bool():
    assert to_int(164000.0) == 164000
    assert to_int(float("nan")) is None
    assert to_bool("true") is True
    assert to_bool(None) is None


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_company_dimension_mapping(context):
    record = build_company_dimension_record(make_raw(), context)
    assert record["company_id"] == "AAPL"
    assert record["company_name"] == "Apple Inc."
    assert record["exchange"] == "NMS"
    assert record["market"] == "us_market"
    assert record["employees"] == 164000
    assert record["active_flag"] is True
    assert record["ipo_date"] == "1980-12-12"
    assert record["source_system"] == "yfinance"
    # Not obtainable from yfinance.
    assert record["cik"] is None and record["cusip"] is None
    assert record["sic_code"] is None and record["naics_code"] is None


def test_market_price_records_grain_and_snapshot(context):
    records = build_market_price_records(make_raw(), context)
    assert len(records) == 3
    assert [r["trade_date"] for r in records] == [
        "2024-09-24",
        "2024-09-25",
        "2024-09-26",
    ]
    assert records[0]["close_price"] == 223.0
    assert records[0]["adjusted_close_price"] == 222.1
    # Snapshot metrics land on the latest row only.
    assert records[-1]["market_cap"] == 3_450_000_000_000
    assert records[-1]["pe_ratio"] == 34.5
    assert records[0]["market_cap"] is None
    assert records[0]["pe_ratio"] is None


def test_financial_statement_period_mode(context):
    records = build_financial_statement_records(make_raw(), context)
    assert len(records) == 2
    latest = records[0]
    assert latest["statement_type"] == "ANNUAL"
    assert latest["fiscal_year"] == 2024
    assert latest["fiscal_quarter"] is None
    assert latest["period_end_date"] == "2024-09-28"
    assert latest["period_start_date"] == "2023-09-29"
    assert latest["revenue"] == 391_035e6
    assert latest["source_record_id"] == "AAPL:ANNUAL:2024-09-28"
    assert latest["filing_date"] is None


def test_derived_metrics(context):
    latest = build_financial_statement_records(make_raw(), context)[0]
    # working_capital = current assets - current liabilities
    assert latest["working_capital"] == pytest.approx(152_987e6 - 176_392e6)
    # net_debt = total debt - (cash + short term investments)
    assert latest["net_debt"] == pytest.approx(106_629e6 - (29_943e6 + 35_228e6))
    # free_cash_flow derived: capex is negative, so it is added
    assert latest["free_cash_flow"] == pytest.approx(118_254e6 - 9_447e6)
    # dividend_per_share is positive despite the negative cash outflow
    assert latest["dividend_per_share"] == pytest.approx(15_234e6 / 15_400e6)
    # ebitda is absent from the stub, so its margin must stay NULL, not 0
    assert latest["ebitda"] is None
    assert latest["ebitda_margin"] is None


def test_fiscal_quarter_follows_the_fiscal_calendar_not_the_calendar_year(context):
    """A September fiscal year end must not produce calendar quarters.

    For a FY ending in September, the quarter ending 31 December is fiscal Q1 of
    the *next* fiscal year.
    """
    raw = make_raw()
    quarter_ends = [
        pd.Timestamp("2024-12-28"),  # FY2025 Q1
        pd.Timestamp("2024-09-28"),  # FY2024 Q4
        pd.Timestamp("2024-06-29"),  # FY2024 Q3
        pd.Timestamp("2024-03-30"),  # FY2024 Q2
    ]
    raw.income_quarterly = pd.DataFrame(
        {end: [100e6] for end in quarter_ends}, index=["Total Revenue"]
    )

    records = build_financial_statement_records(raw, context)
    quarterly = {
        r["period_end_date"]: (r["fiscal_year"], r["fiscal_quarter"])
        for r in records
        if r["statement_type"] == "QUARTERLY"
    }
    assert quarterly["2024-12-28"] == (2025, 1)
    assert quarterly["2024-09-28"] == (2024, 4)
    assert quarterly["2024-06-29"] == (2024, 3)
    assert quarterly["2024-03-30"] == (2024, 2)


def test_fiscal_calendar_defaults_to_calendar_year_without_annual_data(context):
    raw = make_raw()
    raw.income_annual = pd.DataFrame()
    raw.balance_annual = pd.DataFrame()
    raw.cashflow_annual = pd.DataFrame()
    raw.income_quarterly = pd.DataFrame(
        {pd.Timestamp("2024-03-31"): [100e6]}, index=["Total Revenue"]
    )
    record = build_financial_statement_records(raw, context)[0]
    assert (record["fiscal_year"], record["fiscal_quarter"]) == (2024, 1)


def test_ipo_date_from_millisecond_epoch(context):
    """yfinance >=1.0 only exposes firstTradeDateMilliseconds."""
    raw = make_raw()
    raw.info.pop("firstTradeDateEpochUtc")
    raw.info["firstTradeDateMilliseconds"] = 345479400000
    assert build_company_dimension_record(raw, context)["ipo_date"] == "1980-12-12"


def test_financial_statement_statement_mode():
    context = RunContext.create(
        company_id_strategy=COMPANY_ID_STRATEGY_TICKER,
        statement_type_mode=STATEMENT_MODE_STATEMENT,
    )
    records = build_financial_statement_records(make_raw(), context)
    kinds = {r["statement_type"] for r in records}
    assert kinds == {"INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"}
    income_row = next(
        r
        for r in records
        if r["statement_type"] == "INCOME_STATEMENT" and r["fiscal_year"] == 2024
    )
    assert income_row["revenue"] == 391_035e6
    # Balance-sheet fields must not leak into the income-statement row.
    assert income_row.get("total_assets") is None


def test_uuid5_company_id_is_stable_and_shared_across_schemas():
    context = RunContext.create()
    raw = make_raw()
    dimension = build_company_dimension_record(raw, context)
    price = build_market_price_records(raw, context)[0]
    fact = build_financial_statement_records(raw, context)[0]
    assert dimension["company_id"] == price["company_id"] == fact["company_id"]
    assert dimension["company_id"] == RunContext.create().company_id("AAPL")


def test_stub_profile_of_an_unknown_symbol_is_not_treated_as_data():
    """yfinance returns {'trailingPegRatio': None} for unknown symbols."""
    raw = RawTickerData(ticker="NOSUCHTICKERXYZ", info={"trailingPegRatio": None})
    assert not raw.has_profile
    assert not raw.has_any_data


def test_empty_history_yields_no_price_records(context):
    raw = make_raw()
    raw.history = pd.DataFrame()
    assert build_market_price_records(raw, context) == []


# --------------------------------------------------------------------------- #
# load + pipeline
# --------------------------------------------------------------------------- #


def test_build_table_of_zero_records_keeps_columns():
    schema = load_schema("market_price_fact")
    table = build_table([], schema)
    assert table.num_rows == 0
    assert table.schema.names == list(schema.field_names)


def test_run_writes_parquet_and_survives_a_failing_ticker(tmp_path: Path):
    result = run(
        tickers=["AAPL", "BROKEN", "aapl", "MSFT"],
        output_dir=tmp_path,
        extractor=StubExtractor(failing=("BROKEN",)),
    )
    assert result.ok
    assert result.succeeded == ["AAPL", "MSFT"]
    assert "BROKEN" in result.failures
    # Duplicate 'aapl' was normalised away.
    assert result.row_counts["company_dimension"] == 2
    assert result.row_counts["market_price_fact"] == 6
    assert result.row_counts["financial_statement_fact"] == 4

    path = tmp_path / "market_price_fact" / "market_price_fact.parquet"
    table = pq.read_table(path)
    schema = load_schema("market_price_fact")
    assert table.schema.names == list(schema.field_names)
    assert table.num_rows == 6


def test_normalise_tickers():
    assert normalise_tickers([" aapl", "MSFT,NVDA", "aapl"]) == ["AAPL", "MSFT", "NVDA"]


def test_run_rejects_empty_ticker_list(tmp_path: Path):
    with pytest.raises(ValueError):
        run(tickers=["  "], output_dir=tmp_path, extractor=StubExtractor())
