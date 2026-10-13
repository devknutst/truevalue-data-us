"""Offline tests for the SEC EDGAR extract/transform/pipeline layers.

These use a stub extractor and hand-built XBRL-shaped fixtures, so they need
no network access -- the seam is :class:`truevalue.sec_extract.SecExtractor`,
which :func:`truevalue.sec_pipeline.run` accepts by injection.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq
import pytest

from truevalue.extract import ExtractionError
from truevalue.schemas import load_schema
from truevalue.sec_extract import RawSecCompanyData, SecExtractor
from truevalue.sec_pipeline import SEC_SCHEMAS, run
from truevalue.sec_transform import (
    _select_entry,
    build_company_dimension_record,
    build_financial_statement_records,
)
from truevalue.transform import (
    COMPANY_ID_STRATEGY_TICKER,
    STATEMENT_MODE_STATEMENT,
    RunContext,
)


def make_entry(
    end: str,
    val: float,
    fy: int,
    fp: str,
    filed: str,
    start: Optional[str] = None,
    form: str = "10-Q",
) -> Dict[str, Any]:
    """Build one raw XBRL fact entry, matching the shape of the real API."""
    entry: Dict[str, Any] = {
        "end": end,
        "val": val,
        "accn": "0000000000-00-000000",
        "fy": fy,
        "fp": fp,
        "form": form,
        "filed": filed,
    }
    if start:
        entry["start"] = start
    return entry


def tag(entries: List[Dict[str, Any]], unit: str = "USD") -> Dict[str, Any]:
    """Wrap a list of entries as a ``us-gaap`` tag document."""
    return {"units": {unit: entries}}


def make_us_gaap() -> Dict[str, Any]:
    """A trimmed but structurally realistic ``companyfacts["facts"]["us-gaap"]``.

    Covers: a plain quarter (Q1 FY2025), a quarter reported alongside a
    year-to-date duration sharing the same end date (Q2 FY2025 -- the
    year-to-date value must lose), an annual period (FY2025), a period in the
    following fiscal year (Q1 FY2026), a period outside the requested fiscal
    years (FY2024 -- must be excluded), and a restated instant value (Assets,
    Q1 FY2025 -- the later ``filed`` value must win).
    """
    return {
        "RevenueFromContractWithCustomerExcludingAssessedTax": tag(
            [
                make_entry("2024-12-31", 100e6, 2025, "Q1", "2025-01-31", "2024-10-01"),
                make_entry("2025-03-31", 250e6, 2025, "Q2", "2025-05-01", "2024-10-01"),  # YTD
                make_entry("2025-03-31", 150e6, 2025, "Q2", "2025-05-01", "2025-01-01"),  # quarter
                make_entry("2025-09-30", 500e6, 2025, "FY", "2025-10-31", "2024-10-01"),
                make_entry("2025-12-31", 120e6, 2026, "Q1", "2026-01-30", "2025-10-01"),
                make_entry("2024-09-30", 400e6, 2024, "FY", "2024-10-31", "2023-10-01"),
            ]
        ),
        "OperatingIncomeLoss": tag(
            [
                make_entry("2024-12-31", 20e6, 2025, "Q1", "2025-01-31", "2024-10-01"),
                make_entry("2025-03-31", 25e6, 2025, "Q2", "2025-05-01", "2025-01-01"),
                make_entry("2025-09-30", 90e6, 2025, "FY", "2025-10-31", "2024-10-01"),
            ]
        ),
        "NetIncomeLoss": tag(
            [make_entry("2024-12-31", 15e6, 2025, "Q1", "2025-01-31", "2024-10-01")]
        ),
        "IncomeTaxExpenseBenefit": tag(
            [make_entry("2024-12-31", 5e6, 2025, "Q1", "2025-01-31", "2024-10-01")]
        ),
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": tag(
            [make_entry("2024-12-31", 20e6, 2025, "Q1", "2025-01-31", "2024-10-01")]
        ),
        "InterestExpense": tag(
            [make_entry("2024-12-31", 2e6, 2025, "Q1", "2025-01-31", "2024-10-01")]
        ),
        "DepreciationDepletionAndAmortization": tag(
            [make_entry("2024-12-31", 3e6, 2025, "Q1", "2025-01-31", "2024-10-01")]
        ),
        "NetCashProvidedByUsedInOperatingActivities": tag(
            [make_entry("2024-12-31", 30e6, 2025, "Q1", "2025-01-31", "2024-10-01")]
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": tag(
            [make_entry("2024-12-31", 8e6, 2025, "Q1", "2025-01-31", "2024-10-01")]
        ),
        "PaymentsOfDividends": tag(
            [make_entry("2024-12-31", 4e6, 2025, "Q1", "2025-01-31", "2024-10-01")]
        ),
        "EarningsPerShareDiluted": tag(
            [make_entry("2024-12-31", 1.5, 2025, "Q1", "2025-01-31", "2024-10-01")],
            unit="USD/shares",
        ),
        "WeightedAverageNumberOfDilutedSharesOutstanding": tag(
            [make_entry("2024-12-31", 10e6, 2025, "Q1", "2025-01-31", "2024-10-01")],
            unit="shares",
        ),
        "Assets": tag(
            [
                make_entry("2024-12-31", 1000e6, 2025, "Q1", "2025-01-31", form="10-Q"),
                make_entry("2024-12-31", 1050e6, 2025, "Q1", "2025-02-15", form="10-Q/A"),
                make_entry("2025-03-31", 1100e6, 2025, "Q2", "2025-05-01"),
                make_entry("2025-09-30", 1300e6, 2025, "FY", "2025-10-31"),
                make_entry("2025-12-31", 1350e6, 2026, "Q1", "2026-01-30"),
            ]
        ),
        "Liabilities": tag(
            [
                make_entry("2024-12-31", 600e6, 2025, "Q1", "2025-01-31"),
                make_entry("2025-09-30", 750e6, 2025, "FY", "2025-10-31"),
            ]
        ),
        "LiabilitiesCurrent": tag(
            [make_entry("2024-12-31", 200e6, 2025, "Q1", "2025-01-31")]
        ),
        "AssetsCurrent": tag(
            [make_entry("2024-12-31", 400e6, 2025, "Q1", "2025-01-31")]
        ),
        "StockholdersEquity": tag(
            [
                make_entry("2024-12-31", 400e6, 2025, "Q1", "2025-01-31"),
                make_entry("2025-09-30", 550e6, 2025, "FY", "2025-10-31"),
            ]
        ),
        "CashAndCashEquivalentsAtCarryingValue": tag(
            [make_entry("2024-12-31", 150e6, 2025, "Q1", "2025-01-31")]
        ),
        "LongTermDebtCurrent": tag(
            [make_entry("2024-12-31", 30e6, 2025, "Q1", "2025-01-31")]
        ),
        "LongTermDebtNoncurrent": tag(
            [make_entry("2024-12-31", 120e6, 2025, "Q1", "2025-01-31")]
        ),
    }


def make_submissions(
    tickers: Optional[List[str]] = None,
    exchanges: Optional[List[str]] = None,
    foreign: bool = False,
) -> Dict[str, Any]:
    """A trimmed but structurally realistic ``submissions`` document."""
    business_address: Dict[str, Any] = (
        {"isForeignLocation": 1, "country": "UNITED KINGDOM", "countryCode": "GB", "stateOrCountry": None}
        if foreign
        else {"isForeignLocation": 0, "stateOrCountry": "CA", "country": None, "countryCode": None}
    )
    return {
        "cik": "0009999999",
        "entityType": "operating",
        "sic": "7372",
        "sicDescription": "Services-Prepackaged Software",
        "name": "Test Co Inc.",
        "tickers": tickers if tickers is not None else ["TESTCO", "TESTC"],
        "exchanges": exchanges if exchanges is not None else ["Nasdaq", "Nasdaq"],
        "website": "",
        "description": "",
        "addresses": {"business": business_address, "mailing": business_address},
    }


def make_raw(ticker: str = "TESTCO") -> RawSecCompanyData:
    """Assemble a fully populated :class:`RawSecCompanyData` stub."""
    return RawSecCompanyData(
        ticker=ticker,
        cik="0009999999",
        submissions=make_submissions(),
        facts=make_us_gaap(),
    )


class StubSecExtractor:
    """Drop-in replacement for :class:`SecExtractor` in tests."""

    def __init__(self, failing: tuple = ()) -> None:
        self.failing = set(failing)

    def fetch(self, ticker: str) -> RawSecCompanyData:
        symbol = ticker.upper()
        if symbol in self.failing:
            raise ExtractionError("stub failure for {}".format(symbol))
        return make_raw(symbol)


@pytest.fixture()
def context() -> RunContext:
    return RunContext.create(
        company_id_strategy=COMPANY_ID_STRATEGY_TICKER, source_system="sec_edgar"
    )


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def test_sec_extractor_requires_a_user_agent():
    with pytest.raises(ValueError):
        SecExtractor(user_agent="")
    with pytest.raises(ValueError):
        SecExtractor(user_agent="   ")


def test_sec_extractor_accepts_a_valid_user_agent():
    extractor = SecExtractor(user_agent="truevalue-data-us contact@example.com")
    assert extractor.user_agent == "truevalue-data-us contact@example.com"


# --------------------------------------------------------------------------- #
# transform: XBRL fact selection
# --------------------------------------------------------------------------- #


def test_select_entry_prefers_shortest_duration_at_the_latest_end():
    entries = [
        {"end": "2025-03-31", "start": "2024-10-01", "val": 250, "fy": 2025, "fp": "Q2", "filed": "2025-05-01"},
        {"end": "2025-03-31", "start": "2025-01-01", "val": 150, "fy": 2025, "fp": "Q2", "filed": "2025-05-01"},
        {"end": "2024-12-31", "start": "2024-10-01", "val": 100, "fy": 2025, "fp": "Q1", "filed": "2025-01-31"},
    ]
    selected = _select_entry(entries, 2025, "Q2")
    assert selected is not None
    assert selected["val"] == 150


def test_select_entry_breaks_ties_with_the_latest_filed_value():
    entries = [
        {"end": "2024-12-31", "val": 1000, "fy": 2025, "fp": "Q1", "filed": "2025-01-31"},
        {"end": "2024-12-31", "val": 1050, "fy": 2025, "fp": "Q1", "filed": "2025-02-15"},
    ]
    selected = _select_entry(entries, 2025, "Q1")
    assert selected is not None
    assert selected["val"] == 1050


def test_select_entry_returns_none_when_period_absent():
    assert _select_entry([{"end": "2024-12-31", "fy": 2025, "fp": "Q1", "filed": "x"}], 2099, "Q1") is None


# --------------------------------------------------------------------------- #
# transform: financial_statement_fact
# --------------------------------------------------------------------------- #


def test_financial_statement_records_cover_default_fiscal_years_only(context):
    records = build_financial_statement_records(make_raw(), context)
    fiscal_years = {r["fiscal_year"] for r in records}
    assert fiscal_years == {2025, 2026}

    periods = {(r["fiscal_year"], r["statement_type"], r["fiscal_quarter"]) for r in records}
    assert (2025, "ANNUAL", None) in periods
    assert (2025, "QUARTERLY", 1) in periods
    assert (2025, "QUARTERLY", 2) in periods
    assert (2026, "QUARTERLY", 1) in periods


def test_quarter_value_wins_over_year_to_date_sharing_the_same_end_date(context):
    records = build_financial_statement_records(make_raw(), context)
    q2 = next(r for r in records if r["fiscal_year"] == 2025 and r["fiscal_quarter"] == 2)
    assert q2["revenue"] == 150e6
    assert q2["period_start_date"] == "2025-01-01"
    assert q2["period_end_date"] == "2025-03-31"


def test_restated_instant_value_uses_the_latest_filed_figure(context):
    records = build_financial_statement_records(make_raw(), context)
    q1 = next(r for r in records if r["fiscal_year"] == 2025 and r["fiscal_quarter"] == 1)
    assert q1["total_assets"] == 1050e6


def test_filing_date_is_populated_from_the_xbrl_fact(context):
    records = build_financial_statement_records(make_raw(), context)
    q1 = next(r for r in records if r["fiscal_year"] == 2025 and r["fiscal_quarter"] == 1)
    assert q1["filing_date"] == "2025-01-31"


def test_capex_and_dividends_are_stored_as_negative_outflows(context):
    records = build_financial_statement_records(make_raw(), context)
    q1 = next(r for r in records if r["fiscal_year"] == 2025 and r["fiscal_quarter"] == 1)
    assert q1["capital_expenditure"] == -8e6
    assert q1["dividends_paid"] == -4e6
    assert q1["free_cash_flow"] == pytest.approx(30e6 - 8e6)
    assert q1["dividend_per_share"] == pytest.approx(4e6 / 10e6)


def test_derived_metrics(context):
    records = build_financial_statement_records(make_raw(), context)
    q1 = next(r for r in records if r["fiscal_year"] == 2025 and r["fiscal_quarter"] == 1)
    assert q1["ebit"] == pytest.approx(20e6 + 2e6)  # pretax + interest expense
    assert q1["ebitda"] == pytest.approx(20e6 + 2e6 + 3e6)
    assert q1["effective_tax_rate"] == pytest.approx(5e6 / 20e6)
    assert q1["total_debt"] == pytest.approx(30e6 + 120e6)
    assert q1["working_capital"] == pytest.approx(400e6 - 200e6)
    assert q1["net_debt"] == pytest.approx((30e6 + 120e6) - 150e6)
    # gross_profit has no XBRL fact in the fixture -> must stay NULL, not 0.
    assert q1["gross_profit"] is None


def test_financial_statement_statement_mode_does_not_leak_fields_across_statements():
    context = RunContext.create(
        company_id_strategy=COMPANY_ID_STRATEGY_TICKER,
        statement_type_mode=STATEMENT_MODE_STATEMENT,
        source_system="sec_edgar",
    )
    records = build_financial_statement_records(make_raw(), context)
    q1_income = next(
        r
        for r in records
        if r["statement_type"] == "INCOME_STATEMENT"
        and r["fiscal_year"] == 2025
        and r["fiscal_quarter"] == 1
    )
    assert q1_income["revenue"] == 100e6
    assert q1_income.get("total_assets") is None
    q1_balance = next(
        r
        for r in records
        if r["statement_type"] == "BALANCE_SHEET"
        and r["fiscal_year"] == 2025
        and r["fiscal_quarter"] == 1
    )
    assert q1_balance["total_assets"] == 1050e6
    assert q1_balance.get("revenue") is None


def test_no_facts_for_requested_fiscal_years_yields_no_records(context):
    raw = RawSecCompanyData(ticker="EMPTY", cik="0000000001", submissions={}, facts={})
    assert build_financial_statement_records(raw, context) == []


def test_period_with_only_balance_sheet_facts_has_no_filing_date(context):
    """A period with no income/cash-flow duration facts at all falls back to the
    balance-sheet 'end' date for period_end_date and leaves filing_date NULL,
    since only duration facts carry a 'start'/'filed' pair in this fixture."""
    facts = make_us_gaap()
    facts["Assets"]["units"]["USD"].append(
        make_entry("2025-06-30", 1200e6, 2025, "Q3", "2025-08-01")
    )
    raw = RawSecCompanyData(
        ticker="TESTCO", cik="0009999999", submissions=make_submissions(), facts=facts
    )
    q3 = next(
        r for r in build_financial_statement_records(raw, context)
        if r["fiscal_year"] == 2025 and r["fiscal_quarter"] == 3
    )
    assert q3["period_end_date"] == "2025-06-30"
    assert q3["period_start_date"] is None
    assert q3["filing_date"] is None
    assert q3["total_assets"] == 1200e6
    assert q3["revenue"] is None


# --------------------------------------------------------------------------- #
# transform: company_dimension
# --------------------------------------------------------------------------- #


def test_company_dimension_mapping(context):
    record = build_company_dimension_record(make_raw(), context)
    assert record["cik"] == "0009999999"
    assert record["sic_code"] == "7372"
    assert record["industry"] == "Services-Prepackaged Software"
    assert record["exchange"] == "Nasdaq"
    assert record["country"] == "US"
    assert record["currency"] == "USD"
    assert record["active_flag"] is True
    assert record["source_system"] == "sec_edgar"
    # Not obtainable from the SEC APIs.
    assert record["cusip"] is None and record["isin"] is None
    assert record["ipo_date"] is None and record["employees"] is None
    # Empty-string fields from SEC must be normalised to None.
    assert record["website"] is None
    assert record["business_description"] is None


def test_company_dimension_exchange_matches_the_requested_ticker(context):
    raw = make_raw("TESTC")
    record = build_company_dimension_record(raw, context)
    assert record["exchange"] == "Nasdaq"


def test_company_dimension_inactive_when_no_exchange_listed(context):
    raw = RawSecCompanyData(
        ticker="DELISTED",
        cik="0000000002",
        submissions=make_submissions(tickers=[], exchanges=[]),
        facts={},
    )
    record = build_company_dimension_record(raw, context)
    assert record["active_flag"] is False


def test_company_dimension_active_flag_is_per_ticker_not_per_filer(context):
    """A dual-class filer with one delisted class (empty-string exchange) must
    report active_flag per requested ticker, not "any class is still listed"."""
    submissions = make_submissions(
        tickers=["TESTCO", "TESTC2"], exchanges=["Nasdaq", ""]
    )
    listed = build_company_dimension_record(
        RawSecCompanyData(ticker="TESTCO", cik="0009999999", submissions=submissions, facts={}),
        context,
    )
    delisted_class = build_company_dimension_record(
        RawSecCompanyData(ticker="TESTC2", cik="0009999999", submissions=submissions, facts={}),
        context,
    )
    assert listed["active_flag"] is True
    assert delisted_class["exchange"] is None
    assert delisted_class["active_flag"] is False


def test_company_dimension_exchange_falls_back_when_ticker_not_listed(context):
    """A ticker absent from the submissions' own tickers array (e.g. a stale
    mapping) still gets a best-effort exchange rather than NULL."""
    raw = RawSecCompanyData(
        ticker="STALE",
        cik="0000000004",
        submissions=make_submissions(tickers=["TESTCO"], exchanges=["NYSE"]),
        facts={},
    )
    record = build_company_dimension_record(raw, context)
    assert record["exchange"] == "NYSE"


def test_company_dimension_foreign_filer_country(context):
    raw = RawSecCompanyData(
        ticker="FOREIGNCO",
        cik="0000000003",
        submissions=make_submissions(foreign=True),
        facts={},
    )
    record = build_company_dimension_record(raw, context)
    assert record["country"] == "GB"


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


def test_sec_run_writes_only_the_supported_schemas(tmp_path: Path):
    result = run(
        tickers=["TESTCO", "BROKEN"],
        output_dir=tmp_path,
        extractor=StubSecExtractor(failing=("BROKEN",)),
    )
    assert result.ok
    assert result.succeeded == ["TESTCO"]
    assert "BROKEN" in result.failures
    assert set(result.row_counts) == set(SEC_SCHEMAS)
    assert "market_price_fact" not in result.written_paths
    assert not (tmp_path / "market_price_fact").exists()

    company_path = tmp_path / "company_dimension" / "company_dimension.parquet"
    schema = load_schema("company_dimension")
    table = pq.read_table(company_path)
    assert table.schema.names == list(schema.field_names)
    assert table.num_rows == 1


def test_sec_run_requires_an_extractor_or_a_user_agent(tmp_path: Path):
    with pytest.raises(ValueError):
        run(tickers=["TESTCO"], output_dir=tmp_path)


def test_sec_run_rejects_empty_ticker_list(tmp_path: Path):
    with pytest.raises(ValueError):
        run(tickers=["  "], output_dir=tmp_path, extractor=StubSecExtractor())
