"""Transformation layer: maps raw SEC EDGAR payloads onto the target schemas.

Only ``company_dimension`` and ``financial_statement_fact`` are produced here.
The SEC EDGAR APIs carry company profiles and XBRL financial facts, but no
market/price data at all, so ``market_price_fact`` is out of scope for this
source -- see ``README.md``.

Design notes
------------
* The SEC (``us-gaap`` taxonomy) -> target field mappings live in the
  ``SEC_*_FIELD_MAP`` constants below.  Each target field maps to a tuple of
  candidate XBRL tags; the first one with a usable value wins.  Filers do not
  all tag the same concept the same way, so the fallbacks matter.
* A single XBRL concept's fact array mixes the current period with every
  comparative period later filings re-report alongside it, and -- for
  duration concepts -- sometimes both a quarter and a year-to-date figure
  ending on the same date.  :func:`_select_entry` disambiguates deterministically;
  see its docstring.
* Unlike yfinance, SEC facts carry a real ``filed`` date per value, so
  ``filing_date`` is populated here instead of being left ``NULL``.
* Fields the target schema declares but the SEC APIs cannot supply are
  emitted as ``None``.  See :data:`UNAVAILABLE_FIELDS`.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schemas import to_float
from .sec_extract import RawSecCompanyData
from .transform import (
    STATEMENT_MODE_PERIOD,
    STATEMENT_TYPE_ANNUAL,
    STATEMENT_TYPE_BALANCE,
    STATEMENT_TYPE_CASH_FLOW,
    STATEMENT_TYPE_INCOME,
    STATEMENT_TYPE_QUARTERLY,
    RunContext,
)
from .transform_common import build_source_record_id, safe_ratio, safe_sum

logger = logging.getLogger(__name__)

#: Fiscal years ingested when the caller does not specify any.
DEFAULT_FISCAL_YEARS: Tuple[int, ...] = (2025, 2026)

#: Target fields no SEC EDGAR endpoint provides.  Always emitted as NULL.
UNAVAILABLE_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "company_dimension": ("cusip", "isin", "market", "sector", "ipo_date", "employees"),
    "financial_statement_fact": (),
}

# --------------------------------------------------------------------------- #
# SEC us-gaap -> target field maps
# --------------------------------------------------------------------------- #

SEC_INCOME_FIELD_MAP: Mapping[str, Tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "cost_of_revenue": ("CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"),
    "gross_profit": ("GrossProfit",),
    "research_and_development": ("ResearchAndDevelopmentExpense",),
    "selling_general_administrative": (
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ),
    "operating_income": ("OperatingIncomeLoss",),
    "interest_income": ("InvestmentIncomeInterest", "InterestIncomeOther"),
    "interest_expense": ("InterestExpense", "InterestExpenseDebt"),
    "pretax_income": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
    "income_tax_expense": ("IncomeTaxExpenseBenefit",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_basic": ("EarningsPerShareBasic",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "basic_shares_outstanding": ("WeightedAverageNumberOfSharesOutstandingBasic",),
    "diluted_shares_outstanding": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
}

SEC_BALANCE_FIELD_MAP: Mapping[str, Tuple[str, ...]] = {
    "cash_and_cash_equivalents": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "short_term_investments": ("OtherShortTermInvestments", "ShortTermInvestments"),
    "accounts_receivable": ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"),
    "inventory": ("InventoryNet",),
    "total_current_assets": ("AssetsCurrent",),
    "property_plant_equipment": ("PropertyPlantAndEquipmentNet",),
    "goodwill": ("Goodwill",),
    "intangible_assets": (
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet",
    ),
    "total_assets": ("Assets",),
    "accounts_payable": ("AccountsPayableCurrent", "AccountsPayableTradeCurrent"),
    "current_liabilities": ("LiabilitiesCurrent",),
    "short_term_debt": ("LongTermDebtCurrent", "DebtCurrent", "CommercialPaper"),
    "long_term_debt": ("LongTermDebtNoncurrent",),
    "total_liabilities": ("Liabilities",),
    "shareholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "retained_earnings": ("RetainedEarningsAccumulatedDeficit",),
}

SEC_CASH_FLOW_FIELD_MAP: Mapping[str, Tuple[str, ...]] = {
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    # Reported by SEC filers as a positive payment magnitude; negated in
    # _add_derived_values to match this project's "outflow is negative" sign
    # convention (also used for ``dividends_paid``).
    "capital_expenditure": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ),
    "investing_cash_flow": ("NetCashProvidedByUsedInInvestingActivities",),
    "financing_cash_flow": ("NetCashProvidedByUsedInFinancingActivities",),
    "depreciation_and_amortization": (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ),
    "stock_based_compensation": ("ShareBasedCompensation",),
    "dividends_paid": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    "dividend_per_share": ("CommonStockDividendsPerShareDeclared",),
}

#: Target fields reported in a unit other than plain ``USD``.
FIELD_UNITS: Mapping[str, str] = {
    "eps_basic": "USD/shares",
    "eps_diluted": "USD/shares",
    "basic_shares_outstanding": "shares",
    "diluted_shares_outstanding": "shares",
    "dividend_per_share": "USD/shares",
}

#: SEC reports these as positive payment magnitudes; this project's convention
#: (matching :mod:`truevalue.transform`) stores cash outflows as negative.
SIGN_FLIPPED_FIELDS: Tuple[str, ...] = ("capital_expenditure", "dividends_paid")

#: Which target fields belong to which statement, used by ``statement`` mode to
#: blank out everything that does not originate from the row's own statement.
#: Derived fields are assigned to the statement supplying their inputs.
INCOME_DERIVED_FIELDS = ("ebit", "ebitda", "ebitda_margin", "effective_tax_rate")
BALANCE_DERIVED_FIELDS = ("total_debt", "working_capital", "net_working_capital", "net_debt", "invested_capital")
CASH_FLOW_DERIVED_FIELDS = ("free_cash_flow",)

STATEMENT_FIELD_GROUPS: Mapping[str, Tuple[str, ...]] = {
    STATEMENT_TYPE_INCOME: tuple(SEC_INCOME_FIELD_MAP) + INCOME_DERIVED_FIELDS,
    STATEMENT_TYPE_BALANCE: tuple(SEC_BALANCE_FIELD_MAP) + BALANCE_DERIVED_FIELDS,
    STATEMENT_TYPE_CASH_FLOW: tuple(SEC_CASH_FLOW_FIELD_MAP) + CASH_FLOW_DERIVED_FIELDS,
}

#: XBRL fiscal-period labels this pipeline understands; ``FY`` is annual.
_QUARTER_NUMBERS: Mapping[str, int] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
_VALID_FISCAL_PERIODS = frozenset({"Q1", "Q2", "Q3", "Q4", "FY"})


# --------------------------------------------------------------------------- #
# XBRL fact selection
# --------------------------------------------------------------------------- #


def _entries_for_tag(us_gaap: Mapping[str, Any], tag: str, unit: str) -> List[Mapping[str, Any]]:
    """Return the raw fact array for ``tag``/``unit``, or ``[]`` if absent."""
    tag_data = us_gaap.get(tag)
    if not isinstance(tag_data, dict):
        return []
    units = tag_data.get("units")
    if not isinstance(units, dict):
        return []
    entries = units.get(unit)
    return entries if isinstance(entries, list) else []


def _duration_days(entry: Mapping[str, Any]) -> int:
    """Length of a fact's reporting window.

    Returns ``0`` both for a genuine instant (balance-sheet) fact and for a
    duration fact with a missing/malformed ``start``/``end`` -- deliberate,
    since either way the entry has no meaningful duration to compare, and ``0``
    naturally sorts as "shortest" in :func:`_select_entry`'s tie-break. SEC
    always emits canonical ISO dates in practice, so the malformed-date case is
    theoretical.
    """
    start, end = entry.get("start"), entry.get("end")
    if not start or not end:
        return 0
    try:
        return (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError:
        return 0


def _select_entry(
    entries: Sequence[Mapping[str, Any]], fy: int, fp: str
) -> Optional[Mapping[str, Any]]:
    """Pick the one fact representing ``(fy, fp)``'s own current period.

    A concept's fact array mixes the current period with every comparative
    period that later filings re-report alongside it, and -- for duration
    concepts -- sometimes both a quarter and a year-to-date figure ending on
    the same date. Disambiguated, in order, by: latest ``end`` (a filing's own
    current period is always the most recent one it presents), shortest
    duration (picks the single quarter over a year-to-date cumulative), then
    latest ``filed`` (the most recent restatement).
    """
    candidates = [e for e in entries if e.get("fy") == fy and e.get("fp") == fp and e.get("end")]
    if not candidates:
        return None
    max_end = max(e["end"] for e in candidates)
    candidates = [e for e in candidates if e["end"] == max_end]
    min_duration = min(_duration_days(e) for e in candidates)
    candidates = [e for e in candidates if _duration_days(e) == min_duration]
    candidates.sort(key=lambda e: str(e.get("filed", "")))
    return candidates[-1]


def _value_for_field(
    us_gaap: Mapping[str, Any], tags: Sequence[str], unit: str, fy: int, fp: str
) -> Optional[float]:
    """Return the first candidate tag's value for ``(fy, fp)``, or ``None``."""
    for tag in tags:
        entry = _select_entry(_entries_for_tag(us_gaap, tag, unit), fy, fp)
        if entry is not None:
            value = to_float(entry.get("val"))
            if value is not None:
                return value
    return None


def _discover_periods(
    us_gaap: Mapping[str, Any], fiscal_years: Sequence[int]
) -> List[Tuple[int, str]]:
    """Enumerate the distinct ``(fy, fp)`` pairs present within ``fiscal_years``.

    Returns annual periods before quarterly ones, each group ordered by fiscal
    year and quarter descending -- newest first, mirroring
    :mod:`truevalue.transform`.
    """
    wanted_years = set(fiscal_years)
    found = set()
    for tag_data in us_gaap.values():
        if not isinstance(tag_data, dict):
            continue
        units = tag_data.get("units")
        if not isinstance(units, dict):
            continue
        for entries in units.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                fy, fp = entry.get("fy"), entry.get("fp")
                if fy in wanted_years and fp in _VALID_FISCAL_PERIODS:
                    found.add((fy, fp))

    annual = sorted((fy, fp) for fy, fp in found if fp == "FY")
    quarterly = sorted((fy, fp) for fy, fp in found if fp != "FY")
    return list(reversed(annual)) + list(reversed(quarterly))


def _resolve_period_metadata(
    us_gaap: Mapping[str, Any], fy: int, fp: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(period_start_date, period_end_date, filing_date)`` for ``(fy, fp)``.

    Duration concepts (income statement / cash flow) carry both period bounds
    and the filing's ``filed`` date, and are preferred. Instant concepts
    (balance sheet) carry only ``end`` and are used as a fallback for a period
    with no duration facts at all -- ``filing_date`` then stays ``None``.
    """
    for field_tags in list(SEC_INCOME_FIELD_MAP.values()) + list(SEC_CASH_FLOW_FIELD_MAP.values()):
        for tag in field_tags:
            entry = _select_entry(_entries_for_tag(us_gaap, tag, "USD"), fy, fp)
            if entry is not None and entry.get("start"):
                return entry.get("start"), entry.get("end"), entry.get("filed")
    for field_tags in SEC_BALANCE_FIELD_MAP.values():
        for tag in field_tags:
            entry = _select_entry(_entries_for_tag(us_gaap, tag, "USD"), fy, fp)
            if entry is not None and entry.get("end"):
                return None, entry.get("end"), None
    return None, None, None


# --------------------------------------------------------------------------- #
# financial_statement_fact
# --------------------------------------------------------------------------- #


def _build_period_values(us_gaap: Mapping[str, Any], fy: int, fp: str) -> Dict[str, Optional[float]]:
    """Map raw XBRL facts of one fiscal period onto the numeric target fields."""
    values: Dict[str, Optional[float]] = {}

    for field_map in (SEC_INCOME_FIELD_MAP, SEC_BALANCE_FIELD_MAP, SEC_CASH_FLOW_FIELD_MAP):
        for target_field, tags in field_map.items():
            unit = FIELD_UNITS.get(target_field, "USD")
            values[target_field] = _value_for_field(us_gaap, tags, unit, fy, fp)

    for target_field in SIGN_FLIPPED_FIELDS:
        value = values.get(target_field)
        if value is not None:
            values[target_field] = -value

    _add_derived_values(values)
    return values


def _add_derived_values(values: Dict[str, Optional[float]]) -> None:
    """Compute the derived metrics in-place; SEC XBRL has no direct tag for any of them."""
    pretax_income = values.get("pretax_income")
    interest_expense = values.get("interest_expense")
    operating_income = values.get("operating_income")
    if pretax_income is not None and interest_expense is not None:
        values["ebit"] = pretax_income + interest_expense
    else:
        values["ebit"] = operating_income

    depreciation = values.get("depreciation_and_amortization")
    ebit = values.get("ebit")
    values["ebitda"] = safe_sum(ebit, depreciation) if ebit is not None else None

    values["ebitda_margin"] = safe_ratio(values.get("ebitda"), values.get("revenue"))
    values["effective_tax_rate"] = safe_ratio(
        values.get("income_tax_expense"), values.get("pretax_income")
    )

    short_term_debt = values.get("short_term_debt")
    long_term_debt = values.get("long_term_debt")
    values["total_debt"] = safe_sum(short_term_debt, long_term_debt)

    current_assets = values.get("total_current_assets")
    current_liabilities = values.get("current_liabilities")
    cash = values.get("cash_and_cash_equivalents")
    short_term_investments = values.get("short_term_investments")
    total_debt = values.get("total_debt")
    equity = values.get("shareholders_equity")

    if current_assets is not None and current_liabilities is not None:
        values["working_capital"] = current_assets - current_liabilities
        operating_assets = current_assets - (safe_sum(cash, short_term_investments) or 0.0)
        operating_liabilities = current_liabilities - (short_term_debt or 0.0)
        values["net_working_capital"] = operating_assets - operating_liabilities
    else:
        values["working_capital"] = None
        values["net_working_capital"] = None

    if total_debt is not None:
        liquid = safe_sum(cash, short_term_investments) or 0.0
        values["net_debt"] = total_debt - liquid
    else:
        values["net_debt"] = None

    if total_debt is not None and equity is not None:
        values["invested_capital"] = total_debt + equity - (cash or 0.0)
    else:
        values["invested_capital"] = None

    operating_cash_flow = values.get("operating_cash_flow")
    capital_expenditure = values.get("capital_expenditure")
    if operating_cash_flow is not None and capital_expenditure is not None:
        # capital_expenditure is already negative (see SIGN_FLIPPED_FIELDS).
        values["free_cash_flow"] = operating_cash_flow + capital_expenditure
    else:
        values["free_cash_flow"] = None

    if values.get("dividend_per_share") is None:
        dividends_paid = values.get("dividends_paid")
        shares = values.get("basic_shares_outstanding") or values.get("diluted_shares_outstanding")
        if dividends_paid is not None and shares:
            # dividends_paid is already negative; per-share figures are positive.
            values["dividend_per_share"] = abs(dividends_paid) / shares


def build_financial_statement_records(
    raw: RawSecCompanyData,
    context: RunContext,
    fiscal_years: Sequence[int] = DEFAULT_FISCAL_YEARS,
) -> List[Dict[str, Any]]:
    """Map a ticker's XBRL facts onto ``financial_statement_fact`` records.

    The row grain depends on ``context.statement_type_mode``; see
    :mod:`truevalue.transform` for the ``period``/``statement`` semantics,
    shared verbatim by this source.

    Args:
        raw: Extracted SEC payloads for one ticker.
        context: Per-run settings.
        fiscal_years: Which XBRL fiscal years (the ``fy`` label SEC assigns,
            not the calendar year) to build rows for.

    Returns:
        Records ordered by fiscal year and fiscal period descending (annual
        before quarterly within a year) -- the ``(fy, fp)`` label order from
        :func:`_discover_periods`, which coincides with actual period-end order
        except in the rare case of a filer whose fiscal labels and calendar
        period ends disagree.
    """
    us_gaap = raw.facts
    periods = _discover_periods(us_gaap, fiscal_years)
    if not periods:
        logger.warning(
            "%s: no SEC financial facts for fiscal year(s) %s",
            raw.ticker,
            ", ".join(str(year) for year in fiscal_years),
        )
        return []

    records: List[Dict[str, Any]] = []
    for fiscal_year, fiscal_period in periods:
        records.extend(_build_records_for_period(raw, context, us_gaap, fiscal_year, fiscal_period))
    return records


def _build_records_for_period(
    raw: RawSecCompanyData,
    context: RunContext,
    us_gaap: Mapping[str, Any],
    fiscal_year: int,
    fiscal_period: str,
) -> List[Dict[str, Any]]:
    """Build the record(s) for a single fiscal period, honouring the row mode."""
    values = _build_period_values(us_gaap, fiscal_year, fiscal_period)
    period_start_date, period_end_date, filing_date = _resolve_period_metadata(
        us_gaap, fiscal_year, fiscal_period
    )
    is_quarterly = fiscal_period != "FY"
    fiscal_quarter = _QUARTER_NUMBERS.get(fiscal_period) if is_quarterly else None

    common = {
        "company_id": context.company_id(raw.ticker),
        "ticker": raw.ticker,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "period_start_date": period_start_date,
        "period_end_date": period_end_date,
        "filing_date": filing_date,
        "source_system": context.source_system,
        "ingestion_timestamp": context.ingestion_timestamp,
    }

    if context.statement_type_mode == STATEMENT_MODE_PERIOD:
        statement_type = STATEMENT_TYPE_QUARTERLY if is_quarterly else STATEMENT_TYPE_ANNUAL
        record = dict(common)
        record["statement_type"] = statement_type
        record["source_record_id"] = build_source_record_id(
            raw.ticker, statement_type, period_end_date
        )
        record.update(values)
        return [record]

    records: List[Dict[str, Any]] = []
    for statement_type, group_fields in STATEMENT_FIELD_GROUPS.items():
        subset = {name: values.get(name) for name in group_fields}
        if all(value is None for value in subset.values()):
            continue
        record = dict(common)
        record["statement_type"] = statement_type
        record["source_record_id"] = build_source_record_id(
            raw.ticker, statement_type, period_end_date
        )
        record.update(subset)
        records.append(record)
    return records


# --------------------------------------------------------------------------- #
# company_dimension
# --------------------------------------------------------------------------- #


def _exchange_for_ticker(submissions: Mapping[str, Any], ticker: str) -> Optional[str]:
    """Return the exchange matching ``ticker`` within the parallel tickers/exchanges arrays."""
    tickers = submissions.get("tickers") or []
    exchanges = submissions.get("exchanges") or []
    value: Any = None
    if ticker in tickers:
        index = tickers.index(ticker)
        if index < len(exchanges):
            value = exchanges[index]
        elif exchanges:
            # Malformed/short exchanges array; fall back rather than silently
            # returning None for a ticker SEC does list.
            value = exchanges[0]
    elif exchanges:
        value = exchanges[0]
    return str(value) if value else None


def _country_from_business_address(submissions: Mapping[str, Any]) -> Optional[str]:
    """Derive a country code from the submissions ``business`` address.

    SEC's address block has no populated ``country``/``countryCode`` for
    domestic filers -- only ``isForeignLocation`` and a US state in
    ``stateOrCountry``. Foreign filers set ``isForeignLocation`` and populate
    ``countryCode`` instead.
    """
    addresses = submissions.get("addresses")
    if not isinstance(addresses, dict):
        return None
    business = addresses.get("business")
    if not isinstance(business, dict):
        return None
    if business.get("isForeignLocation"):
        return business.get("countryCode") or business.get("country")
    if business.get("stateOrCountry"):
        return "US"
    return None


def build_company_dimension_record(
    raw: RawSecCompanyData, context: RunContext
) -> Dict[str, Any]:
    """Map a ticker's SEC submissions profile onto a ``company_dimension`` record.

    Args:
        raw: Extracted SEC payloads for one ticker.
        context: Per-run settings.

    Returns:
        A target-shaped dict. Coercion to the declared types happens later, in
        :meth:`truevalue.schemas.TargetSchema.coerce`.
    """
    submissions = raw.submissions
    exchange = _exchange_for_ticker(submissions, raw.ticker)

    return {
        "company_id": context.company_id(raw.ticker),
        "ticker": raw.ticker,
        "cik": raw.cik,
        # CUSIP/ISIN are not exposed by the SEC submissions or XBRL APIs.
        "cusip": None,
        "isin": None,
        "company_name": submissions.get("name"),
        "legal_name": submissions.get("name"),
        "exchange": exchange,
        # SEC has no market/MIC equivalent to yfinance's "us_market".
        "market": None,
        "country": _country_from_business_address(submissions),
        # SEC's domestic filer forms (10-K/10-Q) report in USD; a foreign
        # private issuer reporting in another currency is out of scope here.
        "currency": "USD",
        # SEC exposes only the SIC code/description, no broader sector taxonomy.
        "sector": None,
        "industry": submissions.get("sicDescription"),
        "sic_code": submissions.get("sic"),
        "naics_code": None,
        "website": submissions.get("website") or None,
        # Not exposed by SEC EDGAR; would need an external listing-date source.
        "ipo_date": None,
        # Not exposed by SEC EDGAR (no standard XBRL headcount tag).
        "employees": None,
        "business_description": submissions.get("description") or None,
        # Reflects *this* ticker's own listing, not the filer's as a whole --
        # a multi-class filer (e.g. dual-class shares) can have one class
        # delisted while another still trades. EDGAR keeps a deregistered
        # filer's historical data without an explicit active/inactive flag.
        "active_flag": bool(exchange),
        "source_system": context.source_system,
        "created_timestamp": context.ingestion_timestamp,
        "updated_timestamp": context.ingestion_timestamp,
    }


# --------------------------------------------------------------------------- #
# Entry point used by the pipeline
# --------------------------------------------------------------------------- #


def transform_ticker(
    raw: RawSecCompanyData,
    context: RunContext,
    fiscal_years: Sequence[int] = DEFAULT_FISCAL_YEARS,
) -> Dict[str, List[Dict[str, Any]]]:
    """Transform one ticker's raw SEC payloads into the supported target schemas.

    Args:
        raw: Extracted SEC payloads for one ticker.
        context: Per-run settings.
        fiscal_years: Which XBRL fiscal years to build financial statement rows for.

    Returns:
        Mapping of schema name to its list of target-shaped records.
        ``market_price_fact`` is intentionally absent -- SEC EDGAR has no price data.
    """
    return {
        "company_dimension": [build_company_dimension_record(raw, context)],
        "financial_statement_fact": build_financial_statement_records(
            raw, context, fiscal_years
        ),
    }
