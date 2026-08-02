"""Transformation layer: maps raw yfinance payloads onto the target schemas.

Design notes
------------
* The yfinance -> target field mappings live in the ``*_FIELD_MAP`` constants at
  the top of this module.  Each target field maps to a *tuple* of candidate
  yfinance labels; the first one present and non-null wins.  Yahoo renames and
  reorganises statement line items over time, so the fallbacks matter.
* Fields the target schema declares but yfinance cannot supply are emitted as
  ``None`` rather than being guessed.  See :data:`UNAVAILABLE_FIELDS`.
* Derived fields (``ebitda_margin``, ``working_capital``, ...) are computed only
  when all their inputs are present, and prefer a native yfinance line item over
  the computation when Yahoo reports one.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .extract import RawTickerData
from .schemas import to_float

logger = logging.getLogger(__name__)

#: Value written to every ``source_system`` column.
SOURCE_SYSTEM = "yfinance"

#: Namespace for deterministic company_id generation.  Fixed constant: changing
#: it would re-key the entire warehouse.
COMPANY_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

COMPANY_ID_STRATEGY_UUID5 = "uuid5"
COMPANY_ID_STRATEGY_TICKER = "ticker"
COMPANY_ID_STRATEGIES = (COMPANY_ID_STRATEGY_UUID5, COMPANY_ID_STRATEGY_TICKER)

#: One merged row per fiscal period; ``statement_type`` is ANNUAL / QUARTERLY.
STATEMENT_MODE_PERIOD = "period"
#: Three sparse rows per fiscal period, one per financial statement.
STATEMENT_MODE_STATEMENT = "statement"
STATEMENT_MODES = (STATEMENT_MODE_PERIOD, STATEMENT_MODE_STATEMENT)

STATEMENT_TYPE_ANNUAL = "ANNUAL"
STATEMENT_TYPE_QUARTERLY = "QUARTERLY"
STATEMENT_TYPE_INCOME = "INCOME_STATEMENT"
STATEMENT_TYPE_BALANCE = "BALANCE_SHEET"
STATEMENT_TYPE_CASH_FLOW = "CASH_FLOW"

#: Target fields no yfinance endpoint provides.  Always emitted as NULL.
UNAVAILABLE_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "company_dimension": ("cik", "cusip", "sic_code", "naics_code"),
    "financial_statement_fact": ("filing_date",),
    "market_price_fact": (),
}

# --------------------------------------------------------------------------- #
# yfinance -> target field maps
# --------------------------------------------------------------------------- #

INCOME_FIELD_MAP: Mapping[str, Tuple[str, ...]] = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "cost_of_revenue": ("Cost Of Revenue", "Reconciled Cost Of Revenue"),
    "gross_profit": ("Gross Profit",),
    "research_and_development": ("Research And Development",),
    "selling_general_administrative": (
        "Selling General And Administration",
        "Selling General And Administrative",
    ),
    "operating_income": ("Operating Income", "Total Operating Income As Reported"),
    "ebit": ("EBIT",),
    "ebitda": ("EBITDA", "Normalized EBITDA"),
    "interest_income": ("Interest Income", "Interest Income Non Operating"),
    "interest_expense": ("Interest Expense", "Interest Expense Non Operating"),
    "pretax_income": ("Pretax Income",),
    "income_tax_expense": ("Tax Provision", "Income Tax Expense"),
    "net_income": (
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
    ),
    "eps_basic": ("Basic EPS",),
    "eps_diluted": ("Diluted EPS",),
    "basic_shares_outstanding": ("Basic Average Shares",),
    "diluted_shares_outstanding": ("Diluted Average Shares",),
}

BALANCE_FIELD_MAP: Mapping[str, Tuple[str, ...]] = {
    "cash_and_cash_equivalents": (
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ),
    "short_term_investments": ("Other Short Term Investments", "Short Term Investments"),
    "accounts_receivable": ("Accounts Receivable", "Receivables"),
    "inventory": ("Inventory",),
    "total_current_assets": ("Current Assets", "Total Current Assets"),
    "property_plant_equipment": ("Net PPE", "Gross PPE"),
    "goodwill": ("Goodwill",),
    "intangible_assets": (
        "Other Intangible Assets",
        "Goodwill And Other Intangible Assets",
    ),
    "total_assets": ("Total Assets",),
    "accounts_payable": ("Accounts Payable", "Payables"),
    "current_liabilities": ("Current Liabilities", "Total Current Liabilities"),
    "short_term_debt": ("Current Debt", "Current Debt And Capital Lease Obligation"),
    "long_term_debt": ("Long Term Debt", "Long Term Debt And Capital Lease Obligation"),
    "total_debt": ("Total Debt",),
    "total_liabilities": (
        "Total Liabilities Net Minority Interest",
        "Total Liabilities",
    ),
    "shareholders_equity": (
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
    ),
    "retained_earnings": ("Retained Earnings",),
    # Native line items, preferred over the derived computation further below.
    "working_capital": ("Working Capital",),
    "net_debt": ("Net Debt",),
    "invested_capital": ("Invested Capital",),
}

CASH_FLOW_FIELD_MAP: Mapping[str, Tuple[str, ...]] = {
    "operating_cash_flow": (
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ),
    "capital_expenditure": ("Capital Expenditure", "Purchase Of PPE"),
    "free_cash_flow": ("Free Cash Flow",),
    "investing_cash_flow": (
        "Investing Cash Flow",
        "Cash Flow From Continuing Investing Activities",
    ),
    "financing_cash_flow": (
        "Financing Cash Flow",
        "Cash Flow From Continuing Financing Activities",
    ),
    "depreciation_and_amortization": (
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
    ),
    "stock_based_compensation": ("Stock Based Compensation",),
    "dividends_paid": ("Cash Dividends Paid", "Common Stock Dividend Paid"),
}

#: Which target fields belong to which statement, used by ``statement`` mode to
#: blank out everything that does not originate from the row's own statement.
#: Derived fields are assigned to the statement supplying their inputs.
INCOME_DERIVED_FIELDS = ("ebitda_margin", "effective_tax_rate")
BALANCE_DERIVED_FIELDS = (
    "working_capital",
    "net_working_capital",
    "net_debt",
    "invested_capital",
)
CASH_FLOW_DERIVED_FIELDS = ("dividend_per_share",)

STATEMENT_FIELD_GROUPS: Mapping[str, Tuple[str, ...]] = {
    STATEMENT_TYPE_INCOME: tuple(INCOME_FIELD_MAP) + INCOME_DERIVED_FIELDS,
    STATEMENT_TYPE_BALANCE: tuple(BALANCE_FIELD_MAP) + BALANCE_DERIVED_FIELDS,
    STATEMENT_TYPE_CASH_FLOW: tuple(CASH_FLOW_FIELD_MAP) + CASH_FLOW_DERIVED_FIELDS,
}

#: ``Ticker.info`` key -> ``market_price_fact`` column.  These are point-in-time
#: snapshots and are therefore only attached to the most recent trade date.
SNAPSHOT_FIELD_MAP: Mapping[str, str] = {
    "market_cap": "marketCap",
    "enterprise_value": "enterpriseValue",
    "shares_outstanding": "sharesOutstanding",
    "pe_ratio": "trailingPE",
    "price_to_book_ratio": "priceToBook",
    "price_to_sales_ratio": "priceToSalesTrailing12Months",
    "ev_to_ebitda": "enterpriseToEbitda",
    "dividend_yield": "dividendYield",
}


@dataclass(frozen=True)
class RunContext:
    """Per-run settings shared by every transform function.

    Attributes:
        ingestion_timestamp: ISO-8601 UTC stamp written to all audit columns.
        source_system: Value for the ``source_system`` columns.
        company_id_strategy: One of :data:`COMPANY_ID_STRATEGIES`.
        statement_type_mode: One of :data:`STATEMENT_MODES`.
    """

    ingestion_timestamp: str
    source_system: str = SOURCE_SYSTEM
    company_id_strategy: str = COMPANY_ID_STRATEGY_UUID5
    statement_type_mode: str = STATEMENT_MODE_PERIOD

    @classmethod
    def create(
        cls,
        company_id_strategy: str = COMPANY_ID_STRATEGY_UUID5,
        statement_type_mode: str = STATEMENT_MODE_PERIOD,
        source_system: str = SOURCE_SYSTEM,
    ) -> "RunContext":
        """Build a context stamped with the current UTC time.

        Raises:
            ValueError: If a strategy or mode value is not recognised.
        """
        if company_id_strategy not in COMPANY_ID_STRATEGIES:
            raise ValueError(
                "Unknown company_id strategy {!r}; expected one of {}".format(
                    company_id_strategy, COMPANY_ID_STRATEGIES
                )
            )
        if statement_type_mode not in STATEMENT_MODES:
            raise ValueError(
                "Unknown statement_type mode {!r}; expected one of {}".format(
                    statement_type_mode, STATEMENT_MODES
                )
            )
        return cls(
            ingestion_timestamp=utc_now_iso(),
            source_system=source_system,
            company_id_strategy=company_id_strategy,
            statement_type_mode=statement_type_mode,
        )

    def company_id(self, ticker: str) -> str:
        """Return the stable surrogate key for ``ticker``.

        ``uuid5`` yields a deterministic UUID string, so re-running the pipeline
        produces identical keys without any persisted state.  ``ticker`` uses the
        symbol itself, which is more readable but breaks if a symbol is reassigned.
        """
        symbol = ticker.strip().upper()
        if self.company_id_strategy == COMPANY_ID_STRATEGY_TICKER:
            return symbol
        return str(uuid.uuid5(COMPANY_ID_NAMESPACE, "us:{}".format(symbol)))


def utc_now_iso() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _to_date_string(value: Any) -> Optional[str]:
    """Format any date-ish value as ``YYYY-MM-DD``, or ``None`` if unusable."""
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.strftime("%Y-%m-%d")


def _epoch_to_date_string(value: Any, unit: str = "s") -> Optional[str]:
    """Convert a UTC epoch value to ``YYYY-MM-DD``.

    Args:
        value: Epoch offset.
        unit: ``"s"`` for seconds or ``"ms"`` for milliseconds.
    """
    epoch = to_float(value)
    if epoch is None:
        return None
    if unit == "ms":
        epoch /= 1000.0
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return None


def _infer_fiscal_year_end_month(period_ends: Sequence[pd.Timestamp]) -> int:
    """Infer the month a company's fiscal year ends in.

    Yahoo does not label fiscal periods, so the anchor is taken from the annual
    statements: the most frequent month among their period end dates.  Falls back
    to December (calendar year) when there are no annual statements.

    Args:
        period_ends: Period end dates of the annual statements.

    Returns:
        Month number 1-12.
    """
    months = [stamp.month for stamp in period_ends if stamp is not None]
    if not months:
        return 12
    return Counter(months).most_common(1)[0][0]


def _fiscal_year_and_quarter(
    period_end: pd.Timestamp, is_quarterly: bool, fiscal_year_end_month: int
) -> Tuple[int, Optional[int]]:
    """Derive ``fiscal_year`` and ``fiscal_quarter`` relative to the fiscal calendar.

    For a company whose fiscal year ends in September, the quarter ending
    31 December belongs to fiscal Q1 of the *following* fiscal year -- using
    calendar quarters here would mislabel three quarters out of four.

    Args:
        period_end: The period's end date.
        is_quarterly: Whether this is a quarterly period.
        fiscal_year_end_month: Month the fiscal year ends in, 1-12.

    Returns:
        ``(fiscal_year, fiscal_quarter)``; the quarter is ``None`` for annual rows.

    Note:
        Driven by the calendar month of the period end.  Companies on a 52/53-week
        retail calendar occasionally end a period a few days into the next month,
        which shifts that period by one quarter.
    """
    month = period_end.month
    fiscal_year = period_end.year if month <= fiscal_year_end_month else period_end.year + 1
    if not is_quarterly:
        return fiscal_year, None
    fiscal_quarter = ((month - fiscal_year_end_month - 1) % 12) // 3 + 1
    return fiscal_year, fiscal_quarter


def _first_present(source: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    """Return the first non-null numeric value among ``keys``."""
    for key in keys:
        if key in source:
            value = to_float(source[key])
            if value is not None:
                return value
    return None


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide, returning ``None`` on missing inputs or a zero denominator."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _safe_sum(*values: Optional[float]) -> Optional[float]:
    """Sum values, treating ``None`` as zero but returning ``None`` if all are ``None``."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


# --------------------------------------------------------------------------- #
# company_dimension
# --------------------------------------------------------------------------- #


def build_company_dimension_record(
    raw: RawTickerData, context: RunContext
) -> Dict[str, Any]:
    """Map a ticker's profile onto a ``company_dimension`` record.

    Args:
        raw: Extracted payloads for one ticker.
        context: Per-run settings.

    Returns:
        A target-shaped dict.  Coercion to the declared types happens later, in
        :meth:`truevalue.schemas.TargetSchema.coerce`.
    """
    info = raw.info

    # yfinance exposes no delisting flag. Treat "we got a profile and a price"
    # as active; anything else is flagged inactive for downstream review.
    has_price = bool(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    active_flag = raw.has_profile and (has_price or not raw.history.empty)

    return {
        "company_id": context.company_id(raw.ticker),
        "ticker": raw.ticker,
        # cik / cusip / naics_code / sic_code are not exposed by yfinance.
        "cik": None,
        "cusip": None,
        "isin": raw.isin,
        "company_name": info.get("shortName") or info.get("displayName"),
        "legal_name": info.get("longName"),
        "exchange": info.get("exchange"),
        "market": info.get("market"),
        "country": info.get("country"),
        "currency": info.get("currency"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "sic_code": None,
        "naics_code": None,
        "website": info.get("website"),
        # First trade date is a proxy for the IPO date -- it is the earliest date
        # Yahoo has a quote for, which for re-listings is not the true IPO.
        # yfinance <1.0 exposed epoch seconds, >=1.0 only milliseconds.
        "ipo_date": _epoch_to_date_string(info.get("firstTradeDateEpochUtc"), "s")
        or _epoch_to_date_string(info.get("firstTradeDateMilliseconds"), "ms"),
        "employees": info.get("fullTimeEmployees"),
        "business_description": info.get("longBusinessSummary"),
        "active_flag": active_flag,
        "source_system": context.source_system,
        "created_timestamp": context.ingestion_timestamp,
        "updated_timestamp": context.ingestion_timestamp,
    }


# --------------------------------------------------------------------------- #
# market_price_fact
# --------------------------------------------------------------------------- #


def build_market_price_records(
    raw: RawTickerData, context: RunContext
) -> List[Dict[str, Any]]:
    """Map a ticker's daily price history onto ``market_price_fact`` records.

    One record per trading day.  Valuation metrics (``market_cap``, ``pe_ratio``,
    ...) are point-in-time snapshots from ``Ticker.info``; yfinance offers no
    history for them, so they are attached to the latest trade date only and left
    ``NULL`` on all earlier rows rather than back-filled with today's values.

    Args:
        raw: Extracted payloads for one ticker.
        context: Per-run settings.

    Returns:
        Records ordered by ``trade_date`` ascending; empty if there is no history.
    """
    history = raw.history
    if history.empty:
        logger.warning("%s: no price history, skipping market_price_fact", raw.ticker)
        return []

    company_id = context.company_id(raw.ticker)
    records: List[Dict[str, Any]] = []

    for index_value, row in history.iterrows():
        trade_date = _to_date_string(index_value)
        if trade_date is None:
            continue

        close_price = to_float(row.get("Close"))
        records.append(
            {
                "company_id": company_id,
                "ticker": raw.ticker,
                "trade_date": trade_date,
                "open_price": to_float(row.get("Open")),
                "high_price": to_float(row.get("High")),
                "low_price": to_float(row.get("Low")),
                "close_price": close_price,
                # auto_adjust=False keeps both columns; older yfinance builds
                # occasionally omit "Adj Close", so fall back to the raw close.
                "adjusted_close_price": to_float(row.get("Adj Close"))
                if "Adj Close" in row
                else close_price,
                "volume": to_float(row.get("Volume")),
                "market_cap": None,
                "enterprise_value": None,
                "shares_outstanding": None,
                "pe_ratio": None,
                "price_to_book_ratio": None,
                "price_to_sales_ratio": None,
                "ev_to_ebitda": None,
                "dividend_yield": None,
                "source_system": context.source_system,
                "ingestion_timestamp": context.ingestion_timestamp,
            }
        )

    if not records:
        return []

    records.sort(key=lambda record: record["trade_date"])
    _attach_valuation_snapshot(records[-1], raw)
    return records


def _attach_valuation_snapshot(record: Dict[str, Any], raw: RawTickerData) -> None:
    """Write the ``Ticker.info`` valuation snapshot onto the latest price row.

    Note on ``dividend_yield``: the value is passed through exactly as yfinance
    reports it.  yfinance changed the unit from a fraction (0.0044) to a percent
    (0.44) in 0.2.55; normalising here would silently corrupt one of the two, so
    the unit follows the installed yfinance version.
    """
    info = raw.info
    if not info:
        return

    for target_field, info_key in SNAPSHOT_FIELD_MAP.items():
        record[target_field] = to_float(info.get(info_key))

    logger.debug(
        "%s: attached valuation snapshot to trade_date %s",
        raw.ticker,
        record["trade_date"],
    )


# --------------------------------------------------------------------------- #
# financial_statement_fact
# --------------------------------------------------------------------------- #


def _statement_by_period(frame: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, Any]]:
    """Pivot a yfinance statement frame into ``{period_end: {line_item: value}}``.

    yfinance returns statements with line items on the index and period end dates
    on the columns, which is the transpose of what a fact row needs.
    """
    if frame.empty:
        return {}

    by_period: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for column in frame.columns:
        try:
            period_end = pd.Timestamp(column)
        except (ValueError, TypeError):
            logger.debug("Skipping non-date statement column %r", column)
            continue
        if pd.isna(period_end):
            continue
        series = frame[column]
        by_period[period_end] = {
            str(label): value for label, value in series.items()
        }
    return by_period


def _derive_period_start(period_end: pd.Timestamp, is_quarterly: bool) -> Optional[str]:
    """Derive ``period_start_date`` from the period end.

    yfinance reports only the period end, so the start is reconstructed as
    "end minus one period, plus one day".  This is exact for calendar-aligned
    fiscal periods and off by a few days for 52/53-week retail calendars.
    """
    offset = pd.DateOffset(months=3) if is_quarterly else pd.DateOffset(years=1)
    try:
        start = period_end - offset + pd.Timedelta(days=1)
    except (ValueError, OverflowError):
        return None
    return _to_date_string(start)


def _build_period_values(
    income: Mapping[str, Any],
    balance: Mapping[str, Any],
    cash_flow: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    """Map raw line items of one period onto the numeric target fields."""
    values: Dict[str, Optional[float]] = {}

    for target_field, labels in INCOME_FIELD_MAP.items():
        values[target_field] = _first_present(income, labels)
    for target_field, labels in BALANCE_FIELD_MAP.items():
        values[target_field] = _first_present(balance, labels)
    for target_field, labels in CASH_FLOW_FIELD_MAP.items():
        values[target_field] = _first_present(cash_flow, labels)

    _add_derived_values(values)
    return values


def _add_derived_values(values: Dict[str, Optional[float]]) -> None:
    """Compute the derived metrics in-place, leaving natives untouched.

    Fields already populated from a native yfinance line item (``working_capital``,
    ``net_debt``, ``invested_capital``) keep Yahoo's value; only gaps are filled.
    """
    values["ebitda_margin"] = _safe_ratio(values.get("ebitda"), values.get("revenue"))
    values["effective_tax_rate"] = _safe_ratio(
        values.get("income_tax_expense"), values.get("pretax_income")
    )

    current_assets = values.get("total_current_assets")
    current_liabilities = values.get("current_liabilities")
    cash = values.get("cash_and_cash_equivalents")
    short_term_investments = values.get("short_term_investments")
    short_term_debt = values.get("short_term_debt")
    total_debt = values.get("total_debt")
    equity = values.get("shareholders_equity")

    if values.get("working_capital") is None:
        if current_assets is not None and current_liabilities is not None:
            values["working_capital"] = current_assets - current_liabilities

    # Operating (net) working capital: current assets and liabilities excluding
    # the financing items, i.e. cash/short-term investments and short-term debt.
    if current_assets is not None and current_liabilities is not None:
        operating_assets = current_assets - (_safe_sum(cash, short_term_investments) or 0.0)
        operating_liabilities = current_liabilities - (short_term_debt or 0.0)
        values["net_working_capital"] = operating_assets - operating_liabilities
    else:
        values["net_working_capital"] = None

    if values.get("net_debt") is None:
        if total_debt is not None:
            liquid = _safe_sum(cash, short_term_investments) or 0.0
            values["net_debt"] = total_debt - liquid

    if values.get("invested_capital") is None:
        if total_debt is not None and equity is not None:
            values["invested_capital"] = total_debt + equity - (cash or 0.0)

    # Yahoo reports dividends paid as a negative cash outflow; per-share figures
    # are conventionally positive.
    dividends_paid = values.get("dividends_paid")
    shares = values.get("basic_shares_outstanding") or values.get(
        "diluted_shares_outstanding"
    )
    if dividends_paid is not None and shares:
        values["dividend_per_share"] = abs(dividends_paid) / shares
    else:
        values["dividend_per_share"] = None

    if values.get("free_cash_flow") is None:
        operating_cash_flow = values.get("operating_cash_flow")
        capital_expenditure = values.get("capital_expenditure")
        if operating_cash_flow is not None and capital_expenditure is not None:
            # Capital Expenditure is reported as a negative number by Yahoo.
            values["free_cash_flow"] = operating_cash_flow + capital_expenditure


def build_financial_statement_records(
    raw: RawTickerData, context: RunContext
) -> List[Dict[str, Any]]:
    """Map a ticker's statements onto ``financial_statement_fact`` records.

    The row grain depends on ``context.statement_type_mode``:

    * ``period``    -- one merged row per fiscal period, ``statement_type`` is
      ``ANNUAL`` or ``QUARTERLY``.
    * ``statement`` -- three sparse rows per fiscal period, ``statement_type`` is
      ``INCOME_STATEMENT`` / ``BALANCE_SHEET`` / ``CASH_FLOW``; annual periods are
      distinguished from quarterly ones by ``fiscal_quarter IS NULL``.

    Args:
        raw: Extracted payloads for one ticker.
        context: Per-run settings.

    Returns:
        Records ordered by period end descending, annual before quarterly.
    """
    records: List[Dict[str, Any]] = []

    frame_sets = (
        (False, raw.income_annual, raw.balance_annual, raw.cashflow_annual),
        (True, raw.income_quarterly, raw.balance_quarterly, raw.cashflow_quarterly),
    )

    # Anchor the fiscal calendar on the annual statements before mapping any
    # period, so quarterly rows get the correct fiscal year and quarter.
    annual_period_ends = list(
        set(_statement_by_period(raw.income_annual))
        | set(_statement_by_period(raw.balance_annual))
        | set(_statement_by_period(raw.cashflow_annual))
    )
    fiscal_year_end_month = _infer_fiscal_year_end_month(annual_period_ends)
    logger.debug(
        "%s: fiscal year end month inferred as %d", raw.ticker, fiscal_year_end_month
    )

    for is_quarterly, income_frame, balance_frame, cash_flow_frame in frame_sets:
        income = _statement_by_period(income_frame)
        balance = _statement_by_period(balance_frame)
        cash_flow = _statement_by_period(cash_flow_frame)

        period_ends = sorted(
            set(income) | set(balance) | set(cash_flow), reverse=True
        )
        if not period_ends:
            continue

        for period_end in period_ends:
            records.extend(
                _build_records_for_period(
                    raw=raw,
                    context=context,
                    period_end=period_end,
                    is_quarterly=is_quarterly,
                    fiscal_year_end_month=fiscal_year_end_month,
                    income=income.get(period_end, {}),
                    balance=balance.get(period_end, {}),
                    cash_flow=cash_flow.get(period_end, {}),
                )
            )

    if not records:
        logger.warning(
            "%s: no financial statements available, skipping financial_statement_fact",
            raw.ticker,
        )
    return records


def _build_records_for_period(
    raw: RawTickerData,
    context: RunContext,
    period_end: pd.Timestamp,
    is_quarterly: bool,
    fiscal_year_end_month: int,
    income: Mapping[str, Any],
    balance: Mapping[str, Any],
    cash_flow: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Build the record(s) for a single fiscal period, honouring the row mode."""
    values = _build_period_values(income, balance, cash_flow)

    period_end_date = _to_date_string(period_end)
    fiscal_year, fiscal_quarter = _fiscal_year_and_quarter(
        period_end, is_quarterly, fiscal_year_end_month
    )

    common = {
        "company_id": context.company_id(raw.ticker),
        "ticker": raw.ticker,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "period_start_date": _derive_period_start(period_end, is_quarterly),
        "period_end_date": period_end_date,
        # yfinance exposes no SEC filing date; it would require an EDGAR lookup.
        "filing_date": None,
        "source_system": context.source_system,
        "ingestion_timestamp": context.ingestion_timestamp,
    }

    if context.statement_type_mode == STATEMENT_MODE_PERIOD:
        statement_type = (
            STATEMENT_TYPE_QUARTERLY if is_quarterly else STATEMENT_TYPE_ANNUAL
        )
        record = dict(common)
        record["statement_type"] = statement_type
        record["source_record_id"] = _source_record_id(
            raw.ticker, statement_type, period_end_date
        )
        record.update(values)
        return [record]

    records: List[Dict[str, Any]] = []
    for statement_type, group_fields in STATEMENT_FIELD_GROUPS.items():
        subset = {name: values.get(name) for name in group_fields}
        if all(value is None for value in subset.values()):
            # Nothing was reported for this statement in this period.
            continue
        record = dict(common)
        record["statement_type"] = statement_type
        record["source_record_id"] = _source_record_id(
            raw.ticker, statement_type, period_end_date
        )
        record.update(subset)
        records.append(record)
    return records


def _source_record_id(
    ticker: str, statement_type: str, period_end_date: Optional[str]
) -> str:
    """Build a deterministic natural key for lineage back to the source."""
    return "{}:{}:{}".format(ticker, statement_type, period_end_date or "unknown")


# --------------------------------------------------------------------------- #
# Entry point used by the pipeline
# --------------------------------------------------------------------------- #


def transform_ticker(
    raw: RawTickerData, context: RunContext
) -> Dict[str, List[Dict[str, Any]]]:
    """Transform one ticker's raw payloads into all three target schemas.

    Args:
        raw: Extracted payloads for one ticker.
        context: Per-run settings.

    Returns:
        Mapping of schema name to its list of target-shaped records.
    """
    return {
        "company_dimension": [build_company_dimension_record(raw, context)],
        "financial_statement_fact": build_financial_statement_records(raw, context),
        "market_price_fact": build_market_price_records(raw, context),
    }
