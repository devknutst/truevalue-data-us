"""Extraction layer: everything that talks to yfinance.

This module performs no mapping onto the target schemas.  It returns the raw
yfinance payloads (``dict`` for the profile, ``pandas.DataFrame`` for prices and
statements) wrapped in :class:`RawTickerData`.

Every individual yfinance call is isolated: a failure on one endpoint (say, the
quarterly cash flow) degrades that one attribute to an empty payload and is
recorded in :attr:`RawTickerData.errors`, but never aborts the ticker, let alone
the run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: yfinance returns this placeholder instead of an ISIN for most US listings.
_ISIN_PLACEHOLDER = {"-", "", "N/A"}

#: For an unknown symbol yfinance does not raise -- it returns a stub dict such
#: as ``{"trailingPegRatio": None}``, which is truthy but carries no identity.
#: A profile only counts as real if at least one of these keys has a value.
_PROFILE_IDENTITY_KEYS = (
    "shortName",
    "longName",
    "displayName",
    "symbol",
    "exchange",
    "quoteType",
)


class ExtractionError(RuntimeError):
    """Raised when no usable data at all could be retrieved for a ticker."""


@dataclass
class RawTickerData:
    """Raw, untransformed yfinance payloads for a single ticker.

    Attributes:
        ticker: The requested symbol, upper-cased.
        info: ``Ticker.info`` profile dictionary (empty if unavailable).
        isin: ``Ticker.isin``, normalised to ``None`` for yfinance placeholders.
        history: Daily OHLCV frame with unadjusted ``Close`` and ``Adj Close``.
        income_annual / balance_annual / cashflow_annual: Annual statements.
        income_quarterly / balance_quarterly / cashflow_quarterly: Quarterly ones.
        errors: Human-readable messages for endpoints that failed.
    """

    ticker: str
    info: Dict[str, Any] = field(default_factory=dict)
    isin: Optional[str] = None
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    income_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    income_quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow_quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: List[str] = field(default_factory=list)

    @property
    def has_profile(self) -> bool:
        """Whether the profile carries actual identifying content.

        Guards against the stub dict yfinance returns for unknown symbols; see
        :data:`_PROFILE_IDENTITY_KEYS`.
        """
        return any(self.info.get(key) for key in _PROFILE_IDENTITY_KEYS)

    @property
    def has_any_data(self) -> bool:
        """Whether at least one endpoint returned something usable."""
        return self.has_profile or not self.history.empty or self.has_statements

    @property
    def has_statements(self) -> bool:
        """Whether at least one financial statement frame is non-empty."""
        return any(
            not frame.empty
            for frame in (
                self.income_annual,
                self.balance_annual,
                self.cashflow_annual,
                self.income_quarterly,
                self.balance_quarterly,
                self.cashflow_quarterly,
            )
        )


class YFinanceExtractor:
    """Fetches raw data for tickers from yfinance.

    Args:
        history_period: ``Ticker.history`` period, e.g. ``"1y"``, ``"5y"``, ``"max"``.
        history_interval: Bar interval; the target schema is daily-grained, so
            values other than ``"1d"`` will violate the ``trade_date`` grain.
        include_quarterly: Whether to fetch the quarterly statement endpoints.
        max_retries: Attempts per endpoint before giving up.
        retry_backoff_seconds: Base delay for linear backoff between retries.
    """

    def __init__(
        self,
        history_period: str = "1y",
        history_interval: str = "1d",
        include_quarterly: bool = True,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.history_period = history_period
        self.history_interval = history_interval
        self.include_quarterly = include_quarterly
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = retry_backoff_seconds

    def fetch(self, ticker: str) -> RawTickerData:
        """Fetch every endpoint for one ticker.

        Args:
            ticker: Symbol to fetch; case-insensitive.

        Returns:
            The populated :class:`RawTickerData`.  Partially failed endpoints are
            left empty and noted in ``errors``.

        Raises:
            ExtractionError: If yfinance could not be imported, the ``Ticker``
                object could not be constructed, or no endpoint yielded data.
        """
        symbol = ticker.strip().upper()
        raw = RawTickerData(ticker=symbol)

        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ExtractionError(
                "yfinance is not installed; run 'pip install -r requirements.txt'"
            ) from exc

        try:
            handle = yf.Ticker(symbol)
        except Exception as exc:
            raise ExtractionError(
                "Could not create yfinance Ticker for {}: {}".format(symbol, exc)
            ) from exc

        logger.info("Extracting %s", symbol)

        raw.info = self._safe(raw, "info", lambda: dict(handle.info or {})) or {}
        raw.isin = _normalise_isin(self._safe(raw, "isin", lambda: handle.isin))

        # Must go through _safe_frame: `result or pd.DataFrame()` would call
        # bool() on a DataFrame, which pandas rejects as ambiguous.
        raw.history = self._safe_frame(
            raw,
            "history",
            lambda: handle.history(
                period=self.history_period,
                interval=self.history_interval,
                # Keep raw Close *and* Adj Close: the target schema has a column
                # for each. auto_adjust=True would collapse them into one.
                auto_adjust=False,
                actions=True,
            ),
        )

        raw.income_annual = self._safe_frame(raw, "income_stmt", lambda: handle.income_stmt)
        raw.balance_annual = self._safe_frame(raw, "balance_sheet", lambda: handle.balance_sheet)
        raw.cashflow_annual = self._safe_frame(raw, "cashflow", lambda: handle.cashflow)

        if self.include_quarterly:
            raw.income_quarterly = self._safe_frame(
                raw, "quarterly_income_stmt", lambda: handle.quarterly_income_stmt
            )
            raw.balance_quarterly = self._safe_frame(
                raw, "quarterly_balance_sheet", lambda: handle.quarterly_balance_sheet
            )
            raw.cashflow_quarterly = self._safe_frame(
                raw, "quarterly_cashflow", lambda: handle.quarterly_cashflow
            )

        if not raw.has_any_data:
            raise ExtractionError(
                "yfinance returned no usable data for {} (delisted or invalid symbol?)".format(
                    symbol
                )
            )

        logger.info(
            "Extracted %s: profile=%s, price_rows=%d, statements=%s",
            symbol,
            raw.has_profile,
            len(raw.history),
            raw.has_statements,
        )
        return raw

    def _safe(
        self, raw: RawTickerData, endpoint: str, call: Callable[[], T]
    ) -> Optional[T]:
        """Run ``call`` with retries, recording failures instead of raising.

        Args:
            raw: Container whose ``errors`` list collects failure messages.
            endpoint: Endpoint name, used for logging and error messages.
            call: Zero-argument callable performing the yfinance request.

        Returns:
            The call's result, or ``None`` if every attempt failed.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return call()
            except Exception as exc:  # yfinance raises a wide variety of types
                last_error = exc
                logger.debug(
                    "%s: endpoint %s failed on attempt %d/%d: %s",
                    raw.ticker,
                    endpoint,
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        message = "{}: endpoint {} failed after {} attempt(s): {}".format(
            raw.ticker, endpoint, self.max_retries, last_error
        )
        logger.warning(message)
        raw.errors.append(message)
        return None

    def _safe_frame(
        self, raw: RawTickerData, endpoint: str, call: Callable[[], Any]
    ) -> pd.DataFrame:
        """Like :meth:`_safe`, but always returns a ``DataFrame``."""
        result = self._safe(raw, endpoint, call)
        if isinstance(result, pd.DataFrame):
            return result
        return pd.DataFrame()


def _normalise_isin(value: Any) -> Optional[str]:
    """Map yfinance's ISIN placeholders to ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() in _ISIN_PLACEHOLDER:
        return None
    return text
