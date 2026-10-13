"""Extraction layer: everything that talks to the SEC EDGAR APIs.

This module performs no mapping onto the target schemas.  It returns the raw
SEC payloads (the ``submissions`` profile and the ``us-gaap`` slice of
``companyfacts``) wrapped in :class:`RawSecCompanyData`.

Three EDGAR endpoints are used, all under the ``data.sec.gov`` / ``sec.gov``
hosts that make up the public, unauthenticated "EDGAR APIs":

* ``https://www.sec.gov/files/company_tickers.json`` -- ticker -> CIK lookup,
  fetched once per extractor instance and cached.
* ``https://data.sec.gov/submissions/CIK##########.json`` -- the company
  profile (name, SIC code, addresses, exchanges, ...).
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` -- every
  XBRL fact (``us-gaap`` taxonomy) the filer has ever reported, in one call.

SEC's fair-access policy requires a descriptive ``User-Agent`` with contact
information and asks callers to stay well under ~10 requests/second; both are
enforced here rather than left to the caller.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .extract import ExtractionError

logger = logging.getLogger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


@dataclass
class RawSecCompanyData:
    """Raw, untransformed SEC EDGAR payloads for a single ticker.

    Attributes:
        ticker: The requested symbol, upper-cased.
        cik: Ten-digit, zero-padded Central Index Key, once resolved.
        submissions: The ``submissions`` profile document (empty if unavailable).
        facts: The ``facts["us-gaap"]`` slice of ``companyfacts``, keyed by XBRL
            tag name (empty if unavailable).
        errors: Human-readable messages for endpoints that failed.
    """

    ticker: str
    cik: Optional[str] = None
    submissions: Dict[str, Any] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def has_submissions(self) -> bool:
        """Whether the submissions profile carries an actual company name."""
        return bool(self.submissions.get("name"))

    @property
    def has_facts(self) -> bool:
        """Whether at least one XBRL fact was retrieved."""
        return bool(self.facts)

    @property
    def has_any_data(self) -> bool:
        """Whether at least one endpoint returned something usable."""
        return self.has_submissions or self.has_facts


class SecExtractor:
    """Fetches raw company data for tickers from the SEC EDGAR APIs.

    Args:
        user_agent: Value sent as the ``User-Agent`` header. SEC requires a
            descriptive string with contact information, e.g.
            ``"truevalue-data-us contact@example.com"``; requests without one
            are liable to be throttled or blocked.
        max_retries: Attempts per endpoint before giving up on it.
        retry_backoff_seconds: Base delay for linear backoff between retries.
        min_interval_seconds: Minimum spacing enforced between outgoing
            requests, to stay comfortably under SEC's fair-access guidance.
        timeout_seconds: Per-request timeout passed to ``requests``.

    Raises:
        ValueError: If ``user_agent`` is empty.
    """

    def __init__(
        self,
        user_agent: str,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        min_interval_seconds: float = 0.15,
        timeout_seconds: float = 15.0,
    ) -> None:
        if not user_agent or not user_agent.strip():
            raise ValueError(
                "SEC requires a descriptive User-Agent with contact information, "
                "e.g. 'truevalue-data-us contact@example.com'"
            )
        self.user_agent = user_agent
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = retry_backoff_seconds
        self.min_interval_seconds = min_interval_seconds
        self.timeout_seconds = timeout_seconds
        self._last_request_at = 0.0
        self._ticker_to_cik: Optional[Dict[str, str]] = None

    def fetch(self, ticker: str) -> RawSecCompanyData:
        """Fetch every endpoint for one ticker.

        Args:
            ticker: Symbol to fetch; case-insensitive.

        Returns:
            The populated :class:`RawSecCompanyData`. A partially failed
            endpoint is left empty and noted in ``errors``.

        Raises:
            ExtractionError: If the ticker cannot be resolved to a CIK, or no
                endpoint yielded any data at all.
        """
        symbol = ticker.strip().upper()
        raw = RawSecCompanyData(ticker=symbol)

        cik = self._resolve_cik(symbol, raw)
        if cik is None:
            raise ExtractionError(
                "No SEC CIK found for ticker {} (not in company_tickers.json)".format(symbol)
            )
        raw.cik = cik

        logger.info("Extracting %s (CIK %s)", symbol, cik)
        raw.submissions = self._get_json(
            SUBMISSIONS_URL_TEMPLATE.format(cik=cik), raw, "submissions"
        ) or {}
        raw.facts = self._get_us_gaap_facts(cik, raw)

        if not raw.has_any_data:
            raise ExtractionError(
                "SEC EDGAR returned no usable data for {} (CIK {})".format(symbol, cik)
            )

        logger.info(
            "Extracted %s: profile=%s, us-gaap tags=%d",
            symbol,
            raw.has_submissions,
            len(raw.facts),
        )
        return raw

    def _resolve_cik(self, symbol: str, raw: RawSecCompanyData) -> Optional[str]:
        """Look up ``symbol``'s CIK in the cached ticker map."""
        return self._load_ticker_map(raw).get(symbol)

    def _load_ticker_map(self, raw: RawSecCompanyData) -> Dict[str, str]:
        """Fetch and cache the ticker -> CIK map, shared across ``fetch`` calls."""
        if self._ticker_to_cik is None:
            document = self._get_json(TICKER_MAP_URL, raw, "company_tickers") or {}
            self._ticker_to_cik = {
                str(entry["ticker"]).upper(): str(entry["cik_str"]).zfill(10)
                for entry in document.values()
                if isinstance(entry, dict) and "ticker" in entry and "cik_str" in entry
            }
            logger.debug("Loaded %d ticker -> CIK mappings", len(self._ticker_to_cik))
        return self._ticker_to_cik

    def _get_us_gaap_facts(self, cik: str, raw: RawSecCompanyData) -> Dict[str, Any]:
        """Fetch ``companyfacts`` and return only the ``us-gaap`` taxonomy slice."""
        document = self._get_json(
            COMPANYFACTS_URL_TEMPLATE.format(cik=cik), raw, "companyfacts"
        )
        if not document:
            return {}
        facts = document.get("facts", {})
        us_gaap = facts.get("us-gaap", {}) if isinstance(facts, dict) else {}
        return us_gaap if isinstance(us_gaap, dict) else {}

    def _throttle(self) -> None:
        """Sleep just enough to respect ``min_interval_seconds`` between requests."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def _get_json(
        self, url: str, raw: RawSecCompanyData, endpoint: str
    ) -> Optional[Dict[str, Any]]:
        """GET ``url`` as JSON, with retries and rate limiting.

        A 404 is treated as "no data" (common for companyfacts on filers with
        no XBRL history) and returned as ``None`` without being retried or
        logged as an error.

        Args:
            url: Fully-formed URL to fetch.
            raw: Container whose ``errors`` list collects failure messages.
            endpoint: Endpoint name, used for logging and error messages.

        Returns:
            The parsed JSON document, or ``None`` if the resource does not
            exist or every attempt failed.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = requests.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout_seconds,
                )
                self._last_request_at = time.monotonic()
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                document = response.json()
                return document if isinstance(document, dict) else None
            except Exception as exc:  # SEC/network failures come in many shapes
                last_error = exc
                self._last_request_at = time.monotonic()
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
