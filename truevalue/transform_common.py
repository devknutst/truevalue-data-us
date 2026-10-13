"""Helpers shared by every source-specific transform module.

Kept separate from :mod:`truevalue.transform` (yfinance) so that a second data
source's transform module (e.g. :mod:`truevalue.sec_transform`) can reuse the
same small arithmetic and lineage helpers without depending on, or duplicating,
yfinance-specific mapping code.
"""

from __future__ import annotations

from typing import Optional


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide, returning ``None`` on missing inputs or a zero denominator."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def safe_sum(*values: Optional[float]) -> Optional[float]:
    """Sum values, treating ``None`` as zero but returning ``None`` if all are ``None``."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def build_source_record_id(
    ticker: str, statement_type: str, period_end_date: Optional[str]
) -> str:
    """Build a deterministic natural key for lineage back to the source."""
    return "{}:{}:{}".format(ticker, statement_type, period_end_date or "unknown")
