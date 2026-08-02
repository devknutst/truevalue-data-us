"""Orchestration: extract -> transform -> load for a list of tickers.

The pipeline is deliberately fault-tolerant at the ticker boundary.  A ticker
that fails extraction or transformation is logged, recorded in
:class:`RunResult.failures` and skipped; every other ticker still lands in the
output.  Only a failure affecting the whole run (unreadable schema, unwritable
output directory) propagates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .extract import ExtractionError, RawTickerData, YFinanceExtractor
from .load import FORMAT_PARQUET, write_all
from .schemas import ALL_SCHEMAS, TargetSchema, load_all_schemas
from .transform import RunContext, transform_ticker

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Outcome of a pipeline run.

    Attributes:
        succeeded: Tickers that produced at least one record.
        failures: Ticker -> reason for tickers that were skipped entirely.
        warnings: Non-fatal per-endpoint messages collected during extraction.
        row_counts: Number of rows written per schema.
        written_paths: Output paths per schema.
    """

    succeeded: List[str] = field(default_factory=list)
    failures: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    row_counts: Dict[str, int] = field(default_factory=dict)
    written_paths: Dict[str, List[Path]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether at least one ticker made it through."""
        return bool(self.succeeded)


def normalise_tickers(tickers: Sequence[str]) -> List[str]:
    """Upper-case, strip and de-duplicate ticker symbols, preserving order.

    Args:
        tickers: Raw symbols, possibly with whitespace or comma separators.

    Returns:
        Cleaned, unique symbols in first-seen order.
    """
    seen = set()
    cleaned: List[str] = []
    for entry in tickers:
        for part in str(entry).replace(",", " ").split():
            symbol = part.strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                cleaned.append(symbol)
    return cleaned


def read_tickers_file(path: Path) -> List[str]:
    """Read ticker symbols from a text file, one per line.

    Blank lines and ``#`` comments are ignored.

    Args:
        path: File to read.

    Returns:
        The symbols found, uncleaned -- pass through :func:`normalise_tickers`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.split("#", 1)[0] for line in lines]


def run(
    tickers: Sequence[str],
    output_dir: Path,
    context: Optional[RunContext] = None,
    extractor: Optional[YFinanceExtractor] = None,
    schema_dir: Optional[Path] = None,
    output_format: str = FORMAT_PARQUET,
    compression: str = "snappy",
    request_delay_seconds: float = 0.0,
) -> RunResult:
    """Run the full pipeline for ``tickers``.

    Args:
        tickers: Symbols to ingest; cleaned internally.
        output_dir: Root directory for the output datasets.
        context: Run settings; a default context is created if omitted.
        extractor: Extractor to use; a default one is created if omitted.
            Injectable so tests can supply a stub.
        schema_dir: Optional override for the JSON Schema directory.
        output_format: ``"parquet"``, ``"csv"`` or ``"both"``.
        compression: Parquet codec.
        request_delay_seconds: Pause between tickers, to stay friendly to the
            Yahoo endpoints when ingesting long lists.

    Returns:
        The :class:`RunResult` describing what succeeded and what did not.

    Raises:
        ValueError: If ``tickers`` contains no usable symbol.
        truevalue.schemas.SchemaError: If a schema file cannot be loaded.
    """
    symbols = normalise_tickers(tickers)
    if not symbols:
        raise ValueError("No valid ticker symbols supplied")

    run_context = context or RunContext.create()
    yf_extractor = extractor or YFinanceExtractor()
    schemas: Mapping[str, TargetSchema] = load_all_schemas(schema_dir)

    logger.info(
        "Starting run for %d ticker(s): %s", len(symbols), ", ".join(symbols)
    )
    logger.info(
        "company_id strategy=%s, statement_type mode=%s, ingestion_timestamp=%s",
        run_context.company_id_strategy,
        run_context.statement_type_mode,
        run_context.ingestion_timestamp,
    )

    result = RunResult()
    collected: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ALL_SCHEMAS}

    for position, symbol in enumerate(symbols):
        if position and request_delay_seconds > 0:
            time.sleep(request_delay_seconds)

        raw = _extract_one(symbol, yf_extractor, result)
        if raw is None:
            continue

        result.warnings.extend(raw.errors)

        try:
            transformed = transform_ticker(raw, run_context)
        except Exception as exc:  # defensive: never let one ticker kill the run
            message = "transform failed: {}".format(exc)
            logger.exception("%s: %s", symbol, message)
            result.failures[symbol] = message
            continue

        row_total = 0
        for schema_name, records in transformed.items():
            schema = schemas[schema_name]
            for record in records:
                try:
                    collected[schema_name].append(schema.coerce(record))
                    row_total += 1
                except Exception as exc:
                    message = "{}: record rejected for {}: {}".format(
                        symbol, schema_name, exc
                    )
                    logger.warning(message)
                    result.warnings.append(message)

        if row_total:
            result.succeeded.append(symbol)
        else:
            result.failures[symbol] = "no records produced"
            logger.warning("%s: produced no records", symbol)

    result.row_counts = {name: len(rows) for name, rows in collected.items()}
    result.written_paths = write_all(
        records_by_schema=collected,
        schemas=schemas,
        output_dir=Path(output_dir),
        output_format=output_format,
        compression=compression,
    )

    _log_summary(result)
    return result


def _extract_one(
    symbol: str, extractor: YFinanceExtractor, result: RunResult
) -> Optional[RawTickerData]:
    """Extract one ticker, recording failures on ``result`` instead of raising."""
    try:
        return extractor.fetch(symbol)
    except ExtractionError as exc:
        logger.error("%s: %s", symbol, exc)
        result.failures[symbol] = str(exc)
    except Exception as exc:  # unexpected yfinance/network failure
        logger.exception("%s: unexpected extraction failure", symbol)
        result.failures[symbol] = "unexpected extraction failure: {}".format(exc)
    return None


def _log_summary(result: RunResult) -> None:
    """Emit the end-of-run summary."""
    logger.info(
        "Run finished: %d succeeded, %d failed",
        len(result.succeeded),
        len(result.failures),
    )
    for schema_name, count in result.row_counts.items():
        logger.info("  %-28s %6d row(s)", schema_name, count)
    for symbol, reason in result.failures.items():
        logger.warning("  FAILED %s: %s", symbol, reason)
    if result.warnings:
        logger.info("  %d non-fatal warning(s) during extraction", len(result.warnings))
