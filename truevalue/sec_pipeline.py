"""Orchestration: extract -> transform -> load for SEC EDGAR company data.

Mirrors :mod:`truevalue.pipeline`'s per-ticker fault isolation and reuses its
generic ticker-list utilities and :class:`~truevalue.pipeline.RunResult`, but
only writes ``company_dimension`` and ``financial_statement_fact`` --
SEC EDGAR has no market/price data, so ``market_price_fact`` is not produced
by this pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .extract import ExtractionError
from .load import FORMAT_PARQUET, write_all
from .pipeline import RunResult, normalise_tickers
from .schemas import (
    COMPANY_DIMENSION,
    FINANCIAL_STATEMENT_FACT,
    TargetSchema,
    load_all_schemas,
)
from .sec_extract import RawSecCompanyData, SecExtractor
from .sec_transform import DEFAULT_FISCAL_YEARS, transform_ticker
from .transform import RunContext

logger = logging.getLogger(__name__)

#: Schemas this pipeline writes; market_price_fact is out of scope for SEC EDGAR.
SEC_SCHEMAS: Tuple[str, ...] = (COMPANY_DIMENSION, FINANCIAL_STATEMENT_FACT)


def run(
    tickers: Sequence[str],
    output_dir: Path,
    context: Optional[RunContext] = None,
    extractor: Optional[SecExtractor] = None,
    user_agent: Optional[str] = None,
    fiscal_years: Sequence[int] = DEFAULT_FISCAL_YEARS,
    schema_dir: Optional[Path] = None,
    output_format: str = FORMAT_PARQUET,
    compression: str = "snappy",
    request_delay_seconds: float = 0.0,
) -> RunResult:
    """Run the SEC EDGAR ingestion pipeline for ``tickers``.

    Args:
        tickers: Symbols to ingest; cleaned internally.
        output_dir: Root directory for the output datasets.
        context: Run settings; a default context is created if omitted.
        extractor: Extractor to use; a default :class:`SecExtractor` is built
            from ``user_agent`` if omitted. Injectable so tests can supply a stub.
        user_agent: ``User-Agent`` for the default extractor; required unless
            ``extractor`` is given.
        fiscal_years: Which XBRL fiscal years to build financial statement rows for.
        schema_dir: Optional override for the JSON Schema directory.
        output_format: ``"parquet"``, ``"csv"`` or ``"both"``.
        compression: Parquet codec.
        request_delay_seconds: Pause between tickers, on top of whatever
            per-request throttling the extractor does on its own.

    Returns:
        The :class:`truevalue.pipeline.RunResult` describing what succeeded and
        what did not; ``row_counts``/``written_paths`` only ever contain
        :data:`SEC_SCHEMAS`.

    Raises:
        ValueError: If ``tickers`` contains no usable symbol, or neither
            ``extractor`` nor ``user_agent`` was given.
        truevalue.schemas.SchemaError: If a schema file cannot be loaded.
    """
    symbols = normalise_tickers(tickers)
    if not symbols:
        raise ValueError("No valid ticker symbols supplied")

    run_context = context or RunContext.create(source_system="sec_edgar")
    if extractor is None:
        if not user_agent:
            raise ValueError("Either 'extractor' or 'user_agent' must be given")
        extractor = SecExtractor(user_agent=user_agent)

    all_schemas: Mapping[str, TargetSchema] = load_all_schemas(schema_dir)
    schemas = {name: all_schemas[name] for name in SEC_SCHEMAS}

    logger.info(
        "Starting SEC EDGAR run for %d ticker(s): %s", len(symbols), ", ".join(symbols)
    )
    logger.info(
        "company_id strategy=%s, statement_type mode=%s, fiscal_years=%s",
        run_context.company_id_strategy,
        run_context.statement_type_mode,
        ", ".join(str(year) for year in fiscal_years),
    )

    result = RunResult()
    collected: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SEC_SCHEMAS}

    for position, symbol in enumerate(symbols):
        if position and request_delay_seconds > 0:
            time.sleep(request_delay_seconds)

        raw = _extract_one(symbol, extractor, result)
        if raw is None:
            continue

        result.warnings.extend(raw.errors)

        try:
            transformed = transform_ticker(raw, run_context, fiscal_years)
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
    symbol: str, extractor: SecExtractor, result: RunResult
) -> Optional[RawSecCompanyData]:
    """Extract one ticker, recording failures on ``result`` instead of raising."""
    try:
        return extractor.fetch(symbol)
    except ExtractionError as exc:
        logger.error("%s: %s", symbol, exc)
        result.failures[symbol] = str(exc)
    except Exception as exc:  # unexpected SEC/network failure
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
