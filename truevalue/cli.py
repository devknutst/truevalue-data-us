"""Command line entry point for the yfinance -> common schema ingestion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .extract import YFinanceExtractor
from .load import FORMAT_BOTH, FORMAT_CSV, FORMAT_PARQUET, OUTPUT_FORMATS
from .pipeline import RunResult, read_tickers_file, run
from .schemas import SchemaError
from .transform import (
    COMPANY_ID_STRATEGIES,
    COMPANY_ID_STRATEGY_UUID5,
    SOURCE_SYSTEM,
    STATEMENT_MODE_PERIOD,
    STATEMENT_MODES,
    RunContext,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_TICKERS = ("AAPL", "MSFT", "JNJ")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="truevalue-ingest",
        description=(
            "Fetch US company data from yfinance and transform it into the "
            "company_dimension / financial_statement_fact / market_price_fact "
            "target schemas."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m truevalue --tickers AAPL MSFT NVDA\n"
            "  python -m truevalue --tickers-file tickers.txt "
            "--history-period 5y --format both\n"
        ),
    )

    source = parser.add_argument_group("input")
    source.add_argument(
        "-t",
        "--tickers",
        nargs="+",
        metavar="SYMBOL",
        help="US ticker symbols, space or comma separated.",
    )
    source.add_argument(
        "--tickers-file",
        type=Path,
        metavar="PATH",
        help="Text file with one ticker per line ('#' starts a comment).",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for the written datasets.",
    )
    output.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=OUTPUT_FORMATS,
        default=FORMAT_PARQUET,
        help="Output format. Parquet preserves the schema types, CSV does not.",
    )
    output.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec (snappy, zstd, gzip, none).",
    )
    output.add_argument(
        "--schema-dir",
        type=Path,
        default=None,
        help="Override the directory holding the JSON Schema files.",
    )

    extraction = parser.add_argument_group("extraction")
    extraction.add_argument(
        "--history-period",
        default="1y",
        help="Price history window (1mo, 6mo, 1y, 5y, max, ...).",
    )
    extraction.add_argument(
        "--history-interval",
        default="1d",
        help="Bar interval. market_price_fact has a daily grain, so keep '1d'.",
    )
    extraction.add_argument(
        "--no-quarterly",
        action="store_true",
        help="Skip the quarterly statements and ingest annual periods only.",
    )
    extraction.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Attempts per yfinance endpoint before giving up on it.",
    )
    extraction.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Pause between tickers, to go easy on the Yahoo endpoints.",
    )

    mapping = parser.add_argument_group("schema mapping")
    mapping.add_argument(
        "--company-id-strategy",
        choices=COMPANY_ID_STRATEGIES,
        default=COMPANY_ID_STRATEGY_UUID5,
        help=(
            "How to derive company_id. 'uuid5' is a deterministic surrogate key, "
            "'ticker' uses the symbol itself."
        ),
    )
    mapping.add_argument(
        "--statement-type-mode",
        choices=STATEMENT_MODES,
        default=STATEMENT_MODE_PERIOD,
        help=(
            "Row grain of financial_statement_fact. 'period' writes one merged "
            "row per fiscal period (statement_type ANNUAL/QUARTERLY); 'statement' "
            "writes one sparse row per statement (INCOME_STATEMENT/BALANCE_SHEET/"
            "CASH_FLOW)."
        ),
    )
    mapping.add_argument(
        "--source-system",
        default=SOURCE_SYSTEM,
        help="Value written to every source_system column.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only log warnings and errors.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Additionally write the log to this file.",
    )

    return parser


def configure_logging(
    verbose: bool = False, quiet: bool = False, log_file: Optional[Path] = None
) -> None:
    """Set up root logging for a CLI run.

    Args:
        verbose: Enable ``DEBUG`` output.
        quiet: Restrict output to ``WARNING`` and above; overridden by ``verbose``.
        log_file: Optional file to mirror the log into.
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
    )
    # yfinance is chatty at INFO level and mostly duplicates our own messages.
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def resolve_tickers(args: argparse.Namespace) -> List[str]:
    """Collect ticker symbols from the CLI arguments.

    Falls back to :data:`DEFAULT_TICKERS` when neither ``--tickers`` nor
    ``--tickers-file`` is given.

    Args:
        args: Parsed arguments.

    Returns:
        The raw symbol list, cleaned later by the pipeline.

    Raises:
        FileNotFoundError: If ``--tickers-file`` points at a missing file.
    """
    symbols: List[str] = []
    if args.tickers:
        symbols.extend(args.tickers)
    if args.tickers_file:
        symbols.extend(read_tickers_file(args.tickers_file))
    if not symbols:
        logger.warning(
            "No tickers given, falling back to the default sample: %s",
            ", ".join(DEFAULT_TICKERS),
        )
        symbols.extend(DEFAULT_TICKERS)
    return symbols


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` if at least one ticker was ingested, ``1`` if none were, ``2`` on a
        configuration or I/O error that aborted the run.
    """
    args = build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)

    try:
        tickers = resolve_tickers(args)
        context = RunContext.create(
            company_id_strategy=args.company_id_strategy,
            statement_type_mode=args.statement_type_mode,
            source_system=args.source_system,
        )
        extractor = YFinanceExtractor(
            history_period=args.history_period,
            history_interval=args.history_interval,
            include_quarterly=not args.no_quarterly,
            max_retries=args.max_retries,
        )
        result: RunResult = run(
            tickers=tickers,
            output_dir=args.output_dir,
            context=context,
            extractor=extractor,
            schema_dir=args.schema_dir,
            output_format=args.output_format,
            compression=args.compression,
            request_delay_seconds=args.request_delay,
        )
    except (SchemaError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 2
    except OSError as exc:
        logger.error("I/O error: %s", exc)
        return 2

    if not result.ok:
        logger.error("No ticker could be ingested")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
