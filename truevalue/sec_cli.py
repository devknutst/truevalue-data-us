"""Command line entry point for the SEC EDGAR -> common schema ingestion.

Run with ``python -m truevalue.sec_cli``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .cli import configure_logging
from .pipeline import RunResult, read_tickers_file
from .sec_extract import SecExtractor
from .sec_pipeline import run
from .sec_transform import DEFAULT_FISCAL_YEARS
from .load import FORMAT_PARQUET, OUTPUT_FORMATS
from .schemas import SchemaError
from .transform import (
    COMPANY_ID_STRATEGIES,
    COMPANY_ID_STRATEGY_UUID5,
    STATEMENT_MODE_PERIOD,
    STATEMENT_MODES,
    RunContext,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_TICKERS = ("AAPL", "MSFT", "JNJ")
DEFAULT_SOURCE_SYSTEM = "sec_edgar"
#: SEC asks for a descriptive User-Agent with contact info; a bare "let me
#: guess" default would get the account rate-limited or blocked, so this is
#: required rather than defaulted.
USER_AGENT_ENV_VAR = "SEC_EDGAR_USER_AGENT"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="truevalue-sec-ingest",
        description=(
            "Fetch US company data from the SEC EDGAR APIs and transform it "
            "into the company_dimension / financial_statement_fact target "
            "schemas. market_price_fact is out of scope: SEC EDGAR has no "
            "market/price data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m truevalue.sec_cli --tickers AAPL MSFT "
            "--sec-user-agent 'MyCompany contact@example.com'\n"
            "  export SEC_EDGAR_USER_AGENT='MyCompany contact@example.com'\n"
            "  python -m truevalue.sec_cli --tickers-file tickers.txt "
            "--fiscal-years 2024 2025 2026\n"
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
        "--sec-user-agent",
        default=os.environ.get(USER_AGENT_ENV_VAR),
        metavar="STRING",
        help=(
            "User-Agent sent to SEC EDGAR, e.g. 'MyCompany contact@example.com'. "
            "SEC requires a descriptive value with contact information. "
            "Falls back to the {} environment variable.".format(USER_AGENT_ENV_VAR)
        ),
    )
    extraction.add_argument(
        "--fiscal-years",
        nargs="+",
        type=int,
        default=list(DEFAULT_FISCAL_YEARS),
        metavar="YEAR",
        help="XBRL fiscal year(s) (the 'fy' label SEC assigns) to ingest.",
    )
    extraction.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Attempts per SEC endpoint before giving up on it.",
    )
    extraction.add_argument(
        "--min-request-interval",
        type=float,
        default=0.15,
        metavar="SECONDS",
        help="Minimum spacing between outgoing requests to SEC EDGAR.",
    )
    extraction.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Additional pause between tickers, on top of --min-request-interval.",
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
        default=DEFAULT_SOURCE_SYSTEM,
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

    if not args.sec_user_agent:
        logger.error(
            "No SEC User-Agent given. Pass --sec-user-agent or set %s; "
            "SEC requires a descriptive value with contact information.",
            USER_AGENT_ENV_VAR,
        )
        return 2

    try:
        tickers = resolve_tickers(args)
        context = RunContext.create(
            company_id_strategy=args.company_id_strategy,
            statement_type_mode=args.statement_type_mode,
            source_system=args.source_system,
        )
        extractor = SecExtractor(
            user_agent=args.sec_user_agent,
            max_retries=args.max_retries,
            min_interval_seconds=args.min_request_interval,
        )
        result: RunResult = run(
            tickers=tickers,
            output_dir=args.output_dir,
            context=context,
            extractor=extractor,
            fiscal_years=args.fiscal_years,
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
