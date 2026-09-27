# truevalue-data-us

Reads data from yfinance and transforms it into the common schema.

Ingests US company master data, fundamentals and daily prices from yfinance and
writes them into the three target schemas defined under `resources/schema/`:
`company_dimension`, `financial_statement_fact` and `market_price_fact`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Default: Parquet, one year of price history, annual + quarterly statements
python -m truevalue --tickers AAPL MSFT NVDA

# Longer history, both output formats, tickers from a file
python -m truevalue --tickers-file tickers.txt --history-period 5y --format both

# Full option list
python -m truevalue --help
```

Importable as a library -- the pipeline takes an injectable extractor, so it can
be driven from a scheduler or exercised without network access:

```python
from pathlib import Path
from truevalue.pipeline import run
from truevalue.transform import RunContext

result = run(
    tickers=["AAPL", "MSFT"],
    output_dir=Path("output"),
    context=RunContext.create(statement_type_mode="period"),
)
print(result.row_counts, result.failures)
```

## Output layout

```
output/
  company_dimension/company_dimension.parquet
  financial_statement_fact/financial_statement_fact.parquet
  market_price_fact/market_price_fact.parquet
```

Parquet is the default: it carries field order, types and nullability with the
data, so the JSON Schema contract survives the round trip. CSV (`--format csv`)
is meant for hand inspection and loses the type information.

The Arrow schema used at write time is **derived from the JSON Schema files at
runtime** -- no column list is hard-coded. Editing a file under
`resources/schema/` changes the output accordingly.

## Architecture

| Module | Responsibility |
| --- | --- |
| `truevalue/schemas.py` | Loads the JSON Schemas, derives Arrow schemas, coerces types |
| `truevalue/extract.py` | yfinance access only; returns raw payloads, no mapping |
| `truevalue/transform.py` | Maps raw payloads onto the target schemas |
| `truevalue/load.py` | Writes Parquet/CSV |
| `truevalue/pipeline.py` | Orchestration, per-ticker error isolation |
| `truevalue/cli.py` | argparse entry point |

A ticker that fails is logged, recorded in `RunResult.failures` and skipped; the
rest of the run continues. Exit code `0` = at least one ticker ingested,
`1` = none, `2` = configuration or I/O error.

## Schema mapping decisions

The JSON Schemas declare no `required`, no `format` and no enums, so the
following conventions were chosen. All are configurable.

| Topic | Decision | Flag |
| --- | --- | --- |
| `company_id` | Deterministic `uuid5` over the ticker -- stable across runs, no state needed | `--company-id-strategy {uuid5,ticker}` |
| `statement_type` | `period`: one merged row per fiscal period (`ANNUAL`/`QUARTERLY`). `statement`: three sparse rows per period (`INCOME_STATEMENT`/`BALANCE_SHEET`/`CASH_FLOW`), annual identified by `fiscal_quarter IS NULL` | `--statement-type-mode {period,statement}` |
| Date fields | `YYYY-MM-DD` | -- |
| Timestamp fields | ISO-8601 UTC, `YYYY-MM-DDTHH:MM:SSZ` | -- |
| Nullability | All columns present and nullable | -- |

### Fields yfinance does not provide

Always written as `NULL` rather than guessed:

* `company_dimension`: `cik`, `cusip`, `sic_code`, `naics_code` (need SEC EDGAR
  or a security master)
* `financial_statement_fact`: `filing_date` (needs SEC EDGAR)

### Derived values

* `fiscal_year` / `fiscal_quarter` are anchored on the fiscal year end month
  inferred from the annual statements, not on the calendar. Slightly off for
  52/53-week retail calendars.
* `period_start_date` is reconstructed as "period end minus one period plus one
  day"; yfinance reports only the period end.
* `ipo_date` is Yahoo's first trade date, a proxy for the true IPO date.
* Valuation metrics in `market_price_fact` (`market_cap`, `pe_ratio`, ...) are
  point-in-time snapshots from `Ticker.info`. yfinance has no history for them,
  so they are attached to the **latest trade date only** and left `NULL` on
  earlier rows instead of being back-filled with today's values.
* `dividend_yield` is passed through as yfinance reports it. The unit changed
  from a fraction to a percent in yfinance 0.2.55; normalising here would
  silently corrupt one of the two.

## Tests

```bash
python -m pytest tests/ -v
```

The suite uses a stub extractor and needs neither network access nor a yfinance
installation.


