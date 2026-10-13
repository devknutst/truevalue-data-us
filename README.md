# truevalue-data-us

Reads US public-company data from two independent sources -- yfinance and the
SEC EDGAR APIs -- and transforms each into the same three common target
schemas defined under `resources/schema/`: `company_dimension`,
`financial_statement_fact` and `market_price_fact`.

* **yfinance** (`python -m truevalue`) covers all three schemas: company
  master data, fundamentals and daily prices.
* **SEC EDGAR** (`python -m truevalue.sec_cli`) covers `company_dimension` and
  `financial_statement_fact` only. SEC EDGAR has no market/price data at all,
  so it never produces `market_price_fact`; run the yfinance pipeline for that
  schema instead. In return, it gets `cik`, `sic_code` and `filing_date` --
  fields yfinance cannot supply -- straight from the source.

The two pipelines are independent and can be run separately. Give each its own
output directory (`-o`): both write `company_dimension` and
`financial_statement_fact` under the same file names, so a second run into the
same directory overwrites those tables from the first.

Start both entry points as modules (`python -m ...`) from the project root.
`truevalue/__main__.py` can also be run directly as a script (e.g. a PyCharm
"Script path" run configuration). `truevalue/sec_cli.py` cannot: run directly,
it fails with `ImportError: attempted relative import with no known parent
package`, so in PyCharm set its run configuration to "Module name"
(`truevalue.sec_cli`).

## Installation

```bash
pip install -r requirements.txt
```

## Usage: yfinance

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

## Usage: SEC EDGAR

SEC's fair-access policy requires every caller to identify itself with a
descriptive `User-Agent` (name/company plus contact information); requests
without one are liable to be throttled or blocked. Pass it via
`--sec-user-agent` or the `SEC_EDGAR_USER_AGENT` environment variable.

```bash
# Default: Parquet, fiscal years 2025 + 2026, annual + quarterly statements
export SEC_EDGAR_USER_AGENT="MyCompany contact@example.com"
python -m truevalue.sec_cli --tickers AAPL MSFT NVDA

# Different fiscal years, tickers from a file
python -m truevalue.sec_cli --tickers-file tickers.txt --fiscal-years 2023 2024 2025 2026

# Full option list
python -m truevalue.sec_cli --help
```

Also importable as a library, with the same injectable-extractor pattern as
the yfinance pipeline:

```python
from pathlib import Path
from truevalue.sec_pipeline import run
from truevalue.transform import RunContext

result = run(
    tickers=["AAPL", "MSFT"],
    output_dir=Path("output"),
    user_agent="MyCompany contact@example.com",
    context=RunContext.create(statement_type_mode="period", source_system="sec_edgar"),
)
print(result.row_counts, result.failures)
```

## Output layout

Each run writes one folder per schema below its `--output-dir` (default
`output/`). The file names do not carry the source, so run the two pipelines
against separate directories to keep both results:

```bash
python -m truevalue --tickers AAPL MSFT -o output/yfinance
python -m truevalue.sec_cli --tickers AAPL MSFT -o output/sec
```

```
output/
  yfinance/
    company_dimension/company_dimension.parquet
    financial_statement_fact/financial_statement_fact.parquet
    market_price_fact/market_price_fact.parquet    # yfinance pipeline only
  sec/
    company_dimension/company_dimension.parquet
    financial_statement_fact/financial_statement_fact.parquet
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
| `truevalue/load.py` | Writes Parquet/CSV -- shared by both pipelines |
| `truevalue/transform_common.py` | Arithmetic/lineage helpers shared by both transform modules |
| `truevalue/extract.py` | yfinance access only; returns raw payloads, no mapping |
| `truevalue/transform.py` | Maps raw yfinance payloads onto the target schemas |
| `truevalue/pipeline.py` | yfinance orchestration, per-ticker error isolation |
| `truevalue/cli.py` | yfinance argparse entry point |
| `truevalue/sec_extract.py` | SEC EDGAR access only; returns raw payloads, no mapping |
| `truevalue/sec_transform.py` | Maps raw SEC XBRL/profile payloads onto the target schemas |
| `truevalue/sec_pipeline.py` | SEC EDGAR orchestration, per-ticker error isolation |
| `truevalue/sec_cli.py` | SEC EDGAR argparse entry point |

A ticker that fails is logged, recorded in `RunResult.failures` and skipped; the
rest of the run continues. Exit code `0` = at least one ticker ingested,
`1` = none, `2` = configuration or I/O error. Both pipelines share this
contract via the same `RunResult` type.

## Schema mapping decisions (shared)

The JSON Schemas declare no `required`, no `format` and no enums, so the
following conventions were chosen. All are configurable, and shared verbatim
by both pipelines via the same `RunContext`.

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
  or a security master -- the SEC pipeline supplies `cik` and `sic_code`)
* `financial_statement_fact`: `filing_date` (needs SEC EDGAR -- the SEC
  pipeline supplies it directly from each XBRL fact's own `filed` date)

### Derived values (yfinance)

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

### Fields the SEC APIs do not provide

Always written as `NULL` rather than guessed:

* `company_dimension`: `cusip`, `isin`, `market`, `sector`, `ipo_date`,
  `employees` (SEC exposes a SIC code/description but no CUSIP/ISIN, no
  broader sector taxonomy, no listing date and no standard XBRL headcount tag)
* `financial_statement_fact`: none -- every field either has a mapped XBRL tag
  or is derived (see below); a specific filer simply may not have tagged a
  given concept, in which case that one cell is `NULL`

### How SEC XBRL facts are selected

`companyfacts` returns, per XBRL tag, every value the filer has ever reported
-- the current period *and* every comparative period later filings re-report
alongside it, plus (for some tags) both a single-quarter and a
year-to-date duration ending on the same date. For a given `(fiscal_year,
fiscal_quarter)`, `sec_transform._select_entry` disambiguates by, in order:
latest `end` date (a filing's own current period is always the most recent one
it presents), shortest duration (picks the quarter over a year-to-date
cumulative), then latest `filed` date (the most recent restatement wins). This
was verified against Apple's real filings, where it correctly separates a
quarter's own revenue from the cumulative year-to-date figure Apple tags
alongside it.

SEC filers only submit a 10-Q for fiscal Q1-Q3; Q4 is implicit in the annual
10-K, so `financial_statement_fact` never has a standalone Q4 row -- only
`ANNUAL` and `QUARTERLY` (Q1-Q3) rows, per fiscal year requested via
`--fiscal-years` (default: 2025 and 2026).

### Derived values (SEC)

SEC's `us-gaap` taxonomy has no direct tag for these; each is computed only
when its inputs are present:

* `ebit` = `pretax_income + interest_expense`, falling back to
  `operating_income` if either input is missing.
* `ebitda` = `ebit + depreciation_and_amortization`.
* `total_debt` = `short_term_debt + long_term_debt`.
* `working_capital`, `net_working_capital`, `net_debt`, `invested_capital`,
  `free_cash_flow`: same formulas as the yfinance pipeline (see
  `truevalue/transform_common.py`).
* `capital_expenditure` and `dividends_paid` are reported by SEC filers as
  positive payment magnitudes; both are negated to match this project's
  "cash outflow is negative" convention.
* `dividend_per_share` prefers the direct `CommonStockDividendsPerShareDeclared`
  tag, falling back to `abs(dividends_paid) / shares_outstanding`.

## Tests

```bash
python -m pytest tests/ -v
```

Both suites (`test_pipeline_offline.py` for yfinance,
`test_sec_pipeline_offline.py` for SEC EDGAR) use a stub extractor and need
neither network access nor the respective third-party dependency.


