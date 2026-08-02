"""truevalue-data-us: yfinance ingestion into the common target schema.

The package is split along the ETL boundaries:

* :mod:`truevalue.schemas`   -- loads the JSON Schema files and derives Arrow schemas
* :mod:`truevalue.extract`   -- talks to yfinance, returns raw payloads
* :mod:`truevalue.transform` -- maps raw payloads onto the target schemas
* :mod:`truevalue.load`      -- writes the target records to Parquet/CSV
* :mod:`truevalue.pipeline`  -- orchestrates extract -> transform -> load
* :mod:`truevalue.cli`       -- argparse entry point
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
