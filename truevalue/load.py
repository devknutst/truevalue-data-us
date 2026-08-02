"""Loading layer: writes target records to disk.

Parquet is the default because it is the only one of the two supported formats
that carries the schema with the data: types, field order and nullability
survive the round trip, so a downstream reader gets exactly what the JSON Schema
declared.  CSV is offered as a debugging / hand-inspection format and loses the
type information.

Each schema is written to ``<output-dir>/<schema_name>/<schema_name>.<ext>``, so
the output directory can be pointed at directly as a partitioned dataset root.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from .schemas import TargetSchema

logger = logging.getLogger(__name__)

FORMAT_PARQUET = "parquet"
FORMAT_CSV = "csv"
FORMAT_BOTH = "both"
OUTPUT_FORMATS = (FORMAT_PARQUET, FORMAT_CSV, FORMAT_BOTH)


def build_table(records: Sequence[Mapping[str, Any]], schema: TargetSchema) -> pa.Table:
    """Build an Arrow table from already-coerced records.

    Columns are built individually against the Arrow type declared in the JSON
    Schema, so a value that cannot be represented raises here rather than being
    silently cast at write time.

    Args:
        records: Records produced by :meth:`TargetSchema.coerce`.
        schema: The target schema driving column order and types.

    Returns:
        An Arrow table matching ``schema.arrow_schema()`` exactly.  Zero records
        yield an empty table with the full column set.

    Raises:
        pyarrow.ArrowInvalid: If a value does not fit its declared column type.
    """
    arrow_schema = schema.arrow_schema()
    columns = [
        pa.array([record.get(name) for record in records], type=field.type)
        for name, field in zip(arrow_schema.names, arrow_schema)
    ]
    return pa.Table.from_arrays(columns, schema=arrow_schema)


def write_records(
    records: Sequence[Mapping[str, Any]],
    schema: TargetSchema,
    output_dir: Path,
    output_format: str = FORMAT_PARQUET,
    compression: str = "snappy",
) -> List[Path]:
    """Write one schema's records to disk.

    Args:
        records: Coerced records for this schema.
        schema: The target schema being written.
        output_dir: Root output directory; a subdirectory per schema is created.
        output_format: One of :data:`OUTPUT_FORMATS`.
        compression: Parquet codec, e.g. ``"snappy"``, ``"zstd"``, ``"none"``.

    Returns:
        Paths actually written.

    Raises:
        ValueError: If ``output_format`` is not recognised.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            "Unknown output format {!r}; expected one of {}".format(
                output_format, OUTPUT_FORMATS
            )
        )

    table = build_table(records, schema)
    target_dir = Path(output_dir) / schema.name
    target_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    if output_format in (FORMAT_PARQUET, FORMAT_BOTH):
        path = target_dir / "{}.parquet".format(schema.name)
        pq.write_table(table, path, compression=compression)
        written.append(path)

    if output_format in (FORMAT_CSV, FORMAT_BOTH):
        path = target_dir / "{}.csv".format(schema.name)
        pa_csv.write_csv(table, path)
        written.append(path)

    logger.info(
        "Wrote %d row(s) for %s to %s",
        table.num_rows,
        schema.name,
        ", ".join(str(path) for path in written),
    )
    return written


def write_all(
    records_by_schema: Mapping[str, Sequence[Mapping[str, Any]]],
    schemas: Mapping[str, TargetSchema],
    output_dir: Path,
    output_format: str = FORMAT_PARQUET,
    compression: str = "snappy",
) -> Dict[str, List[Path]]:
    """Write every schema's records.

    Args:
        records_by_schema: Coerced records keyed by schema name.
        schemas: Loaded schemas keyed by name.
        output_dir: Root output directory.
        output_format: One of :data:`OUTPUT_FORMATS`.
        compression: Parquet codec.

    Returns:
        Written paths keyed by schema name.
    """
    written: Dict[str, List[Path]] = {}
    for schema_name, schema in schemas.items():
        written[schema_name] = write_records(
            records=records_by_schema.get(schema_name, []),
            schema=schema,
            output_dir=output_dir,
            output_format=output_format,
            compression=compression,
        )
    return written
