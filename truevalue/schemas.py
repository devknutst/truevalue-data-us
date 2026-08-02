"""Loading of the target JSON Schema definitions and type coercion.

The JSON Schema files under ``resources/schema`` are the single source of truth
for field names, field order and data types.  Nothing in this package hard-codes
a column list: the Arrow schema used when writing output is derived from the
JSON Schema at runtime, so a change to a schema file propagates automatically.

The schema files declare neither ``required`` nor a nullable union type.  Since
a columnar output format has no notion of an "absent" field, every field is
materialised as a nullable column of the declared type.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pyarrow as pa

logger = logging.getLogger(__name__)

#: Default location of the JSON Schema files, relative to the repository root.
DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "resources" / "schema"

COMPANY_DIMENSION = "company_dimension"
FINANCIAL_STATEMENT_FACT = "financial_statement_fact"
MARKET_PRICE_FACT = "market_price_fact"

#: All schemas handled by the pipeline, in load order (dimension before facts).
ALL_SCHEMAS: Tuple[str, ...] = (
    COMPANY_DIMENSION,
    FINANCIAL_STATEMENT_FACT,
    MARKET_PRICE_FACT,
)

_JSON_TYPE_TO_ARROW: Mapping[str, pa.DataType] = {
    "string": pa.string(),
    "number": pa.float64(),
    "integer": pa.int64(),
    "boolean": pa.bool_(),
}


class SchemaError(RuntimeError):
    """Raised when a schema file is missing, malformed or uses an unknown type."""


def _is_missing(value: Any) -> bool:
    """Return ``True`` for values that must be represented as SQL ``NULL``.

    Covers ``None``, ``float('nan')``, ``numpy.nan``, ``pandas.NaT`` and
    ``pandas.NA``.  Applied to scalars only -- array-likes are rejected by the
    coercion helpers instead.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    # pandas/numpy scalars: pd.isna is the only reliable check, but it returns an
    # array for array-likes, so guard the result explicitly.
    try:
        import pandas as pd

        result = pd.isna(value)
    except Exception:  # pragma: no cover - pandas always present in practice
        return False
    return result is True


def to_string(value: Any) -> Optional[str]:
    """Coerce ``value`` to a stripped ``str``, mapping empty/missing to ``None``."""
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def to_float(value: Any) -> Optional[float]:
    """Coerce ``value`` to ``float``, mapping missing and non-finite to ``None``."""
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def to_int(value: Any) -> Optional[int]:
    """Coerce ``value`` to ``int``, mapping missing and non-finite to ``None``.

    Floats are truncated towards zero, which is safe here because the only
    integer target fields (``employees``, ``fiscal_year``, ``fiscal_quarter``)
    are conceptually whole numbers already.
    """
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def to_bool(value: Any) -> Optional[bool]:
    """Coerce ``value`` to ``bool``, mapping missing to ``None``."""
    if _is_missing(value):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
        return None
    return bool(value)


_COERCERS = {
    "string": to_string,
    "number": to_float,
    "integer": to_int,
    "boolean": to_bool,
}


@dataclass(frozen=True)
class TargetSchema:
    """A single target schema loaded from its JSON Schema definition.

    Attributes:
        name: Schema title, e.g. ``"company_dimension"``.
        path: Location the definition was loaded from.
        fields: ``(field_name, json_type)`` pairs in declaration order.
    """

    name: str
    path: Path
    fields: Tuple[Tuple[str, str], ...]

    @property
    def field_names(self) -> Tuple[str, ...]:
        """Field names in schema declaration order."""
        return tuple(name for name, _ in self.fields)

    def arrow_schema(self) -> pa.Schema:
        """Build the ``pyarrow`` schema, preserving declaration order.

        All fields are nullable -- see the module docstring for the rationale.
        """
        return pa.schema(
            [
                pa.field(name, _JSON_TYPE_TO_ARROW[json_type], nullable=True)
                for name, json_type in self.fields
            ]
        )

    def empty_record(self) -> Dict[str, Any]:
        """Return a record with every schema field present and set to ``None``."""
        return {name: None for name in self.field_names}

    def coerce(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Project ``record`` onto this schema and coerce every value to its type.

        Fields absent from ``record`` become ``None``.  Fields present in
        ``record`` but not in the schema are dropped with a warning -- they would
        otherwise silently disappear at write time.

        Args:
            record: Raw target-shaped mapping produced by the transform layer.

        Returns:
            A new dict containing exactly the schema fields, in schema order.
        """
        unknown = set(record) - set(self.field_names)
        if unknown:
            logger.warning(
                "Dropping %d field(s) not declared in schema %s: %s",
                len(unknown),
                self.name,
                ", ".join(sorted(unknown)),
            )

        coerced: Dict[str, Any] = {}
        for name, json_type in self.fields:
            coerced[name] = _COERCERS[json_type](record.get(name))
        return coerced


def load_schema(name: str, schema_dir: Optional[Path] = None) -> TargetSchema:
    """Load a single target schema by name.

    Args:
        name: Schema name without extension, e.g. ``"market_price_fact"``.
        schema_dir: Directory holding the ``.json`` files.  Defaults to
            ``resources/schema`` next to the package.

    Returns:
        The parsed :class:`TargetSchema`.

    Raises:
        SchemaError: If the file is missing, unparseable, declares no
            properties, or uses a JSON type without an Arrow equivalent.
    """
    directory = Path(schema_dir) if schema_dir is not None else DEFAULT_SCHEMA_DIR
    path = directory / "{}.json".format(name)

    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError as exc:
        raise SchemaError("Schema file not found: {}".format(path)) from exc
    except json.JSONDecodeError as exc:
        raise SchemaError("Schema file {} is not valid JSON: {}".format(path, exc)) from exc

    properties = document.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise SchemaError("Schema {} declares no properties".format(path))

    fields: List[Tuple[str, str]] = []
    for field_name, definition in properties.items():
        json_type = definition.get("type") if isinstance(definition, dict) else None
        if json_type not in _JSON_TYPE_TO_ARROW:
            raise SchemaError(
                "Field {}.{} has unsupported type {!r}; expected one of {}".format(
                    name, field_name, json_type, sorted(_JSON_TYPE_TO_ARROW)
                )
            )
        fields.append((field_name, json_type))

    logger.debug("Loaded schema %s with %d fields from %s", name, len(fields), path)
    return TargetSchema(name=document.get("title", name), path=path, fields=tuple(fields))


@lru_cache(maxsize=None)
def _load_all_cached(schema_dir: Optional[str]) -> Dict[str, TargetSchema]:
    """Cache-friendly loader keyed by the stringified directory."""
    directory = Path(schema_dir) if schema_dir else None
    return {name: load_schema(name, directory) for name in ALL_SCHEMAS}


def load_all_schemas(schema_dir: Optional[Path] = None) -> Dict[str, TargetSchema]:
    """Load every target schema, keyed by name.

    Args:
        schema_dir: Optional override for the schema directory.

    Returns:
        Mapping of schema name to :class:`TargetSchema`.  Results are cached per
        directory, so repeated calls are cheap.
    """
    return dict(_load_all_cached(str(schema_dir) if schema_dir else None))
