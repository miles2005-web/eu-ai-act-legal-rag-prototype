"""Shared models and utilities for structured assessment runs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import re
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import uuid4

if TYPE_CHECKING:
    from src.assessment.facts import AssessmentFacts
    from src.assessment.findings import Finding


class TriState(str, Enum):
    """A legal fact that may be confirmed, rejected, or not yet known."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


_STABLE_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
)


def validate_stable_identifier(value: str, *, field_name: str) -> str:
    """Validate the common persisted identifier syntax without generating IDs."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not _STABLE_IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must be 1-128 characters and contain only letters, "
            "numbers, '.', '_', ':', or '-'"
        )
    return value


class AssessmentRunStatus(str, Enum):
    """Lifecycle state of an in-memory assessment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def new_identifier() -> str:
    """Return a stable string representation for a new domain identifier."""

    return str(uuid4())


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


class FrozenDict(Mapping[_KeyT, _ValueT], Generic[_KeyT, _ValueT]):
    """Small dependency-free immutable mapping used by canonical contracts."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[_KeyT, _ValueT] | None = None) -> None:
        self._data = dict(values or {})

    def __getitem__(self, key: _KeyT) -> _ValueT:
        return self._data[key]

    def __iter__(self) -> Iterator[_KeyT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict[_KeyT, _ValueT]:
        return self

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"


def freeze_value(value: Any) -> Any:
    """Recursively isolate JSON-like values from caller mutation."""

    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(
            {
                deepcopy(key): freeze_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze_value(item) for item in value), key=repr))
    return deepcopy(value)


def to_primitive(value: Any) -> Any:
    """Convert nested domain values into JSON-compatible primitives."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            model_field.name: to_primitive(getattr(value, model_field.name))
            for model_field in fields(value)
            if not model_field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value


class SerializableModel:
    """Mixin providing a persistence-neutral dictionary representation."""

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        primitive = to_primitive(self)
        if not isinstance(primitive, dict):
            raise TypeError("Domain model did not serialize to a dictionary")
        return primitive

    @classmethod
    def from_dict(cls, payload: dict[str, Any]):
        """Hydrate a typed model without changing source-version metadata."""

        return model_from_dict(cls, payload)


_SerializableT = TypeVar("_SerializableT", bound=SerializableModel)


def model_from_dict(
    model_type: type[_SerializableT],
    payload: dict[str, Any],
) -> _SerializableT:
    """Deserialize dataclass models using their declared field types.

    This is an explicit compatibility boundary, not a migration operation:
    absent fields retain their declared defaults and version fields are kept
    exactly as supplied.
    """

    if not isinstance(payload, dict):
        raise TypeError(f"{model_type.__name__} payload must be an object")
    if not is_dataclass(model_type):
        raise TypeError("model_type must be a dataclass model")

    model_fields = {item.name: item for item in fields(model_type)}
    required = getattr(model_type, "REQUIRED_SERIALIZED_FIELDS", frozenset())
    missing_required = sorted(required.difference(payload))
    if missing_required:
        raise ValueError(
            f"{model_type.__name__} payload is missing required serialized "
            f"fields: {', '.join(missing_required)}"
        )
    unknown = sorted(set(payload).difference(model_fields))
    if unknown:
        raise ValueError(
            f"unknown {model_type.__name__} fields: {', '.join(unknown)}"
        )
    type_hints = get_type_hints(model_type)
    values = {
        name: _from_primitive(type_hints.get(name, model_field.type), value)
        for name, value in payload.items()
        if (model_field := model_fields[name]).init
    }
    return model_type(**values)


def _from_primitive(expected_type: Any, value: Any) -> Any:
    if expected_type is Any:
        return deepcopy(value)
    origin = get_origin(expected_type)
    arguments = get_args(expected_type)
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise TypeError("serialized collection must be a list")
        item_type = arguments[0] if arguments else Any
        converted = [_from_primitive(item_type, item) for item in value]
        return converted if origin is list else tuple(converted)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError("serialized mapping must be an object")
        key_type, item_type = arguments or (Any, Any)
        return {
            _from_primitive(key_type, key): _from_primitive(item_type, item)
            for key, item in value.items()
        }
    if origin in (UnionType, Union):
        if value is None and type(None) in arguments:
            return None
        errors: list[Exception] = []
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return _from_primitive(candidate, value)
            except (TypeError, ValueError) as exc:
                errors.append(exc)
        raise TypeError(
            f"value {value!r} does not match {expected_type!r}"
        ) from (errors[-1] if errors else None)
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return value if isinstance(value, expected_type) else expected_type(value)
    if expected_type is datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)
    if expected_type is date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)
    if (
        isinstance(expected_type, type)
        and is_dataclass(expected_type)
        and issubclass(expected_type, SerializableModel)
    ):
        if isinstance(value, expected_type):
            return deepcopy(value)
        return model_from_dict(expected_type, value)
    if expected_type in (str, int, float, bool):
        if not isinstance(value, expected_type):
            raise TypeError(
                f"expected {expected_type.__name__}, got {type(value).__name__}"
            )
    return deepcopy(value)


@dataclass(slots=True)
class AssessmentRun(SerializableModel):
    """A versioned, reproducible execution over an immutable fact snapshot."""

    case_id: str
    facts_snapshot: AssessmentFacts
    id: str = field(default_factory=new_identifier)
    ruleset_version: str = "2.0.0"
    questionnaire_version: str = "2.0.0"
    corpus_version: str | None = None
    authorized_rule_ids: list[str] = field(default_factory=list)
    input_fingerprint: str | None = None
    status: AssessmentRunStatus = AssessmentRunStatus.PENDING
    findings: list[Finding] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    error_message: str | None = None

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset({"id"})

    def __post_init__(self) -> None:
        # A run must retain the facts it evaluated even if the draft case changes.
        self.facts_snapshot = deepcopy(self.facts_snapshot)
        validate_stable_identifier(self.id, field_name="id")
        self.authorized_rule_ids = list(self.authorized_rule_ids)
        if any(
            not isinstance(rule_id, str) or not rule_id.strip()
            for rule_id in self.authorized_rule_ids
        ):
            raise ValueError("authorized_rule_ids must contain non-empty strings")
        if len(set(self.authorized_rule_ids)) != len(self.authorized_rule_ids):
            raise ValueError("authorized_rule_ids must not contain duplicates")
        if self.input_fingerprint is not None and (
            not isinstance(self.input_fingerprint, str)
            or not self.input_fingerprint.strip()
        ):
            raise ValueError("input_fingerprint must be non-empty when provided")
