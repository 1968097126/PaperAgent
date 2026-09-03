"""Small JSON-schema subset used to validate ToolDef arguments without a new dependency."""
from __future__ import annotations

from typing import Any


class ToolInputError(ValueError):
    pass


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def validate_tool_input(schema: dict, params: Any) -> None:
    input_schema = schema.get("input_schema", {})
    if not isinstance(params, dict):
        raise ToolInputError("tool input must be an object")
    required = input_schema.get("required", [])
    missing = [name for name in required if name not in params]
    if missing:
        raise ToolInputError(f"missing required parameter(s): {', '.join(missing)}")
    properties = input_schema.get("properties", {})
    if input_schema.get("additionalProperties") is False:
        unknown = sorted(set(params) - set(properties))
        if unknown:
            raise ToolInputError(f"unknown parameter(s): {', '.join(unknown)}")
    for name, value in params.items():
        rule = properties.get(name)
        if not rule:
            continue
        expected = rule.get("type")
        if expected and not _matches_type(value, expected):
            raise ToolInputError(f"parameter '{name}' must be {expected}")
        if "enum" in rule and value not in rule["enum"]:
            raise ToolInputError(f"parameter '{name}' must be one of {rule['enum']}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                raise ToolInputError(f"parameter '{name}' must be >= {rule['minimum']}")
            if "maximum" in rule and value > rule["maximum"]:
                raise ToolInputError(f"parameter '{name}' must be <= {rule['maximum']}")
        if isinstance(value, str) and rule.get("minLength", 0) and len(value.strip()) < rule["minLength"]:
            raise ToolInputError(f"parameter '{name}' must not be empty")
