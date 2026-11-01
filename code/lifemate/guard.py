"""Deterministic argument repairs and read provenance before logged tool calls."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
import json
import math
import re
from zoneinfo import ZoneInfo

from .history import history_tool_pairs


class GuardError(ValueError):
    pass


def canonical_arguments(args):
    def numbers(value):
        if isinstance(value, dict):
            return {k: numbers(v) for k, v in value.items()}
        if isinstance(value, list):
            return [numbers(v) for v in value]
        if type(value) is float and math.isfinite(value) and value.is_integer():
            return int(value)
        return value
    return json.dumps(numbers(args), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def succeeded(result):
    # No success flag is required by the ToolEnv interface. Explicit errors fail.
    return result is not None and not (isinstance(result, dict) and (
        result.get("ok") is False or result.get("success") is False or result.get("error")
        or result.get("status") in ("error", "failed", "failure")))


def _temporal(value, schema, context):
    fmt, pattern = schema.get("format"), schema.get("pattern")
    candidates = []
    try:
        if fmt == "date":
            try:
                parsed = date.fromisoformat(value)
            except ValueError:
                # DD/MM is safe only when month-first is impossible (or equal).
                pieces = value.split("/")
                if len(pieces) != 3 or (int(pieces[0]) <= 12 and pieces[0] != pieces[1]):
                    return value
                parsed = datetime.strptime(value, "%d/%m/%Y").date()
            candidates = [parsed.isoformat()]
        elif fmt == "date-time":
            if not re.match(r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}", value):
                return value  # datetime.fromisoformat would invent midnight for a date.
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                zone = context.get("timezone")
                if not zone:
                    return value  # Never invent a timezone or use host-local time.
                tz = ZoneInfo(zone)
                first, second = parsed.replace(tzinfo=tz, fold=0), parsed.replace(tzinfo=tz, fold=1)
                if first.utcoffset() != second.utcoffset():
                    return value  # Ambiguous/nonexistent daylight-saving local time.
                parsed = first
            candidates = [parsed.isoformat()]
        elif fmt == "time" or pattern:
            match = re.fullmatch(r"(\d{1,2})(?:\s*[:hH]\s*|\s+giờ\s+)(\d{2})(?::(\d{2}))?", value)
            if match:
                parsed = time(int(match[1]), int(match[2]), int(match[3] or 0))
            else:
                if not re.match(r"^\d{2}:\d{2}", value):
                    return value
                parsed = time.fromisoformat(value.replace("Z", "+00:00"))
            candidates = [parsed.isoformat()]
            if parsed.second == 0 and parsed.microsecond == 0 and parsed.tzinfo is None:
                candidates.append(parsed.strftime("%H:%M"))
        if pattern:
            candidates = [c for c in candidates if re.search(pattern, c)]
    except (ValueError, TypeError, KeyError, re.error):
        return value
    return candidates[0] if candidates else value


def repair_arguments(value, schema, context=None):
    """Only lossless, schema-directed repairs. ToolEnv remains the validator.

    Unknown fields/defaults/IDs are never invented or dropped. Composed schemas
    are left to the validator rather than guessing a oneOf/anyOf branch.
    """
    context = context or {}
    if not isinstance(schema, dict) or any(k in schema for k in ("oneOf", "anyOf", "allOf", "$ref")):
        return deepcopy(value)
    kind = schema.get("type")
    if isinstance(kind, list):
        kinds = [k for k in kind if k != "null"]
        if value is None or len(kinds) != 1:
            return deepcopy(value)
        kind = kinds[0]
    if kind in ("integer", "number") and not isinstance(value, bool):
        if isinstance(value, (str, int, float)):
            try:
                number = Decimal(str(value).strip())
                if number.is_finite():
                    if kind == "integer" and number == number.to_integral_value():
                        value = int(number)
                    elif kind == "number":
                        candidate = float(number)
                        if math.isfinite(candidate) and Decimal(str(candidate)) == number:
                            value = candidate
            except (ValueError, InvalidOperation, OverflowError):
                pass
    elif kind == "boolean" and isinstance(value, str):
        if value.strip().lower() in ("true", "false"):
            value = value.strip().lower() == "true"
    elif kind == "string":
        if type(value) is int or (type(value) is float and math.isfinite(value)):
            value = str(value)
        if isinstance(value, str):
            options = schema.get("enum", [])
            if value not in options:
                matches = [v for v in options if isinstance(v, str) and v.casefold() == value.strip().casefold()]
                if len(matches) == 1:
                    value = matches[0]
            value = _temporal(value, schema, context)
    if kind == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", {})
        return {k: repair_arguments(v, props.get(k, extra), context) for k, v in value.items()}
    if kind == "array" and isinstance(value, list):
        return [repair_arguments(v, schema.get("items", {}), context) for v in value]
    return deepcopy(value)


def validated_arguments(env, name, args, schema, context):
    """Preserve arguments already accepted by ToolEnv, otherwise try one repair."""
    canonical_arguments(args)  # Reject non-JSON NaN/Infinity before any call.
    def validate(candidate):
        try:
            result = env.validate(name, candidate)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return result
    result = validate(args)
    if isinstance(result, dict) and result.get("ok"):
        return args
    repaired = repair_arguments(args, schema, context)
    if json.dumps(repaired, sort_keys=True, allow_nan=False) != json.dumps(args, sort_keys=True, allow_nan=False):
        result = validate(repaired)
        if isinstance(result, dict) and result.get("ok"):
            return repaired
    raise GuardError(f"Schema invalid: {result!r}")


def _refs(schema, value, app, field=""):
    if not isinstance(schema, dict):
        return
    if "$ref" in schema:
        raise GuardError("Unresolved schema $ref; resolve ID metadata before writing")
    # Require provenance conservatively for every referenced composed branch;
    # never select a branch merely to bypass a resource constraint.
    for keyword in ("allOf", "anyOf", "oneOf"):
        for branch in schema.get(keyword, []):
            yield from _refs(branch, value, app, field)
    ref = schema.get("x-ref")
    if not ref and (field == "id" or field.endswith(("_id", "_ids"))):
        stem = field.removesuffix("_ids").removesuffix("_id")
        ref = f"{app}.{stem}s" if field != "id" else f"{app}.*"
    if ref:
        if not isinstance(ref, str) or not ref.startswith(app + "."):
            raise GuardError(f"Unsupported ID reference {ref!r}; inspect the schema before writing")
        for entry in value if isinstance(value, list) else [value]:
            if entry is not None:
                yield ref, entry, field
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _refs(schema.get("properties", {}).get(key, schema.get("additionalProperties", {})), child, app, key)
    elif isinstance(value, list):
        for child in value:
            yield from _refs(schema.get("items", {}), child, app)


def _id_key(value):
    if type(value) not in (str, int) or str(value) == "":
        raise GuardError("Referenced ID must be a nonempty string or integer")
    return str(value)


class CallGuard:
    def __init__(self, catalog, history=()):
        self.catalog = catalog
        self.reads = set()
        self.known = {}
        for turn in history:
            for call, result in history_tool_pairs(turn):
                if call["name"] in catalog:
                    self.observe(call["name"], call["arguments"], result)
        # Historical reads establish ID provenance, but may be refreshed this turn.
        self.reads.clear()

    def check(self, name, args):
        spec = self.catalog[name]
        if spec.get("kind") == "read" and (name, canonical_arguments(args)) in self.reads:
            raise GuardError("Duplicate read without intervening successful state change")
        if spec.get("kind") == "write":
            app = name.split(".")[0]
            for ref, value, field in _refs(spec.get("parameters", {}), args, app):
                if _id_key(value) not in self.known.get(ref, set()):
                    raise GuardError(f"Unread ID for {field or ref}: {value!r}; read {ref} first")

    def observe(self, name, args, result):
        if not succeeded(result):
            return
        spec = self.catalog[name]
        if spec.get("kind") == "write":
            self.reads.clear()
            return
        if spec.get("kind") != "read":
            return
        self.reads.add((name, canonical_arguments(args)))
        app = name.split(".")[0]
        refs = set()
        for tool, other in self.catalog.items():
            if tool.startswith(app + ".") and other.get("kind") == "write":
                # Traverse schema, independently of which write arguments are proposed.
                def scan(schema, field=""):
                    if not isinstance(schema, dict):
                        return
                    try:
                        for ref, _, _ in _refs(schema, "", app, field):
                            refs.add(ref)
                    except GuardError:
                        pass  # An unsupported write ref is blocked by check(), not by a read.
                    for key, child in schema.get("properties", {}).items():
                        scan(child, key)
                    scan_items = schema.get("items")
                    if isinstance(scan_items, dict):
                        scan(scan_items)
                    for keyword in ("allOf", "anyOf", "oneOf"):
                        for branch in schema.get(keyword, []):
                            scan(branch, field)
                scan(other.get("parameters", {}))
        for ref in refs:
            collection = ref.split(".", 1)[1]
            singular = collection[:-3] + "y" if collection.endswith("ies") else collection.removesuffix("s")
            accepted = {collection, singular, singular + "_ids"}
            id_fields = {"id", singular + "_id"}
            found = set()
            def collect(node, in_collection=False):
                if isinstance(node, list):
                    for child in node:
                        if in_collection and type(child) in (str, int) and str(child):
                            found.add(_id_key(child))
                        else:
                            collect(child, in_collection)
                elif isinstance(node, dict):
                    for key, child in node.items():
                        if (in_collection or collection == "*") and key in id_fields and type(child) in (str, int) and str(child):
                            found.add(_id_key(child))
                        if key == singular + "_id" and type(child) in (str, int) and str(child):
                            found.add(_id_key(child))
                        # A nested object's generic id belongs to that object,
                        # not automatically to its enclosing resource collection.
                        collect(child, key in accepted)
            # Unwrapped list/detail results are trusted only for a matching read name.
            matching_read = bool(accepted.intersection(name.split(".", 1)[1].split("_")))
            collect(result, matching_read)
            self.known.setdefault(ref, set()).update(found)
