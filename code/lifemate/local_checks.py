"""Bounded normalization, arithmetic and advisory schema inspection.

No helper imports a model, ToolEnv, networking or application state. A successful
schema inspection is not proof of entity existence or task correctness.
"""
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, localcontext
import json
import math
import re
import unicodedata


_LIMIT = 100
_WARNING = "Checks supplied operands only; their source and the user's intended goal are not verified."


def _spec(name, description, properties, required):
    return {"name": name, "kind": "local", "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}


_COMPARISON = {"type": "string", "enum": ["lt", "le", "eq", "ge", "gt"]}
_TOOLS = {
    "local_text.normalize": _spec(
        "local_text.normalize",
        "Chuẩn hóa đúng một email hoặc số điện thoại đọc bằng lời. Không đoán ký tự hay chữ số bị mất; "
        "trả ambiguous/unsupported khi chưa rõ. Không sửa trực tiếp arguments của tool ứng dụng.",
        {"kind": {"type": "string", "enum": ["email", "phone"]},
         "text": {"type": "string", "minLength": 1, "maxLength": 1024}}, ["kind", "text"]),
    "local_checks.evaluate": _spec(
        "local_checks.evaluate",
        "Tính tổng, so sánh, tìm giá trị nhỏ nhất hoặc lọc bản ghi theo số; tối đa 100 phần tử. "
        "Chỉ kiểm toán hạng được truyền, không xác nhận đúng mục tiêu. "
        "sum/minimum dùng values; compare dùng left/right/comparison; filter dùng records/field/right/comparison.",
        {"operation": {"type": "string", "enum": ["sum", "compare", "minimum", "filter"]},
         "values": {"type": "array", "items": {"type": "number"}, "maxItems": _LIMIT},
         "left": {"type": "number"}, "right": {"type": "number"},
         "comparison": _COMPARISON,
         "records": {"type": "array", "items": {"type": "object"}, "maxItems": _LIMIT},
         "field": {"type": "string", "minLength": 1, "maxLength": 100}}, ["operation"]),
    "local_schema.inspect": _spec(
        "local_schema.inspect",
        "Kiểm tra cấu trúc arguments với schema thật của tool được phép: required, type, enum, x-one-of, "
        "giới hạn số và mảng. Chỉ tư vấn; không thay env.validate, không chứng minh ID tồn tại.",
        {"target_tool": {"type": "string", "minLength": 1, "maxLength": 150},
         "arguments": {"type": "object"}}, ["target_tool", "arguments"]),
}


def schemas():
    return deepcopy(_TOOLS)


def _same(left, right):
    # JSON numbers may compare across int/float; booleans never equal 0/1.
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


_METADATA = {"title", "description", "default", "examples", "$schema", "$id", "$comment",
             "x-ref", "x-match", "readOnly", "writeOnly", "deprecated"}
_SUPPORTED = {"type", "properties", "required", "additionalProperties", "items",
              "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
              "exclusiveMinimum", "exclusiveMaximum", "enum", "const",
              "anyOf", "oneOf", "allOf", "x-one-of", "format"}
_TYPES = {
    "null": lambda v: v is None,
    "boolean": lambda v: type(v) is bool,
    "integer": lambda v: type(v) is int,
    "number": lambda v: type(v) in (int, float) and (type(v) is int or math.isfinite(v)),
    "string": lambda v: isinstance(v, str),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _error(errors, path, code, message):
    errors.append({"path": path, "code": code, "message": message})


def _check(value, schema, path, errors, unsupported, depth=0, budget=None):
    budget = [0] if budget is None else budget
    budget[0] += 1
    if depth > 20 or budget[0] > 10000:
        unsupported.append({"path": path, "keyword": "inspection_limit"})
        return
    if schema is True:
        return
    if schema is False:
        _error(errors, path, "forbidden", "The schema forbids this value.")
        return
    if not isinstance(schema, dict):
        unsupported.append({"path": path, "keyword": "malformed_schema"})
        return
    unsupported.extend({"path": path, "keyword": key}
                       for key in sorted(schema.keys() - _SUPPORTED - _METADATA))
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind is not None else []
    if any(not isinstance(k, str) or k not in _TYPES for k in kinds):
        unsupported.append({"path": path, "keyword": "type"})
    elif kinds and not any(_TYPES[k](value) for k in kinds):
        _error(errors, path, "type", f"Expected {' or '.join(kinds)}; received {type(value).__name__}.")
        return
    if type(value) is float and not math.isfinite(value):
        _error(errors, path, "non_finite", "NaN and infinity are not JSON numbers.")
        return
    if "enum" in schema:
        if not isinstance(schema["enum"], list):
            unsupported.append({"path": path, "keyword": "enum"})
        elif not any(_same(value, choice) for choice in schema["enum"]):
            _error(errors, path, "enum", "Value is outside the declared enum.")
    if "const" in schema and not _same(value, schema["const"]):
        _error(errors, path, "const", "Value differs from the declared constant.")
    for keyword in ("anyOf", "oneOf"):
        if keyword not in schema:
            continue
        branches = schema[keyword]
        if not isinstance(branches, list) or not branches:
            unsupported.append({"path": path, "keyword": keyword})
            continue
        outcomes = []
        for branch in branches:
            branch_errors, branch_unknown = [], []
            _check(value, branch, path, branch_errors, branch_unknown, depth + 1, budget)
            outcomes.append((branch_errors, branch_unknown))
        passed = sum(not e and not u for e, u in outcomes)
        unknown = any(not e and u for e, u in outcomes)
        if (keyword == "anyOf" and not passed and not unknown) or (
                keyword == "oneOf" and (passed > 1 or (not passed and not unknown))):
            _error(errors, path, keyword, f"Value does not satisfy {keyword}.")
        elif (keyword == "oneOf" and unknown) or (keyword == "anyOf" and not passed and unknown):
            unsupported.append({"path": path, "keyword": keyword})
    if "allOf" in schema:
        if isinstance(schema["allOf"], list):
            for branch in schema["allOf"]:
                _check(value, branch, path, errors, unsupported, depth + 1, budget)
        else:
            unsupported.append({"path": path, "keyword": "allOf"})
    if isinstance(value, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(props, dict):
            unsupported.append({"path": path, "keyword": "properties"})
            props = {}
        if isinstance(required, list) and all(isinstance(k, str) for k in required):
            for key in required:
                if key not in value:
                    _error(errors, f"{path}.{key}", "required", "Required parameter is missing.")
        else:
            unsupported.append({"path": path, "keyword": "required"})
        groups = schema.get("x-one-of")
        if groups is not None:
            if (not isinstance(groups, list) or not groups or
                    any(not isinstance(g, list) or not g or
                        any(not isinstance(k, str) for k in g) for g in groups)):
                unsupported.append({"path": path, "keyword": "x-one-of"})
            elif not any(all(k in value for k in group) for group in groups):
                _error(errors, path, "x-one-of", "Provide at least one complete declared parameter group.")
        for key, child in value.items():
            if key in props:
                _check(child, props[key], f"{path}.{key}", errors, unsupported, depth + 1, budget)
            # patternProperties could permit an otherwise unknown property.
            elif "patternProperties" not in schema:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    _error(errors, f"{path}.{key}", "additionalProperties", "Undeclared parameter is forbidden.")
                elif isinstance(extra, dict):
                    _check(child, extra, f"{path}.{key}", errors, unsupported, depth + 1, budget)
    if isinstance(value, list):
        _length_checks(value, schema, path, errors, unsupported, "Items")
        if "items" in schema:
            if isinstance(schema["items"], (dict, bool)):
                for i, child in enumerate(value):
                    _check(child, schema["items"], f"{path}[{i}]", errors, unsupported, depth + 1, budget)
            else:
                unsupported.append({"path": path, "keyword": "items"})
    if isinstance(value, str):
        _length_checks(value, schema, path, errors, unsupported, "Length")
        if "format" in schema:
            form = schema["format"]
            try:
                if form == "date":
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                        raise ValueError
                    date.fromisoformat(value)
                elif form == "date-time":
                    parsed = datetime.fromisoformat(value)
                    if parsed.tzinfo is None or parsed.utcoffset() is None or "T" not in value:
                        raise ValueError
                else:
                    unsupported.append({"path": path, "keyword": f"format:{form}"})
            except ValueError:
                _error(errors, path, "format", f"Expected {form}.")
    if type(value) in (int, float):
        for keyword, predicate in (
                ("minimum", lambda a, b: a >= b), ("maximum", lambda a, b: a <= b),
                ("exclusiveMinimum", lambda a, b: a > b), ("exclusiveMaximum", lambda a, b: a < b)):
            if keyword in schema:
                bound = schema[keyword]
                if type(bound) not in (int, float) or (type(bound) is float and not math.isfinite(bound)):
                    unsupported.append({"path": path, "keyword": keyword})
                elif not predicate(value, bound):
                    _error(errors, path, keyword, f"Value violates {keyword}={bound}.")


def _length_checks(value, schema, path, errors, unsupported, suffix):
    for prefix, compare in (("min", lambda a, b: a >= b), ("max", lambda a, b: a <= b)):
        key = prefix + suffix
        if key not in schema:
            continue
        bound = schema[key]
        if type(bound) is not int or bound < 0:
            unsupported.append({"path": path, "keyword": key})
        elif not compare(len(value), bound):
            _error(errors, path, key, f"Length violates {key}={bound}.")


def inspect_call(name, arguments, env_tools):
    """Advisory inspection of the exposed schema, never a call to env.validate."""
    errors, unsupported = [], []
    if not isinstance(env_tools, dict) or not isinstance(name, str) or name not in env_tools:
        _error(errors, "$.target_tool", "tool_not_allowed", "Target is not an exposed application tool.")
    elif not isinstance(arguments, dict):
        _error(errors, "$.arguments", "type", "Tool arguments must be an object.")
    else:
        spec = env_tools[name]
        if not isinstance(spec, dict):
            unsupported.append({"path": "$.arguments", "keyword": "malformed_schema"})
        else:
            _check(arguments, spec.get("parameters"), "$.arguments", errors, unsupported)
    return {"status": "invalid" if errors else "unsupported" if unsupported else "valid",
            "scope": "declared_schema_only", "errors": errors, "unsupported": unsupported,
            "warnings": ["Advisory only: ToolEnv remains authoritative. Entity existence and business conditions are not checked."]}


def _arguments(name, args):
    if not isinstance(name, str) or name not in _TOOLS or not isinstance(args, dict):
        raise ValueError("Unknown helper or non-object arguments.")
    try:
        encoded = json.dumps(args, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Arguments must be finite JSON values.") from exc
    if len(encoded) > 100000:
        raise ValueError("Helper arguments exceed the inspection limit.")
    errors, unsupported = [], []
    _check(args, _TOOLS[name]["parameters"], "$", errors, unsupported)
    if errors or unsupported:
        raise ValueError(f"Invalid helper arguments: {errors or unsupported}")


_DIGITS = {"không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4",
           "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9"}
_EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_{}|~\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)+")


def _supported_email(value):
    if not _EMAIL.fullmatch(value) or len(value) > 254:
        return False
    local, domain = value.rsplit("@", 1)
    return (len(local) <= 64 and not local.startswith(".") and not local.endswith(".")
            and ".." not in local and all(len(label) <= 63 for label in domain.split(".")))


def _normalize(kind, text):
    original = text
    text = unicodedata.normalize("NFC", text).strip()
    if kind == "email":
        if _supported_email(text):
            return {"status": "matched", "value": text, "evidence": {"input": original}}
        if len(re.findall(r"@|a\s+còng", text, re.I)) > 1:
            return {"status": "ambiguous", "candidates": [], "evidence": {"input": original}}
        converted = re.sub(r"\ba\s+còng\b", "@", text, flags=re.I)
        converted = re.sub(r"\bchấm\b", ".", converted, flags=re.I)
        converted = re.sub(r"\bgạch\s+dưới\b", "_", converted, flags=re.I)
        converted = re.sub(r"\bgạch\s+ngang\b", "-", converted, flags=re.I)
        converted = re.sub(r"\s*([@._\-])\s*", r"\1", converted)
        # Spaces elsewhere remain: never silently turn a full sentence into an email.
        if _supported_email(converted):
            return {"status": "matched", "value": converted, "evidence": {"input": original}}
        return {"status": "unsupported", "evidence": {"input": original},
                "reason": "Input is not one unambiguous supported email expression."}
    if re.search(r"\b(?:hoặc|hay)\b|[/;]", text, re.I):
        return {"status": "ambiguous", "candidates": [], "evidence": {"input": original}}
    literal = re.sub(r"[\s().,\-]", "", text)
    if re.fullmatch(r"\+?\d{7,15}", literal):
        return {"status": "matched", "value": literal, "evidence": {"input": original}}
    words = re.sub(r"[,.\-()]", " ", text.lower()).split()
    if words and all(word in _DIGITS or word.isascii() and word.isdigit() for word in words):
        digits = "".join(_DIGITS.get(word, word) for word in words)
        if re.fullmatch(r"\d{7,15}", digits):
            return {"status": "matched", "value": digits, "evidence": {"input": original}}
    return {"status": "unsupported", "evidence": {"input": original},
            "reason": "Input contains unsupported words or an incomplete phone number; no digits were guessed."}


def _decimal(value):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("Expected a finite JSON number, not a string or boolean.")
    number = Decimal(str(value))
    if abs(number) > Decimal("1e18"):
        raise ValueError("Numeric operands must be within -1e18 to 1e18.")
    return number


def _number(value):
    return int(value) if value == value.to_integral_value() else float(value)


def _compare(left, right, op):
    return {"lt": left < right, "le": left <= right, "eq": left == right,
            "ge": left >= right, "gt": left > right}[op]


def _evaluate(args):
    operation = args["operation"]
    required = {"sum": {"operation", "values"}, "minimum": {"operation", "values"},
                "compare": {"operation", "left", "right", "comparison"},
                "filter": {"operation", "records", "field", "right", "comparison"}}[operation]
    if args.keys() != required:
        raise ValueError(f"{operation} requires exactly {sorted(required)}.")
    with localcontext() as ctx:
        ctx.prec = 50
        if operation in ("sum", "minimum"):
            values = [_decimal(v) for v in args["values"]]
            if operation == "minimum" and not values:
                raise ValueError("minimum requires at least one value.")
            value = sum(values, Decimal(0)) if operation == "sum" else min(values)
            result = {"value": _number(value)}
            if operation == "minimum":
                result["indices"] = [i for i, v in enumerate(values) if v == value]
        elif operation == "compare":
            result = {"value": _compare(_decimal(args["left"]), _decimal(args["right"]), args["comparison"])}
        else:
            expected = _decimal(args["right"])
            selected, indices = [], []
            for i, record in enumerate(args["records"]):
                if args["field"] not in record:
                    raise ValueError(f"Record {i} has no field {args['field']!r}.")
                if _compare(_decimal(record[args["field"]]), expected, args["comparison"]):
                    selected.append(deepcopy(record))
                    indices.append(i)
            result = {"records": selected, "indices": indices}
    return {"status": "ok", "result": result, "warnings": [_WARNING]}


def execute(name, args, env_tools):
    _arguments(name, args)
    if name == "local_text.normalize":
        return _normalize(args["kind"], args["text"])
    if name == "local_schema.inspect":
        return inspect_call(args["target_tool"], args["arguments"], env_tools)
    return _evaluate(args)
