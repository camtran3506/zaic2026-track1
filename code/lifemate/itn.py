"""Conservative Vietnamese ITN. Annotations are evidence, never application actions.

No model, labels, hidden application state, filesystem or external services.
Offsets refer to the supplied transcript (Python Unicode code points, end exclusive).
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import re
import unicodedata

ITN_POLICY = "vi-explicit-spans-v1"

_DIGITS = {"không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4",
           "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9"}
_EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_{}|~\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)+")


def _supported_email(value):
    if not _EMAIL.fullmatch(value) or len(value) > 254:
        return False
    local, domain = value.rsplit("@", 1)
    return (len(local) <= 64 and not local.startswith(".") and not local.endswith(".")
            and ".." not in local and all(len(label) <= 63 for label in domain.split(".")))


def normalize_span(kind, text):
    if kind not in {"email", "phone", "money"} or not isinstance(text, str):
        raise ValueError("Expected kind email/phone/money and one text span")
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
    if kind == "money":
        return normalize_money(original)
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


_ONES = {**{word: int(value) for word, value in _DIGITS.items()}, "mốt": 1, "tư": 4, "lăm": 5}
_SCALES = {"tỷ": 10**9, "tỉ": 10**9, "triệu": 10**6, "nghìn": 1000, "ngàn": 1000}
_CURRENCY = re.compile(r"\s*(?:đồng|vnd|₫)\s*$", re.I)


def _under_hundred(words):
    if len(words) == 1 and words[0] in _DIGITS:
        return int(_DIGITS[words[0]])
    if words and words[0] == "mười":
        if len(words) == 1:
            return 10
        if len(words) == 2 and words[1] in _ONES and _ONES[words[1]] > 0:
            return 10 + _ONES[words[1]]
    if len(words) in (2, 3) and words[0] in _DIGITS and 2 <= int(_DIGITS[words[0]]) <= 9 and words[1] == "mươi":
        if len(words) == 2:
            return int(_DIGITS[words[0]]) * 10
        if words[2] in _ONES and _ONES[words[2]] > 0:
            return int(_DIGITS[words[0]]) * 10 + _ONES[words[2]]
    raise ValueError("Unsupported number grammar")


def _integer_words(words):
    if len(words) >= 2 and words[0] in _DIGITS and words[1] == "trăm":
        total = int(_DIGITS[words[0]]) * 100
        rest = words[2:]
        if not rest:
            return total
        if rest[0] in ("linh", "lẻ"):
            if len(rest) == 2 and rest[1] in _DIGITS and int(_DIGITS[rest[1]]) > 0:
                return total + int(_DIGITS[rest[1]])
            raise ValueError("Unsupported remainder")
        # 'hai trăm năm' can be colloquial 250 or literal 205: do not choose.
        if len(rest) == 1:
            raise ValueError("Ambiguous hundreds remainder")
        return total + _under_hundred(rest)
    return _under_hundred(words)


def _coefficient(words):
    if len(words) == 1 and re.fullmatch(r"[0-9]+(?:\.[0-9]{3})*(?:,[0-9]+)?", words[0]):
        if re.fullmatch(r"[0-9]{1,3},[0-9]{3}", words[0]):
            raise ValueError("Ambiguous decimal versus grouped thousands")
        if len(words[0]) > 30:
            raise ValueError("Amount too large")
        return Decimal(words[0].replace(".", "").replace(",", "."))
    return Decimal(_integer_words(words))


def normalize_money(text):
    """VND-style amounts only; no currency/exchange/ambiguous shorthand guesses."""
    original = text
    expression = unicodedata.normalize("NFC", text).strip().casefold()
    currency = "VND" if _CURRENCY.search(expression) else None
    expression = _CURRENCY.sub("", expression).strip()
    unsupported = {"status": "unsupported", "evidence": {"input": original},
                   "reason": "Use one explicit amount with descending scales; ambiguous shorthand is not inferred."}
    if re.search(r"\b(?:hoặc|hay)\b|[/;]", expression):
        return {"status": "ambiguous", "candidates": [], "evidence": {"input": original}}
    if len(expression) > 1024:
        return unsupported
    words = expression.split()
    try:
        if not words:
            return unsupported
        if words[-1] == "rưỡi":
            if len(words) < 3 or words[-2] not in _SCALES:
                return unsupported
            coefficient = _coefficient(words[:-2])
            if coefficient != coefficient.to_integral_value():
                return unsupported
            value = (coefficient + Decimal("0.5")) * _SCALES[words[-2]]
        else:
            total, previous, pending = Decimal(0), 10**12, []
            for word in words:
                if word not in _SCALES:
                    pending.append(word)
                    continue
                scale = _SCALES[word]
                if scale >= previous or not pending:
                    return unsupported
                coefficient = _coefficient(pending)
                if coefficient <= 0 or coefficient >= 1000:
                    return unsupported
                total += coefficient * scale
                previous, pending = scale, []
            if pending:
                # A remainder after millions/billions is commonly shorthand.
                if previous in (10**9, 10**6):
                    return unsupported
                tail = _coefficient(pending)
                if previous == 1000 and tail >= 1000:
                    return unsupported
                total += tail
            value = total
        if not value.is_finite() or not 0 <= value <= 10**15 or value != value.to_integral_value():
            return unsupported
        return {"status": "matched", "value": int(value), "currency": currency,
                "evidence": {"input": original},
                "warnings": ["Amount only; recipient, intent and authorization are not inferred."]}
    except (ValueError, InvalidOperation):
        return unsupported


# The detector requires explicit phone markers, an @/spoken @ email, or a VND
# currency suffix. Bare numbers, account IDs and vague financial prose stay text.
_DIGIT_WORD = r"(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|[0-9]+)"
_PHONE_SPAN = re.compile(
    rf"(?<!\w)(?:số điện thoại|số máy|sđt|gọi(?:\s+đến)?\s+số|nhắn(?:\s+tin)?\s+(?:đến\s+)?số)"
    rf"\s*(?:là\s*|:\s*)?(?P<value>\+?{_DIGIT_WORD}(?:[\s().,\-]+{_DIGIT_WORD})*)(?!\w)", re.I)
_EMAIL_SPAN = re.compile(
    r"(?<![\w@.+\-])(?:[a-z0-9_+\-]+(?:\s*(?:chấm|gạch dưới|gạch ngang|\.)\s*[a-z0-9_+\-]+)*)"
    r"\s*(?:a\s+còng|@)\s*"
    r"(?:[a-z0-9\-]+\s*(?:chấm|\.)\s*)+"
    r"(?:com|vn|org|net|edu|gov|io|test)(?:\s*(?:chấm|\.)\s*vn)?"
    r"(?!\w|@|\.[a-z0-9]|\s+chấm\s+[a-z0-9])", re.I)
_LITERAL_EMAIL = re.compile(
    r"(?<![\w@.!#$%&'*+/=?^_`{|}~\-])[A-Za-z0-9.!#$%&'*+/=?^_{}|~\-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)+"
    r"(?![\w@\-]|\.[A-Za-z0-9])")
_MONEY_WORD = r"(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mốt|tư|lăm|mười|mươi|trăm|linh|lẻ|tỷ|tỉ|triệu|nghìn|ngàn|rưỡi|[0-9]+(?:\.[0-9]{3})*(?:,[0-9]+)?)"
_MONEY_SPAN = re.compile(rf"(?<![\w.,])(?P<value>{_MONEY_WORD}(?:\s+{_MONEY_WORD})*\s*(?:đồng|vnd|₫))(?!\w)", re.I)


@lru_cache(maxsize=128)
def _cached_candidates(text):
    candidates = []
    has_email = "@" in text or re.search(r"a\s+còng", text, re.I)
    has_currency = re.search(r"\b(?:đồng|vnd)\b|₫", text, re.I)
    for kind, pattern in (("phone", _PHONE_SPAN), ("email", _EMAIL_SPAN),
                          ("email", _LITERAL_EMAIL), ("money", _MONEY_SPAN)):
        if (kind == "email" and not has_email) or (kind == "money" and not has_currency):
            continue
        for match in pattern.finditer(text):
            start, end = match.span("value") if "value" in pattern.groupindex else match.span()
            source = text[start:end]
            if len(source) > 1024:
                continue
            # Never interpret a positive suffix of an explicitly negative amount.
            if kind == "money" and re.search(r"(?:\bâm|\btrừ|-)\s*$", text[max(0, start-8):start], re.I):
                continue
            if kind == "phone" and re.match(r"\s+(?:mươi|trăm|linh|lẻ|rưỡi|phẩy)\b", text[end:], re.I):
                continue
            result = normalize_span(kind, source)
            if result["status"] != "matched":
                continue
            candidate = {"kind": kind, "source_text": source, "start": start, "end": end,
                         "value": result["value"]}
            if kind == "money":
                candidate["currency"] = result["currency"]
            # Mixed spoken/literal email can yield a suffix match as well.
            # Keep the complete expression, never a second shortened address.
            if any(row["kind"] == kind and row["start"] <= start and row["end"] >= end for row in candidates):
                continue
            candidates[:] = [row for row in candidates if not (
                row["kind"] == kind and start <= row["start"] and end >= row["end"])]
            candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda row: (row["start"], row["end"], row["kind"])))


def detect_candidates(text):
    """Return all supported explicit spans, preserving intent/negation in source.

    This does not rewrite a transcript or select a recipient/argument. Repeated
    prompt steps reuse cached detection, with copies preventing caller mutation.
    """
    if not isinstance(text, str):
        raise TypeError("ITN input must be text")
    return deepcopy(list(_cached_candidates(text)))


_FALLBACK_MONEY = re.compile(
    rf"(?<!\w)(?P<cue>số tiền|chuyển tiền|trả tiền|chuyển|trả|thanh toán)\s+"
    rf"(?P<value>{_MONEY_WORD}(?:\s+{_MONEY_WORD})*)(?!\w)", re.I)


def fallback_money_requests(text):
    """SFT-only proposals with financial cues but no explicit currency suffix.

    Inference keeps these decisions with the LLM; no bare monetary rewrite occurs.
    """
    automatic = detect_candidates(text)
    requests = []
    for match in _FALLBACK_MONEY.finditer(text):
        start, end = match.span("value")
        if any(row["kind"] == "money" and row["start"] <= start and row["end"] >= end for row in automatic):
            continue
        span = text[start:end]
        if (match["cue"].casefold() in {"chuyển", "trả"}
                and not any(word in _SCALES for word in span.casefold().split())):
            continue
        if normalize_span("money", span)["status"] == "matched":
            requests.append({"name": "local_text.normalize", "arguments": {"kind": "money", "text": span}})
    return requests
