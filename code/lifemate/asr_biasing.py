"""Contextual BPE hotword extraction (1A) and post-ASR code normalization (2A)."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import re
from typing import Any

VALID_HOTWORD_MODES = ("off", "auto")
DEFAULT_ASR_HOTWORDS_SCORE = 1.5

_DIGIT_WORDS = {
    "0": "KHÔNG", "1": "MỘT", "2": "HAI", "3": "BA", "4": "BỐN",
    "5": "NĂM", "6": "SÁU", "7": "BẢY", "8": "TÁM", "9": "CHÍN",
}

_SPOKEN_DIGIT_MAP = {
    "không": "0", "linh": "0", "lẻ": "0",
    "một": "1", "mốt": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4", "tư": "4",
    "năm": "5", "lăm": "5", "nhăm": "5",
    "sáu": "6",
    "bảy": "7", "bẩy": "7",
    "tám": "8",
    "chín": "9",
}

_ENTITY_KEYS = frozenset({
    "name", "title", "lot_name", "operator", "restaurant", "artist", "channel",
    "channels", "album", "playlist", "contact", "company", "organization", "venue",
    "place", "location", "sender", "recipient", "course", "book", "show", "program",
    "station", "airline", "origin", "destination", "brand", "store", "shop", "hotel",
    "symbol", "ticker",
})

_STOP_STARTS = frozenset({
    "DẠ", "VÂNG", "MÌNH", "BẠN", "EM", "ANH", "CHỊ", "CÔ", "CHÚ", "BÁC", "ÔNG", "BÀ",
    "TÔI", "HIỆN", "HÔM", "NGÀY", "LÚC", "TỔNG", "ĐÃ", "CÒN", "CÓ", "OK", "ĐỂ", "VÀO",
    "TẠI", "TRÊN", "Ở", "VỚI", "THỨ", "TUẦN", "THÁNG", "NĂM", "SÁNG", "TRƯA", "CHIỀU",
    "TỐI", "ĐÊM", "MAI", "NAY", "QUA", "SAU", "TRƯỚC", "GẦN", "QUANH", "NGOÀI", "TRONG",
    "THEO", "XEM", "ĐẶT", "TẠO", "HỦY", "XÓA", "MỞ", "BẬT", "TẮT", "CHUYỂN", "TÌM",
    "NHẮN", "GỌI", "GỬI", "KIỂM", "CHÚC", "CẢM", "XIN", "RÕ", "CHẮC", "TRỜI", "NHIỆT",
    "GIÁ", "TIỀN", "PHÍ", "SỐ", "LỊCH", "VIỆC", "BÀI", "CHUYẾN", "TUYẾN", "TÀU", "XE",
    "KÊNH", "PHIM", "QUÁN", "NHÀ", "BÃI", "GA", "SÂN", "ĐƯỜNG", "PHƯỜNG", "QUẬN", "THÀNH",
    "NẾU", "KHI", "NHƯ", "VÀ", "HOẶC", "NHƯNG", "THÌ", "LÀ", "MÀ", "CỦA", "CHO", "CHỌN",
    "ĐANG", "CẬU", "CHỊU", "NÀY", "ĐÓ", "ĐÂY", "KIA", "XONG", "VỪA", "MỚI",
})

_SINGLE_UNIT = r"(?:không|linh|lẻ|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín)"
_UNIT_NO_ZERO = r"(?:một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín)"
_Q_END_FOLLOW = r"(?=\s*(?:$|[?!]|(?:thì|nhỉ|nhé|nha|ạ|đâu|đấy|cơ|chứ|hả|rồi)\b))"
_DIGIT_NO_Q_KHONG = rf"(?:{_UNIT_NO_ZERO}|không(?!{_Q_END_FOLLOW}))"
_DIGIT_ALL = rf"(?:{_UNIT_NO_ZERO}|không)"
_TENS_TAIL = r"(?:mốt|hai|ba|bốn|tư|lăm|sáu|bảy|bẩy|tám|chín)"
_TWO_DIGIT_TENS = (
    rf"(?:mười(?:\s+{_UNIT_NO_ZERO})?"
    rf"|(?:hai|ba|bốn|tư|năm|sáu|bảy|bẩy|tám|chín)\s+mươi(?:\s+{_TENS_TAIL})?"
    rf"|(?:hai|ba|bốn|tư|năm|sáu|bảy|bẩy|tám|chín)\s+(?:mốt|tư|lăm))"
)
_SPOKEN_CODE_NUM = rf"(?:{_TWO_DIGIT_TENS}|{_UNIT_NO_ZERO}|không)"
_FLIGHT_TRAIN_CODE_NUM = (
    rf"(?:{_TWO_DIGIT_TENS}"
    rf"|{_UNIT_NO_ZERO}\s+{_DIGIT_ALL}\s+{_DIGIT_NO_Q_KHONG}"
    rf"|{_UNIT_NO_ZERO}\s+{_DIGIT_NO_Q_KHONG}"
    rf"|{_UNIT_NO_ZERO}|không)"
)
_SPOKEN_NUM_TOK = r"(?:không|linh|lẻ|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|bẩy|tám|chín|mười|mươi)"
_TRAILING_DIGIT_CONT = (
    r"(?:linh|lẻ|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|bẩy|tám|chín|mười|mươi"
    r"|không\s+(?:không|một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|mười))"
)
_BAD_FOLLOW = (
    rf"(?!\s+(?:{_TRAILING_DIGIT_CONT}|giờ|tiếng|phút|giây|lúc|chút|tí|lần|ngày|hôm\s+nữa|tuần|tháng|năm|"
    r"người|cái|vé|chiếc|chỗ|con|bạn|việc|lít|ký|cân|triệu|nghìn|ngàn|trăm|chục|đồng)\b)"
)
_CLAUSE_FOLLOW = r"(?:\s+(?:thì|nhỉ|nhé|nha|ạ|đâu|đấy|cơ|chứ|hả|rồi)\b)"


def load_tokens_vocab_chars(tokens_path: Path | None) -> set[str] | None:
    """Load supported character set from a sherpa-onnx tokens.txt file."""
    if tokens_path is None or not Path(tokens_path).is_file():
        return None
    chars: set[str] = set()
    for line in Path(tokens_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        tok = parts[0].replace("\u2581", "")
        if tok.startswith("<") and tok.endswith(">"):
            continue
        chars.update(tok)
    return chars


def ensure_bpe_vocab(model_path: Path, vocab_dir: Path | None = None) -> Path | None:
    """Return an existing bpe.vocab path or synthesize one from bpe.model inside vocab_dir."""
    model_path = Path(model_path)
    direct = model_path / "bpe.vocab"
    if direct.is_file():
        return direct
    bpe_model = model_path / "bpe.model"
    if not bpe_model.is_file() or vocab_dir is None:
        return None
    import sentencepiece as spm

    vocab_dir = Path(vocab_dir)
    vocab_dir.mkdir(parents=True, exist_ok=True)
    target = vocab_dir / "bpe.vocab"
    if target.is_file():
        return target
    sp = spm.SentencePieceProcessor(model_file=str(bpe_model))
    tmp = vocab_dir / "bpe.vocab.tmp"
    with tmp.open("w", encoding="utf-8") as handle:
        for idx in range(sp.get_piece_size()):
            handle.write(f"{sp.id_to_piece(idx)}\t{sp.get_score(idx)}\n")
    tmp.replace(target)
    return target


def _expand_alnum_token(tok: str) -> str:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", tok)
    if not match:
        return tok
    letters, digits = match.group(1).upper(), match.group(2)
    if len(digits) <= 3:
        spoken = " ".join(_DIGIT_WORDS[d] for d in digits)
        if len(letters) == 2:
            return f"{' '.join(letters)} {spoken}"
        return f"{letters} {spoken}"
    return letters


def sanitize_hotword_phrase(
    text: str,
    *,
    allow_single: bool = False,
    vocab_chars: set[str] | None = None,
) -> str:
    """Normalize a hotword candidate to uppercase tokens representable in tokens.txt."""
    if not text:
        return ""
    text = text.replace("+", " CỘNG ").replace("&", " ").replace("-", " ")
    tokens: list[str] = []
    for raw in re.split(r"\s+", text.strip()):
        cleaned = re.sub(r"[^\w]", "", raw, flags=re.UNICODE)
        if not cleaned:
            continue
        if any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
            tokens.extend(_expand_alnum_token(cleaned).split())
        elif cleaned.isdigit():
            if len(cleaned) <= 2:
                tokens.extend(_DIGIT_WORDS[d] for d in cleaned)
        else:
            tokens.append(cleaned.upper())
    phrase = " ".join(tokens).strip()
    if vocab_chars is not None:
        phrase = "".join(ch if (ch in vocab_chars or ch == " ") else " " for ch in phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    words = phrase.split()
    if not words or len(words) > 5 or len(phrase) < 4:
        return ""
    if words[0] in _STOP_STARTS and (len(words) == 1 or (len(words) == 2 and words[1] in _STOP_STARTS)):
        return ""
    if len(words) == 1:
        if not allow_single or not words[0].isascii() or len(words[0]) < 4:
            return ""
    return phrase


def _walk_entity_values(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in _ENTITY_KEYS:
                if isinstance(val, str):
                    out.append(val)
                elif isinstance(val, list):
                    out.extend(item for item in val if isinstance(item, str))
            else:
                _walk_entity_values(val, out)
    elif isinstance(obj, list):
        for val in obj:
            _walk_entity_values(val, out)


def _walk_context_entities(ctx: dict[str, Any], out: list[str]) -> None:
    """Collect entity names from visible structured context state, skipping flags/timestamps."""
    if not isinstance(ctx, dict):
        return
    skip_top = {"installed_apps", "user_profile", "now", "today", "date", "time", "timezone", "location", "device"}
    collections = ctx.get("collections")
    if isinstance(collections, dict):
        _walk_entity_values(collections, out)
    for key, val in ctx.items():
        if key in skip_top or key == "collections":
            continue
        if isinstance(val, (dict, list)):
            _walk_entity_values(val, out)


def extract_sample_hotwords(
    item: dict[str, Any],
    *,
    up_to_turn: int | None = None,
    vocab_chars: set[str] | None = None,
    max_phrases: int = 24,
) -> str:
    """Extract clean sample-level BPE hotwords from user_profile, context entities, and prior assistant turns."""
    if not isinstance(item, dict):
        return ""
    primary: list[tuple[str, bool]] = []
    secondary: list[tuple[str, bool]] = []
    ctx = item.get("context") or {}
    profile = ctx.get("user_profile") or {}
    name = (profile.get("name") or "").strip()
    if name:
        primary.append((name, False))
    for addr_key in ("home_address", "work_address"):
        addr = profile.get(addr_key) or ""
        for seg in addr.split(","):
            cleaned = re.sub(r"^\s*\d+\w*\s+", "", seg).strip()
            if cleaned and len(cleaned.split()) >= 2:
                primary.append((cleaned, False))
    ctx_entities: list[str] = []
    _walk_context_entities(ctx, ctx_entities)
    for entity in ctx_entities:
        primary.append((entity, True))

    history = item.get("history") or []
    turn_limit = len(history) if up_to_turn is None else max(0, up_to_turn)
    for turn in history[:turn_limit]:
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        text = turn.get("text") or ""
        for quoted in re.findall(r"['\"]([^'\"]{3,45})['\"]", text):
            primary.append((quoted, True))
        for code in re.findall(r"\b(?:[A-Za-z]{1,4}-?\d{1,3}|[A-Z][A-Za-z0-9]{0,4}\+\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9+]+)\b", text):
            primary.append((code.replace("-", ""), False))
        multi_spans: list[tuple[int, int]] = []
        for match in re.finditer(r"\b[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ0-9+]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ0-9+]+){1,4}", text):
            multi_spans.append((match.start(), match.end()))
            cap = match.group(0)
            words = cap.split()
            if words[0].upper() in _STOP_STARTS:
                if len(words) < 3:
                    continue
                cap = " ".join(words[1:])
            primary.append((cap, False))
        # Standalone ASCII brand/entity tokens (CamelCase or single non-sentence-initial capitalized words)
        for match in re.finditer(r"\b([A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]{4,})\b", text):
            word = match.group(1)
            is_camel = bool(re.search(r"[a-z][A-Z]", word))
            in_multi = any(start <= match.start() and match.end() <= end for start, end in multi_spans)
            if not is_camel and (in_multi or match.start() == 0 or text[match.start() - 1] in ".!?\n"):
                continue
            if word.upper() not in _STOP_STARTS and word.isascii():
                secondary.append((word, True))
        entities: list[str] = []
        _walk_entity_values(turn.get("tool_results"), entities)
        for entity in entities:
            primary.append((entity, True))

    seen: set[str] = set()
    final: list[str] = []
    for raw, allow_single in primary:
        phrase = sanitize_hotword_phrase(raw, allow_single=allow_single, vocab_chars=vocab_chars)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        final.append(phrase)
        if len(final) >= max_phrases:
            break
    for raw, allow_single in secondary:
        if len(final) >= max_phrases:
            break
        phrase = sanitize_hotword_phrase(raw, allow_single=allow_single, vocab_chars=vocab_chars)
        if not phrase or phrase in seen:
            continue
        if any(f" {phrase} " in f" {existing} " for existing in final):
            continue
        seen.add(phrase)
        final.append(phrase)
    return " / ".join(final)


def _spoken_to_digits(seq: str) -> str:
    words = [w.lower() for w in seq.split()]
    if words == ["mười"]:
        return "10"
    if len(words) == 2 and words[0] == "mười" and words[1] in _SPOKEN_DIGIT_MAP:
        return "1" + _SPOKEN_DIGIT_MAP[words[1]]
    if len(words) == 2 and words[1] == "mươi" and words[0] in _SPOKEN_DIGIT_MAP:
        return _SPOKEN_DIGIT_MAP[words[0]] + "0"
    if len(words) == 2 and words[0] in _SPOKEN_DIGIT_MAP and words[1] in {"mốt", "tư", "lăm"}:
        return _SPOKEN_DIGIT_MAP[words[0]] + _SPOKEN_DIGIT_MAP[words[1]]
    if len(words) == 3 and words[0] in _SPOKEN_DIGIT_MAP and words[1] == "mươi" and words[2] in _SPOKEN_DIGIT_MAP:
        return _SPOKEN_DIGIT_MAP[words[0]] + _SPOKEN_DIGIT_MAP[words[2]]
    if all(w in _SPOKEN_DIGIT_MAP for w in words):
        return "".join(_SPOKEN_DIGIT_MAP[w] for w in words)
    return ""


def _replace_alnum_with_clause_guard(prefix: str, spoken_seq: str, suffix_text: str) -> str:
    digits = _spoken_to_digits(spoken_seq)
    if not digits:
        return f"{prefix} {spoken_seq}"
    return f"{prefix}{digits}"


_SPACED_CODE_PREFIX_RULES: tuple[tuple[str, str, str], ...] = (
    (r"s\s*e", "se", _FLIGHT_TRAIN_CODE_NUM),
    (r"r\s*v", "rv", _FLIGHT_TRAIN_CODE_NUM),
    (r"(?:v\s*t\s*v|h\s*t\s*v|v\s*o\s*v|s\s*c\s*t\s*v)", "", _SPOKEN_CODE_NUM),
)


def post_normalize_asr_text(text: str) -> str:
    """Deterministically normalize separated spoken codes and unambiguous domain patterns."""
    if not isinstance(text, str):
        raise ValueError("Transcript must be a string")
    t = text
    # 1) Unambiguous separated letter prefixes + spoken code number (e.g., s e năm -> se5, rv 720 -> rv720, h t v 7 -> htv7).
    for prefix_pat, fixed_prefix, num_pat in _SPACED_CODE_PREFIX_RULES:
        t = re.sub(
            rf"\b({prefix_pat})\s+({num_pat})({_CLAUSE_FOLLOW})?{_BAD_FOLLOW}",
            lambda m, fp=fixed_prefix: (
                _replace_alnum_with_clause_guard(fp or re.sub(r"\s+", "", m.group(1)).lower(), m.group(2), m.group(3) or "")
                + (m.group(3) or "")
            ),
            t,
            flags=re.I,
        )
    # 2) Train code prefix N A (disambiguated by train context so "quả na" is never touched).
    t = re.sub(
        rf"\b(vé\s+tàu|chuyến\s+tàu|tàu|chuyến)\s+n\s*a\s+({_SPOKEN_CODE_NUM}){_BAD_FOLLOW}",
        lambda m: f"{m.group(1)} na{_spoken_to_digits(m.group(2))}",
        t,
        flags=re.I,
    )
    # 3) Bus route codes in transit context.
    t = re.sub(
        rf"\b(tuyến|buýt|xe\s+buýt)\s+r\s+({_SPOKEN_CODE_NUM}){_BAD_FOLLOW}",
        lambda m: f"{m.group(1)} r{_spoken_to_digits(m.group(2))}",
        t,
        flags=re.I,
    )
    # 4) Rice / product variety codes in explicit context (st21, st24, st25, etc.).
    t = re.sub(
        rf"\b(gạo)\s+s\s*t\s+(hai\s+năm|{_SPOKEN_CODE_NUM}){_BAD_FOLLOW}",
        lambda m: f"{m.group(1)} st{'25' if m.group(2).lower() == 'hai năm' else _spoken_to_digits(m.group(2))}",
        t,
        flags=re.I,
    )
    # 5) Fuel grades: RON 92/95/98 and E5/E10.
    t = re.sub(
        rf"\b(?:ron|rôn)\s+(chín\s+(?:mươi\s+)?(?:hai|lăm|tám)|chín\s+mươi\s+năm){_BAD_FOLLOW}",
        lambda m: "ron " + (
            "95" if ("lăm" in m.group(1).lower() or "năm" in m.group(1).lower())
            else ("98" if "tám" in m.group(1).lower() else "92")
        ),
        t,
        flags=re.I,
    )
    t = re.sub(
        rf"\b(xăng|lít|đổ|bơm|coi|xem|giá|loại)\s+e\s+(năm|mười){_BAD_FOLLOW}",
        lambda m: f"{m.group(1)} e{'5' if m.group(2).lower() == 'năm' else '10'}",
        t,
        flags=re.I,
    )
    # 6) Coffee G7 & ISO paper sizes A0..A6.
    t = re.sub(rf"\b(cà\s+phê)\s+g\s+bảy{_BAD_FOLLOW}", r"\1 g7", t, flags=re.I)
    t = re.sub(
        rf"\b(giấy|khổ|tờ|trang|in)\s+a\s+(không|một|hai|ba|bốn|tư|năm|sáu){_BAD_FOLLOW}",
        lambda m: f"{m.group(1)} a{_SPOKEN_DIGIT_MAP[m.group(2).lower()]}",
        t,
        flags=re.I,
    )
    # 7) Spoken "meo" for email in strict email-verb context, and spaced "s j c" gold code.
    t = re.sub(
        r"\b(gửi|đọc|check|kiểm\s+tra|soạn|trả\s+lời|hộp\s+thư)\s+(?:e\s+)?meo\b",
        r"\1 mail",
        t,
        flags=re.I,
    )
    t = re.sub(r"\b(vàng|giá|lượng|chỉ)\s+s\s*j\s*c\b", r"\1 sjc", t, flags=re.I)
    return t


_VI_ASCII_SYLLABLE_RE = re.compile(
    r"^(?:ngh|ng|nh|ch|tr|th|ph|kh|gh|gi|qu|[bcdghklmnpqrstvx])?"
    r"(?:uye|uoi|uay|uya|ieu|yeu|oai|oao|oau|oay|ai|ao|au|ay|eo|eu|ia|ie|iu|oa|oe|oi|ua|ue|ui|uo|uu|[aeiouy])"
    r"(?:ng|nh|ch|[cmnpt])?$"
)
_VI_ASCII_GUARD_WORDS = frozenset({
    "anh", "em", "chi", "co", "chu", "bac", "ong", "ba", "toi", "minh", "ban", "con",
    "khoa", "linh", "trang", "long", "dung", "hung", "phuong", "thanh", "binh", "quang",
    "xuan", "trinh", "ngan", "thao", "trung", "khang", "hanh", "phuc", "hoang", "loan",
    "nam", "lan", "mai", "hoa", "thu", "tam", "sau", "bay", "bon", "chin", "muoi",
    "sang", "trua", "chieu", "ngay", "thang", "tuan", "gio", "phut", "nha", "nhe",
    "nghe", "nghen", "hen", "roi", "xong", "trong", "ngoai", "phan", "tran", "nguyen",
    "pham", "huynh", "dang", "bui", "doan", "dinh", "duong", "luong", "truong", "vuong",
    "quoc", "viet", "nhat", "han", "phap", "duc", "uc", "my",
})
_ENGLISH_INFLECTION_SUFFIXES = ("s", "es", "ed", "ing", "er", "ers", "ly")


def _is_vietnamese_ascii_word(word: str) -> bool:
    wl = word.lower()
    if not wl.isascii():
        return True
    if wl in _VI_ASCII_GUARD_WORDS:
        return True
    return bool(
        _VI_ASCII_SYLLABLE_RE.match(wl)
        and not re.search(r"c(?=[eiy])", wl)
        and not re.search(r"k(?=[aou])", wl)
        and not re.search(r"^q[^u]", wl)
    )


def _is_morphological_variant(source_word: str, target_word: str) -> bool:
    """Avoid shrinking valid inflected English words into shorter context stems (e.g. grilled -> grill)."""
    if source_word == target_word:
        return True
    for suf in _ENGLISH_INFLECTION_SUFFIXES:
        if source_word == target_word + suf or target_word == source_word + suf:
            return True
    return False


def _lev_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        ndp = [i]
        for j, y in enumerate(b, 1):
            ndp.append(min(ndp[-1] + 1, dp[j] + 1, dp[j - 1] + (0 if x == y else 1)))
        dp = ndp
    return 1.0 - dp[-1] / m


def _phonetic_code(text: str) -> str:
    s = text.lower()
    s = re.sub(r"ght?", "t", s)
    s = re.sub(r"ph", "f", s)
    s = re.sub(r"th", "t", s)
    s = re.sub(r"qu", "w", s)
    s = re.sub(r"ck|c|k|q", "k", s)
    s = re.sub(r"[dtđ]", "t", s)
    s = re.sub(r"[bpv]", "b", s)
    s = re.sub(r"[szx]", "s", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def _consonant_skeleton(text: str) -> str:
    p = _phonetic_code(text)
    if not p:
        return ""
    return p[0] + re.sub(r"[aeiouyw]+", "", p[1:])


def _oov_similarity(a: str, b: str) -> float:
    r1 = _lev_ratio(a, b)
    r2 = _lev_ratio(_phonetic_code(a), _phonetic_code(b))
    base = max(r1, r2)
    s1, s2 = _consonant_skeleton(a), _consonant_skeleton(b)
    if min(len(s1), len(s2)) >= 3:
        return 0.65 * base + 0.35 * _lev_ratio(s1, s2)
    return base


def _prefer_surface(existing: str | None, candidate: str) -> str:
    if not existing:
        return candidate
    if any(ch.isupper() for ch in candidate) and not any(ch.isupper() for ch in existing):
        return candidate
    return existing


def _extract_context_repair_targets(
    item: dict[str, Any], up_to_turn: int | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    raw_cands: list[str] = []
    ctx = item.get("context") or {}
    prof = ctx.get("user_profile") or {}
    for key in ("name", "home_address", "work_address"):
        if prof.get(key):
            raw_cands.append(prof[key])
    _walk_context_entities(ctx, raw_cands)
    history = item.get("history") or []
    limit = len(history) if up_to_turn is None else max(0, up_to_turn)
    for turn in history[:limit]:
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        text = turn.get("text") or ""
        raw_cands.extend(re.findall(r"['\"]([^'\"]{3,45})['\"]", text))
        raw_cands.extend(re.findall(r"\b[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ0-9+]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ0-9+]+){1,4}", text))
        raw_cands.extend(re.findall(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b", text))
        raw_cands.extend(
            m.group(1)
            for m in re.finditer(r"(?<=[^.!?\n]\s)\b([A-Z][a-z]{3,})\b", text)
        )
        # Also include standalone non-Vietnamese ASCII words appearing in prior assistant turns.
        for tok in re.findall(r"\b([A-Za-z]{5,})\b", text):
            if not _is_vietnamese_ascii_word(tok):
                raw_cands.append(tok)
        _walk_entity_values(turn.get("tool_results"), raw_cands)

    singles: dict[str, str] = {}
    phrases: dict[str, str] = {}
    for cand in raw_cands:
        surface = re.sub(r"[^\w\s]", " ", cand.replace("-", " ").replace("'", " ")).strip()
        surface = re.sub(r"\s+", " ", surface)
        surf_ws = surface.split()
        low_ws = [w.lower() for w in surf_ws]
        if 2 <= len(surf_ws) <= 4:
            key = " ".join(low_ws)
            phrases[key] = _prefer_surface(phrases.get(key), surface)
        if len(surf_ws) >= 3:
            tail_key = " ".join(low_ws[-2:])
            tail_surf = " ".join(surf_ws[-2:])
            phrases[tail_key] = _prefer_surface(phrases.get(tail_key), tail_surf)
        for sw, lw in zip(surf_ws, low_ws):
            if lw.isascii() and lw.isalpha() and len(lw) >= 4 and lw not in _VI_ASCII_GUARD_WORDS:
                singles[lw] = _prefer_surface(singles.get(lw), sw)
    return singles, phrases


def _apply_surface_casing(raw_span: str, canonical: str, full_text: str) -> str:
    if full_text.isupper():
        return canonical.upper()
    if any(ch.isupper() for ch in canonical):
        return canonical
    if raw_span and raw_span[0].isupper() and canonical:
        return canonical[0].upper() + canonical[1:]
    return canonical


def repair_sample_entities(text: str, item: dict[str, Any], *, up_to_turn: int | None = None) -> str:
    """Repair OOV ASCII entity slips using unambiguous entities from the same sample's context."""
    if not isinstance(text, str) or not isinstance(item, dict) or not text:
        return text
    ctx_tokens, ctx_phrases = _extract_context_repair_targets(item, up_to_turn=up_to_turn)
    if not ctx_tokens and not ctx_phrases:
        return text
    matches = list(re.finditer(r"[A-Za-zÀ-ỹ0-9+]+", text))
    if not matches:
        return text
    words = [m.group(0) for m in matches]
    w_low = [w.lower() for w in words]

    edits: list[tuple[int, int, str]] = []
    i = 0
    while i < len(words):
        w = w_low[i]
        if i + 1 < len(words):
            w1, w2 = w_low[i], w_low[i + 1]
            # 1) Two-word entity phrase match before split-word merge.
            if (
                w1.isalpha()
                and w2.isalpha()
                and w1.isascii()
                and w2.isascii()
                and w1 not in _VI_ASCII_GUARD_WORDS
                and w2 not in _VI_ASCII_GUARD_WORDS
                and (not _is_vietnamese_ascii_word(w1) or not _is_vietnamese_ascii_word(w2))
                and not (w1 in ctx_tokens and w2 in ctx_tokens)
            ):
                phrase_scored: dict[str, float] = {}
                for ph_key, ph_surf in ctx_phrases.items():
                    p_ws = ph_key.split()
                    if len(p_ws) != 2 or not all(x.isascii() for x in p_ws):
                        continue
                    pw1, pw2 = p_ws
                    if f"{words[i]} {words[i + 1]}" == ph_surf:
                        continue
                    s1, s2 = _oov_similarity(w1, pw1), _oov_similarity(w2, pw2)
                    if (
                        (min(s1, s2) >= 0.72 and 0.5 * (s1 + s2) >= 0.82)
                        or (s2 >= 0.85 and len(w1) >= 4 and _phonetic_code(w1)[0] == _phonetic_code(pw1)[0] and s1 >= 0.58)
                        or (s1 >= 0.85 and s2 >= 0.82)
                    ):
                        sc = 0.5 * s1 + 0.5 * s2
                        if ph_key not in phrase_scored or sc > phrase_scored[ph_key]:
                            phrase_scored[ph_key] = sc
                if phrase_scored:
                    sorted_ph = sorted(
                        phrase_scored.items(),
                        key=lambda kv: (kv[1], 1 if kv[0].endswith(w2[-1]) else 0),
                        reverse=True,
                    )
                    best_ph, best_sc = sorted_ph[0]
                    distinct = [kv for kv in sorted_ph if kv[0].rstrip("s") != best_ph.rstrip("s")]
                    if not distinct or best_sc - distinct[0][1] >= 0.08:
                        edits.append((i, i + 2, ctx_phrases[best_ph]))
                        i += 2
                        continue

            # 2) Two adjacent split foreign tokens -> single context entity (e.g. sao tel -> SaoTel).
            joined = w1 + w2
            if (
                w1.isascii()
                and w2.isascii()
                and w1.isalpha()
                and w2.isalpha()
                and len(joined) >= 5
                and w1 not in _VI_ASCII_GUARD_WORDS
                and w2 not in _VI_ASCII_GUARD_WORDS
                and w1 not in ctx_tokens
                and w2 not in ctx_tokens
                and (not _is_vietnamese_ascii_word(w1) or not _is_vietnamese_ascii_word(w2))
            ):
                scored: list[tuple[float, str]] = []
                pj = _phonetic_code(joined)
                for ct in ctx_tokens:
                    pct = _phonetic_code(ct)
                    if (pj[0] != pct[0] and len(joined) < 6) or abs(len(joined) - len(ct)) > 3:
                        continue
                    sc = _oov_similarity(joined, ct)
                    if joined == ct or (sc >= 0.82 and (pj[0] == pct[0] or sc >= 0.85)):
                        scored.append((1.0 if joined == ct else sc, ct))
                scored.sort(reverse=True)
                if scored and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.08):
                    edits.append((i, i + 2, ctx_tokens[scored[0][1]]))
                    i += 2
                    continue

        # 3) Single OOV ASCII token -> unambiguous context entity token.
        if (
            w.isascii()
            and w.isalpha()
            and len(w) >= 4
            and not _is_vietnamese_ascii_word(w)
            and w not in ctx_tokens
        ):
            scored_s: list[tuple[float, str]] = []
            pw = _phonetic_code(w)
            for ct in ctx_tokens:
                if _is_morphological_variant(w, ct):
                    continue
                pct = _phonetic_code(ct)
                same_init = pw[0] == pct[0]
                if not same_init and min(len(w), len(ct)) < 6:
                    continue
                sc = _oov_similarity(w, ct)
                if len(w) >= 5 and ct.startswith(w) and 1 <= len(ct) - len(w) <= 2:
                    sc = max(sc, 0.84)
                req = 0.80 if same_init else 0.84
                if sc >= req:
                    scored_s.append((sc, ct))
            scored_s.sort(reverse=True)
            if scored_s and (len(scored_s) == 1 or scored_s[0][0] - scored_s[1][0] >= 0.08):
                edits.append((i, i + 1, ctx_tokens[scored_s[0][1]]))
        i += 1

    if not edits:
        return text
    out_parts: list[str] = []
    last_pos = 0
    for s_idx, e_idx, rep in edits:
        cs = matches[s_idx].start()
        ce = matches[e_idx - 1].end()
        out_parts.append(text[last_pos:cs])
        out_parts.append(_apply_surface_casing(text[cs:ce], rep, text))
        last_pos = ce
    out_parts.append(text[last_pos:])
    return "".join(out_parts)


def hotwords_enabled(asr: Any) -> bool:
    """Return True only for ASR instances supporting per-stream BPE hotwords."""
    return bool(
        asr is not None
        and getattr(asr, "supports_hotwords", False)
        and getattr(asr, "hotwords_mode", "off") != "off"
    )


def transcribe_with_item(asr: Any, audio_file: Path, item: dict[str, Any], *, up_to_turn: int | None = None) -> str:
    """Transcribe one utterance with contextual hotwords and sample-scoped entity repair when enabled."""
    if hotwords_enabled(asr):
        hw = extract_sample_hotwords(item, up_to_turn=up_to_turn, vocab_chars=getattr(asr, "vocab_chars", None))
        text = asr.transcribe(audio_file, hotwords=hw) if hw else asr.transcribe(audio_file)
    else:
        text = asr.transcribe(audio_file)
    if getattr(asr, "post_normalize", False):
        text = repair_sample_entities(text, item, up_to_turn=up_to_turn)
    return text


class _ContextAwareHistoryASR:
    """Lightweight ASR adapter preserving single-arg history_text callers with non-leaky hotwords."""

    def __init__(self, asr: Any, item: dict[str, Any], split_dir: Path):
        self._asr = asr
        self._item = item
        self._vocab = getattr(asr, "vocab_chars", None)
        self._turn_queues: dict[str, deque[int]] = {}
        root = Path(split_dir).resolve()
        for idx, turn in enumerate(item.get("history") or []):
            if not isinstance(turn, dict) or turn.get("text") is not None:
                continue
            rel = turn.get("audio") or turn.get("audio_path")
            if isinstance(rel, str) and rel.strip():
                key = str((root / rel).resolve())
                self._turn_queues.setdefault(key, deque()).append(idx)

    def transcribe(self, path: Path) -> str:
        resolved = str(Path(path).resolve())
        queue = self._turn_queues.get(resolved)
        up_to = queue.popleft() if queue else None
        return transcribe_with_item(self._asr, path, self._item, up_to_turn=up_to)


def history_asr_for_item(asr: Any, item: dict[str, Any], split_dir: Path) -> Any:
    """Wrap an ASR engine to attach non-leaky turn hotwords and entity repair during history resolution."""
    if not hotwords_enabled(asr) and not getattr(asr, "post_normalize", False):
        return asr
    return _ContextAwareHistoryASR(asr, item, split_dir)


def collect_batch_audio_hotwords(
    items: list[dict[str, Any]],
    split_dir: Path,
    vocab_chars: set[str] | None,
    *,
    skip_main_fn=None,
) -> tuple[list[Path], list[str]]:
    """Collect deduplicated (resolved_audio_path, hotwords) pairs across items and history turns."""
    from .data import audio_path

    paths: list[Path] = []
    hotwords_list: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if skip_main_fn is None or not skip_main_fn(item):
            if item.get("audio"):
                p = audio_path(split_dir, item["audio"])
                hw = extract_sample_hotwords(item, vocab_chars=vocab_chars)
                key = (str(p.resolve()), hw)
                if key not in seen:
                    seen.add(key)
                    paths.append(p)
                    hotwords_list.append(hw)
        for idx, turn in enumerate(item.get("history") or []):
            if not isinstance(turn, dict) or turn.get("text") is not None:
                continue
            rel = turn.get("audio") or turn.get("audio_path")
            if isinstance(rel, str) and rel.strip():
                p = audio_path(split_dir, rel)
                hw = extract_sample_hotwords(item, up_to_turn=idx, vocab_chars=vocab_chars)
                key = (str(p.resolve()), hw)
                if key not in seen:
                    seen.add(key)
                    paths.append(p)
                    hotwords_list.append(hw)
    return paths, hotwords_list
