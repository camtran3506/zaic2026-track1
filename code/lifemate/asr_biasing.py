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
    "album", "playlist", "contact", "company", "organization", "venue", "place",
    "location", "sender", "recipient", "course", "book", "show", "program",
    "station", "airline", "origin", "destination",
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

_SINGLE_UNIT = r"(?:một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|tám|chín)"
_CHANNEL_NUM = rf"(?:mười(?:\s+{_SINGLE_UNIT})?|hai\s+mươi(?:\s+(?:mốt|hai|ba|tư|bốn|lăm))?|{_SINGLE_UNIT})"
_SPOKEN_NUM_TOK = r"(?:không|linh|lẻ|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|bẩy|tám|chín|mười|mươi)"
_RV_NUM_SEQ = rf"(?:{_SPOKEN_NUM_TOK}(?:\s+{_SPOKEN_NUM_TOK}){{0,4}})"
_BAD_FOLLOW = (
    r"(?!\s+(?:giờ|tiếng|phút|giây|lúc|chút|tí|lần|ngày|hôm|tuần|tháng|năm|"
    r"người|cái|vé|chiếc|chỗ|con|bạn|việc|lít|ký|cân|triệu|nghìn|ngàn|trăm|chục|đồng)\b)"
)
_CLAUSE_FOLLOW = r"(?:\s+(?:thì|nhỉ|nhé|nha|ạ|đâu|đấy|cơ|đi|chứ|hả|rồi)\b)"


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
        if letters in {"SE", "NA"}:
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
            if key in _ENTITY_KEYS and isinstance(val, str):
                out.append(val)
            else:
                _walk_entity_values(val, out)
    elif isinstance(obj, list):
        for val in obj:
            _walk_entity_values(val, out)


def extract_sample_hotwords(
    item: dict[str, Any],
    *,
    up_to_turn: int | None = None,
    vocab_chars: set[str] | None = None,
    max_phrases: int = 24,
) -> str:
    """Extract clean sample-level BPE hotwords from user_profile, prior assistant turns, and apps."""
    if not isinstance(item, dict):
        return ""
    candidates: list[tuple[str, bool]] = []
    ctx = item.get("context") or {}
    profile = ctx.get("user_profile") or {}
    name = (profile.get("name") or "").strip()
    if name:
        candidates.append((name, False))
    for addr_key in ("home_address", "work_address"):
        addr = profile.get(addr_key) or ""
        for seg in addr.split(","):
            cleaned = re.sub(r"^\s*\d+\w*\s+", "", seg).strip()
            if cleaned and len(cleaned.split()) >= 2:
                candidates.append((cleaned, False))

    history = item.get("history") or []
    turn_limit = len(history) if up_to_turn is None else max(0, up_to_turn)
    for turn in history[:turn_limit]:
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        text = turn.get("text") or ""
        for quoted in re.findall(r"['\"]([^'\"]{3,45})['\"]", text):
            candidates.append((quoted, True))
        for code in re.findall(r"\b(?:[A-Za-z]{1,4}\d{1,3}|S\+\s*Sport)\b", text):
            candidates.append((code, False))
        for cap in re.findall(r"\b[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ0-9+]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ0-9+]+){1,4}", text):
            words = cap.split()
            if words[0].upper() in _STOP_STARTS:
                if len(words) < 3:
                    continue
                cap = " ".join(words[1:])
            candidates.append((cap, False))
        entities: list[str] = []
        _walk_entity_values(turn.get("tool_results"), entities)
        for entity in entities:
            candidates.append((entity, True))

    apps = {k for k, v in (ctx.get("installed_apps") or {}).items() if v}
    if "public_transit" in apps or "train_bus_tickets" in apps:
        for phrase in ("XE BUÝT RA GA", "CHUYẾN BUÝT", "TÀU S E BA", "TÀU S E NĂM", "TÀU S E BẢY"):
            candidates.append((phrase, False))
    if "gold_price" in apps:
        candidates.append(("VÀNG SJC", False))

    seen: set[str] = set()
    final: list[str] = []
    for raw, allow_single in candidates:
        phrase = sanitize_hotword_phrase(raw, allow_single=allow_single, vocab_chars=vocab_chars)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        final.append(phrase)
        if len(final) >= max_phrases:
            break
    return " / ".join(final)


def _spoken_to_digits(seq: str) -> str:
    words = [w.lower() for w in seq.split()]
    if words == ["mười"]:
        return "10"
    if len(words) == 2 and words[0] == "mười" and words[1] in _SPOKEN_DIGIT_MAP:
        return "1" + _SPOKEN_DIGIT_MAP[words[1]]
    if len(words) == 2 and words[0] == "hai" and words[1] == "mươi":
        return "20"
    if len(words) == 3 and words[0] in _SPOKEN_DIGIT_MAP and words[1] == "mươi" and words[2] in _SPOKEN_DIGIT_MAP:
        return _SPOKEN_DIGIT_MAP[words[0]] + _SPOKEN_DIGIT_MAP[words[2]]
    out: list[str] = []
    for word in words:
        if word in _SPOKEN_DIGIT_MAP:
            out.append(_SPOKEN_DIGIT_MAP[word])
        elif word == "mười":
            out.append("10")
        elif word == "mươi":
            continue
        else:
            return ""
    return "".join(out)


def _replace_alnum_with_clause_guard(prefix: str, spoken_seq: str, suffix_text: str) -> str:
    words = spoken_seq.split()
    kept_tail = ""
    if len(words) >= 2 and words[-1].lower() == "không" and re.match(_CLAUSE_FOLLOW, suffix_text, re.I):
        kept_tail = " " + words.pop()
    digits = _spoken_to_digits(" ".join(words))
    if not digits:
        return prefix + spoken_seq + kept_tail
    return f"{prefix}{digits}{kept_tail}"


def post_normalize_asr_text(text: str) -> str:
    """Deterministically normalize separated spoken codes and unambiguous ASR slips before text norm."""
    if not isinstance(text, str):
        raise ValueError("Transcript must be a string")
    t = text
    t = re.sub(
        rf"\bs\s*e\s+({_CHANNEL_NUM})(\s+(?:thì|nhỉ|nhé|nha|ạ|đâu|đấy|cơ|đi|chứ|hả|rồi))?{_BAD_FOLLOW}",
        lambda m: _replace_alnum_with_clause_guard("se", m.group(1), m.group(2) or "") + (m.group(2) or ""),
        t,
        flags=re.I,
    )
    t = re.sub(
        rf"\b(tàu|chuyến)\s+n\s*a\s+({_CHANNEL_NUM}){_BAD_FOLLOW}",
        lambda m: f"{m.group(1)} na{_spoken_to_digits(m.group(2))}",
        t,
        flags=re.I,
    )
    t = re.sub(
        rf"\br\s*v\s+({_RV_NUM_SEQ})({_CLAUSE_FOLLOW})?{_BAD_FOLLOW}",
        lambda m: _replace_alnum_with_clause_guard("rv", m.group(1), m.group(2) or "") + (m.group(2) or ""),
        t,
        flags=re.I,
    )
    t = re.sub(
        rf"\b(vtv|htv|vov|sctv)\s+({_CHANNEL_NUM}){_BAD_FOLLOW}",
        lambda m: f"{m.group(1).lower()}{_spoken_to_digits(m.group(2))}",
        t,
        flags=re.I,
    )
    t = re.sub(
        rf"\b(tuyến|buýt)\s+r\s+({_CHANNEL_NUM}){_BAD_FOLLOW}",
        lambda m: f"{m.group(1)} r{_spoken_to_digits(m.group(2))}",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(ron|rôn)\s+(chín\s+(?:mươi\s+)?(?:lăm|năm)|chín\s+(?:mươi\s+)?hai)\b",
        lambda m: "ron 95" if ("lăm" in m.group(2).lower() or "năm" in m.group(2).lower()) else "ron 92",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(xăng|lít|đổ)\s+e\s+(năm|mười)\b",
        lambda m: f"{m.group(1)} e{'5' if m.group(2).lower() == 'năm' else '10'}",
        t,
        flags=re.I,
    )
    t = re.sub(r"\b(cà\s+phê)\s+g\s+bảy\b", r"\1 g7", t, flags=re.I)
    t = re.sub(
        r"\b(giấy|khổ|tờ|in)\s+a\s+(không|một|hai|ba|bốn|năm)\b",
        lambda m: f"{m.group(1)} a{_SPOKEN_DIGIT_MAP[m.group(2).lower()]}",
        t,
        flags=re.I,
    )
    t = re.sub(r"\bs\s+cộng\s+sport\b", "s+ sport", t, flags=re.I)
    t = re.sub(r"\b(?:gia|da|ra)\s+lô\b", "zalo", t, flags=re.I)
    t = re.sub(
        r"\b(gửi|đọc|check|kiểm\s+tra|xem|mở|nhận|trả\s+lời|viết|soạn|hộp\s+thư|nhắn)\s+meo\b",
        r"\1 mail",
        t,
        flags=re.I,
    )
    t = re.sub(r"\b(vàng|giá|lượng|chỉ)\s+(?:ét\s+di\s+xi|s\s+j\s+c)\b", r"\1 sjc", t, flags=re.I)
    return t


def hotwords_enabled(asr: Any) -> bool:
    """Return True only for ASR instances supporting per-stream BPE hotwords."""
    return bool(
        asr is not None
        and getattr(asr, "supports_hotwords", False)
        and getattr(asr, "hotwords_mode", "off") != "off"
    )


def transcribe_with_item(asr: Any, audio_file: Path, item: dict[str, Any], *, up_to_turn: int | None = None) -> str:
    """Transcribe one utterance, passing contextual hotwords only when supported by the engine."""
    if hotwords_enabled(asr):
        hw = extract_sample_hotwords(item, up_to_turn=up_to_turn, vocab_chars=getattr(asr, "vocab_chars", None))
        if hw:
            return asr.transcribe(audio_file, hotwords=hw)
    return asr.transcribe(audio_file)


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
    """Wrap an ASR engine to attach non-leaky turn hotwords during history resolution when enabled."""
    if not hotwords_enabled(asr):
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
