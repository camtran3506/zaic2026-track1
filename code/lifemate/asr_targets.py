"""Gipformer BPE target formatting, spoken-number normalization, and unverified-label checks."""
from __future__ import annotations

import difflib
from pathlib import Path
import re
import unicodedata

DEFAULT_TOKENS_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯ"
    "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊ"
    "ỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
)

_UNITS = ["KHÔNG", "MỘT", "HAI", "BA", "BỐN", "NĂM", "SÁU", "BẢY", "TÁM", "CHÍN"]
_TONE_OA = {"À": "ÒA", "Á": "ÓA", "Ả": "ỎA", "Ã": "ÕA", "Ạ": "ỌA"}
_TONE_OE = {"È": "ÒE", "É": "ÓE", "Ẻ": "ỎE", "Ẽ": "ÕE", "Ẹ": "ỌE"}
_TONE_UY = {"Ỳ": "ÙY", "Ý": "ÚY", "Ỷ": "ỦY", "Ỹ": "ŨY", "Ỵ": "ỤY"}

_NEEDS_SPOKEN_SYMBOL_RE = re.compile(r"[+%&@/°]")


def load_vocab_chars(tokens_path: Path) -> set[str]:
    """Extract allowed uppercase single characters from SentencePiece tokens.txt."""
    chars: set[str] = set()
    with Path(tokens_path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            sym = parts[0]
            if sym.startswith("<") or sym.startswith("#"):
                continue
            for ch in sym.replace("\u2581", ""):
                chars.add(unicodedata.normalize("NFC", ch).upper())
    return chars


def model_style(text: str) -> str:
    """Align old-style Vietnamese tone marks (HOÀ, KHOÁ, HUỶ, THUỶ) to Gipformer BPE style."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"O([ÀÁẢÃẠ])(?=\s|$)", lambda m: _TONE_OA[m.group(1)], text)
    text = re.sub(r"O([ÈÉẺẼẸ])(?=\s|$)", lambda m: _TONE_OE[m.group(1)], text)
    text = re.sub(r"(?<!Q)U([ỲÝỶỸỴ])(?=\s|$)", lambda m: _TONE_UY[m.group(1)], text)
    return text


def _plain(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).upper()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return text.split()


def needs_target_normalization(text: str, vocab_chars: set[str] | frozenset[str] | None = None) -> bool:
    """True when text still contains digits, spoken symbols, or characters absent from BPE vocab."""
    if any(ch.isdigit() for ch in text) or _NEEDS_SPOKEN_SYMBOL_RE.search(text):
        return True
    allowed = vocab_chars or DEFAULT_TOKENS_CHARS
    words = _plain(text)
    if not words:
        return True
    return any(ch not in allowed for word in words for ch in word)


def normalize_spoken_symbols(raw: str) -> str:
    """Normalize common spoken symbols (&, %, @, °, +, date/time/decimal delimiters) consistently."""
    text = unicodedata.normalize("NFC", raw).upper()
    text = re.sub(r"([A-ZĐ0-9]+)\+", r"\1__PLUS ", text)
    text = re.sub(r"\+", " CỘNG ", text)
    text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1H\2", text)
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})(?:/\d{2,4})?\b", r"\1 THÁNG \2", text)
    text = re.sub(r"\b(\d+),(\d)\b", r"\1 PHẨY \2", text)
    text = re.sub(r"\b(\d+)\s*%", r"\1 PHẦN TRĂM", text)
    text = text.replace("%", " PHẦN TRĂM ")
    text = text.replace("&", " VÀ ")
    text = text.replace("@", " A CÒNG ")
    text = text.replace("°C", " ĐỘ C ").replace("°", " ĐỘ ")
    return text


def format_human_target(text: str, hypothesis: str = "") -> str:
    """BPE-compatible formatting for human-edited audio_text.

    Human annotators accurately converted numbers to words. All other
    symbol normalizations (&, %, +, @, °, time/date delimiters), punctuation removal,
    uppercase, and model_style tone alignment are applied identically to history audio.
    """
    pre = normalize_spoken_symbols(text)
    if "__PLUS" in pre:
        tokens = pre.split()
        resolved: list[str] = []
        hyp_upper = hypothesis.upper()
        for tok in tokens:
            if tok.endswith("__PLUS"):
                stem = tok[:-6]
                plus_word = "PLUS" if "PLUS" in hyp_upper or "PLƠT" in hyp_upper else "CỘNG"
                resolved.append(f"{stem} {plus_word}".strip() if stem else plus_word)
            else:
                resolved.append(tok)
        pre = " ".join(resolved)
    words = _plain(pre)
    return model_style(" ".join(words))


def _spell_up(n: int) -> str:
    """Vietnamese spoken reading of 0 <= n < 10^9 in uppercase."""
    if n == 0:
        return "KHÔNG"
    out: list[str] = []
    for scale, word in ((10**6, "TRIỆU"), (1000, "NGHÌN"), (1, "")):
        k, n = divmod(n, scale)
        if not k:
            continue
        h, r = divmod(k, 100)
        t, u = divmod(r, 10)
        part: list[str] = []
        if h or out:
            part += [_UNITS[h], "TRĂM"]
        if t == 0 and u and (h or out):
            part.append("LINH")
        if t == 1:
            part.append("MƯỜI")
        elif t:
            part += [_UNITS[t], "MƯƠI"]
        if u:
            part.append(
                "MỐT" if u == 1 and t > 1
                else "LĂM" if u == 5 and t
                else "TƯ" if u == 4 and t > 1
                else _UNITS[u]
            )
        out += part + ([word] if word else [])
    return " ".join(out)


def _int_variants(n: int) -> set[str]:
    std = _spell_up(n)
    out = {std, std.replace("NGHÌN", "NGÀN"), std.replace("LINH", "LẺ")}
    if 21 <= n <= 99 and n % 10:
        t, u = divmod(n, 10)
        last = {1: "MỐT", 4: "TƯ", 5: "LĂM"}.get(u, _UNITS[u])
        out |= {f"{_UNITS[t]} {last}", f"{_UNITS[t]} {_UNITS[u]}"}
    if 100 <= n <= 999:
        h, r = divmod(n, 100)
        t, u = divmod(r, 10)
        if r and u == 0:
            out |= {
                f"{_UNITS[h]} TRĂM {_UNITS[t] if t > 1 else 'MƯỜI'}",
                f"TRĂM {_UNITS[t]}" if h == 1 and t > 1 else std,
            }
        if t > 1 and u:
            last = {1: "MỐT", 4: "TƯ", 5: "LĂM"}.get(u, _UNITS[u])
            out.add(f"{_UNITS[h]} TRĂM {_UNITS[t]} {last}")
    for scale, word in ((10**6, "TRIỆU"), (1000, "NGHÌN")):
        if n >= scale and n % scale and n % (scale // 10) == 0:
            a, b = divmod(n, scale)
            d = b // (scale // 10)
            for w in (word, "NGÀN" if word == "NGHÌN" else word):
                out.add(f"{_spell_up(a)} {w} {_UNITS[d]}")
                if d == 5:
                    out.add(f"{_spell_up(a)} {w} RƯỠI")
    return {" ".join(x.split()) for x in out}


def _digit_variants(d: str) -> list[str]:
    digits = " ".join(_UNITS[int(c)] for c in d)
    if (d.startswith("0") and len(d) > 1) or (len(d) >= 9 and int(d) % 1000):
        return [digits]
    std = _spell_up(int(d))
    pool = _int_variants(int(d))
    if len(d) >= 3:
        pool = pool | {digits}
    # Keep standard spelling first when no acoustic hypothesis is supplied.
    return [std, *sorted(pool - {std})]


def _token_variants(tok: str) -> list[str] | None:
    """Candidate uppercase spoken readings for one reference token carrying digits or markers."""
    t = tok.upper()
    if t.endswith("__PLUS"):
        stem = t[:-6]
        if not stem:
            return ["CỘNG", "PLUS"]
        base_opts = _token_variants(stem) if any(c.isdigit() for c in stem) else [stem]
        if not base_opts:
            return None
        ordered = [f"{b} CỘNG" for b in base_opts] + [f"{b} PLUS" for b in base_opts] + base_opts
        return list(dict.fromkeys(x.strip() for x in ordered if x.strip()))
    m = re.fullmatch(r"(\d+)", t)
    if m:
        return _digit_variants(m.group(1))
    m = re.fullmatch(r"(\d+)K", t)
    if m:
        base = _int_variants(int(m.group(1)))
        return sorted({f"{b} {w}" for b in base for w in ("NGHÌN", "NGÀN", "CA")})
    m = re.fullmatch(r"(\d+)H(\d*)", t)
    if m:
        h, mi = int(m.group(1)), m.group(2)
        hs = _int_variants(h)
        if not mi or int(mi) == 0:
            return sorted({f"{x} GIỜ" for x in hs} | {f"{x} GIỜ ĐÚNG" for x in hs})
        ms = _int_variants(int(mi))
        out = {f"{x} GIỜ {y}" for x in hs for y in ms} | {f"{x} GIỜ {y} PHÚT" for x in hs for y in ms}
        if int(mi) == 30:
            out |= {f"{x} GIỜ RƯỠI" for x in hs} | {f"{x} RƯỠI" for x in hs}
        return sorted(out)
    # Mixed alphanumeric tokens such as RON95, E5, SCTV9, RV107, 8A3, A4.
    parts = re.findall(r"\d+|[A-ZĐ]+", t)
    if parts and "".join(parts) == t and any(p.isdigit() for p in parts):
        combos: list[list[str]] = [[]]
        for part in parts:
            opts = _digit_variants(part) if part.isdigit() else [part]
            combos = [prefix + choice.split() for prefix in combos for choice in opts][:400]
        return sorted({" ".join(c) for c in combos})
    return None


def _wdist(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _preprocess_symbols_and_numbers(raw: str) -> list[str]:
    """Rejoin grouped numbers, times, dates, decimals and preserve + markers before splitting punctuation."""
    text = normalize_spoken_symbols(raw)
    # Temporarily guard __PLUS from punctuation splitting on '_', then restore marker cleanly.
    text = text.replace("__PLUS", "ZZPLUSMARKER")
    tokens = _plain(text)
    return [tok.replace("ZZPLUSMARKER", "__PLUS") for tok in tokens]


def target_v3(reference: str, hypothesis: str = "") -> str | None:
    """Resolve digit/symbol spans using Vietnamese readings closest to base hypothesis."""
    ref = _preprocess_symbols_and_numbers(reference)
    if not ref:
        return None
    if not any(any(c.isdigit() for c in w) or "__PLUS" in w for w in ref):
        return model_style(" ".join(ref))
    hyp = _plain(hypothesis or "")
    out: list[str] = []
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        seg = ref[i1:i2]
        if not any(any(c.isdigit() for c in w) or "__PLUS" in w for w in seg):
            out.extend(seg)
            continue
        options: list[list[str]] = [[]]
        for w in seg:
            if any(c.isdigit() for c in w) or "__PLUS" in w:
                v = _token_variants(w)
                if not v:
                    return None
                options = [o + x.split() for o in options for x in v][:400]
            else:
                options = [o + [w] for o in options]
        spoken = hyp[j1:j2] if (op == "replace" and hyp) else []
        best = min(options, key=lambda o: (_wdist(o, spoken), len(o))) if spoken else options[0]
        out.extend(best)
    text = model_style(" ".join(out))
    return text if text and not any(c.isdigit() for c in text) else None


def normalized_word_error_rate(reference_target: str, hypothesis: str) -> float:
    """Compute word-level edit error rate (with model_style normalization) between two utterances."""
    ref_words = format_human_target(reference_target).split()
    hyp_words = format_human_target(hypothesis).split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    return _wdist(ref_words, hyp_words) / len(ref_words)


def encode_bpe_ids(target: str, sp, vocab_size: int | None = None) -> list[int] | None:
    """Encode target with SentencePiece and reject empty, <unk>, or out-of-range IDs."""
    ids = sp.encode(target, out_type=int) if target else []
    if not ids:
        return None
    unk_id = sp.unk_id() if hasattr(sp, "unk_id") else -1
    if unk_id >= 0 and unk_id in ids:
        return None
    if any(t <= 0 or (vocab_size is not None and t >= vocab_size) for t in ids):
        return None
    return ids


def build_asr_target(
    text: str,
    *,
    hypothesis: str = "",
    is_history: bool = False,
    vocab_chars: set[str] | frozenset[str] | None = None,
    sp=None,
    vocab_size: int | None = None,
    max_history_wer: float | None = None,
) -> tuple[str | None, dict]:
    """Build BPE-ready training target selectively applying normalization/denoising.

    Human annotators accurately converted numbers to words in current-turn audio_text.
    All other transformations (symbol normalization, punctuation removal, uppercase,
    model_style tone alignment, and BPE checks) are applied identically across all utterances.
    Leftover turns still carrying digits run target_v3 digit expansion.
    Noisy-label rejection (max_history_wer) applies only to unverified history turns.
    """
    allowed = vocab_chars or DEFAULT_TOKENS_CHARS
    needs_norm = needs_target_normalization(text, allowed)
    if not is_history and not needs_norm:
        target = format_human_target(text, hypothesis=hypothesis)
        if not target or any(ch not in allowed and ch != " " for ch in target):
            return None, {"mode": "human_verified", "reason": "invalid_vocab_char"}
        if sp is not None and encode_bpe_ids(target, sp, vocab_size) is None:
            return None, {"mode": "human_verified", "reason": "bpe_unk"}
        return target, {
            "mode": "human_verified",
            "normalized_digits": False,
            "noise_filtered": False,
        }

    target = target_v3(text, hypothesis)
    if not target or any(ch not in allowed and ch != " " for ch in target):
        return None, {
            "mode": "history" if is_history else "leftover_main",
            "normalized_digits": True,
            "reason": "unresolved_target",
        }
    if sp is not None and encode_bpe_ids(target, sp, vocab_size) is None:
        return None, {
            "mode": "history" if is_history else "leftover_main",
            "normalized_digits": True,
            "reason": "bpe_unk",
        }
    if is_history and max_history_wer is not None and hypothesis:
        wer = normalized_word_error_rate(target, hypothesis)
        if wer > max_history_wer:
            return None, {
                "mode": "history",
                "normalized_digits": True,
                "noise_filtered": True,
                "reason": "high_history_wer",
                "wer": round(wer, 4),
            }
    return target, {
        "mode": "history" if is_history else "leftover_main",
        "normalized_digits": needs_norm,
        "noise_filtered": False,
    }
