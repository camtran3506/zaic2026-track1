"""One validated ToolEnv execution boundary shared by inference and SFT replay.

Automatic guards are Python middleware, not model-callable helper tools. This
module uses only the provided env's validate/call methods; it never creates an
instance, reads state, or edits/reconstructs the official call log.
"""
from datetime import date as _date, timedelta as _timedelta
import json
import re
import unicodedata

from .eval_trace import exception_details
from .execution_policy import check_tool_access, verify_result, _canonical
from .history import collect_ids


_MAP_ADMIN = frozenset({"quan", "phuong", "q", "p", "tp", "huyen", "thanh", "pho"})
_NON_FILTER_PARAMS = frozenset({"query", "servings", "limit", "top_k", "offset", "page", "passengers", "sort", "order"})
_TOOL_MATCH_MODES = {
    "maps.search_place": "token_or",
    "news.search": "token_or",
    "ecommerce.search": "token_half",
    "recipes.search": "token_half",
    "grocery.search": "token_half",
}
_SHOP_STOP = frozenset({
    "tim", "kiem", "cong", "thuc", "cach", "nau", "lam", "mon", "mua", "cho",
    "toi", "minh", "em", "anh", "chi", "cua", "va", "voi", "loai", "cai",
    "chiec", "cac", "nhung", "mot", "giup", "xem", "don", "hang", "san", "pham",
})
_MULTI_HONORIFICS = frozenset({
    "cô giáo", "thầy giáo", "bác sĩ", "giáo sư", "tiến sĩ", "kỹ sư", "luật sư",
})
_SINGLE_HONORIFICS = frozenset({
    "cô", "thầy", "bác", "chú", "dì", "cậu", "mợ", "anh", "chị", "em",
    "ông", "bà", "bạn", "bé",
})
_SPATIAL_LANDMARK_RE = re.compile(
    r"(?:đối\s+diện|bên\s+cạnh|kế\s+bên|gần|cạnh|sát|ngay|cách|ở\s+chỗ|quanh|khu\s+vực|khu|ở|tại|trên|dưới)\s+"
    r"(?P<phrase>(?:nhà\s+hàng|khách\s+sạn|cửa\s+hàng|siêu\s+thị|công\s+viên|bệnh\s+viện|"
    r"nhà\s+hát|chung\s+cư|tòa\s+nhà|bến\s+xe|trung\s+tâm|quán|tiệm|chợ|ga|bến|trường|"
    r"đường|phố|cầu|hồ|rạp|sân)\s+[^\W_]+(?:\s+[^\W_]+){0,4})",
    re.IGNORECASE | re.UNICODE,
)
_STANDALONE_LANDMARK_RE = re.compile(
    r"(?<!\w)(?P<phrase>(?:nhà\s+hàng|khách\s+sạn|công\s+viên|bệnh\s+viện|nhà\s+hát|"
    r"chung\s+cư|siêu\s+thị|đêm\s+nhạc|buổi\s+diễn|quán|tiệm|liveshow)"
    r"\s+[^\W_]+(?:\s+[^\W_]+){0,4})",
    re.IGNORECASE | re.UNICODE,
)
_NOUN_PREFIXES = (
    "nha hang", "khach san", "cua hang", "sieu thi", "cong vien", "benh vien",
    "nha hat", "chung cu", "toa nha", "ben xe", "trung tam", "dem nhac", "buoi dien",
    "quan", "tiem", "cho", "ga", "ben", "truong", "duong", "pho", "cau", "ho",
    "rap", "san", "liveshow",
)
_LANDMARK_BOUNDARY_WORDS = frozenset({
    "à", "ạ", "em", "anh", "chị", "mình", "nha", "nhé", "nghe", "nhen",
    "rồi", "đi", "với", "lúc", "tối", "sáng", "chiều", "trưa", "đêm", "hôm",
    "nay", "mai", "qua", "được", "thì", "coi", "xem", "tìm", "giùm", "dùm", "hộ",
    "cho", "là", "mà", "này", "đó", "kia", "ấy", "nào", "gì", "đâu", "ở",
    "tại", "của", "và", "nhưng", "vậy", "nên", "để", "còn", "đã", "sẽ", "đang",
    "ngon", "rẻ", "mới", "cũ", "gần", "đẹp", "nhất", "luôn", "xong", "hay", "hoặc",
})
_READ_NAME_SUFFIXES = (".search", ".search_place", ".list", ".get", ".get_quote", ".get_portfolio", ".find", ".read", ".check")


def _fold_search(text):
    """Fold text to lowercase ASCII without diacritics, matching zaic_tools._fold."""
    if text is None:
        return ""
    folded = unicodedata.normalize("NFD", str(text).lower()).replace("đ", "d")
    return " ".join("".join(c for c in folded if not unicodedata.combining(c)).split())


def _search_tokens(text):
    """Word token set after folding, matching zaic_tools._tokens."""
    return set(re.findall(r"[^\W_]+", _fold_search(text)))


def _shop_tokens(text):
    """Meaningful ordered token list matching zaic_tools.apps._shop_match."""
    words = _fold_search(text).split()
    return [w for w in words if w not in _SHOP_STOP] or words


def _parse_obs_result(obs):
    raw = obs.get("result")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _is_epoch_advancing_write(obs, parsed):
    if not isinstance(parsed, dict) or parsed.get("ok") is not True or parsed.get("done") is False:
        return False
    policy = (obs.get("automatic_execution_feedback") or {}).get("repeat_policy")
    if policy == "allowed":
        return True
    if policy in ("reuse_read_until_state_change", "blocked_until_state_change", "blocked_for_sample"):
        return False
    name = obs.get("name", "")
    if not isinstance(name, str) or name.startswith(("sys.", "local_")):
        return False
    return not any(name.endswith(suffix) for suffix in _READ_NAME_SUFFIXES)


def _collect_epoch_read_state(name, observations):
    """Summarize prior executed calls for `name` since the last state-changing write."""
    exact_sigs = set()
    empty_calls = []
    failed_quotes = {}
    for obs in observations or ():
        if not isinstance(obs, dict) or "result" not in obs or obs.get("scope") == "local":
            continue
        parsed = _parse_obs_result(obs)
        if _is_epoch_advancing_write(obs, parsed):
            exact_sigs.clear()
            empty_calls.clear()
            failed_quotes.clear()
            continue
        if obs.get("name") != name:
            continue
        prev_args = obs.get("arguments")
        if not isinstance(prev_args, dict):
            continue
        sig = _canonical(prev_args)
        if sig is not None:
            exact_sigs.add(sig)
        status = (obs.get("verification") or {}).get("status")
        if status is None and parsed is not None:
            status = verify_result(name, parsed).get("status")
        if status == "empty":
            empty_calls.append(prev_args)
        elif status == "failure" and "query" not in prev_args:
            for k, val in prev_args.items():
                if isinstance(val, str) and val.strip():
                    failed_quotes[(k, val.strip().upper())] = val
    return exact_sigs, empty_calls, failed_quotes


def _strip_honorific_query(query):
    """Strip leading Vietnamese honorifics/titles when a proper name follows."""
    if not isinstance(query, str):
        return ""
    words = query.strip().split()
    if len(words) < 2:
        return ""
    nfc_words = [unicodedata.normalize("NFC", re.sub(r"[^\w]", "", w, flags=re.UNICODE).casefold()) for w in words]
    if len(words) >= 3 and " ".join(nfc_words[:2]) in _MULTI_HONORIFICS:
        candidate = " ".join(words[2:]).strip()
        if candidate:
            return candidate
    if nfc_words[0] in _SINGLE_HONORIFICS:
        candidate = " ".join(words[1:]).strip()
        if candidate:
            return candidate
    return ""


def _trim_landmark_phrase(phrase):
    """Trim trailing discourse particles and return `(full_phrase, bare_name)` if valid."""
    raw_words = [w.strip(".,;:!?\"'()[]{}") for w in phrase.strip().split()]
    raw_words = [w for w in raw_words if w]
    if len(raw_words) < 2:
        return None, None
    folded_joined = _fold_search(" ".join(raw_words))
    prefix_len = 0
    for prefix in _NOUN_PREFIXES:
        if folded_joined == prefix or folded_joined.startswith(prefix + " "):
            prefix_len = len(prefix.split())
            break
    if prefix_len == 0 or len(raw_words) <= prefix_len:
        return None, None
    kept = list(raw_words[:prefix_len])
    for word in raw_words[prefix_len:]:
        nfc = unicodedata.normalize("NFC", word.casefold())
        if not nfc or nfc in _LANDMARK_BOUNDARY_WORDS:
            break
        kept.append(word)
    tail = kept[prefix_len:]
    if not tail or (len(" ".join(tail)) < 2):
        return None, None
    full = " ".join(kept)
    bare = " ".join(tail)
    return full, (bare if _fold_search(bare) != _fold_search(full) else None)


def _extract_landmark_queries(transcript):
    """Extract Vietnamese venue/landmark anchor phrases from `transcript`."""
    if not isinstance(transcript, str) or not transcript.strip():
        return []
    seen_folded = set()
    results = []

    def add_candidate(cand):
        if not cand:
            return
        folded = _fold_search(cand)
        if not folded or len(folded) < 2 or folded in seen_folded:
            return
        seen_folded.add(folded)
        results.append(cand)

    for pattern in (_SPATIAL_LANDMARK_RE, _STANDALONE_LANDMARK_RE):
        for match in pattern.finditer(transcript):
            full, bare = _trim_landmark_phrase(match.group("phrase"))
            add_candidate(full)
            add_candidate(bare)
    return results


def _active_filters(args):
    return {k: v for k, v in args.items() if k != "query" and v not in (None, "", [], {})}


def _filters_cover_or_equal(prev_args, curr_args, *, ignore_keys=()):
    prev_filters = _active_filters(prev_args)
    for key, value in prev_filters.items():
        if key in ignore_keys:
            continue
        if curr_args.get(key) != value:
            return False
    return True


def _search_match_mode(name, spec=None):
    """Infer search matching semantics from schema metadata (`x-match`/description) with safe defaults."""
    if isinstance(spec, dict):
        x_match = str(spec.get("x-match") or "").strip().lower()
        if x_match in {"token_or", "token_half", "substring"}:
            return x_match
        desc = _fold_search(spec.get("description") or "")
        if "chi can mot tu" in desc or "bat ky tu" in desc:
            return "token_or"
    return _TOOL_MATCH_MODES.get(name, "substring")


def _search_ignore_tokens(name, spec=None):
    props = ((spec or {}).get("parameters") or {}).get("properties") or {}
    if name.startswith("maps.") or "near" in props:
        return _MAP_ADMIN
    return frozenset()


def _dead_tokens_from_empty(name, empty_calls, curr_args, spec=None, mode=None):
    mode = mode or _search_match_mode(name, spec)
    admin_stop = _search_ignore_tokens(name, spec)
    dead = set()
    for prev_args in empty_calls:
        prev_q = prev_args.get("query")
        if not isinstance(prev_q, str) or _fold_search(prev_q) in ("", "*"):
            continue
        if mode == "token_or":
            if not _active_filters(prev_args) or _filters_cover_or_equal(prev_args, curr_args, ignore_keys=_NON_FILTER_PARAMS):
                dead |= (_search_tokens(prev_q) - {"*"} - admin_stop)
        elif mode == "token_half":
            if _filters_cover_or_equal(prev_args, curr_args, ignore_keys=_NON_FILTER_PARAMS):
                prev_toks = _shop_tokens(prev_q)
                if 1 <= len(prev_toks) <= 2:
                    dead.update(prev_toks)
    return dead


def _is_doomed_search_query(name, query, curr_args, empty_calls, spec=None):
    """Return True only when prior empty reads mathematically guarantee 0 hits."""
    if not isinstance(query, str) or _fold_search(query) in ("", "*"):
        return False
    if not empty_calls:
        return False
    mode = _search_match_mode(name, spec)
    admin_stop = _search_ignore_tokens(name, spec)
    if mode == "token_or":
        dead = _dead_tokens_from_empty(name, empty_calls, curr_args, spec=spec, mode=mode)
        curr_toks = _search_tokens(query) - {"*"} - admin_stop
        return bool(curr_toks and curr_toks <= dead)
    if mode == "token_half":
        dead = _dead_tokens_from_empty(name, empty_calls, curr_args, spec=spec, mode=mode)
        curr_toks = _shop_tokens(query)
        if not curr_toks:
            return False
        if all(tok in dead for tok in curr_toks):
            return True
        curr_folded = _fold_search(query)
        need = max(1, (len(curr_toks) + 1) // 2)
        live_count = sum(1 for tok in curr_toks if tok not in dead)
        for prev_args in empty_calls:
            prev_q = prev_args.get("query")
            prev_folded = _fold_search(prev_q) if isinstance(prev_q, str) else ""
            if (len(prev_folded) >= 2 and prev_folded in curr_folded
                    and _filters_cover_or_equal(prev_args, curr_args, ignore_keys=_NON_FILTER_PARAMS)
                    and live_count < need):
                return True
        return False
    # Substring matching tools (base.py / contacts / event_tickets / library / etc.)
    curr_folded = _fold_search(query)
    for prev_args in empty_calls:
        prev_q = prev_args.get("query")
        if not isinstance(prev_q, str):
            continue
        prev_folded = _fold_search(prev_q)
        if (len(prev_folded) >= 2 and prev_folded in curr_folded
                and _filters_cover_or_equal(prev_args, curr_args, ignore_keys=_NON_FILTER_PARAMS)):
            return True
    return False


def _context_fuzzy_candidates(name, query, context, history):
    if not isinstance(query, str) or not query.strip():
        return []
    from .local_knowledge import _similarity, knowledge_records
    records = knowledge_records(context, history)
    app = name.split(".", 1)[0] if isinstance(name, str) and "." in name else ""
    scored = []
    seen = set()
    for entry in records:
        if app and not entry.get("collection", "").startswith(app + "."):
            continue
        row = entry.get("record") or {}
        label = next((row[k] for k in ("name", "title", "label", "symbol") if isinstance(row.get(k), str) and row[k].strip()), None)
        if not label:
            continue
        score, _ = _similarity(query, label)
        if score >= 0.25 and label not in seen:
            seen.add(label)
            scored.append((score, label, row.get("id")))
    scored.sort(key=lambda item: -item[0])
    out = []
    for _, label, rec_id in scored[:3]:
        out.append({"id": rec_id, "name": label} if isinstance(rec_id, str) else {"name": label})
    return out


_COMPANION_LIST_RANK_TOKENS = ("list", "portfolio", "all", "catalog", "restaurants", "items", "tracks", "habits", "vouchers", "bills", "events", "alerts", "locks")


def _companion_list_options(name, enabled, observations=()):
    """Discover untried parameterless read tools in the same app from `enabled`."""
    if not isinstance(enabled, dict) or not isinstance(name, str) or "." not in name:
        return []
    app = name.split(".", 1)[0]
    scored = []
    for cand_name, cand_spec in sorted(enabled.items()):
        if cand_name == name or not cand_name.startswith(app + ".") or not isinstance(cand_spec, dict):
            continue
        if cand_spec.get("kind") != "read":
            continue
        cand_required = (cand_spec.get("parameters") or {}).get("required") or []
        if cand_required:
            continue
        cand_exact, _, _ = _collect_epoch_read_state(cand_name, observations)
        empty_sig = _canonical({})
        if empty_sig is not None and empty_sig in cand_exact:
            continue
        method = cand_name.split(".", 1)[1].lower()
        rank = next((idx for idx, tok in enumerate(_COMPANION_LIST_RANK_TOKENS) if tok in method), len(_COMPANION_LIST_RANK_TOKENS))
        if any(dep in method for dep in ("history", "recent", "booking")):
            rank += 10
        scored.append((rank, cand_name))
    scored.sort()
    return [{"name": cand_name, "arguments": {}} for _, cand_name in scored]


def _list_fallback_options(name, spec, base_args, exact_sigs, enabled, *, observations=()):
    """Build untried list-all / companion tool call options from tool schemas."""
    params = (spec or {}).get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    kept_required = {k: base_args[k] for k in required if k != "query" and k in base_args}
    candidates = []
    if spec is not None and "query" in props:
        if "query" not in required and _canonical(kept_required) not in exact_sigs:
            candidates.append({"name": name, "arguments": kept_required})
        wildcard_args = {**kept_required, "query": "*"}
        if _canonical(wildcard_args) not in exact_sigs:
            candidates.append({"name": name, "arguments": wildcard_args})
    elif spec is None:
        wildcard_args = {**kept_required, "query": "*"}
        if _canonical(wildcard_args) not in exact_sigs:
            candidates.append({"name": name, "arguments": wildcard_args})
    candidates.extend(_companion_list_options(name, enabled, observations))
    return candidates


def _untried_query_alternatives(
    name,
    query,
    template_args,
    transcript,
    empty_calls,
    exact_sigs,
    *,
    spec=None,
    honorific_first=True,
    limit=3,
):
    """Return untried, non-doomed `(query_text, args_dict)` pairs from honorifics/landmarks."""
    stripped = _strip_honorific_query(query)
    landmarks = _extract_landmark_queries(transcript)
    raw = ([stripped, *landmarks] if honorific_first else [*landmarks, stripped])
    base = {k: v for k, v in template_args.items() if k != "query"}
    seen_folded = {_fold_search(query)}
    selected = []
    for cand in raw:
        folded = _fold_search(cand)
        if not cand or not folded or folded in seen_folded:
            continue
        seen_folded.add(folded)
        cand_args = {**base, "query": cand}
        cand_sig = _canonical(cand_args)
        if (cand_sig is not None and cand_sig in exact_sigs) or _is_doomed_search_query(name, cand, cand_args, empty_calls, spec=spec):
            continue
        selected.append((cand, cand_args))
        if len(selected) >= limit:
            break
    return selected


_TRAVEL_SEARCH_TOOLS = frozenset({"flights.search", "train_bus_tickets.search"})
_FLIGHT_AIRPORTS = {
    "SGN": ("Hồ Chí Minh", "Sài Gòn", "Tân Sơn Nhất", "TP HCM", "TP.HCM", "TPHCM", "HCM", "HCMC"),
    "HAN": ("Hà Nội", "Nội Bài"),
    "DAD": ("Đà Nẵng",),
    "HUI": ("Huế", "Phú Bài"),
    "VCA": ("Cần Thơ",),
    "DLI": ("Đà Lạt", "Liên Khương"),
    "UIH": ("Quy Nhơn", "Phù Cát"),
    "PQC": ("Phú Quốc",),
    "CXR": ("Nha Trang", "Cam Ranh"),
    "VDH": ("Đồng Hới", "Quảng Bình"),
    "HPH": ("Hải Phòng", "Cát Bi"),
    "VII": ("Vinh",),
    "THD": ("Thanh Hoá", "Thanh Hóa", "Thọ Xuân"),
    "VCL": ("Chu Lai", "Quảng Nam"),
    "TBB": ("Tuy Hoà", "Tuy Hòa", "Phú Yên"),
    "PXU": ("Pleiku", "Gia Lai"),
    "BMV": ("Buôn Ma Thuột", "Đắk Lắk"),
    "VCS": ("Côn Đảo",),
    "VKG": ("Rạch Giá",),
    "CAH": ("Cà Mau",),
    "DIN": ("Điện Biên",),
    "VDO": ("Vân Đồn", "Quảng Ninh"),
    # International hubs observed across splits or common regional routes
    "TPE": ("Đài Bắc", "Taipei", "Taoyuan"),
    "KHH": ("Cao Hùng", "Kaohsiung"),
    "SIN": ("Singapore", "Changi"),
    "KUL": ("Kuala Lumpur",),
    "BKK": ("Bangkok", "Băng Cốc", "Suvarnabhumi"),
    "DMK": ("Don Mueang",),
    "ICN": ("Seoul", "Incheon"),
    "GMP": ("Gimpo",),
    "NRT": ("Tokyo", "Narita"),
    "HND": ("Haneda",),
    "KIX": ("Osaka", "Kansai"),
    "HKG": ("Hong Kong", "Hồng Kông"),
    "PNH": ("Phnom Penh", "Phnôm Pênh"),
    "VTE": ("Vientiane", "Viêng Chăn"),
    "MNL": ("Manila",),
    "CGK": ("Jakarta",),
    "DPS": ("Bali", "Denpasar"),
    "SYD": ("Sydney",),
    "MEL": ("Melbourne",),
    "CDG": ("Paris", "Charles de Gaulle"),
    "LHR": ("London", "Luân Đôn", "Heathrow"),
    "FRA": ("Frankfurt",),
}
_MULTI_AIRPORT_EQUIV = {
    "BKK": frozenset({"BKK", "DMK"}),
    "DMK": frozenset({"BKK", "DMK"}),
    "ICN": frozenset({"ICN", "GMP"}),
    "GMP": frozenset({"ICN", "GMP"}),
    "NRT": frozenset({"NRT", "HND"}),
    "HND": frozenset({"NRT", "HND"}),
}
_AIRPORT_GENERIC = frozenset({"thanh", "pho", "tp", "san", "bay", "tinh", "thanhpho", "sanbay", "ga", "city", "airport"})
_AIRPORT_WORD_ALIASES = {}
for _iata_code, _names in _FLIGHT_AIRPORTS.items():
    _variants = set()
    for _raw in _names:
        _w = tuple(re.findall(r"[^\W_]+", _fold_search(_raw)))
        if _w:
            _variants.add(_w)
            if len(_w) > 1:
                _variants.add(("".join(_w),))
    _AIRPORT_WORD_ALIASES[_iata_code] = tuple(sorted(_variants, key=lambda x: (-len(x), x)))

_AIRPORT_FLAT_PHRASES = sorted(
    [(w, code) for code, variants in _AIRPORT_WORD_ALIASES.items() for w in variants],
    key=lambda item: (-len(item[0]), -len("".join(item[0])), item[1]),
)

_DEST_CUE_RE = re.compile(
    r"\b(?:đi|tới|đến|về|bay\s+(?:đi|tới|đến|về|ra|vào)|chuyến\s+(?:bay\s+)?[a-z0-9]{2,8}\s+(?:đi|tới|đến|về))"
    r"\s+(?:(?:thành\s+phố|tp\.?|sân\s+bay|tỉnh)\s+)?(?P<place>[^\W\d_]+(?:\s+[^\W\d_]+){0,3})",
    re.UNICODE,
)
_ORIGIN_CUE_RE = re.compile(
    r"\b(?:từ|xuất\s+phát(?:\s+(?:từ|tại|ở))?|khởi\s+hành(?:\s+(?:từ|tại|ở))?)"
    r"\s+(?:(?:thành\s+phố|tp\.?|sân\s+bay|tỉnh)\s+)?(?P<place>[^\W\d_]+(?:\s+[^\W\d_]+){0,3})",
    re.UNICODE,
)
_VI_SPOKEN_ONES = {
    "khong": 0, "mot": 1, "mots": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4,
    "nam": 5, "lam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
}


def _nfc_and_folded_words(text):
    raw_words = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).lower())
    return tuple(raw_words), tuple(_fold_search(w) for w in raw_words)


def _is_valid_alias_hit(alias, raw_words, folded_words, pos):
    if list(folded_words[pos:pos + len(alias)]) != list(alias):
        return False
    # Check diacritics before tone folding so "Vĩnh/Vịnh ..." never collides with Vinh (VII).
    if alias == ("vinh",):
        if raw_words[pos] != "vinh":
            return False
        if pos + 1 < len(folded_words) and folded_words[pos + 1] in {"du", "danh", "hanh"}:
            return False
    return True


def _resolve_airport_iata(value):
    """Map city/airport name or code to canonical 3-letter uppercase IATA if known."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    upper = cleaned.upper()
    if cleaned.isascii() and len(upper) == 3 and upper in _FLIGHT_AIRPORTS:
        return upper
    raw_words, q_words = _nfc_and_folded_words(cleaned)
    if not q_words:
        return None
    for alias_words, code in _AIRPORT_FLAT_PHRASES:
        n_len = len(alias_words)
        if len(q_words) < n_len:
            continue
        for idx in range(len(q_words) - n_len + 1):
            if _is_valid_alias_hit(alias_words, raw_words, q_words, idx):
                leftover = q_words[:idx] + q_words[idx + n_len:]
                if set(leftover) <= _AIRPORT_GENERIC:
                    return code
    return None


def _transcript_mentions_airport(transcript, iata):
    if not isinstance(transcript, str) or not isinstance(iata, str):
        return False
    equiv_codes = _MULTI_AIRPORT_EQUIV.get(iata.upper(), (iata.upper(),))
    raw_words, t_words = _nfc_and_folded_words(transcript)
    for code in equiv_codes:
        aliases = _AIRPORT_WORD_ALIASES.get(code)
        if not aliases:
            continue
        for i in range(len(t_words)):
            for alias in aliases:
                if _is_valid_alias_hit(alias, raw_words, t_words, i):
                    return True
    return False


def _match_leading_airport_iata(phrase):
    raw_words, words = _nfc_and_folded_words(phrase)
    if not words:
        return None
    for alias_words, code in _AIRPORT_FLAT_PHRASES:
        n_len = len(alias_words)
        if len(words) >= n_len and _is_valid_alias_hit(alias_words, raw_words, words, 0):
            return code
    return None


def _extract_flight_cues(transcript):
    if not isinstance(transcript, str) or not transcript.strip():
        return set(), set()
    nfc_lower = unicodedata.normalize("NFC", transcript).lower()
    dest_cues = {c for m in _DEST_CUE_RE.finditer(nfc_lower) if (c := _match_leading_airport_iata(m.group("place")))}
    orig_cues = {c for m in _ORIGIN_CUE_RE.finditer(nfc_lower) if (c := _match_leading_airport_iata(m.group("place")))}
    return orig_cues, dest_cues


def _normalize_flight_search_args(arguments, transcript, observations):
    """Normalize Vietnamese city/airport strings to IATA codes and fix contradictory destination/origin cues."""
    cleaned = dict(arguments)
    for field in ("origin", "destination"):
        val = cleaned.get(field)
        if isinstance(val, str):
            code = _resolve_airport_iata(val)
            if code:
                cleaned[field] = code
    if not isinstance(transcript, str) or not transcript.strip():
        return cleaned
    exact_sigs, _, _ = _collect_epoch_read_state("flights.search", observations)
    orig_cues, dest_cues = _extract_flight_cues(transcript)
    for field, cues, other_field in (("destination", dest_cues, "origin"), ("origin", orig_cues, "destination")):
        if len(cues) != 1:
            continue
        cue_iata = next(iter(cues))
        curr_val = cleaned.get(field)
        equiv = _MULTI_AIRPORT_EQUIV.get(cue_iata, frozenset({cue_iata}))
        if (isinstance(curr_val, str) and curr_val.upper() not in equiv
                and cue_iata != cleaned.get(other_field)
                and not _transcript_mentions_airport(transcript, curr_val)):
            candidate = {**cleaned, field: cue_iata}
            cand_sig = _canonical(candidate)
            if cand_sig is None or cand_sig not in exact_sigs:
                cleaned = candidate
    return cleaned


def _parse_vi_small_number(words):
    """Parse spoken Vietnamese integer 1..31 from folded words."""
    if not words:
        return None
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])
    if words[0] in ("mung", "mong") and len(words) >= 2:
        words = words[1:]
    if len(words) == 1:
        return _VI_SPOKEN_ONES.get(words[0])
    if words == ["muoi", "lam"]:
        return 15
    if words[0] == "muoi":
        unit = _VI_SPOKEN_ONES.get(words[1]) if len(words) == 2 else None
        return (10 + unit) if unit is not None and 1 <= unit <= 9 else None
    if len(words) >= 2 and words[1] in ("muoi",):
        tens = _VI_SPOKEN_ONES.get(words[0])
        if tens is None or tens not in (2, 3):
            return None
        if len(words) == 2:
            return tens * 10
        unit = _VI_SPOKEN_ONES.get(words[2]) if len(words) == 3 else None
        return (tens * 10 + unit) if unit is not None and 1 <= unit <= 9 else None
    if len(words) == 2 and words[0] in ("hai", "ba"):
        tens = _VI_SPOKEN_ONES[words[0]]
        unit = _VI_SPOKEN_ONES.get(words[1])
        if unit is not None and 1 <= unit <= 9:
            return tens * 10 + unit
    return None


def _extract_transcript_dates(transcript, searched_date, context=None):
    """Extract explicit calendar dates mentioned in transcript for travel date mismatch checks."""
    out = []
    if not isinstance(transcript, str) or not transcript.strip():
        return out
    try:
        from .local_assist import solve, source_dates
        for d_text, _, _ in source_dates(transcript):
            res = solve({"task": "date", "inputs": {"date_text": d_text}}, context or {})
            val = (res.get("result") or {}).get("date") if res.get("status") == "resolved" else None
            if isinstance(val, str) and val not in out:
                out.append(val)
    except Exception:
        pass
    folded = _fold_search(transcript)
    curr_date = None
    if isinstance(context, dict) and isinstance(context.get("current_time"), str):
        try:
            curr_date = _date.fromisoformat(context["current_time"][:10])
        except ValueError:
            curr_date = None
    ref_year = searched_date.year if searched_date is not None else (curr_date.year if curr_date is not None else None)
    if ref_year is not None:
        pat = re.compile(r"\b(?:(?:ngay|hom)\s+)?((?:mung|mong)\s+\w+|(?:\d{1,2}|[a-z]+(?:\s+[a-z]+){0,2}))\s+thang\s+(\d{1,2}|[a-z]+(?:\s+[a-z]+){0,1})\b")
        for m in pat.finditer(folded):
            day_words = m.group(1).split()
            day = None
            for start in range(len(day_words)):
                day = _parse_vi_small_number(day_words[start:])
                if day is not None:
                    break
            month_words = m.group(2).split()
            month = 1 if month_words == ["gieng"] else _parse_vi_small_number(month_words)
            if day and month and 1 <= day <= 31 and 1 <= month <= 12:
                try:
                    parsed_date = _date(ref_year, month, day)
                except ValueError:
                    continue
                if curr_date is not None and ref_year == curr_date.year and (curr_date - parsed_date).days > 180:
                    try:
                        parsed_date = _date(ref_year + 1, month, day)
                    except ValueError:
                        pass
                iso = parsed_date.isoformat()
                if iso not in out:
                    out.append(iso)
    return out


_VI_WEEKDAYS = {
    "thu hai": 0, "thu ba": 1, "thu tu": 2, "thu nam": 3,
    "thu sau": 4, "thu bay": 5, "chu nhat": 6,
}
_NON_RELATIVE_TOI_PREFIX = re.compile(r"\btu(?:\s+ngay)?\s*$")
_NON_RELATIVE_TOI_NEXT = re.compile(
    r"^(?:thu\s+(?:hai|ba|tu|nam|sau|bay)|chu\s+nhat|cuoi\s+tuan|het|ngay|thang|nam|nay|do|noi|"
    r"san\s+bay|ga|ben|thanh\s+pho|tp|tinh|nha|cong\s+ty|vung\s+tau|ha\s+long|ninh\s+binh|"
    r"phan\s+thiet|hoi\s+an|sa\s+pa|sapa|mui\s+ne)\b"
)


def _relative_weekday_dates(folded, curr_date):
    """Resolve explicit '<weekday> tuần sau/tới' and '<weekday> tới' against `curr_date`."""
    if curr_date is None or not folded:
        return []
    out = []
    next_wk_pat = re.compile(r"\b(thu\s+(?:hai|ba|tu|nam|sau|bay)|chu\s+nhat)\s+(?:tuan\s+(?:sau|toi))\b")
    for m in next_wk_pat.finditer(folded):
        wd = _VI_WEEKDAYS.get(" ".join(m.group(1).split()))
        if wd is not None:
            next_monday = curr_date + _timedelta(days=7 - curr_date.weekday())
            out.append(((next_monday + _timedelta(days=wd)).isoformat(), "đúng thứ thuộc tuần sau"))
    upcoming_pat = re.compile(r"\b(thu\s+(?:hai|ba|tu|nam|sau|bay)|chu\s+nhat)\s+toi\b")
    for m in upcoming_pat.finditer(folded):
        # Skip prepositional "tới" in ranges ("từ thứ hai tới thứ tư") or routes ("thứ năm tới Đà Nẵng").
        if _NON_RELATIVE_TOI_PREFIX.search(folded[:m.start()]):
            continue
        after = folded[m.end():].lstrip()
        if _NON_RELATIVE_TOI_NEXT.match(after) or _match_leading_airport_iata(after) is not None:
            continue
        wd = _VI_WEEKDAYS.get(" ".join(m.group(1).split()))
        if wd is not None:
            ahead = (wd - curr_date.weekday()) % 7 or 7
            out.append(((curr_date + _timedelta(days=ahead)).isoformat(), "đúng ngày thứ tới theo lịch"))
    return out


def _build_travel_search_fallback(name, arguments, empty_calls, exact_sigs, *, transcript="", context=None):
    """Suggest exact transcript dates, relative weekdays, or +7d shifts instead of blind +1d scans."""
    date_str = arguments.get("date")
    try:
        searched_date = _date.fromisoformat(date_str) if isinstance(date_str, str) else None
    except ValueError:
        searched_date = None
    curr_date = None
    if isinstance(context, dict) and isinstance(context.get("current_time"), str):
        try:
            curr_date = _date.fromisoformat(context["current_time"][:10])
        except ValueError:
            curr_date = None

    candidates = []
    notes = []

    def add_date(iso_str, reason):
        if not isinstance(iso_str, str) or iso_str == date_str or iso_str in candidates:
            return
        cand_sig = _canonical({**arguments, "date": iso_str})
        if cand_sig is None or cand_sig not in exact_sigs:
            candidates.append(iso_str)
            notes.append((iso_str, reason))

    for explicit_iso in _extract_transcript_dates(transcript, searched_date, context):
        if searched_date is None or explicit_iso != date_str:
            add_date(explicit_iso, "ngày nêu trực tiếp trong yêu cầu")

    folded = _fold_search(transcript)
    for rel_iso, rel_reason in _relative_weekday_dates(folded, curr_date):
        add_date(rel_iso, rel_reason)

    has_next_week = bool(re.search(r"\b(?:tuan sau|tuan toi)\b", folded))
    if searched_date is not None and has_next_week:
        anchor_dates = []
        for prev in empty_calls:
            if isinstance(prev.get("date"), str):
                try:
                    anchor_dates.append(_date.fromisoformat(prev["date"]))
                except ValueError:
                    pass
        anchor_dates.append(searched_date)
        for d in anchor_dates:
            if curr_date is None or (d - curr_date).days < 7:
                add_date((d + _timedelta(days=7)).isoformat(), "dịch +7 ngày cho cụm 'tuần sau/tuần tới'")

    if not candidates and not empty_calls:
        return None
    parts = [f"Không tìm thấy chuyến phù hợp với ngày {date_str or '(chưa rõ)'}."]
    if candidates:
        primary_iso, primary_reason = notes[0]
        parts.append(f"Thử trực tiếp date='{primary_iso}' ({primary_reason}); không dò tăng từng ngày.")
    else:
        parts.append("Kiểm tra lại điểm đi/đến và ngày chính xác trong yêu cầu; không lặp dò từng ngày.")
    return {
        "failed_search": {k: arguments.get(k) for k in ("origin", "destination", "date") if k in arguments},
        "suggested_dates": candidates,
        "instruction": " ".join(parts),
    }


def _collect_epoch_read_details(name, observations):
    """Collect current-epoch prior calls with their returned item IDs for stagnation detection."""
    calls = []
    for obs in observations or ():
        if not isinstance(obs, dict) or "result" not in obs or obs.get("scope") == "local":
            continue
        parsed = _parse_obs_result(obs)
        if _is_epoch_advancing_write(obs, parsed):
            calls.clear()
            continue
        if obs.get("name") != name or not isinstance(obs.get("arguments"), dict):
            continue
        items = []
        if isinstance(parsed, dict) and parsed.get("ok") is True:
            for val in parsed.values():
                if isinstance(val, list) and all(isinstance(row, dict) for row in val):
                    items = val
                    break
        ids = tuple(str(row["id"]) for row in items if row.get("id") is not None)
        calls.append((obs["arguments"], items, ids))
    return calls


def _norm_travel_place(value):
    if not isinstance(value, str):
        return ""
    iata = _resolve_airport_iata(value)
    return iata if iata else _fold_search(value)


def _check_travel_route_date_loop(name, arguments, observations):
    if name not in _TRAVEL_SEARCH_TOOLS or not isinstance(arguments, dict) or not observations:
        return None
    _, empty_calls, _ = _collect_epoch_read_state(name, observations)
    if len(empty_calls) < 3:
        return None
    route = (_norm_travel_place(arguments.get("origin")), _norm_travel_place(arguments.get("destination")))
    if not any(route):
        return None
    same_route_empty = sum(
        1 for prev in empty_calls
        if (_norm_travel_place(prev.get("origin")), _norm_travel_place(prev.get("destination"))) == route
    )
    if same_route_empty >= 3:
        instr = (
            f"Đã tìm {name} cho tuyến này {same_route_empty} lần trong cùng trạng thái nhưng đều rỗng; "
            "không tiếp tục dò từng ngày. Hãy rà soát lại ngày/mã sân bay theo yêu cầu hoặc hoàn tất phần còn lại."
        )
        return {
            "stage": "loop_guard",
            "name": name,
            "arguments": arguments,
            "error": instr,
            "feedback": {"reason": "route_date_scan_limit", "empty_route_attempts": same_route_empty, "instruction": instr},
        }
    return None


def _check_stagnant_search_loop(name, arguments, spec, context, observations):
    """Stop query subset loops that keep returning identical non-matching result IDs."""
    query = arguments.get("query") if isinstance(arguments, dict) else None
    if not isinstance(query, str) or _fold_search(query) in ("", "*") or not observations:
        return None
    epoch_calls = _collect_epoch_read_details(name, observations)
    if len(epoch_calls) < 2:
        return None
    curr_filters = _active_filters(arguments)
    curr_toks = _search_tokens(query) - {"*"} - (_MAP_ADMIN if name == "maps.search_place" else frozenset())
    if not curr_toks:
        return None
    by_ids = {}
    for prev_args, items, ids in epoch_calls:
        prev_q = prev_args.get("query")
        if not ids or not isinstance(prev_q, str) or _fold_search(prev_q) in ("", "*"):
            continue
        if _active_filters(prev_args) != curr_filters:
            continue
        entry = by_ids.setdefault(ids, {"tokens": set(), "items": items, "count": 0})
        entry["count"] += 1
        entry["tokens"] |= (_search_tokens(prev_q) - {"*"} - (_MAP_ADMIN if name == "maps.search_place" else frozenset()))
    props = (spec.get("parameters") or {}).get("properties") or {} if isinstance(spec, dict) else {}
    is_place_search = (name == "maps.search_place") or ("near" in props)
    for ids, info in by_ids.items():
        if info["count"] < 2 or not (curr_toks <= info["tokens"]):
            continue
        exact_item_match = any(
            curr_toks <= (
                _search_tokens(" ".join(str(item.get(k, "")) for k in ("name", "title", "address", "category", "detail")))
            )
            for item in info["items"]
        )
        if exact_item_match and info["count"] < 3:
            continue
        profile_hint = ""
        if is_place_search and isinstance(context, dict):
            profile = context.get("user_profile")
            if isinstance(profile, dict):
                hits = []
                for key in ("work_address", "home_address"):
                    val = profile.get(key)
                    if isinstance(val, str) and (_search_tokens(val) - _MAP_ADMIN) & info["tokens"]:
                        hits.append(f"{key}={val!r}")
                if hits:
                    profile_hint = " Kiểm tra địa chỉ có sẵn trong user_context.user_profile (" + ", ".join(hits) + ") thay vì tìm lặp lại."
        instr = (
            f"Tìm kiếm {name} đã trả về lặp lại cùng tập ID {list(ids)} qua {info['count']} truy vấn trước; "
            f"không thử lại biến thể con của truy vấn đã bão hòa.{profile_hint}"
        )
        return {
            "stage": "loop_guard",
            "name": name,
            "arguments": arguments,
            "error": instr,
            "feedback": {"reason": "repeated_search_results", "repeated_ids": list(ids), "instruction": instr},
        }
    return None


def _has_empty_epoch_read(ref_app, observations):
    """Return True if a current-epoch read call on `ref_app` returned an empty result."""
    saw_empty = False
    for obs in observations or ():
        if not isinstance(obs, dict) or "result" not in obs or obs.get("scope") == "local":
            continue
        parsed = _parse_obs_result(obs)
        if _is_epoch_advancing_write(obs, parsed):
            saw_empty = False
            continue
        obs_name = obs.get("name", "")
        if isinstance(obs_name, str) and obs_name.startswith(ref_app + "."):
            status = (obs.get("verification") or {}).get("status")
            if status is None and parsed is not None:
                status = verify_result(obs_name, parsed).get("status")
            if status == "empty":
                saw_empty = True
    return saw_empty


def _find_xref_mismatch(value, schema, observed_ids, key_name=""):
    if not isinstance(schema, dict):
        return None
    ref = schema.get("x-ref")
    if ref and value is not None:
        ref_app = str(ref).split(".", 1)[0]
        allowed = observed_ids.get(ref_app, set())
        if not isinstance(value, str) or value not in allowed:
            return ref, ref_app, key_name or "id"
    if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        props = schema["properties"]
        for key, child in value.items():
            if key in props:
                found = _find_xref_mismatch(child, props[key], observed_ids, key)
                if found is not None:
                    return found
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for child in value:
            found = _find_xref_mismatch(child, schema["items"], observed_ids, key_name)
            if found is not None:
                return found
    return None


def _extract_lookup_arg(arguments, spec):
    """Return `(key, value)` for a non-query key-lookup read call if present."""
    params = (spec or {}).get("parameters") or {}
    props = params.get("properties") or {}
    required = params.get("required") or []
    for key in (*required, *props.keys(), *arguments.keys()):
        if key == "query":
            continue
        prop_spec = props.get(key)
        if isinstance(prop_spec, dict) and prop_spec.get("x-ref"):
            continue
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            return key, val
    return None, None


def _is_key_lookup_tool(name, spec):
    if name in _TRAVEL_SEARCH_TOOLS:
        return False
    params = (spec or {}).get("parameters") or {}
    props = params.get("properties") or {}
    if spec is None:
        return False
    if "query" in props:
        return False
    required = list(params.get("required") or [])
    if len(required) != 1:
        return False
    req_key = required[0]
    req_prop = props.get(req_key)
    return isinstance(req_prop, dict) and req_prop.get("type") == "string" and not req_prop.get("x-ref")


def build_search_fallback(
    name,
    arguments,
    spec,
    *,
    transcript="",
    context=None,
    history=(),
    enabled=None,
    observations=(),
):
    """Build non-replayable prompt fallback suggestions after an empty read or failed lookup."""
    if not isinstance(arguments, dict):
        return None
    exact_sigs, empty_calls, _ = _collect_epoch_read_state(name, observations)
    current_sig = _canonical(arguments)
    if current_sig is not None:
        exact_sigs.add(current_sig)
    prior_and_current_empty = list(empty_calls)

    params = (spec or {}).get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])

    if name in _TRAVEL_SEARCH_TOOLS:
        return _build_travel_search_fallback(
            name, arguments, empty_calls, exact_sigs, transcript=transcript, context=context
        )

    if _is_key_lookup_tool(name, spec) or (spec is None and "query" not in arguments and isinstance(arguments.get("symbol"), str)):
        lookup_key, lookup_val = _extract_lookup_arg(arguments, spec)
        if not lookup_key:
            return None
        fallback_calls = _list_fallback_options(name, spec, arguments, exact_sigs, enabled, observations=observations)
        fallback_call = fallback_calls[0] if fallback_calls else None
        context_candidates = _context_fuzzy_candidates(name, lookup_val, context, history)
        if not fallback_call and not context_candidates:
            return None
        if fallback_call:
            args_repr = json.dumps(fallback_call["arguments"], ensure_ascii=False, separators=(",", ":"))
            instr = (
                f"Không tìm thấy {lookup_key} '{lookup_val}'. Không lặp lại giá trị cũ khác chữ hoa/thường; "
                f"hãy gọi {fallback_call['name']}({args_repr}) để xem danh sách hiện có."
            )
        else:
            instr = f"Không tìm thấy {lookup_key} '{lookup_val}'. Kiểm tra lại danh sách trong context/history."
        fallback = {
            "failed_value": lookup_val,
            "alternative_queries": [],
            "fallback_call": fallback_call,
            "instruction": instr,
        }
        if lookup_key == "symbol":
            fallback["failed_symbol"] = lookup_val
        if context_candidates:
            fallback["context_candidates"] = context_candidates
        return fallback

    query = arguments.get("query")
    if not isinstance(query, str) or _fold_search(query) in ("", "*"):
        return None
    prior_and_current_empty.append(arguments)
    kept_query_base = {
        k: v for k, v in arguments.items()
        if k != "query" and (k in required or k in _NON_FILTER_PARAMS)
    }
    optional_filter_keys = [
        k for k, v in arguments.items()
        if k != "query" and k not in required and k not in _NON_FILTER_PARAMS and v not in (None, "", [], {})
    ]
    relaxed_args = {**kept_query_base, "query": query}
    relaxed_sig = _canonical(relaxed_args)
    relaxed_call = (
        {"name": name, "arguments": relaxed_args}
        if optional_filter_keys and (relaxed_sig is None or relaxed_sig not in exact_sigs)
        else None
    )

    alternative_queries = [
        cand for cand, _ in _untried_query_alternatives(
            name, query, kept_query_base, transcript, prior_and_current_empty, exact_sigs,
            spec=spec, honorific_first=True,
        )
    ]
    fallback_calls = _list_fallback_options(
        name, spec, kept_query_base, exact_sigs, enabled, observations=observations
    )
    fallback_call = fallback_calls[0] if fallback_calls else None
    context_candidates = _context_fuzzy_candidates(name, query, context, history)
    if not alternative_queries and relaxed_call is None and fallback_call is None and not context_candidates:
        return None

    parts = [f"Không tìm thấy '{query}'."]
    if relaxed_call is not None:
        parts.append(f"Thử bỏ bộ lọc tuỳ chọn ({', '.join(optional_filter_keys)})")
    if alternative_queries:
        parts.append(f"Thử alternative_queries (như '{alternative_queries[0]}')")
    if fallback_call is not None:
        args_repr = json.dumps(fallback_call["arguments"], ensure_ascii=False, separators=(",", ":"))
        prefix = "hoặc gọi" if (alternative_queries or relaxed_call) else "Hãy gọi"
        parts.append(f"{prefix} {fallback_call['name']}({args_repr}) để xem toàn bộ danh sách;")
    parts.append("không lặp lại từ khóa hoặc bộ lọc đã rỗng.")
    fallback = {
        "failed_query": query,
        "alternative_queries": alternative_queries,
        "fallback_call": fallback_call,
        "instruction": " ".join(parts),
    }
    if relaxed_call is not None:
        fallback["relaxed_call"] = relaxed_call
    if context_candidates:
        fallback["context_candidates"] = context_candidates
    return fallback


def _redirect_doomed_search(name, cleaned, spec, transcript, observations):
    """Redirect guaranteed-empty retries to untried landmark/honorific/list fallbacks."""
    exact_sigs, empty_calls, failed_lookups = _collect_epoch_read_state(name, observations)
    curr_sig = _canonical(cleaned)
    # Never alter exact repeats so LoopMonitor blocks them as repeated reads.
    if curr_sig is not None and curr_sig in exact_sigs:
        return cleaned

    if "query" not in cleaned and _is_key_lookup_tool(name, spec):
        lookup_key, lookup_val = _extract_lookup_arg(cleaned, spec)
        if lookup_key and lookup_val:
            prior_val = failed_lookups.get((lookup_key, lookup_val.strip().upper()))
            if prior_val is not None:
                return {**cleaned, lookup_key: prior_val}
        return cleaned

    query = cleaned.get("query")
    if not isinstance(query, str) or not _is_doomed_search_query(name, query, cleaned, empty_calls, spec=spec):
        return cleaned

    required = set(((spec or {}).get("parameters") or {}).get("required") or [])
    base_args = {k: cleaned[k] for k in required if k != "query" and k in cleaned}
    alternatives = _untried_query_alternatives(
        name, query, base_args, transcript, empty_calls, exact_sigs,
        spec=spec, honorific_first=False, limit=1,
    )
    if alternatives:
        return alternatives[0][1]

    for fallback_call in _list_fallback_options(name, spec, base_args, exact_sigs, None, observations=observations):
        if fallback_call["name"] == name:
            return fallback_call["arguments"]
    return cleaned


def prepare_call_arguments(name, arguments, spec, context=None, transcript="", observations=()):
    """Clean optional empty filters, unmentioned default profile city, and guaranteed-empty retries."""
    if not isinstance(arguments, dict) or not isinstance(spec, dict):
        return arguments
    cleaned = dict(arguments)
    if spec.get("kind") == "read":
        required = set((spec.get("parameters") or {}).get("required") or [])
        cleaned = {
            k: v for k, v in cleaned.items()
            if k in required or v not in (None, "", [], {})
        }
    if name in {"weather.get_current", "weather.get_forecast"} and isinstance(cleaned.get("location"), str):
        profile = context.get("user_profile") if isinstance(context, dict) else None
        city = profile.get("city") if isinstance(profile, dict) else None
        if (isinstance(city, str) and city.strip()
                and cleaned["location"].casefold().strip() == city.casefold().strip()
                and city.casefold() not in (transcript or "").casefold()):
            cleaned.pop("location", None)
    if name == "flights.search":
        cleaned = _normalize_flight_search_args(cleaned, transcript, observations)
    if observations and spec.get("kind") == "read":
        cleaned = _redirect_doomed_search(name, cleaned, spec, transcript, observations)
    return cleaned


def _check_xref_node(value, schema, observed_ids, key_name=""):
    if not isinstance(schema, dict):
        return None
    ref = schema.get("x-ref")
    if ref and value is not None:
        ref_app = str(ref).split(".", 1)[0]
        allowed = observed_ids.get(ref_app, set())
        if not isinstance(value, str) or value not in allowed:
            return {"ok": False, "error": f"{key_name or 'id'} must come from an observed {ref} ID; call a read tool first"}
    if isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            for key, child in value.items():
                if key in props:
                    err = _check_xref_node(child, props[key], observed_ids, key)
                    if err is not None:
                        return err
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for child in value:
            err = _check_xref_node(child, schema["items"], observed_ids, key_name)
            if err is not None:
                return err
    return None


def check_observed_xref(arguments, spec, observed_ids):
    """Verify any present x-ref argument came from visible history, context, or prior reads."""
    if observed_ids is None or not isinstance(arguments, dict) or not isinstance(spec, dict):
        return None
    return _check_xref_node(arguments, spec.get("parameters"), observed_ids)


def precheck_call(name, arguments, catalog, enabled, monitor, local_names=(), *,
                  transcript='', context=None, history=(), observations=()):
    access = check_tool_access(name, catalog, enabled, local_names)
    if not access["allowed"]:
        return {"stage": "precheck", "name": name, "arguments": arguments,
                "error": access["instruction"], "access": access}
    if not isinstance(arguments, dict) or _canonical(arguments) is None:
        return {"stage": "precheck", "name": name, "arguments": arguments,
                "error": "arguments must be a finite JSON object"}
    if access["scope"] == "local":
        return None
    target = arguments.get("tool_name")
    if name == "sys.ask_missing_info" and (not isinstance(target, str) or target not in enabled or target == name):
        return {"stage": "precheck", "name": name, "arguments": arguments,
                "error": "Missing-info tool_name must be another available application tool."}
    if name == "sys.ask_missing_info":
        from .missing_info_guard import check_missing_info
        feedback = check_missing_info(arguments, enabled, transcript=transcript, context=context,
                                      history=history, observations=observations)
        if feedback:
            return {"stage": "missing_info_guard", "name": name, "arguments": arguments,
                    "error": feedback['instruction'], "feedback": feedback}
    spec_curr = enabled[name]
    params_spec = spec_curr.get("parameters") or {}
    props = params_spec.get("properties") or {}
    required_params = params_spec.get("required") or []
    if spec_curr.get("kind") == "read" and required_params:
        missing_req = [k for k in required_params if arguments.get(k) in (None, "", [], {})]
        undeclared = [k for k in arguments if props and k not in props]
        if missing_req and undeclared:
            req_str = ", ".join(missing_req)
            unk_str = ", ".join(undeclared)
            return {
                "stage": "precheck",
                "name": name,
                "arguments": arguments,
                "error": (
                    f"{name} thiếu tham số bắt buộc `{req_str}` "
                    f"(tham số không thuộc schema: `{unk_str}`). "
                    f"Hãy kiểm tra tool_schemas và truyền `{missing_req[0]}` hợp lệ."
                ),
                "feedback": {
                    "reason": "missing_required_parameters",
                    "required_param": missing_req[0],
                    "missing_required": missing_req,
                    "undeclared_params": undeclared,
                },
            }
    feedback = monitor.check(name, arguments, enabled[name].get("kind", "unknown"))
    if feedback:
        return {"stage": "loop_guard", "name": name, "arguments": arguments,
                "error": feedback["instruction"], "feedback": feedback}
    travel_loop = _check_travel_route_date_loop(name, arguments, observations)
    if travel_loop:
        return travel_loop
    stagnant_loop = _check_stagnant_search_loop(name, arguments, enabled[name], context, observations)
    if stagnant_loop:
        return stagnant_loop
    return None


def update_observed_ids(name, arguments, result, kind, observed_ids):
    """Share successful-result ID provenance with environment and recorded replay."""
    if (observed_ids is not None and isinstance(result, dict)
            and result.get('ok') is True and result.get('done') is not False):
        app = name.split('.', 1)[0]
        collect_ids(result, observed_ids, app)
        if kind == 'write':
            collect_ids(arguments, observed_ids, app)


def _suggest_ref_reads(ref_app, enabled):
    """Suggest installed read tools in `ref_app` along with their required parameter signatures."""
    if not isinstance(enabled, dict):
        return []
    hints = []
    for t_name, t_spec in enabled.items():
        if (isinstance(t_name, str) and t_name.startswith(f"{ref_app}.")
                and isinstance(t_spec, dict) and t_spec.get("kind") == "read"):
            req_keys = ((t_spec.get("parameters") or {}).get("required") or [])
            sig = ", ".join(f'"{k}": "..."' for k in req_keys)
            priority = 0 if any(tok in t_name for tok in (".find", ".search", ".list", ".get")) else 1
            hints.append((priority, t_name, f"{t_name}({{{sig}}})"))
    hints.sort()
    return [item[2] for item in hints[:3]]


def attempt_env_call(
    env,
    name,
    arguments,
    kind,
    monitor,
    call_index,
    *,
    emit=None,
    spec=None,
    observed_ids=None,
    transcript="",
    context=None,
    history=(),
    enabled=None,
    observations=(),
):
    """Caller must precheck first. Execution ALWAYS requires explicit ok:true."""
    emit = emit or (lambda event, **data: None)
    emit("validate_start", name=name, arguments=arguments)
    try:
        validation = env.validate(name, arguments)
    except Exception as exc:
        details = exception_details(exc)
        monitor.record_technical_failure(name, arguments, details, stage="validate")
        emit("validate_error", name=name, arguments=arguments, exception=details)
        return {"executed": False, "feedback": {
            "error": str(exc), "stage": "validate", "name": name, "arguments": arguments,
            "exception": details, "origin": "validation_exception"}}
    accepted = isinstance(validation, dict) and validation.get("ok") is True
    emit("validate_result", name=name, arguments=arguments, result=validation, accepted=accepted)
    if not accepted:
        schema_rejected = isinstance(validation, dict) and validation.get("ok") is False
        if schema_rejected:
            monitor.record_validation_failure(name, arguments, validation)
        else:
            monitor.record_technical_failure(name, arguments, validation, stage="validate_protocol")
        return {"executed": False, "feedback": {
            "error": f"Schema invalid: {validation!r}" if schema_rejected else f"Invalid validation response: {validation!r}",
            "stage": "validate", "name": name, "arguments": arguments, "validation": validation,
            "origin": "schema_rejection" if schema_rejected else "validation_protocol"}}
    if observed_ids is not None and spec is not None:
        xref_error = check_observed_xref(arguments, spec, observed_ids)
        if xref_error is not None:
            emit("xref_guard_rejected", name=name, arguments=arguments, validation=xref_error)
            error_msg = xref_error["error"]
            mismatch = _find_xref_mismatch(arguments, spec.get("parameters"), observed_ids)
            if mismatch is not None:
                ref, ref_app, param_key = mismatch
                if not observed_ids.get(ref_app) and _has_empty_epoch_read(ref_app, observations):
                    error_msg += (
                        f" Collection {ref} đã đọc và rỗng []; không tồn tại ID hợp lệ để thực thi {name}. "
                        f"Không thử lại {name}; hãy xử lý các yêu cầu độc lập còn lại hoặc kết thúc bằng done."
                    )
                elif not observed_ids.get(ref_app):
                    read_hints = _suggest_ref_reads(ref_app, enabled)
                    if read_hints:
                        error_msg += (
                            f" Chưa có dữ liệu {ref} nào được quan sát; hãy gọi {', '.join(read_hints)} "
                            f"trước để quan sát {param_key} hợp lệ rồi mới gọi {name}."
                        )
            return {"executed": False, "feedback": {
                "error": error_msg, "stage": "xref_guard", "name": name,
                "arguments": arguments, "validation": xref_error, "origin": "observed_id_guard"}}
    emit("call_start", name=name, arguments=arguments, call_index=call_index)
    execution_exception = None
    try:
        result = env.call(name, arguments)
    except Exception as exc:
        execution_exception = exception_details(exc)
        # This envelope is agent-generated, not an organizer ok:false response.
        result = {"execution_exception": execution_exception}
        emit("call_error", name=name, arguments=arguments, call_index=call_index,
             exception=execution_exception, agent_result=result)
    else:
        emit("call_result", name=name, arguments=arguments, call_index=call_index, result=result)
    update_observed_ids(name, arguments, result, kind, observed_ids)
    verification = monitor.record_result(name, arguments, result, kind,
                                         call_index=call_index, execution_exception=execution_exception)
    if kind == "read" and (
        verification.get("status") == "empty"
        or (
            verification.get("status") == "failure"
            and _is_key_lookup_tool(name, spec)
            and _companion_list_options(name, enabled, observations)
        )
    ):
        fallback = build_search_fallback(
            name, arguments, spec,
            transcript=transcript, context=context, history=history,
            enabled=enabled, observations=observations,
        )
        if fallback is not None:
            verification["search_fallback"] = fallback
            if fallback.get("instruction"):
                verification["instruction"] = fallback["instruction"]
            if hasattr(monitor, "attach_search_fallback"):
                monitor.attach_search_fallback(name, arguments, fallback)
    emit("execution_verified", name=name, verification=verification)
    if verification.get("contract_violation"):
        emit("execution_contract_violation", name=name, arguments=arguments, result=result)
    return {"executed": True, "result": result, "verification": verification,
            "automatic_execution_feedback": monitor.automatic_feedback(kind, verification, call_index)}


def result_observation(name, arguments, outcome, serialize):
    observation = {"name": name, "arguments": arguments, "result": serialize(outcome["result"]),
                   "automatic_execution_feedback": outcome["automatic_execution_feedback"]}
    if outcome["verification"]["status"] in ("failure", "empty", "unknown"):
        observation["verification"] = outcome["verification"]
    return observation
