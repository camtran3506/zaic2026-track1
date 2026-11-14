"""One validated ToolEnv execution boundary shared by inference and SFT replay.

Automatic guards are Python middleware, not model-callable helper tools. This
module uses only the provided env's validate/call methods; it never creates an
instance, reads state, or edits/reconstructs the official call log.
"""
import json
import re
import unicodedata

from .eval_trace import exception_details
from .execution_policy import check_tool_access, verify_result, _canonical
from .history import collect_ids


_MAP_ADMIN = frozenset({"quan", "phuong", "q", "p", "tp", "huyen", "thanh", "pho"})
_SHOP_MATCH_TOOLS = frozenset({"ecommerce.search", "recipes.search", "grocery.search"})
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
        elif name == "stocks.get_quote" and status == "failure":
            sym = prev_args.get("symbol")
            if isinstance(sym, str) and sym.strip():
                failed_quotes[sym.strip().upper()] = sym
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


def _dead_tokens_from_empty(name, empty_calls, curr_args):
    dead = set()
    ignore = {"servings"} if name == "recipes.search" else ()
    for prev_args in empty_calls:
        prev_q = prev_args.get("query")
        if not isinstance(prev_q, str) or _fold_search(prev_q) in ("", "*"):
            continue
        if name == "maps.search_place":
            if not _active_filters(prev_args) or _filters_cover_or_equal(prev_args, curr_args):
                dead |= (_search_tokens(prev_q) - {"*"} - _MAP_ADMIN)
        elif name in _SHOP_MATCH_TOOLS:
            if _filters_cover_or_equal(prev_args, curr_args, ignore_keys=ignore):
                prev_toks = _shop_tokens(prev_q)
                if 1 <= len(prev_toks) <= 2:
                    dead.update(prev_toks)
    return dead


def _is_doomed_search_query(name, query, curr_args, empty_calls):
    """Return True only when prior empty reads mathematically guarantee 0 hits."""
    if not isinstance(query, str) or _fold_search(query) in ("", "*"):
        return False
    if not empty_calls:
        return False
    if name == "maps.search_place":
        dead = _dead_tokens_from_empty(name, empty_calls, curr_args)
        curr_toks = _search_tokens(query) - {"*"} - _MAP_ADMIN
        return bool(curr_toks and curr_toks <= dead)
    if name in _SHOP_MATCH_TOOLS:
        dead = _dead_tokens_from_empty(name, empty_calls, curr_args)
        curr_toks = _shop_tokens(query)
        if not curr_toks:
            return False
        if all(tok in dead for tok in curr_toks):
            return True
        curr_folded = _fold_search(query)
        need = max(1, (len(curr_toks) + 1) // 2)
        live_count = sum(1 for tok in curr_toks if tok not in dead)
        ignore = {"servings"} if name == "recipes.search" else ()
        for prev_args in empty_calls:
            prev_q = prev_args.get("query")
            prev_folded = _fold_search(prev_q) if isinstance(prev_q, str) else ""
            if (len(prev_folded) >= 2 and prev_folded in curr_folded
                    and _filters_cover_or_equal(prev_args, curr_args, ignore_keys=ignore)
                    and live_count < need):
                return True
        return False
    # Substring matching tools (base.py / contacts / event_tickets / news / library / etc.)
    curr_folded = _fold_search(query)
    for prev_args in empty_calls:
        prev_q = prev_args.get("query")
        if not isinstance(prev_q, str):
            continue
        prev_folded = _fold_search(prev_q)
        if (len(prev_folded) >= 2 and prev_folded in curr_folded
                and _filters_cover_or_equal(prev_args, curr_args)):
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


def _list_fallback_options(name, spec, base_args, exact_sigs, enabled):
    """Build untried list-all / companion tool call options."""
    params = (spec or {}).get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    kept_required = {k: base_args[k] for k in required if k != "query" and k in base_args}
    if name == "stocks.get_quote":
        if enabled is None or "stocks.get_portfolio" in enabled:
            return [{"name": "stocks.get_portfolio", "arguments": {}}]
        return []
    if spec is not None and "query" not in props:
        return []
    candidates = []
    if spec is not None and "query" not in required:
        if _canonical(kept_required) not in exact_sigs:
            candidates.append(kept_required)
    wildcard_args = {**kept_required, "query": "*"}
    if _canonical(wildcard_args) not in exact_sigs:
        candidates.append(wildcard_args)
    return [{"name": name, "arguments": args} for args in candidates]


def _untried_query_alternatives(
    name,
    query,
    template_args,
    transcript,
    empty_calls,
    exact_sigs,
    *,
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
        if (cand_sig is not None and cand_sig in exact_sigs) or _is_doomed_search_query(name, cand, cand_args, empty_calls):
            continue
        selected.append((cand, cand_args))
        if len(selected) >= limit:
            break
    return selected


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
    """Build non-replayable prompt fallback suggestions after an empty read or failed stock quote."""
    if not isinstance(arguments, dict):
        return None
    exact_sigs, empty_calls, _ = _collect_epoch_read_state(name, observations)
    current_sig = _canonical(arguments)
    if current_sig is not None:
        exact_sigs.add(current_sig)
    prior_and_current_empty = list(empty_calls)

    if name == "stocks.get_quote":
        symbol = arguments.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        fallback_calls = _list_fallback_options(name, spec, arguments, exact_sigs, enabled)
        fallback_call = fallback_calls[0] if fallback_calls else None
        context_candidates = _context_fuzzy_candidates(name, symbol, context, history)
        if not fallback_call and not context_candidates:
            return None
        instr = (
            f"Không tìm thấy mã '{symbol}'. Không lặp lại cùng mã khác chữ hoa/thường; "
            f"hãy gọi stocks.get_portfolio({{}}) để xem danh sách mã hiện có."
            if fallback_call else
            f"Không tìm thấy mã '{symbol}'. Kiểm tra lại danh sách trong context/history."
        )
        fallback = {"failed_symbol": symbol, "alternative_queries": [], "fallback_call": fallback_call,
                    "instruction": instr}
        if context_candidates:
            fallback["context_candidates"] = context_candidates
        return fallback

    query = arguments.get("query")
    if not isinstance(query, str) or _fold_search(query) in ("", "*"):
        return None
    prior_and_current_empty.append(arguments)
    unfiltered_check = {k: v for k, v in arguments.items() if k not in ("near", "max_price")}
    alternative_queries = [
        cand for cand, _ in _untried_query_alternatives(
            name, query, unfiltered_check, transcript, prior_and_current_empty, exact_sigs, honorific_first=True
        )
    ]
    fallback_calls = _list_fallback_options(name, spec, unfiltered_check, exact_sigs, enabled)
    fallback_call = fallback_calls[0] if fallback_calls else None
    context_candidates = _context_fuzzy_candidates(name, query, context, history)
    if not alternative_queries and fallback_call is None and not context_candidates:
        return None

    parts = [f"Không tìm thấy '{query}'."]
    if alternative_queries:
        parts.append(f"Thử alternative_queries (như '{alternative_queries[0]}')")
    if fallback_call is not None:
        args_repr = json.dumps(fallback_call["arguments"], ensure_ascii=False, separators=(",", ":"))
        prefix = "hoặc gọi" if alternative_queries else "Hãy gọi"
        parts.append(f"{prefix} {fallback_call['name']}({args_repr}) để xem toàn bộ danh sách;")
    parts.append("không nối thêm ngày/giờ hay near vào từ khóa đã rỗng.")
    fallback = {
        "failed_query": query,
        "alternative_queries": alternative_queries,
        "fallback_call": fallback_call,
        "instruction": " ".join(parts),
    }
    if context_candidates:
        fallback["context_candidates"] = context_candidates
    return fallback


def _redirect_doomed_search(name, cleaned, spec, transcript, observations):
    """Redirect guaranteed-empty retries to untried landmark/honorific/list fallbacks."""
    exact_sigs, empty_calls, failed_quotes = _collect_epoch_read_state(name, observations)
    curr_sig = _canonical(cleaned)
    # Never alter exact repeats so LoopMonitor blocks them as repeated reads.
    if curr_sig is not None and curr_sig in exact_sigs:
        return cleaned

    if name == "stocks.get_quote" and isinstance(cleaned.get("symbol"), str):
        norm_sym = cleaned["symbol"].strip().upper()
        if norm_sym and norm_sym in failed_quotes:
            return {**cleaned, "symbol": failed_quotes[norm_sym]}
        return cleaned

    query = cleaned.get("query")
    if not isinstance(query, str) or not _is_doomed_search_query(name, query, cleaned, empty_calls):
        return cleaned

    required = set(((spec or {}).get("parameters") or {}).get("required") or [])
    base_args = {k: cleaned[k] for k in required if k != "query" and k in cleaned}
    alternatives = _untried_query_alternatives(
        name, query, base_args, transcript, empty_calls, exact_sigs, honorific_first=False, limit=1
    )
    if alternatives:
        return alternatives[0][1]

    for fallback_call in _list_fallback_options(name, spec, base_args, exact_sigs, None):
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
    feedback = monitor.check(name, arguments, enabled[name].get("kind", "unknown"))
    if feedback:
        return {"stage": "loop_guard", "name": name, "arguments": arguments,
                "error": feedback["instruction"], "feedback": feedback}
    return None


def update_observed_ids(name, arguments, result, kind, observed_ids):
    """Share successful-result ID provenance with environment and recorded replay."""
    if (observed_ids is not None and isinstance(result, dict)
            and result.get('ok') is True and result.get('done') is not False):
        app = name.split('.', 1)[0]
        collect_ids(result, observed_ids, app)
        if kind == 'write':
            collect_ids(arguments, observed_ids, app)


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
            return {"executed": False, "feedback": {
                "error": xref_error["error"], "stage": "xref_guard", "name": name,
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
        or (name == "stocks.get_quote" and verification.get("status") == "failure")
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
