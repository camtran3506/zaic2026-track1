"""Conservative missing-info checks over visible evidence, with no execution/LLM.

Only schema constants/defaults, explicit field assignments, unambiguous temporal
spans and exact entity mentions are resolved. Similarity alone, arbitrary numbers
and an ID merely occurring somewhere in history never establish user intent.
"""
from copy import deepcopy
import json
import re
import unicodedata


MISSING_INFO_POLICY = 'visible-evidence-missing-info-v1'
_TASKS = {
    'alarm': r'báo thức',
    'reminders': r'nhắc',
    'calendar': r'lịch|cuộc họp|cuộc hẹn',
    'weather': r'thời tiết|dự báo|mưa|nắng',
    'flights': r'chuyến bay|vé máy bay',
    'train_bus_tickets': r'vé tàu|vé xe',
    'banking': r'chuyển|thanh toán',
    'ewallet': r'chuyển|nạp|thanh toán',
    'phone': r'\bgọi\b',
    'sms': r'\bnhắn\b|\bsms\b',
    'email': r'\bemail\b|\bgửi thư\b',
}
_UNCERTAIN = re.compile(r'\b(?:nếu|hoặc|hay|khoảng|tầm|chừng|kém|trước|sau|đừng|không)\b', re.I)
_CONTACT_TOOLS = {'phone.call', 'sms.send', 'email.send', 'ewallet.topup_mobile'}


def _fold(value):
    # Diacritics distinguish names from ordinary words (Phan vs "phần").
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', value).casefold()))


def _mentioned(value, text):
    value, text = _fold(value), _fold(text)
    return bool(value and re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', text))


def _scope(target, transcript):
    """Retain one explicit task clause; never borrow operands from another task."""
    pattern = _TASKS.get(target.split('.', 1)[0])
    if pattern is None:
        return transcript
    clauses = re.split(r'[;!?]|(?<!\d)\.(?!\d)|\b(?:rồi|sau đó|tiếp theo)\b', transcript, flags=re.I)
    selected = [s for s in clauses if re.search(pattern, s, re.I)]
    if len(selected) != 1:
        return ''
    text = selected[0]
    app = target.split('.', 1)[0]
    if any(re.search(p, text, re.I) for a, p in _TASKS.items()
           if a != app and a not in {'banking', 'ewallet'}):
        return ''
    return text


def _valid_value(value, prop):
    from .local_checks import inspect_call
    schema = {'check': {'parameters': {'type': 'object', 'properties': {'value': prop},
                                       'required': ['value']}}}
    return (value not in (None, '', [], {}) and
            inspect_call('check', {'value': value}, schema)['status'] == 'valid')


def _temporal_values(target, text, props, context):
    from .local_assist import solve, source_clocks, source_dates
    if not text or _UNCERTAIN.search(text):
        return {}
    # Generic new apps need explicit tool-specific field assignments instead of
    # assuming which of several actions owns a natural-language time span.
    if target.split('.', 1)[0] not in {'alarm', 'reminders', 'calendar', 'weather', 'flights', 'train_bus_tickets'}:
        return {}
    dates, clocks = source_dates(text), source_clocks(text)
    if len(dates) != 1 or len(clocks) > 1:
        return {}
    result = {}
    for key, prop in props.items():
        if key == 'date':
            task, inputs, output = 'date', {'date_text': dates[0][0]}, 'date'
        elif key in {'time', 'start'} and clocks:
            task = 'datetime'
            inputs = {'date_text': dates[0][0], 'time_text': clocks[0][0]}
            output = 'datetime'
        else:
            continue
        resolved = solve({'task': task, 'inputs': inputs}, context)
        if resolved.get('status') != 'resolved':
            continue
        value = resolved.get('result', {}).get(output)
        if prop.get('format') == 'time' and isinstance(value, str) and 'T' in value:
            value = value.split('T', 1)[1][:8]
        if _valid_value(value, prop):
            result[key] = (value, {'source': 'transcript_and_time_context', 'text': text.strip()})
    return result


def _records(collection, context, history, observations, tools):
    """Gather the referenced collection only, including current successful reads."""
    from .batching import observation_children
    from .local_knowledge import _READS, knowledge_records
    app, _, field = collection.partition('.')
    turns = list(history)
    for obs in observation_children(observations):
        try:
            value = json.loads(obs['result']) if isinstance(obs.get('result'), str) else obs.get('result')
        except ValueError:
            continue
        if isinstance(value, dict) and value.get('ok') is True and value.get('done') is not False:
            turns.append({'role': 'assistant', 'tool_calls': [obs], 'tool_results': [value]})
    # A successful write invalidates older read snapshots. A new read or an
    # explicit current context collection can establish the values again.
    visible = []
    for turn in turns:
        if not isinstance(turn, dict) or turn.get('role', turn.get('speaker')) != 'assistant':
            continue
        calls, results = turn.get('tool_calls') or [], turn.get('tool_results') or []
        if not isinstance(calls, list) or not isinstance(results, list) or len(calls) != len(results):
            continue
        for call, result in zip(calls, results):
            if (not isinstance(call, dict) or not isinstance(result, dict)
                    or not isinstance(call.get('name'), str) or not isinstance(call.get('arguments'), dict)):
                continue
            name = call.get('name', '')
            if (isinstance(name, str) and name.split('.', 1)[0] == app
                    and tools.get(name, {}).get('kind') == 'write'
                    and result.get('ok') is True and result.get('done') is not False):
                visible.clear()
            visible.append({'role': 'assistant', 'tool_calls': [call], 'tool_results': [result]})
    if collection in _READS:
        return [r['record'] for r in knowledge_records(context, visible) if r['collection'] == collection]
    rows = []
    # The declared x-ref supplies the response field for new app schemas too.
    for turn in visible:
        calls, results = turn.get('tool_calls') or [], turn.get('tool_results') or []
        for call, result in zip(calls, results):
            name = call.get('name', '')
            if (name.split('.', 1)[0] == app and tools.get(name, {}).get('kind') == 'read'
                    and isinstance(result, dict) and result.get('ok') is True and result.get('done') is not False
                    and isinstance(result.get(field), list)):
                if not call.get('arguments'):
                    rows.clear()
                rows.extend(r for r in result[field] if isinstance(r, dict))
    collections = context.get('collections')
    explicit = collections.get(collection) if isinstance(collections, dict) else None
    nested = context.get(app)
    for values in (explicit, nested.get(field) if isinstance(nested, dict) else None):
        if isinstance(values, list):
            rows = [r for r in values if isinstance(r, dict)]
    return list({json.dumps(r.get('id', r), sort_keys=True): r for r in rows}.values())


def _entity_value(target, argument, prop, text, context, history, observations, tools):
    collection = prop.get('x-ref')
    projection = 'id'
    if not collection and argument in {'phone', 'phone_number', 'email'} and target in _CONTACT_TOOLS:
        collection, projection = 'contacts.contacts', 'email' if argument == 'email' else 'phone'
    if not isinstance(collection, str) or not text or _UNCERTAIN.search(text):
        return None
    candidates, ids, matches = [], {}, {}
    for row in _records(collection, context, history, observations, tools):
        value = row.get(projection, row.get(argument))
        if not _valid_value(value, prop):
            continue
        labels = [row.get(k) for k in ('name', 'title', 'label', 'relation')]
        labels += row.get('aliases', []) if isinstance(row.get('aliases'), list) else []
        labels = [v for v in labels if isinstance(v, str) and _fold(v)]
        identity = json.dumps(value, sort_keys=True)
        candidates.append((identity, labels))
        explicit_id = (projection == 'id' and isinstance(value, str)
                       and re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', text))
        if explicit_id:
            ids[identity] = value
        named = [v for v in labels if _mentioned(v, text)]
        if named:
            matches[identity] = (value, named)
    if len(ids) == 1:
        return next(iter(ids.values())), {'source': 'explicit_visible_id', 'collection': collection}
    if ids:
        return None
    if len(matches) != 1:
        return None
    identity, (value, labels) = next(iter(matches.items()))
    # "Lộc" remains ambiguous when the visible list also contains "Anh Lộc".
    if not any(not any(other != identity and any(_mentioned(label, v) for v in names)
                       for other, names in candidates) for label in labels):
        return None
    return value, {'source': 'visible_collection', 'collection': collection}


def check_missing_info(arguments, tools, *, transcript='', context=None, history=(), observations=()):
    """Return actionable rejection feedback, or None if missingness is unresolved.

    No gold values, final_state, schema examples or rejected call arguments are
    evidence. Optional fields can still be necessary for the user's specific task.
    """
    target, params = arguments.get('tool_name'), arguments.get('params')
    if not isinstance(target, str) or target not in tools:
        return None  # Access checks own this error.
    spec = tools[target].get('parameters') or {}
    props = spec.get('properties') or {}
    if (not isinstance(params, list) or not params
            or any(not isinstance(p, str) or not p for p in params)
            or len(set(params)) != len(params)):
        return {'reason': 'invalid_missing_info_params',
                'instruction': 'params phải là danh sách không rỗng gồm tên tham số thiếu, không trùng lặp.'}
    unknown = [p for p in params if p not in props]
    if unknown:
        return {'reason': 'unknown_missing_info_parameters', 'tool_name': target,
                'unknown_params': unknown, 'available_params': sorted(props),
                'instruction': 'Tên tham số không thuộc tool đích. Đối chiếu đúng tool/schema và tiếp tục yêu cầu.'}
    context = context if isinstance(context, dict) else {}
    text = _scope(target, transcript)
    known = {}
    for key, prop in props.items():
        if not isinstance(prop, dict):
            continue
        for marker in ('const', 'default'):
            # A schema default is a fallback, not proof the user selected it.
            if marker == 'default' and not re.search(r'\bmặc định\b', text, re.I):
                continue
            if marker in prop and _valid_value(prop[marker], prop):
                known[key] = (deepcopy(prop[marker]), {'source': 'schema_' + marker})
                break
        choices = prop.get('enum')
        if key not in known and isinstance(choices, list) and len(choices) == 1 and _valid_value(choices[0], prop):
            known[key] = (deepcopy(choices[0]), {'source': 'schema_single_enum'})
        # A literal field assignment binds a value without guessing its role.
        assigned = [] if _UNCERTAIN.search(text) else re.findall(
            r'(?<!\w)' + re.escape(key) + r'\s*[:=]\s*("[^"\n]*"|\d+(?:\.\d+)?|true|false)(?!\w)', text, re.I)
        values = []
        for raw in assigned:
            try:
                value = json.loads(raw.lower() if raw.lower() in {'true', 'false'} else raw)
            except ValueError:
                continue
            if _valid_value(value, prop):
                values.append(value)
        if values and len({json.dumps(v) for v in values}) == 1:
            known[key] = (values[0], {'source': 'explicit_field_assignment'})
        if key not in known:
            resolved = _entity_value(target, key, prop, text, context, history, observations, tools)
            if resolved is not None:
                known[key] = resolved
    known.update(_temporal_values(target, text, props, context))
    available = {p: known[p] for p in params if p in known}
    # x-one-of alternatives: a known valid alternative removes the need to ask
    # for a different member of that same group (ordinary required still apply).
    required = set(spec.get('required') or [])
    groups = spec.get('x-one-of') or []
    oneof_fields = {p for group in groups for p in group}
    satisfied = next((g for g in groups if all(p in known for p in g)), None)
    unnecessary = [p for p in params if p not in required and p in oneof_fields] if satisfied else []
    if not available and not unnecessary:
        return None
    include = set(available) | set(satisfied or [])
    return {'reason': 'information_already_available', 'tool_name': target,
            'known_arguments': {p: deepcopy(known[p][0]) for p in include},
            'evidence': {p: known[p][1] for p in include},
            'unresolved_params': [p for p in params if p not in available and p not in unnecessary],
            'instruction': 'Thông tin đã có trong dữ kiện nhìn thấy; dùng known_arguments và tiếp tục hành động. '
                           'Nếu còn thiếu trường khác, chỉ hỏi phần thực sự chưa xác định.'}
