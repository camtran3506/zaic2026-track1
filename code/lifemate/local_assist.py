"""Fixed Python task workflows. No ToolEnv, model, files, network or labels.

The caller supplies source operands, never a list of executable steps. Internal
calendar/normalizer outputs are joined here and emitted to optional debug traces.
"""
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import re
import unicodedata
import json

from .local_time import execute_local_tool, _context_zone, _timestamp, _MAX_MINUTES
from .itn import normalize_span
from .local_checks import inspect_call
from .temporal_text import CLOCK as _CLOCK, NUMBER, number as _number


ASSIST_TOOL = 'local_assist.solve'
ASSIST_POLICY = 'fixed-python-task-workflows-v1'
ASSIST_INSTRUCTIONS = """
- Ưu tiên local_assist.solve cho tác vụ date/datetime, departure_time, duration,
  normalize_fields đã đủ dữ liệu. Gửi task và inputs nguồn; Python tự chọn thứ tự
  các bước, tra lịch, ghép ngày/giờ và truyền kết quả trung gian. Không cần gọi
  calendar_reference rồi tự tính ngày. Không gửi danh sách hàm/steps/references.
- date dùng date_text; datetime dùng date_text+time_text hoặc timestamp ISO có offset.
  departure_time dùng date_text+arrival_time hoặc arrival ISO, travel_minutes và
  early_minutes đã biết (0 nếu yêu cầu xác định không có khoảng đó). duration dùng
  date_text+start_time+end_time (end_date_text nếu khác ngày) hoặc start+end ISO.
  normalize_fields dùng fields=[{id,kind,text},...], kind là phone/email/money.
- Sao chép đoạn nguồn và số phút đã có; không bịa năm, chọn tuần/người, mặc định
  thời gian di chuyển hay tự suy qua nửa đêm. missing/ambiguous/unsupported/partial
  là phản hồi cần xem xét, không phải quyền đoán hoặc done. Solver không gọi env,
  không tự ghi ứng dụng. Dùng ITN đã có khi đủ; helper đơn và batch vẫn là fallback.
"""


def schemas():
    text = {'type': 'string', 'minLength': 1, 'maxLength': 256}
    properties = {key: deepcopy(text) for key in (
        'date_text', 'end_date_text', 'time_text', 'arrival_time', 'start_time', 'end_time',
        'timestamp', 'arrival', 'start', 'end')}
    properties.update({key: {'type': 'integer', 'minimum': 0, 'maximum': _MAX_MINUTES}
                       for key in ('travel_minutes', 'early_minutes')})
    properties['fields'] = {'type': 'array', 'minItems': 1, 'items': {
        'type': 'object', 'additionalProperties': False, 'required': ['id', 'kind', 'text'],
        'properties': {'id': text, 'kind': {'type': 'string', 'enum': ['phone', 'email', 'money']},
                       'text': {'type': 'string', 'minLength': 1, 'maxLength': 1024}}}}
    return {ASSIST_TOOL: {'name': ASSIST_TOOL, 'kind': 'local',
        'description': 'Resolve a date/datetime, departure time, duration or multiple text fields in one Python workflow. '
                       'Inputs are source spans/known operands. Returns resolved, missing, ambiguous, unsupported or partial; '
                       'never calls application tools. Use explicit week/year and timezone; no guessed travel time or overnight rollover.',
        'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['task', 'inputs'],
            'properties': {'task': {'type': 'string', 'enum': ['date', 'datetime', 'departure_time', 'duration', 'normalize_fields']},
                           'inputs': {'type': 'object', 'additionalProperties': False, 'properties': properties}}}}}


class ResolutionIssue(Exception):
    def __init__(self, status, reason, fields=(), candidates=()):
        self.status, self.reason, self.fields, self.candidates = status, reason, list(fields), list(candidates)


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().strip(' .,').split())


PART = r'(?:sáng|trưa|chiều|tối|đêm)'
WEEKDAY = r'(?:thứ (?:hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật)'
DAY_WORD = r'(?:\d+|'+NUMBER+r')'
DATE_SPAN = re.compile(r'(?<!\w)(?:'+PART+r'\s+)?(?:'+WEEKDAY+r'(?:\s+tuần\s+(?:này|sau|trước))?'
    r'|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|(?:hôm|ngày)\s+(?:nay|mai|kia|qua|mốt)'
    r'|'+DAY_WORD+r'\s+ngày\s+(?:nữa|trước)|mai|mốt|nay|ni|qua)(?!\w)', re.I)


def source_dates(text):
    return [(m.group(), m.start(), m.end()) for m in DATE_SPAN.finditer(text)]


def source_clocks(text):
    """Retain exact clock spans and adjacent dayparts, not calculated timestamps."""
    clocks = []
    # Seconds must not be silently discarded by the legacy minute-only regex.
    pattern = re.compile(r'(?<!\w)\d{1,2}:\d{2}:\d{2}(?!\w)|'+_CLOCK.pattern)
    for match in pattern.finditer(text.casefold()):
        start, end = match.span()
        before = re.search(r'\b'+PART+r'\s*$', text[:start], re.I)
        after = re.match(r'\s*'+PART+r'\b', text[end:], re.I)
        if before:
            start = before.start()
        if after:
            end += after.end()
        clocks.append((text[start:end], start, end))
    return clocks


class Workflow:
    def __init__(self, context, trace=None):
        self.context = {k: deepcopy(context[k]) for k in ('current_time', 'timezone') if k in context}
        self.trace = trace or (lambda event, **data: None)
        self.reference = None

    def step(self, operation, inputs, fn):
        self.trace('assist_step_start', operation=operation, inputs=deepcopy(inputs))
        try:
            result = fn()
        except Exception as exc:
            self.trace('assist_step_error', operation=operation, error=str(exc),
                       status=exc.status if isinstance(exc, ResolutionIssue) else 'unsupported')
            raise
        self.trace('assist_step_result', operation=operation,
                   result=json.loads(json.dumps(result, ensure_ascii=False, default=str)))
        return result

    def primitive(self, name, arguments):
        return self.step(name, arguments, lambda: execute_local_tool(name, arguments, self.context))

    def calendar(self):
        if self.reference is None:
            if not self.context.get('current_time'):
                raise ResolutionIssue('missing', 'Sample current_time is required for local date resolution.', ['context.current_time'])
            self.reference = self.primitive('local_time.calendar_reference', {})
        return self.reference

    def zone(self):
        if not self.context.get('timezone') and not self.context.get('current_time'):
            raise ResolutionIssue('missing', 'A sample timezone/offset is required.', ['context.timezone'])
        fallback = _timestamp(self.context['current_time'], 'context.current_time').tzinfo if self.context.get('current_time') else None
        return _context_zone(self.context, fallback)

    def day(self, raw):
        def resolve():
            text = normalized(raw)
            part_match = re.match('^('+PART+r')\s+', text)
            part = part_match[1] if part_match else None
            if part:
                text = text[part_match.end():]
            if re.search(r'\b(?:hoặc|hay)\b|;', text):
                raise ResolutionIssue('ambiguous', 'More than one date alternative.')
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
                return date.fromisoformat(text), part
            if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}', text):
                d, m, y = map(int, text.split('/'))
                return date(y, m, d), part
            ref = self.calendar()
            aliases = {'nay': 'hôm nay', 'ni': 'hôm nay', 'ngày nay': 'hôm nay',
                       'ngày mốt': 'ngày kia', 'mốt': 'ngày kia', 'mai': 'ngày mai',
                       'qua': 'hôm qua', 'ngày qua': 'hôm qua', 'hôm mai': 'ngày mai', 'hôm mốt': 'ngày kia'}
            key = aliases.get(text, text)
            if key in ref['relative_dates']:
                return date.fromisoformat(ref['relative_dates'][key]), part
            if key == 'hôm kia':
                return date.fromisoformat(ref['relative_dates']['hôm nay'])-timedelta(days=2), part
            delta = re.fullmatch('('+DAY_WORD+r')\s+ngày\s+(nữa|trước)', text)
            if delta:
                days = _number(delta[1]) * (1 if delta[2] == 'nữa' else -1)
                return date.fromisoformat(ref['relative_dates']['hôm nay'])+timedelta(days=days), part
            weekday = re.fullmatch('('+WEEKDAY+r')(?:\s+tuần\s+(này|sau|trước))?', text)
            if weekday:
                day_name = weekday[1]
                digit = re.fullmatch(r'thứ ([2-7])', day_name)
                if digit:
                    day_name = ('thứ hai', 'thứ ba', 'thứ tư', 'thứ năm', 'thứ sáu', 'thứ bảy')[int(digit[1])-2]
                if not weekday[2]:
                    raise ResolutionIssue('ambiguous', 'Weekday requires an explicit week.',
                        candidates=[{'week': week, 'date': dates[day_name]} for week, dates in ref['weeks'].items()])
                return date.fromisoformat(ref['weeks']['tuần '+weekday[2]][day_name]), part
            raise ResolutionIssue('unsupported', 'Use one supported relative date, weekday+week or date with an explicit year.')
        return self.step('resolve_date', {'date_text': raw}, resolve)

    def clock(self, raw, hint=None):
        def resolve():
            text = normalized(raw)
            if re.search(r'\b(?:hoặc|hay|khoảng|tầm|chừng|kém)\b', text):
                raise ResolutionIssue('ambiguous', 'Clock is approximate, alternative or unsupported subtraction wording.')
            parts = re.findall(r'\b'+PART+r'\b', text)
            if len(set(parts)) > 1 or (parts and hint and parts[0] != hint):
                raise ResolutionIssue('ambiguous', 'Conflicting dayparts.')
            part = parts[0] if parts else hint
            text = re.sub(r'\b'+PART+r'\b', '', text).strip()
            literal = re.fullmatch(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', text)
            if literal:
                h, m, s = int(literal[1]), int(literal[2]), int(literal[3] or 0)
            else:
                match = _CLOCK.fullmatch(text)
                if not match:
                    raise ResolutionIssue('unsupported', 'Use one HH:MM[:SS] or supported Vietnamese clock span.')
                h = _number(match['hour']); m = 30 if match['half'] else _number(match['minute']) if match['minute'] else 0
                s = 0
            if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
                raise ResolutionIssue('unsupported', 'Clock outside the valid range.')
            if part:
                if part == 'sáng' and h >= 12 or part in {'tối', 'đêm'} and h == 12:
                    raise ResolutionIssue('ambiguous', 'Daypart conflicts with the hour or leaves midnight date unclear.')
                if part == 'trưa':
                    if h in (1, 2): h += 12
                    elif h not in (11, 12, 13, 14):
                        raise ResolutionIssue('ambiguous', 'Unsupported noon clock.')
                elif part in {'chiều', 'tối', 'đêm'} and 1 <= h < 12:
                    if part == 'đêm':
                        raise ResolutionIssue('ambiguous', 'Night clock needs an explicit date/24-hour clock.')
                    h += 12
                if ((part == 'chiều' and not 13 <= h <= 18)
                        or (part == 'tối' and not 17 <= h <= 23)
                        or (part == 'đêm' and not 19 <= h <= 23)):
                    raise ResolutionIssue('ambiguous', 'Hour and daypart do not identify one supported clock.')
            elif not literal and 1 <= h <= 12:
                raise ResolutionIssue('ambiguous', 'Spoken clock needs a morning/afternoon/evening qualifier.')
            return time(h, m, s)
        return self.step('resolve_clock', {'time_text': raw, 'daypart': hint}, resolve)

    def stamp(self, inputs, *, iso_key, date_key, clock_key):
        if iso_key in inputs:
            if date_key in inputs or clock_key in inputs:
                raise ResolutionIssue('ambiguous', 'Use either an ISO timestamp or date/clock spans, not both.')
            result = _timestamp(inputs[iso_key], 'inputs.'+iso_key)
            return result.astimezone(_context_zone(self.context, result.tzinfo))
        missing = ['inputs.'+k for k in (date_key, clock_key) if k not in inputs]
        if missing:
            raise ResolutionIssue('missing', 'Date and clock operands are required.', missing)
        day, hint = self.day(inputs[date_key])
        clock = self.clock(inputs[clock_key], hint)
        zone = self.zone()
        def combine():
            value = datetime.combine(day, clock, tzinfo=zone)
            back = value.astimezone(timezone.utc).astimezone(zone)
            if back.replace(tzinfo=None) != value.replace(tzinfo=None):
                raise ResolutionIssue('unsupported', 'Nonexistent local time at DST transition.')
            if value.utcoffset() != value.replace(fold=1).utcoffset():
                raise ResolutionIssue('ambiguous', 'Ambiguous local time needs an explicit ISO offset.')
            return value.isoformat()
        return datetime.fromisoformat(self.step('combine_datetime', {'date': day.isoformat(), 'time': clock.isoformat()}, combine))


_INPUTS = {'date': {'date_text'}, 'datetime': {'timestamp', 'date_text', 'time_text'},
    'departure_time': {'arrival', 'date_text', 'arrival_time', 'travel_minutes', 'early_minutes'},
    'duration': {'start', 'end', 'date_text', 'end_date_text', 'start_time', 'end_time'},
    'normalize_fields': {'fields'}}


def solve(arguments, context, *, trace=None):
    """Execute a whitelisted workflow and expose final values plus source evidence."""
    if not isinstance(arguments, dict) or set(arguments) != {'task', 'inputs'}:
        raise ValueError('Task request must contain exactly task and inputs')
    validation = inspect_call(ASSIST_TOOL, arguments, schemas())
    if validation['status'] != 'valid':
        raise ValueError(f"Invalid task inputs: {validation.get('errors', validation['status'])}")
    task, inputs = arguments['task'], arguments['inputs']
    if not isinstance(task, str) or task not in _INPUTS or not isinstance(inputs, dict):
        raise ValueError('Unknown task or non-object inputs')
    if set(inputs)-_INPUTS[task]:
        raise ValueError(f'Unexpected fields for {task}: {sorted(set(inputs)-_INPUTS[task])}')
    if not isinstance(context, dict):
        raise ValueError('Task context must be an object')
    workflow = Workflow(context, trace)
    evidence = {'inputs': deepcopy(inputs)}
    response = {'task': task, 'policy': ASSIST_POLICY, 'evidence': evidence}
    try:
        if task == 'normalize_fields':
            if not inputs.get('fields'):
                raise ResolutionIssue('missing', 'Provide source text fields.', ['inputs.fields'])
            if len({field['id'] for field in inputs['fields']}) != len(inputs['fields']):
                raise ValueError('Field IDs must be unique')
            fields, sources = {}, {}
            for field in inputs['fields']:
                output = workflow.step('local_text.normalize', field,
                    lambda f=field: normalize_span(f['kind'], f['text']))
                fields[field['id']] = {k: deepcopy(v) for k, v in output.items() if k not in {'evidence', 'warnings'}}
                sources[field['id']] = {'kind': field['kind'], 'text': field['text']}
            matched = all(r['status'] == 'matched' for r in fields.values())
            response.update(status='resolved' if matched else 'ambiguous' if any(r['status']=='ambiguous' for r in fields.values()) else 'partial' if any(r['status']=='matched' for r in fields.values()) else 'unsupported',
                            result={'fields': fields})
            response['evidence'] = {'fields': sources}
            return response
        evidence['context'] = deepcopy(workflow.context)
        if task == 'date':
            if 'date_text' not in inputs:
                raise ResolutionIssue('missing', 'Provide a source date span.', ['inputs.date_text'])
            day, _ = workflow.day(inputs['date_text'])
            result = {'date': day.isoformat()}
        elif task == 'datetime':
            result = {'datetime': workflow.stamp(inputs, iso_key='timestamp', date_key='date_text', clock_key='time_text').isoformat()}
        elif task == 'departure_time':
            missing = ['inputs.'+k for k in ('travel_minutes', 'early_minutes') if k not in inputs]
            if missing:
                raise ResolutionIssue('missing', 'Do not guess travel/early minutes.', missing)
            arrival = workflow.stamp(inputs, iso_key='arrival', date_key='date_text', clock_key='arrival_time')
            result = workflow.primitive('local_time.departure_time', {
                'arrival': arrival.isoformat(), 'travel_minutes': inputs['travel_minutes'], 'early_minutes': inputs['early_minutes']})
            result.pop('ok')
        else:
            if ('start' in inputs) != ('end' in inputs):
                raise ResolutionIssue('missing', 'Provide both ISO endpoints.', ['inputs.end' if 'start' in inputs else 'inputs.start'])
            if 'start' in inputs:
                if set(inputs)-{'start', 'end'}:
                    raise ResolutionIssue('ambiguous', 'Use ISO endpoints or date/clock spans, not both.')
                start = workflow.stamp({'start': inputs['start']}, iso_key='start', date_key='date_text', clock_key='start_time')
                end = workflow.stamp({'end': inputs['end']}, iso_key='end', date_key='end_date_text', clock_key='end_time')
            else:
                start = workflow.stamp(inputs, iso_key='start', date_key='date_text', clock_key='start_time')
                end_inputs = {'end_date_text': inputs.get('end_date_text', inputs['date_text'])}
                if 'end_time' in inputs: end_inputs['end_time'] = inputs['end_time']
                end = workflow.stamp(end_inputs, iso_key='end', date_key='end_date_text', clock_key='end_time')
            if end.astimezone(timezone.utc) < start.astimezone(timezone.utc):
                raise ResolutionIssue('ambiguous', 'End precedes start; specify the actual end date, no overnight rollover is inferred.',
                                      ['inputs.end' if 'start' in inputs else 'inputs.end_date_text'])
            result = workflow.primitive('local_time.duration_between', {'start': start.isoformat(), 'end': end.isoformat()})
            result.pop('ok')
        return {**response, 'status': 'resolved', 'result': result}
    except ResolutionIssue as issue:
        return {**response, 'status': issue.status, 'reason': issue.reason,
                **({'missing': issue.fields} if issue.fields else {}),
                **({'candidates': issue.candidates} if issue.candidates else {})}
    except (ValueError, OverflowError, OSError) as exc:
        return {**response, 'status': 'unsupported', 'reason': str(exc)}
