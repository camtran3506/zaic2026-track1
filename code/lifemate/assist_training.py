"""Task-level SFT demonstrations from source spans and already observed results.

This selector never receives a gold action. It changes local demonstrations only;
the caller still replays official application calls once in their original order.
"""
from copy import deepcopy
import json
import re

from .local_assist import ASSIST_TOOL, solve, source_dates, source_clocks
from .temporal_training import (_EARLY, _number, _route_minutes, _TEMPORAL_ENV,
                                _TEMPORAL_REQUEST)


ASSIST_TRAINING_POLICY = 'observable-python-task-workflows-v1'


def _key(call):
    return call['name'], json.dumps(call['arguments'], ensure_ascii=False, sort_keys=True)


def task_demonstrations(transcript, context, enabled, observations, primitives, *, normalize_fields=True, include_date_tasks=True):
    """Prefer a fixed solver workflow; retain unrelated/unsupported primitives.

    Exact request spans, sample date anchors and genuine earlier read outputs are
    the only operands. No guessed default date, travel time or future tool data.
    """
    previous = {_key(row) for row in observations
                if row.get('scope') == 'local' and isinstance(row.get('arguments'), dict)}
    text = transcript.casefold()
    dates, clocks = source_dates(transcript), source_clocks(transcript)
    candidates, consumed = [], set()

    def add(task, inputs):
        call = {'name': ASSIST_TOOL, 'arguments': {'task': task, 'inputs': deepcopy(inputs)}}
        try:
            output = solve(call['arguments'], context)
        except ValueError:
            return False
        if output['status'] != 'resolved':
            return False
        if _key(call) not in previous:
            candidates.append(call)
            previous.add(_key(call))
        return True  # Already observed is also a reason not to repeat primitives.

    # A completed event update or uniquely identified event interval is already
    # observable. The legacy source selector has checked its provenance.
    for index, primitive in enumerate(primitives):
        if primitive['name'] == 'local_time.departure_time':
            if 'báo thức trước đó' in text and add('departure_time', primitive['arguments']):
                consumed.add(index)
        elif primitive['name'] == 'local_time.duration_between' and 'dời' in text:
            if add('duration', primitive['arguments']):
                consumed.add(index)

    temporal = any(name.split('.')[0] in _TEMPORAL_ENV for name in enabled)
    temporal = temporal and bool(_TEMPORAL_REQUEST.search(text) or clocks)
    covered_calendar = bool(consumed)
    # Extracting a short clock must not erase "kém", approximations or alternatives
    # from its surrounding request. Unsupported source arithmetic stays unresolved.
    uncertain = re.search(r'\b(?:kém|khoảng|tầm|chừng|hoặc|hay)\b', text)
    bare_date = len(dates) == 1 and re.fullmatch(r'mai|mốt|nay|ni|qua', dates[0][0], re.I)
    safe_date = not bare_date or dates[0][1] == 0
    if temporal and len(dates) == 1 and not uncertain and safe_date:
        date_text = dates[0][0]
        early = list(_EARLY.finditer(text))
        on_time = re.search(r'(?:có mặt|tới nơi|đến nơi)', text)
        route_task = (len(clocks) == 1 and 'maps.get_route' in enabled
                      and (len(early) == 1 or (not early and on_time)))
        if route_task:
            travel = _route_minutes(observations)
            # Do not teach a redundant date-only call before the route read.
            # The facade resolves the date itself once its operands exist.
            covered_calendar = True
            if travel is not None:
                covered = add('departure_time', {'date_text': date_text, 'arrival_time': clocks[0][0],
                    'travel_minutes': travel, 'early_minutes': _number(early[0]['minutes']) if early else 0})
                if covered:
                    consumed.update(i for i, p in enumerate(primitives) if p['name'] == 'local_time.departure_time')
                else:
                    covered_calendar = False
        elif len(clocks) == 2:
            a, b = clocks
            if re.search(r'\btừ\s*$', text[:a[1]]) and re.fullmatch(r'\s*(?:tới|đến|-)\s*', text[a[2]:b[1]]):
                if add('duration', {'date_text': date_text, 'start_time': a[0], 'end_time': b[0]}):
                    covered_calendar = True
                    consumed.update(i for i, p in enumerate(primitives) if p['name'] == 'local_time.duration_between')
        elif len(clocks) == 1 and include_date_tasks:
            covered_calendar = add('datetime', {'date_text': date_text, 'time_text': clocks[0][0]}) or covered_calendar
        elif not clocks and include_date_tasks:
            covered_calendar = add('date', {'date_text': date_text}) or covered_calendar

    if covered_calendar:
        consumed.update(i for i, p in enumerate(primitives) if p['name'] == 'local_time.calendar_reference')

    # Clear ITN candidates were excluded by specialist_demonstrations already.
    # Never teach a second normalization for a source field resolved in a prior
    # facade call (including fields split across token-sized requests).
    normalized = set()
    for row in observations:
        if row.get('scope') == 'local' and row.get('name') == ASSIST_TOOL:
            args = row.get('arguments', {})
            if args.get('task') == 'normalize_fields':
                normalized.update((f['kind'], f['text']) for f in args['inputs'].get('fields', []))
    fields = []
    for index, primitive in enumerate(primitives):
        if primitive['name'] != 'local_text.normalize':
            continue
        args = primitive['arguments']
        key = args['kind'], args['text']
        if key in normalized:
            consumed.add(index)
        elif normalize_fields:
            fields.append({'id': f'field_{len(fields)+1}', **deepcopy(args)})
            consumed.add(index)
            normalized.add(key)
    if fields:
        candidates.append({'name': ASSIST_TOOL, 'arguments': {
            'task': 'normalize_fields', 'inputs': {'fields': fields}}})
    return candidates + [deepcopy(p) for i, p in enumerate(primitives) if i not in consumed]
