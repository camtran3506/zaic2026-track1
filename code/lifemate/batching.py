"""Bounded planning batches; no ToolEnv access, threads, or extra LLMs here.

The agent dispatches actual children through the ordinary execution boundary.
SFT groups canonical replay rows afterwards, without replaying any call twice.
"""
from copy import deepcopy
import json
import re
import unicodedata

from .context import DEFAULT_MAX_NEW_TOKENS, MAX_CONTEXT_TOKENS


BATCH_TOOL = 'local_batch.run'
BATCH_POLICY = 'token-bounded-helpers-and-reads-v2'
MAX_READ_BATCH_CALLS = 4
BATCH_INSTRUCTIONS = """
- Có thể gọi local_batch.run với calls=[{name,arguments}, ...] để gộp các helper
  độc lập HOẶC 2–4 tool có kind=read độc lập vào một lượt. Không trộn hai nhóm.
  Helper không giới hạn số lượng theo batch/sample; chỉ gộp khi arguments và output
  vừa ngân sách token. Nhóm quá dài hoặc cần kết quả trước thì chia sang lượt sau.
  Mọi toán hạng phải đã biết trước nhóm; không tham chiếu kết quả tương lai.
  Nếu bộ kiểm tra chưa xác minh được toán hạng, gọi riêng từng tool như bình thường.
- Write, tool chưa rõ kind, sys.ask_missing_info và batch lồng nhau luôn gọi riêng.
  Env reads trong batch vẫn được validate/thực thi/ghi log từng call theo thứ tự.
  Khi batch bị ngắt, kiểm tra executed của từng result; remaining_calls là phần
  chưa được thử sau điểm ngắt. Đọc phản hồi rồi quyết định tiếp, không tự phát lại cả batch.
"""


def batch_schema():
    return {'name': BATCH_TOOL, 'kind': 'local',
        'description': 'Gộp các helper độc lập theo ngân sách token, không có trần số helper; '
                       'env read giới hạn 2–4 call/nhóm. Không write/meta/nested batch. '
                       'Kết quả từng child và phần chưa thực thi được trả chung; không giảm số env calls.',
        'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['calls'],
            'properties': {'calls': {'type': 'array', 'minItems': 2,
                'items': {'type': 'object', 'additionalProperties': False, 'required': ['name', 'arguments'],
                    'properties': {'name': {'type': 'string'}, 'arguments': {'type': 'object'}}}}}}}


def observation_children(observations):
    """Expose real child evidence to existing Python planners, not wrapper operands."""
    result = []
    for obs in observations:
        if obs.get('name') == BATCH_TOOL and obs.get('scope') == 'batch':
            response = json.loads(obs['result'])
            result.extend(response['results'])
        else:
            result.append(obs)
    return result


def successful(observation):
    if 'result' not in observation:
        return False
    result = observation['result']
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return False
    if not isinstance(result, dict):
        return False
    if observation.get('verification', {}).get('status') in {'empty', 'failure', 'unknown'}:
        return False
    if 'ok' in result:
        return result['ok'] is True and result.get('done') is not False
    return result.get('status') in {'ok', 'matched', 'valid', 'resolved'}


def batch_observation(calls, results):
    """One honest controller response; no wrapper or synthetic child in env log."""
    children = [{**deepcopy(r), 'executed': 'result' in r} for r in results]
    response = {'status': 'completed' if len(results) == len(calls) and all(map(successful, results))
                else 'interrupted', 'results': children,
                'remaining_calls': deepcopy(calls[len(results):])}
    return {'name': BATCH_TOOL, 'arguments': {'calls': deepcopy(calls)}, 'scope': 'batch',
            'result': json.dumps(response, ensure_ascii=False, default=str, separators=(',', ':'))}


def _normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def _values(value):
    if isinstance(value, dict):
        # Observations carry JSON strings as results; parse those as data only.
        for key, item in value.items():
            if key == 'result' and isinstance(item, str):
                try:
                    item = json.loads(item)
                except ValueError:
                    pass
            yield from _values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _values(item)
    else:
        yield value


def visible_values(payload):
    # Exclude schema examples and planned/rejected arguments: neither is evidence
    # that an entity ID exists or that a previous read produced an operand.
    results = [o['result'] for o in observation_children(payload.get('previous_actions_and_results', []))
               if successful(o)]
    evidence = [payload.get('request_transcript', ''), payload.get('user_context', {}),
                payload.get('conversation_history', []), payload.get('itn_candidates', []),
                payload.get('itn_history_candidates', []), [{'result': r} for r in results]]
    return list(_values(evidence))


def requires_observation(payload):
    """Conservatively retain a planning boundary for explicit control flow."""
    text = _normalized(' '.join([payload.get('request_transcript', '')] +
        [t.get('text', '') for t in payload.get('conversation_history', []) if t.get('role') == 'user']))
    text = ''.join(c for c in unicodedata.normalize('NFD', text.replace('đ', 'd'))
                   if unicodedata.category(c) != 'Mn')
    return bool(re.search(r'\b(neu|tuy|truong hop|dua vao|roi moi|xong moi|sau khi|if|unless|otherwise|depending|only after)\b'
                          r'|\b(khi|co|khong)\b.{0,160}\bthi\b', text))


def arguments_ready(call, schemas, evidence):
    """Conservative literal/schema-constant check, shared by runtime and SFT.

    This proves availability, not that the user's intent was understood. A call
    rejected here can still be proposed on its own through ordinary validation.
    """
    strings = [_normalized(v) for v in evidence if isinstance(v, str)]
    props = schemas.get(call['name'], {}).get('parameters', {}).get('properties', {})
    references = None

    def schema_references(value):
        if isinstance(value, dict):
            if isinstance(value.get('x-ref'), str):
                yield value['x-ref']
            for child in value.values():
                yield from schema_references(child)
        elif isinstance(value, list):
            for child in value:
                yield from schema_references(child)

    def ready(value, prop, key):
        nonlocal references
        if any(type(value) is type(v) and value == v for v in prop.get('enum', [])):
            return True
        if any(type(value) is type(prop[k]) and value == prop[k] for k in ('const', 'default') if k in prop):
            return True
        if isinstance(value, dict):
            return all(ready(v, prop.get('properties', {}).get(k, {}), k) for k, v in value.items())
        if isinstance(value, list):
            return all(ready(v, prop.get('items', {}), key) for v in value)
        if isinstance(value, str):
            if not value:
                return True
            if key in {'target_tool', 'collection', 'tool_name'}:
                if references is None:
                    references = set(schemas) | set(schema_references(schemas))
                if value in references:
                    return True
            needle = _normalized(value)
            return any(re.search(r'(?<!\w)' + re.escape(needle) + r'(?!\w)', text) for text in strings)
        if value is None or type(value) is bool:
            return True
        if type(value) in (int, float):
            if key in {'limit', 'offset', 'max_results'}:
                return True  # Literal pagination controls, still checked against schema.
            return any(type(v) is type(value) and v == value for v in evidence) or any(
                re.search(r'(?<![\w.])' + re.escape(str(value)) + r'(?!\w|\.\d)', text) for text in strings)
        return False

    return all(ready(value, props.get(key, {}), key) for key, value in call['arguments'].items())


def batch_error(arguments, schemas, local_names, payload, *, env_remaining=8):
    """Preflight every member before any execution. Never call env.validate here."""
    from .execution_policy import _canonical
    from .local_checks import inspect_call
    if not isinstance(arguments, dict) or set(arguments) != {'calls'}:
        return 'Batch arguments must contain only calls.'
    calls = arguments['calls']
    if not isinstance(calls, list) or len(calls) < 2:
        return 'Batch requires at least two calls; one helper/tool can be called directly.'
    categories, seen = set(), set()
    evidence = visible_values(payload)
    for index, call in enumerate(calls, 1):
        if not isinstance(call, dict) or set(call) != {'name', 'arguments'}:
            return 'Each batch member must contain exactly name and arguments.'
        name, args = call['name'], call['arguments']
        if not isinstance(name, str) or name == BATCH_TOOL or name not in schemas:
            return f'Batch member {index}: unknown, disallowed or nested batch tool.'
        if name in local_names:
            categories.add('local')
        elif not name.startswith(('sys.', 'local_')) and schemas[name].get('kind') == 'read':
            categories.add('read')
        else:
            return f'Batch member {index} ({name}): write, meta and unknown-kind tools must be called separately.'
        canonical = _canonical(args)
        if not isinstance(args, dict) or canonical is None:
            return 'Batch member arguments must be a finite JSON object.'
        if (name, canonical) in seen:
            return f'Batch member {index} ({name}): do not repeat an identical call inside a batch.'
        seen.add((name, canonical))
        inspection = inspect_call(name, args, schemas)
        if inspection['status'] == 'invalid':
            return f'Batch member {index} ({name}) violates its declared schema: {inspection["errors"]}'
        if not arguments_ready(call, schemas, evidence):
            return f'Batch member {index} ({name}): cannot verify all operands before the batch. Call separately and observe results first.'
    if len(categories) != 1:
        return 'Batch helpers and env reads separately.'
    if 'read' in categories and requires_observation(payload):
        return 'Conditional request: call env reads separately and observe each result before planning the next.'
    if 'read' in categories and len(calls) > MAX_READ_BATCH_CALLS:
        return f'Env read batch exceeds {MAX_READ_BATCH_CALLS} calls; split into smaller read batches.'
    if 'read' in categories and len(calls) > env_remaining:
        return 'Batch exceeds the remaining per-sample env call budget.'
    return None


def group_examples(examples, *, tokenizer=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, max_length=MAX_CONTEXT_TOKENS):
    """Group adjacent canonical/alias targets, never use future results as input.

    The canonical replay has already run once. Each trajectory is processed
    independently, after alias/noise augmentation, before it is written to SFT.
    """
    from .local_helpers import helper_tool_schemas
    from .llm import chat_token_ids
    local_names = helper_tool_schemas()
    groups, output = {}, []
    trajectory_key = None
    index = 0
    while index < len(examples):
        row = deepcopy(examples[index])
        key = (row['sample_id'], json.dumps(row.get('augmentation'), sort_keys=True))
        if key != trajectory_key:
            groups, trajectory_key = {}, key
        payload = json.loads(row['messages'][1]['content'])
        original_obs = payload['previous_actions_and_results']
        projected, offset = [], 0
        while offset < len(original_obs):
            if offset in groups:
                end, calls = groups[offset]
                if end > len(original_obs):
                    raise ValueError('SFT batch crosses an observation boundary')
                projected.append(batch_observation(calls, original_obs[offset:end]))
                offset = end
            else:
                projected.append(original_obs[offset]); offset += 1
        payload['previous_actions_and_results'] = projected
        row['messages'][1]['content'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        schemas = {s['name']: s for s in payload['available_tools']['tool_schemas']}
        target = json.loads(row['messages'][-1]['content'])
        calls = []
        if target.get('action') == 'call' and target['name'] != BATCH_TOOL:
            calls.append({'name': target['name'], 'arguments': target['arguments']})
        # Without a tokenizer there is no defensible token budget for arbitrarily
        # many helpers. Keep them single; finetune.py always supplies a tokenizer.
        can_group_helpers = tokenizer is not None or target.get('name') not in local_names
        prefix = None
        for next_index in range(index+1, len(examples)):
            if not calls or not can_group_helpers:
                break
            following = examples[next_index]
            # Never cross source sample or alias trajectory boundaries.
            if (following['sample_id'] != row['sample_id'] or
                    following.get('augmentation') != row.get('augmentation')):
                break
            after = json.loads(following['messages'][1]['content'])['previous_actions_and_results']
            if len(after) != len(original_obs)+len(calls) or not successful(after[-1]):
                break
            next_target = json.loads(following['messages'][-1]['content'])
            if next_target.get('action') != 'call':
                break
            proposal = calls+[{'name': next_target['name'], 'arguments': next_target['arguments']}]
            error = batch_error({'calls': proposal}, schemas, local_names, payload,
                env_remaining=8-sum(o.get('scope') != 'local' for o in original_obs))
            if error:
                break
            candidate = {'action': 'call', 'name': BATCH_TOOL, 'arguments': {'calls': proposal}}
            row['messages'][-1]['content'] = json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))
            if tokenizer is not None:
                if prefix is None:
                    prefix = chat_token_ids(tokenizer, row['messages'][:-1], True)
                full = chat_token_ids(tokenizer, row['messages'], False)
                if len(full)-len(prefix) > max_new_tokens:
                    row['messages'][-1]['content'] = json.dumps(
                        {'action': 'call', 'name': BATCH_TOOL, 'arguments': {'calls': calls}},
                        ensure_ascii=False, separators=(',', ':')) if len(calls) > 1 else json.dumps(target, ensure_ascii=False, separators=(',', ':'))
                    break
            calls = proposal
        if len(calls) > 1:
            groups[len(original_obs)] = (len(original_obs)+len(calls), calls)
        row['step'] = len(output)
        output.append(row)
        index += max(1, len(calls))
    if tokenizer is not None and any(len(chat_token_ids(tokenizer, row['messages'], False)) > max_length
                                     for row in output):
        # Wrapper overhead must never make an otherwise usable trajectory fail
        # the full context budget. Keep all original targets/results instead.
        return deepcopy(examples)
    return output
