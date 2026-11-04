"""ITN is source-scoped preprocessing shared by training and inference."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from lifemate.agent import AgentConfig, LifeMateAgent, build_step_prompt
from lifemate.function_calling import native_chat
from lifemate.itn import ITN_POLICY, detect_candidates, normalize_span
from lifemate.local_helpers import execute_helper, helper_tool_schemas
from lifemate.runtime import RUNTIME_MANIFEST, load_runtime_manifest
from lifemate.sft import replay_sample, validate_prompt_policy, SFT_PROMPT_POLICY
from lifemate.local_helpers import LOCAL_HELPER_POLICY
from lifemate.local_time import LOCAL_TIME_POLICY
from lifemate.text import TEXT_NORMALIZATION
from test_local_time_runtime import temporal_artifact
from test_python_helpers_integration import CATALOG, Env, Planner, DONE, call, sample, prepared


@pytest.mark.parametrize('text,kind,value', [
    ('Gọi số không chín một hai ba bốn năm sáu bảy tám.', 'phone', '0912345678'),
    ('Số điện thoại là +84 912 345 678.', 'phone', '+84912345678'),
    ('Nhắn tin đến số 0912-345-678.', 'phone', '0912345678'),
    ('Email Lan chấm Data a còng example chấm com chấm vn.', 'email', 'Lan.Data@example.com.vn'),
    ('Gửi Team.Data@example.test.', 'email', 'Team.Data@example.test'),
    ('Gửi a+b@somewhere.dev.', 'email', 'a+b@somewhere.dev'),
    ('Email Lan chấm Data@example.com.', 'email', 'Lan.Data@example.com'),
    ("Email o'connor@example.com.", 'email', "o'connor@example.com"),
    ('Trả hai triệu rưỡi đồng.', 'money', 2500000),
    ('Trả một triệu hai trăm nghìn đồng.', 'money', 1200000),
    ('Trả hai trăm linh năm đồng.', 'money', 205),
    ('Trả một nghìn hai trăm đồng.', 'money', 1200),
    ('Trả 250.000 VND.', 'money', 250000),
    ('Trả 1,5 triệu đồng.', 'money', 1500000),
    ('Trả một tỷ hai trăm triệu đồng.', 'money', 1200000000),
])
def test_explicit_spans_keep_provenance_and_match_the_same_helper(text, kind, value):
    rows = detect_candidates(text)
    assert len(rows) == 1
    row = rows[0]
    assert row['kind'] == kind and row['value'] == value
    assert text[row['start']:row['end']] == row['source_text']
    result = execute_helper('local_text.normalize', {'kind': kind, 'text': row['source_text']}, {}, [], {})
    assert result['value'] == value and result['status'] == 'matched'
    if kind == 'money':
        assert row['currency'] == result['currency'] == 'VND'


@pytest.mark.parametrize('text', [
    'Mã tài khoản 0912345678.', 'Tôi có hai triệu.', 'Đặt lịch năm giờ.',
    'Gọi bà Năm.', 'Email a..b@example.com.', 'Email a@@example.com.',
    'Số điện thoại 091234567890123456.', 'Số điện thoại 0912345678abc.',
    'Số điện thoại không chín một hai ba bốn năm sáu bảy tám mươi.',
    'Trả hai triệu năm đồng.', 'Trả hai trăm năm đồng.',
    'Trả một nghìn triệu đồng.', 'Trả 1.5 triệu đồng.',
    'Trả âm hai triệu đồng.', 'Trả -250.000 VND.', 'Trả 200 USD.', 'Trả 250,000 đồng.', 'Trả 1,500 triệu đồng.',
])
def test_unclear_or_unsupported_forms_are_not_guessed(text):
    assert detect_candidates(text) == []


def test_negation_and_multiple_values_remain_available_without_choosing_intent():
    text = 'Đừng gọi số 0912345678; gọi số 0987654321. Trả 100 đồng, không trả 200 đồng.'
    rows = detect_candidates(text)
    assert [r['value'] for r in rows] == ['0912345678', '0987654321', 100, 200]
    rows[0]['value'] = 'MUTATION'
    assert detect_candidates(text)[0]['value'] == '0912345678'


@pytest.mark.parametrize('text', ['hai triệu năm', '1.5 triệu đồng', 'hai trăm năm', 'một nghìn triệu', '9999999999999999999999999999999999999999'])
def test_manual_money_helper_does_not_guess_shorthand(text):
    assert normalize_span('money', text)['status'] == 'unsupported'


def test_manual_helper_remains_available_for_unmarked_spans():
    assert detect_candidates('Chuyển hai triệu rưỡi.') == []
    assert execute_helper('local_text.normalize', {'kind': 'money', 'text': 'hai triệu rưỡi'}, {}, [], {})['value'] == 2500000
    assert normalize_span('money', 'hai triệu hoặc ba triệu')['status'] == 'ambiguous'
    assert 'money' in helper_tool_schemas()['local_text.normalize']['parameters']['properties']['kind']['enum']


def test_shared_prompt_preserves_history_tool_values_and_native_annotations():
    history = [{'role': 'assistant', 'text': 'Email X a còng example chấm com.',
                'tool_calls': [{'name': 'notes.search', 'arguments': {'query': 'ABC@Example.COM'}}],
                'tool_results': [{'ok': True, 'literal': 'ABC@Example.COM'}]}]
    original = deepcopy(history)
    _, prompt = build_step_prompt('Gọi số 0912345678.', {}, history, CATALOG, [])
    payload = json.loads(prompt)
    assert payload['itn_policy'] == ITN_POLICY
    assert payload['request_transcript'] == 'Gọi số 0912345678.'
    assert payload['itn_candidates'][0]['value'] == '0912345678'
    assert payload['itn_history_candidates'][0]['history_index'] == 0
    assert payload['conversation_history'][0]['tool_calls'] == original[0]['tool_calls']
    assert payload['conversation_history'][0]['tool_results'] == original[0]['tool_results']
    assert history == original
    chat, _ = native_chat([{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}])
    request = next(json.loads(row['content']) for row in chat if row['role'] == 'user' and 'request_transcript' in json.loads(row['content']))
    assert request['itn_candidates'] == payload['itn_candidates']


def test_sft_inference_share_automatic_itn_without_a_helper_turn_or_env_effect():
    text = 'Gọi số 0912345678, trả hai triệu đồng.'
    entry = {**sample(), 'audio_text': text, 'history': [], 'label': {'tool_calls': [
        {'name': 'notes.append', 'arguments': {'text': '2000000'}}]}}
    prepared_row = {**prepared(), 'transcript': text}
    env = Env()
    examples = replay_sample(entry, CATALOG, env, Path('/unused'), prepared_transcript=prepared_row)
    targets = [json.loads(e['messages'][-1]['content']) for e in examples]
    assert targets == [call('notes.append', text='2000000'), DONE]
    planner = Planner(targets)
    inference_env = Env()
    stats = LifeMateAgent(None, planner, CATALOG, AgentConfig()).run(entry, inference_env, Path('/unused'), 'public_test', prepared_row)
    training = json.loads(examples[0]['messages'][1]['content'])
    inference = json.loads(planner.requests[0][1])
    assert training == inference
    assert len(training['itn_candidates']) == 2
    assert stats['local_executed'] == 0 and stats['executed'] == 1
    assert env.get_tool_calls() == inference_env.get_tool_calls() == entry['label']['tool_calls']
    assert env.get_state_hash() == inference_env.get_state_hash()
    changed = deepcopy(entry)
    changed['label']['tool_calls'][0]['arguments']['text'] = 'FUTURE'
    other = replay_sample(changed, CATALOG, Env(), Path('/unused'), prepared_transcript=prepared_row)
    assert examples[0]['messages'][1]['content'] == other[0]['messages'][1]['content']


def test_itn_manifest_incompatibility_requires_repreparation(temporal_artifact):
    base, artifact, tokenizer, manifest = temporal_artifact
    assert manifest['itn_policy'] == ITN_POLICY
    manifest.pop('itn_policy')
    (artifact / RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='ITN policy'):
        load_runtime_manifest(base, tokenizer, artifact)


def test_sft_manifest_requires_itn_policy():
    manifest = {"prompt_policy": SFT_PROMPT_POLICY, "text_normalization": TEXT_NORMALIZATION,
                "local_helper_policy": LOCAL_HELPER_POLICY, "local_time_policy": LOCAL_TIME_POLICY,
                "itn_policy": ITN_POLICY}
    validate_prompt_policy(manifest)
    manifest.pop("itn_policy")
    with pytest.raises(ValueError, match="rerun prepare_sft"):
        validate_prompt_policy(manifest)
