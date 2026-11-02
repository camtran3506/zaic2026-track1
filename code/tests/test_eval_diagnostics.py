"""Post-hoc error categories preserve scoring, duplicates and stop semantics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.eval_diagnostics import DiagnosticSummary, analyze_rollout
from lifemate.scoring import call_f1, state_tool_score
from test_baseline import CATALOG, FakeEnv, StubASR, StubLLM, item


def call(name='smart_home.get_status', **arguments):
    return {'name': name, 'arguments': arguments}


def analyze(predicted, gold, *, stop='done', same_state=False, catalog=CATALOG):
    return analyze_rollout(predicted, gold, 'right' if same_state else 'wrong', 'right', catalog, stop)


def test_matching_ignores_order_and_counts_duplicate_calls_once():
    read = call()
    write = call('smart_home.light_set', device_id='light_living', power='on')
    predicted, gold = [write, read, read], [read, write]
    original = deepcopy((predicted, gold, CATALOG))
    diagnostic = analyze(predicted, gold, same_state=True)
    assert diagnostic['error_types'] == ['extra_calls']
    assert diagnostic['call_counts'] == {'predicted': 3, 'gold': 2, 'matched': 2,
                                         'missing': 0, 'extra': 1, 'wrong_arguments': 0}
    assert len(diagnostic['extra_prediction_call_indices']) == 1
    assert call_f1(predicted, gold, CATALOG)[1] == len(diagnostic['matched_call_indices'])
    assert (predicted, gold, CATALOG) == original


def test_diagnostics_use_maximum_matching_and_scorer_fuzzy_rules():
    catalog = {'app.tool': {'parameters': {'properties': {'text': {'x-match': 'fuzzy'}}}}}
    def fuzzy(text): return call('app.tool', text=text)
    gold = [fuzzy('a b c'), fuzzy('a b d')]
    predicted = [fuzzy('a b c d'), fuzzy('a c')]
    diagnostic = analyze(predicted, gold, same_state=True, catalog=catalog)
    assert not diagnostic['error_types']
    assert diagnostic['call_counts']['matched'] == 2
    assert analyze([fuzzy('a b c x y')], [fuzzy('a b c d e')], catalog=catalog)['wrong_argument_pairs']
    assert not analyze([fuzzy('CAFE\u0301')], [fuzzy('café')], same_state=True, catalog=catalog)['error_types']


def test_wrong_arguments_overlap_missing_extra_and_describe_each_difference():
    predicted = [call('smart_home.light_set', device_id=1, power='off', extra=True)]
    gold = [call('smart_home.light_set', device_id='1', power='on', required='value')]
    diagnostic = analyze(predicted, gold)
    assert set(diagnostic['error_types']) == {'state_hash_mismatch', 'missing_calls', 'extra_calls', 'wrong_arguments'}
    pair = diagnostic['wrong_argument_pairs'][0]
    assert pair['prediction_index'] == pair['gold_index'] == 0
    assert {d['argument']: d['kind'] for d in pair['differences']} == {
        'device_id': 'type_mismatch', 'power': 'value_mismatch', 'extra': 'unexpected', 'required': 'missing'}
    assert not diagnostic['unattempted_gold_call_indices']  # Attempted, not evidence of stopping early.
    assert state_tool_score(predicted, gold, 'wrong', 'right', CATALOG) == 0


def test_full_matches_take_priority_before_same_name_argument_diagnosis():
    predicted = [call('smart_home.light_set', device_id='wrong'), call('smart_home.light_set', device_id='b')]
    gold = [call('smart_home.light_set', device_id='b'), call('smart_home.light_set', device_id='a')]
    diagnostic = analyze(predicted, gold)
    assert diagnostic['matched_call_indices'] == [{'prediction_index': 1, 'gold_index': 0}]
    assert diagnostic['wrong_argument_pairs'][0]['prediction_index'] == 0
    assert diagnostic['wrong_argument_pairs'][0]['gold_index'] == 1


def test_wrong_tool_is_missing_and_extra_without_claiming_argument_error():
    diagnostic = analyze([call('other.tool')], [call()])
    assert diagnostic['unattempted_gold_call_indices'] == [0]
    assert not diagnostic['wrong_argument_pairs']
    assert set(diagnostic['error_types']) == {'state_hash_mismatch', 'missing_calls', 'extra_calls', 'early_stop'}


@pytest.mark.parametrize('stop,early', [('done', True), ('max_calls', False), ('max_invalid', False),
                                     ('time_limit', False), ('ask_missing_info', False), ('unknown', False)])
def test_early_stop_requires_explicit_done_wrong_state_and_unattempted_gold(stop, early):
    diagnostic = analyze([], [call()], stop=stop)
    assert ('early_stop' in diagnostic['error_types']) is early
    assert 'early_stop' not in analyze([], [call()], stop=stop, same_state=True)['error_types']


def test_hash_mismatch_is_reported_even_when_empty_gold_scoring_rule_passes():
    diagnostic = analyze([], [])
    assert diagnostic['error_types'] == ['state_hash_mismatch']
    assert state_tool_score([], [], 'wrong', 'right', CATALOG) == 1
    assert analyze([call()], [], same_state=True)['error_types'] == ['extra_calls']


def test_summary_counts_samples_and_calls_separately_and_groups_transcript_sources():
    summary = DiagnosticSummary()
    summary.add(analyze([], [call(), call()]), 'asr')
    summary.add(analyze([call(), call()], [call()], same_state=True), 'audio_text')
    summary.add(analyze([], [], same_state=True), 'legacy')
    report = summary.as_dict()
    assert report['samples'] == 3
    assert report['error_samples']['missing_calls'] == 1
    assert report['call_counts']['missing'] == 2
    assert report['error_rates']['missing_calls'] == pytest.approx(1 / 3)
    assert report['by_transcript_source']['asr']['error_samples']['early_stop'] == 1
    assert report['by_transcript_source']['audio_text']['error_samples']['extra_calls'] == 1
    assert report['by_transcript_source']['unknown']['samples'] == 1
    assert summary.metrics()['eval_missing_calls'] == 2
    assert summary.metrics()['eval_early_stop_samples'] == 1
    json.dumps(report)  # Report/Trainer metrics contain serializable counters, not Counter objects.


@pytest.mark.parametrize('mode,reason', [('done', 'done'), ('max_calls', 'max_calls'),
    ('max_invalid', 'max_invalid'), ('time_limit', 'time_limit'),
    ('no_tools', 'no_available_tools'), ('ask', 'ask_missing_info')])
def test_agent_records_stop_reason_without_additional_model_or_tool_calls(mode, reason):
    entry = item()
    config = AgentConfig()
    if mode == 'max_calls':
        config.max_calls = 1
        outputs = ['{"action":"call","name":"smart_home.get_status","arguments":{}}']
    elif mode == 'max_invalid':
        config.max_invalid = 1
        outputs = ['not a decision']
    elif mode == 'time_limit':
        config.sample_seconds = 0
        outputs = []
    elif mode == 'no_tools':
        entry['context']['installed_apps'] = {}
        outputs = []
    elif mode == 'ask':
        outputs = ['{"action":"call","name":"sys.ask_missing_info","arguments":'
                   '{"tool_name":"smart_home.light_set","params":["device_id"]}}']
    else:
        outputs = ['{"action":"done"}']
    class LLM(StubLLM):
        count = 0
        def generate(self, system, user):
            self.count += 1
            return super().generate(system, user)
    llm, env = LLM(outputs), FakeEnv()
    stats = LifeMateAgent(StubASR(), llm, CATALOG, config).run(entry, env, Path('/unused'), 'train')
    assert stats['stop_reason'] == reason
    assert llm.count == len(outputs)
    assert len(env.calls) == (1 if mode in ('max_calls', 'ask') else 0)
