import pytest

from lifemate.scoring import calls_match, max_matches, state_tool_score, token_f1

CATALOG = {'app.tool': {'parameters': {'properties': {'text': {'x-match': 'fuzzy'},
                                                     'id': {'type': 'integer'}}}}}


def call(text='a b c', **kwargs):
    return {'name': 'app.tool', 'arguments': {'text': text, **kwargs}}


def test_state_gate_and_empty_gold_rule():
    gold = [call()]
    assert state_tool_score(gold, gold, 'wrong', 'right', CATALOG) == 0
    assert state_tool_score(gold, gold, 'right', 'right', CATALOG) == 1
    assert state_tool_score([], gold, 'right', 'right', CATALOG) == 0
    assert state_tool_score([], [], 'different', 'hash', CATALOG) == 1
    assert state_tool_score(gold, [], 'hash', 'hash', CATALOG) == 0


def test_fuzzy_strict_threshold_normalization_and_exact_arguments():
    assert token_f1('CAFE\u0301', 'café') == 1
    assert token_f1('a b c x y', 'a b c d e') == 0.6
    assert not calls_match(call('a b c x y'), call('a b c d e'), CATALOG)
    assert calls_match(call('Café', id=2), call('cafe\u0301', id=2), CATALOG)
    assert not calls_match(call(id=True), call(id=1), CATALOG)
    assert not calls_match(call(id={'nested': [True]}), call(id={'nested': [1]}), CATALOG)


def test_maximum_matching_is_order_independent_and_counts_duplicates():
    # The first prediction matches both golds; the second can only match gold 0.
    golds = [call('a b c'), call('a b d')]
    preds = [call('a b c d'), call('a c')]
    assert max_matches(preds, golds, CATALOG) == 2
    assert max_matches(preds[::-1], golds[::-1], CATALOG) == 2
    assert state_tool_score([call(), call(), call()], [call()], 'same', 'same', CATALOG) == 0.5
    # An unmatched schema-invalid submitted call contributes to U as well.
    bad = {'name': 'app.tool', 'arguments': {'invalid': True}}
    assert state_tool_score([call(), bad], [call()], 'same', 'same', CATALOG) == pytest.approx(2 / 3)


def test_missing_information_scored_as_a_normal_call():
    ask = {'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'app.tool', 'params': ['id']}}
    assert state_tool_score([ask], [ask], 's', 's', {}) == 1
