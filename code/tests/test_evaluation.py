import json
import sys
from types import SimpleNamespace

import pytest

from lifemate.evaluation import StateToolEvaluator, score_callback
from lifemate.sft import UnexposedToolError, split_records
from test_baseline import CATALOG, FakeEnv
from test_sft import labeled


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import prepare_sft
    source = tmp_path / 'train'
    source.mkdir()
    records = [dict(labeled(), id=f'train_{i}') for i in range(4)]
    (source / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (source / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': list(CATALOG.values())}]}))
    instances = []

    class Env(FakeEnv):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.power = 'off'
            instances.append(self)

        def call(self, name, args):
            result = super().call(name, args)
            if name == 'smart_home.light_set':
                self.power = args.get('power', self.power)
            return result

        def get_state_hash(self): return f'real-fixture-api:{self.power}'

    monkeypatch.setitem(sys.modules, 'zaic_tools', SimpleNamespace(ToolEnv=Env))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    out = tmp_path / 'sft'
    monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--data-root', str(source), '--output-dir', str(out)])
    prepare_sft.main()
    return out, instances, source, records


def test_rollouts_use_fresh_envs_without_gold_leakage_and_apply_state_gate(prepared, tmp_path):
    out, instances, _, _ = prepared
    manifest = json.loads((out / 'manifest.json').read_text())
    evaluator = StateToolEvaluator(out, manifest)
    saved = evaluator.rows[0]
    assert 'label' not in saved['item']
    assert saved['gold_state_hash'] == 'real-fixture-api:on'

    class LLM:
        power = 'on'

        def generate(self, system, user):
            payload = json.loads(user)
            assert 'gold_state_hash' not in payload and 'gold_tool_calls' not in payload
            observations = payload['previous_actions_and_results']
            if not observations:
                return '{"action":"call","name":"smart_home.get_status","arguments":{}}'
            if len(observations) == 1:
                return json.dumps({'action': 'call', 'name': 'smart_home.light_set',
                    'arguments': {'device_id': 'light_living', 'power': self.power}})
            return '{"action":"done"}'

    llm = LLM()
    initial = len(instances)
    try:
        good = evaluator.evaluate(llm, tmp_path / 'good.jsonl')
        assert good['eval_state_tool_score'] == 100
        assert good['eval_rollout_samples'] == 1  # Counts original IDs, not SFT steps.
        llm.power = 'off'
        bad = evaluator.evaluate(llm, tmp_path / 'bad.jsonl')
        assert bad['eval_state_tool_score'] == 0
        assert bad['eval_tool_call_f1'] == 50
        assert len(instances) == initial + 2
        assert instances[-1] is not instances[-2]
    finally:
        evaluator.close()


def test_selection_requires_unchanged_validation_data(prepared):
    out, _, _, _ = prepared
    manifest = json.loads((out / 'manifest.json').read_text())
    with (out / 'validation_rollouts.jsonl').open('a') as f:
        f.write('{}\n')
    with pytest.raises(ValueError, match='changed'):
        StateToolEvaluator(out, manifest)


def test_mean_is_over_original_samples_not_sft_steps_or_calls(prepared, tmp_path, monkeypatch):
    import prepare_sft
    _, _, source, records = prepared
    _, valid = split_records(records, 0.5)
    empty = next(r for r in records if r['id'] == valid[0]['id'])
    empty['label']['tool_calls'] = []
    empty['context']['installed_apps'] = {}
    # A prediction must now read before writing. Make the nonempty gold have
    # one extra post-write read, retaining a partial-score case for aggregation.
    for record in records:
        if record['label']['tool_calls']:
            record['label']['tool_calls'].append({'name': 'smart_home.get_status', 'arguments': {}})
    (source / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    out = tmp_path / 'two-samples'
    monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--data-root', str(source),
        '--output-dir', str(out), '--validation-fraction', '0.5'])
    prepare_sft.main()
    evaluator = StateToolEvaluator(out, json.loads((out / 'manifest.json').read_text()))

    class LLM:
        def generate(self, system, user):
            observations = json.loads(user)['previous_actions_and_results']
            if len(observations) >= 2:
                return '{"action":"done"}'
            if not observations:
                return '{"action":"call","name":"smart_home.get_status","arguments":{}}'
            return '{"action":"call","name":"smart_home.light_set","arguments":{"device_id":"light_living","power":"on"}}'

    try:
        result = evaluator.evaluate(LLM(), tmp_path / 'mean.jsonl')
        assert result['eval_rollout_samples'] == 2
        assert result['eval_state_tool_score'] == pytest.approx(100 * (1 + 4 / 5) / 2)
    finally:
        evaluator.close()


def test_supplied_gold_hash_must_agree_with_replay(prepared, tmp_path, monkeypatch):
    import prepare_sft
    _, _, source, records = prepared
    _, valid = split_records(records)
    next(r for r in records if r['id'] == valid[0]['id'])['label']['state_hash'] = 'wrong-gold'
    (source / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--data-root', str(source),
        '--output-dir', str(tmp_path / 'bad-hash')])
    with pytest.raises(ValueError, match='differs from the provided gold hash'):
        prepare_sft.main()


def test_validation_samples_cannot_be_silently_skipped(prepared, tmp_path, monkeypatch):
    import prepare_sft
    _, _, source, records = prepared
    _, valid = split_records(records)
    original = prepare_sft.replay_sample

    def replay(record, *args, **kwargs):
        if record['id'] == valid[0]['id']:
            raise UnexposedToolError('unavailable')
        return original(record, *args, **kwargs)

    monkeypatch.setattr(prepare_sft, 'replay_sample', replay)
    monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--data-root', str(source),
        '--output-dir', str(tmp_path / 'skip'), '--missing-tool-policy', 'skip'])
    with pytest.raises(ValueError):
        prepare_sft.main()


def test_best_checkpoint_uses_score_even_when_loss_improves(tmp_path):
    import torch
    from transformers import Trainer, TrainingArguments
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to='none',
        eval_strategy='steps', save_strategy='steps', load_best_model_at_end=True,
        metric_for_best_model='state_tool_score', greater_is_better=True)
    trainer = Trainer(model=torch.nn.Linear(1, 1), args=args, eval_dataset=[{'x': 0}])
    trainer.state.global_step = 1
    assert trainer._determine_best_metric({'eval_state_tool_score': 80, 'eval_loss': 5}, trial=None)
    trainer.state.global_step = 2
    assert not trainer._determine_best_metric({'eval_state_tool_score': 60, 'eval_loss': 1}, trial=None)
    assert trainer.state.best_metric == 80
    assert trainer.state.best_global_step == 1


@pytest.mark.parametrize('fail', [False, True])
def test_callback_restores_training_mode_and_unsloth_state(tmp_path, fail):
    import torch
    from transformers import TrainerState, TrainerControl
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'a': 0})))
    model = torch.nn.Embedding(1, 2)
    model.get_input_embeddings = lambda: model
    model.config = SimpleNamespace(use_cache=False)
    calls = []

    def evaluate(llm, path):
        assert not model.training and model.config.use_cache
        if fail:
            raise RuntimeError('GPU/evaluation failure')
        return {'eval_state_tool_score': 75}

    fast = SimpleNamespace(for_inference=lambda m: calls.append('inference'),
                           for_training=lambda m: calls.append('training'))
    callback = score_callback(SimpleNamespace(evaluate=evaluate), tok, tmp_path, fast)
    state = TrainerState(global_step=1)
    state.max_steps = 1
    final_control = callback.on_step_end(SimpleNamespace(eval_strategy='steps'), state, TrainerControl())
    assert final_control.should_evaluate and final_control.should_save
    metrics = {'eval_loss': 2}
    if fail:
        with pytest.raises(RuntimeError):
            callback.on_evaluate(None, state, TrainerControl(), model=model, metrics=metrics)
    else:
        callback.on_evaluate(None, state, TrainerControl(), model=model, metrics=metrics)
        assert metrics['eval_state_tool_score'] == 75
    assert model.training and not model.config.use_cache
    assert calls == ['inference', 'training']
