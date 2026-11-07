"""Real backward/optimizer tests: stop corrupt updates and retain exact batches."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainerCallback, TrainingArguments, PretrainedConfig

from lifemate.numerics import (DEBUG_BATCH_KEY, NUMERICS_POLICY, NumericsGuard,
                               NonFiniteTrainingError, guarded_trainer_class)
from lifemate.sft import AssistantCollator, SupervisedDataset
from test_sft import TemplateTokenizer


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, bad_value):
        ctx.bad_value = bad_value
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, ctx.bad_value), None


class TinyLoss(torch.nn.Module):
    def __init__(self, *, bad_at=None, bad_value=float('nan'), bad_loss=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self.config = PretrainedConfig()
        self.bad_at, self.bad_value, self.bad_loss = bad_at, bad_value, bad_loss

    # An explicit signature verifies debug metadata never leaks to the model.
    def forward(self, input_ids, labels, attention_mask=None):
        bad = self.bad_at is not None and bool(input_ids.eq(self.bad_at).any())
        weight = BadGradient.apply(self.weight, self.bad_value) if bad and not self.bad_loss else self.weight
        logits = weight * input_ids.float()
        mask = labels.ne(-100)
        loss = ((logits-labels.float()).square()*mask).sum()/mask.sum()
        if bad and self.bad_loss:
            loss = loss*self.bad_value
        return {'loss': loss, 'logits': logits}


def dataset(count=8):
    return [{'input_ids': [i+1, 1], 'attention_mask': [1, 1], 'labels': [-100, 2],
             DEBUG_BATCH_KEY: {'sample_id': f'train_{i}', 'sft_step': i, 'source_line': i+1,
                               'transcript_source': 'asr', 'augmentation_variant': i % 3}}
            for i in range(count)]


def make_trainer(tmp_path, model, *, guarded=True, accumulation=2, steps=2, data=None):
    base = guarded_trainer_class(Trainer) if guarded else Trainer
    class SequentialTrainer(base):
        def _get_train_sampler(self, *args, **kwargs):
            return SequentialSampler(self.train_dataset)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, max_steps=steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
        learning_rate=.01, lr_scheduler_type='constant', max_grad_norm=1.,
        eval_strategy='no', save_strategy='no', logging_steps=1, logging_nan_inf_filter=False,
        report_to='none', remove_unused_columns=False, disable_tqdm=True, dataloader_num_workers=0, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    data = data or dataset()
    if not guarded:
        data = [{k: v for k, v in row.items() if k != DEBUG_BATCH_KEY} for row in data]
    trainer = SequentialTrainer(model=model, args=args, train_dataset=data,
        data_collator=AssistantCollator(0), optimizers=(optimizer, None))
    return trainer, optimizer


def failure(tmp_path):
    files = list(Path(tmp_path).glob('debug_numerics/*/failure.json'))
    assert len(files) == 1
    return files[0], json.loads(files[0].read_text())


def test_finite_guard_preserves_training_weights_optimizer_and_loss(tmp_path):
    models = [TinyLoss(), TinyLoss()]
    runs = []
    for guarded, model in zip((False, True), models):
        trainer, optimizer = make_trainer(tmp_path/str(guarded), model, guarded=guarded)
        trainer.train()
        runs.append((trainer, optimizer))
    torch.testing.assert_close(models[0].weight, models[1].weight, rtol=0, atol=0)
    assert runs[0][0].state.global_step == runs[1][0].state.global_step == 2
    losses = [[r['loss'] for r in trainer.state.log_history if 'loss' in r] for trainer, _ in runs]
    assert losses[0] == losses[1]
    assert not list(tmp_path.glob('**/failure.json'))
    assert not runs[1][0].numerics_guard.handles
    assert not runs[1][0].numerics_guard.window


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), -float('inf')])
def test_bad_gradient_is_caught_before_clipping_or_optimizer_and_records_accumulation(tmp_path, bad_value, monkeypatch):
    from safetensors.torch import load_file
    model = TinyLoss(bad_at=2, bad_value=bad_value)
    original = model.weight.detach().clone()
    trainer, optimizer = make_trainer(tmp_path, model)
    clipped, stepped = [], []
    original_clip = Trainer._clip_grad_norm
    def clip(self, *args, **kwargs):
        clipped.append(1)
        return original_clip(self, *args, **kwargs)
    monkeypatch.setattr(Trainer, '_clip_grad_norm', clip)
    optimizer.register_step_post_hook(lambda *args: stepped.append(1))
    with pytest.raises(NonFiniteTrainingError, match='after_backward.*optimizer step 1'):
        trainer.train()
    assert not clipped and not stepped and not optimizer.state
    assert torch.equal(model.weight, original)
    assert trainer.state.global_step == 0
    path, report = failure(tmp_path)
    assert report['policy'] == NUMERICS_POLICY and report['stage'] == 'after_backward'
    assert report['completed_optimizer_steps'] == 0 and report['target_optimizer_step'] == 1
    assert [s['sample_id'] for b in report['microbatches'] for s in b['samples']] == ['train_0', 'train_1']
    assert report['bad_gradients'][0]['name'] == 'weight'
    assert report['bad_gradients'][0]['nan_elements' if str(bad_value)=='nan' else 'inf_elements'] == 1
    assert report['microbatches'][1]['supervised_tokens'] == 1
    tensors = load_file(str(path.parent/report['batch_file']))
    assert tensors['microbatch_002.input_ids'].tolist() == [[2, 1]]
    assert tensors['microbatch_002.labels'].tolist() == [[-100, 2]]
    assert not trainer.numerics_guard.handles and not trainer.numerics_guard.window


def test_failing_step_three_preserves_previous_updates_and_reports_only_current_window(tmp_path):
    trainer, optimizer = make_trainer(tmp_path, TinyLoss(bad_at=6), steps=4)
    trainer.args.save_strategy = 'steps'
    trainer.args.save_steps = 1
    states = []
    optimizer.register_step_post_hook(lambda *args: states.append(trainer.model.weight.detach().clone()))
    with pytest.raises(NonFiniteTrainingError):
        trainer.train()
    assert trainer.state.global_step == len(states) == 2
    assert torch.equal(trainer.model.weight, states[-1])
    assert (tmp_path/'checkpoint-2/trainer_state.json').is_file()
    assert not (tmp_path/'checkpoint-3').exists()
    _, report = failure(tmp_path)
    assert report['target_optimizer_step'] == 3 and report['completed_optimizer_steps'] == 2
    assert [s['sample_id'] for b in report['microbatches'] for s in b['samples']] == ['train_4', 'train_5']


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_forward_loss_fails_before_backward(tmp_path, value):
    trainer, optimizer = make_trainer(tmp_path, TinyLoss(bad_at=1, bad_value=value, bad_loss=True))
    with pytest.raises(NonFiniteTrainingError, match='train_loss'):
        trainer.train()
    assert trainer.model.weight.grad is None and not optimizer.state
    _, report = failure(tmp_path)
    assert report['stage'] == 'train_loss' and report['scalar'] in {'nan', 'inf'}


def test_optimizer_pre_hook_catches_mutation_after_callback_checks(tmp_path):
    trainer, optimizer = make_trainer(tmp_path, TinyLoss())
    initial = trainer.model.weight.detach().clone()
    class CorruptAfterGuard(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            model.weight.grad.fill_(float('nan'))
    trainer.add_callback(CorruptAfterGuard())
    with pytest.raises(NonFiniteTrainingError, match='optimizer_step'):
        trainer.train()
    assert not optimizer.state and torch.equal(trainer.model.weight, initial)
    assert failure(tmp_path)[1]['stage'] == 'optimizer_step'


def test_nonfinite_norm_is_rejected_even_if_grad_values_are_finite(tmp_path, monkeypatch):
    trainer, optimizer = make_trainer(tmp_path, TinyLoss())
    monkeypatch.setattr(Trainer, '_clip_grad_norm', lambda *args: torch.tensor(float('inf')))
    with pytest.raises(NonFiniteTrainingError, match='gradient_norm'):
        trainer.train()
    assert not optimizer.state
    assert failure(tmp_path)[1]['stage'] == 'gradient_norm'


def test_nonfinite_reported_norm_cannot_reach_optimizer(tmp_path, monkeypatch):
    trainer, optimizer = make_trainer(tmp_path, TinyLoss())
    monkeypatch.setattr(Trainer, '_get_grad_norm', lambda *args, **kwargs: float('nan'))
    with pytest.raises(NonFiniteTrainingError, match='gradient_norm'):
        trainer.train()
    assert not optimizer.state


def test_report_write_error_still_blocks_optimizer(tmp_path, monkeypatch):
    import safetensors.torch
    trainer, optimizer = make_trainer(tmp_path, TinyLoss(bad_at=1))
    def no_space(*args, **kwargs):
        raise OSError('No space left on device')
    monkeypatch.setattr(safetensors.torch, 'save_file', no_space)
    with pytest.raises(NonFiniteTrainingError, match='could not write diagnostics.*No space'):
        trainer.train()
    assert not optimizer.state and not trainer.numerics_guard.handles


def test_eval_nonfinite_loss_is_not_logged_as_a_good_checkpoint(tmp_path):
    trainer, optimizer = make_trainer(tmp_path, TinyLoss(bad_at=1, bad_loss=True))
    trainer.eval_dataset = dataset(2)
    with pytest.raises(NonFiniteTrainingError, match='eval_loss'):
        trainer.evaluate()
    report = failure(tmp_path)[1]
    assert report['stage'] == 'eval_loss' and report['microbatches'][0]['samples'][0]['sample_id'] == 'train_0'
    assert not optimizer.state


def test_debug_metadata_preserves_assistant_mask_and_is_separate_from_model_tensors():
    example = {'sample_id': 'train_debug', 'step': 2, 'transcript_source': 'asr',
               'augmentation': {'variant': 1}, 'messages': [
                   {'role': role, 'content': ''} for role in ('system', 'user', 'assistant')]}
    plain = SupervisedDataset([example], TemplateTokenizer(), 6)
    debug = SupervisedDataset([example], TemplateTokenizer(), 6, include_debug_metadata=True, source_path='/train.jsonl')
    assert {k: v for k, v in debug[0].items() if k != DEBUG_BATCH_KEY} == plain[0]
    batch = AssistantCollator(0)([debug[0]])
    assert batch['labels'].tolist() == [[-100, -100, -100, 7, 8, 9]]
    assert batch[DEBUG_BATCH_KEY][0]['source_line'] == 1
    batch[DEBUG_BATCH_KEY][0]['sample_id'] = 'mutated'
    assert debug[0][DEBUG_BATCH_KEY]['sample_id'] == 'train_debug'


def test_optimizer_wrappers_are_unwrapped_and_existing_hooks_survive(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(2))
    model = torch.nn.ParameterList([parameter])
    trainer = SimpleNamespace(args=SimpleNamespace(output_dir=str(tmp_path), gradient_accumulation_steps=1),
                              state=SimpleNamespace(global_step=0, epoch=0))
    # SimpleNamespace cannot be weak-referenced; real Trainer instances can.
    class Owner: pass
    owner = Owner(); owner.args, owner.state = trainer.args, trainer.state
    guard = NumericsGuard(owner)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    calls = []
    optimizer.register_step_post_hook(lambda *args: calls.append(1))
    guard.attach_optimizer(SimpleNamespace(optimizer=SimpleNamespace(optimizer=optimizer)), model)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    guard.close()
    optimizer.step()
    assert calls == [1, 1] and torch.allclose(parameter, torch.full_like(parameter, .8))


def test_existing_runtime_training_step_patch_is_preserved(tmp_path, monkeypatch):
    original = Trainer.training_step
    calls = []
    def patched(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Trainer, 'training_step', patched)
    trainer, _ = make_trainer(tmp_path, TinyLoss(), steps=1)
    trainer.train()
    assert calls == [1, 1] and trainer.state.global_step == 1


def test_real_hybrid_qwen_lora_gradient_failure_is_blocked(tmp_path):
    from peft import LoraConfig, get_peft_model
    from test_prefix_cache import tiny_model
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = get_peft_model(tiny_model(conditional=True), LoraConfig(
            r=2, task_type='CAUSAL_LM', target_modules=['in_proj_qkv', 'k_proj']))
        name, parameter = next((n, p) for n, p in model.named_parameters() if p.requires_grad)
        handle = parameter.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        originals = {n: p.detach().clone() for n, p in model.named_parameters()}
        trainer, optimizer = make_trainer(tmp_path, model, steps=1)
        with pytest.raises(NonFiniteTrainingError):
            trainer.train()
        assert not optimizer.state
        assert all(torch.equal(originals[n], p) for n, p in model.named_parameters())
        assert name in [row['name'] for row in failure(tmp_path)[1]['bad_gradients']]
        handle.remove()
    finally:
        torch.set_num_threads(previous)


def test_numerics_module_does_not_import_frameworks_before_unsloth():
    import subprocess, sys
    result = subprocess.run([sys.executable, '-c',
        'import sys; import lifemate.numerics; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
