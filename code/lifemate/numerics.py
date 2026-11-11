"""Fail before optimizer mutation on nonfinite loss/gradients; no extra model.

Imports of Torch/Transformers stay lazy so Unsloth can patch them first. Debug
batches contain tokenized user data and stay beside the training artifacts.
"""
from collections import defaultdict
from copy import deepcopy
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import uuid
import weakref


NUMERICS_POLICY = 'finite-loss-and-gradients-v1'
DEBUG_BATCH_KEY = '_lifemate_debug_samples'


class NonFiniteTrainingError(FloatingPointError):
    pass


def _number(value):
    if hasattr(value, 'detach'):
        value = value.detach().item()
    value = float(value)
    return value if math.isfinite(value) else str(value)


class NumericsGuard:
    def __init__(self, trainer, context=None):
        self.trainer = weakref.ref(trainer)
        self.output_dir = Path(trainer.args.output_dir)
        self.context = dict(context or {})
        self.window = []
        self.handles = []
        self.in_training_step = False

    @staticmethod
    def batch(metadata, inputs):
        import torch
        tensors = {key: value.detach() for key, value in inputs.items()
                   if key in {'input_ids', 'attention_mask', 'labels', 'loss_weights'} and isinstance(value, torch.Tensor)}
        return {'samples': deepcopy(metadata or []), 'tensors': tensors}

    def start_microbatch(self, metadata, inputs):
        self.window.append(self.batch(metadata, inputs))

    def check_loss(self, loss, *, phase, metadata=None, inputs=None):
        import torch
        finite = bool(torch.isfinite(loss.detach()).all().item()) if isinstance(loss, torch.Tensor) else math.isfinite(float(loss))
        if not finite:
            batches = None if phase == 'train' else [self.batch(metadata, inputs or {})]
            self.fail(phase+'_loss', scalar=loss, batches=batches)
        if phase == 'train' and self.window and 'loss' not in self.window[-1]:
            self.window[-1]['loss'] = _number(loss)

    def check_scalar(self, value, stage):
        if value is not None and not math.isfinite(float(_number(value))):
            self.fail(stage, scalar=value)

    def check_gradients(self, model, stage, optimizer=None):
        import torch
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        if optimizer is None:
            parameters = list(model.named_parameters())
        else:
            parameters = [(names.get(id(p), f'optimizer.group_{i}.param_{j}'), p)
                          for i, group in enumerate(optimizer.param_groups) for j, p in enumerate(group['params'])]
        grouped, seen = defaultdict(list), set()
        for name, parameter in parameters:
            if parameter.grad is None or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            grad = parameter.grad.detach()
            values = grad.coalesce().values() if grad.is_sparse else grad
            grouped[str(values.device)].append((name, values, torch.isfinite(values).all()))
        bad = []
        # One synchronization per device on the healthy path, not per parameter.
        for entries in grouped.values():
            flags = torch.stack([entry[2] for entry in entries])
            if bool(flags.all().item()):
                continue
            for (name, values, _), finite in zip(entries, flags.cpu().tolist()):
                if not finite:
                    bad.append({'name': name, 'shape': list(values.shape), 'dtype': str(values.dtype),
                                'device': str(values.device), 'elements': values.numel(),
                                'nan_elements': int(torch.isnan(values).sum().item()),
                                'inf_elements': int(torch.isinf(values).sum().item())})
        if bad:
            self.fail(stage, gradients=bad)

    def attach_optimizer(self, optimizer, model):
        # Accelerate wraps AdamW; attach to the actual optimizer that mutates the
        # parameters, so a patched training loop cannot skip the final barrier.
        seen = set()
        while getattr(optimizer, 'optimizer', None) is not None and id(optimizer) not in seen:
            seen.add(id(optimizer))
            optimizer = optimizer.optimizer
        self.remove_hooks()
        if not callable(getattr(optimizer, 'register_step_pre_hook', None)):
            raise RuntimeError('Numerics guard requires an optimizer supporting step pre-hooks')
        self.handles.append(optimizer.register_step_pre_hook(
            lambda opt, args, kwargs: self.check_gradients(model, 'optimizer_step', opt)))
        self.handles.append(optimizer.register_step_post_hook(
            lambda opt, args, kwargs: self.window.clear()))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def close(self):
        self.remove_hooks()
        self.window.clear()

    def fail(self, stage, *, scalar=None, gradients=None, batches=None):
        import torch
        from safetensors.torch import save_file
        trainer = self.trainer()
        state = trainer.state
        target_step = int(state.global_step)+1
        batches = self.window if batches is None else batches
        samples = [sample for batch in batches for sample in batch['samples']]
        message = (f'Nonfinite training numerics at {stage}, before optimizer step {target_step}; '
                   f'samples={[s.get("sample_id") for s in samples]}; '
                   f'parameters={[g["name"] for g in gradients or []]}')
        folder = self.output_dir/'debug_numerics'/f'step-{target_step:06d}-{uuid.uuid4().hex[:8]}'
        logging.critical(message)
        try:
            folder.mkdir(parents=True, exist_ok=False)
            tensors, descriptions = {}, []
            for index, batch in enumerate(batches, 1):
                description = {'microbatch': index, 'samples': batch['samples'], 'shapes': {}}
                if 'loss' in batch:
                    description['loss'] = batch['loss']
                for name, value in batch['tensors'].items():
                    tensors[f'microbatch_{index:03d}.{name}'] = value.to('cpu').contiguous().clone()
                    description['shapes'][name] = list(value.shape)
                if 'labels' in batch['tensors']:
                    description['supervised_tokens'] = int((batch['tensors']['labels'] != -100).sum().item())
                descriptions.append(description)
            if tensors:
                save_file(tensors, str(folder/'batches.safetensors'))
            versions = {}
            for package in ('torch', 'transformers', 'accelerate', 'peft', 'unsloth', 'unsloth-zoo',
                            'triton', 'flash-linear-attention', 'fla-core', 'causal-conv1d'):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
            device = next(trainer.model.parameters()).device
            gpu = ({'name': torch.cuda.get_device_name(device),
                    'capability': list(torch.cuda.get_device_capability(device))} if device.type == 'cuda' else None)
            report = {'policy': NUMERICS_POLICY, 'stage': stage, 'completed_optimizer_steps': int(state.global_step),
                      'target_optimizer_step': target_step, 'epoch': state.epoch,
                      'gradient_accumulation_steps': trainer.args.gradient_accumulation_steps,
                      'scalar': _number(scalar) if scalar is not None else None,
                      'bad_gradients': gradients or [], 'microbatches': descriptions,
                      'context': self.context, 'versions': versions,
                      'device': str(device), 'cuda_runtime': torch.version.cuda, 'gpu': gpu,
                      'batch_file': 'batches.safetensors' if tensors else None,
                      'note': 'No optimizer update/checkpoint was performed by the failing step; tokenized user data is included.'}
            (folder/'failure.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        except Exception as exc:
            # Disk-full/report errors must never allow a corrupt optimizer step.
            raise NonFiniteTrainingError(message+f'; could not write diagnostics to {folder}: {exc}') from exc
        raise NonFiniteTrainingError(message+f'; diagnostics: {folder/"failure.json"}')


def guarded_trainer_class(base_trainer):
    """Subclass the already patched Trainer; preserve Unsloth's training methods."""
    from transformers import TrainerCallback

    class GuardCallback(TrainerCallback):
        def __init__(self, guard):
            self.guard = guard

        def on_train_begin(self, args, state, control, optimizer=None, model=None, **kwargs):
            self.guard.attach_optimizer(optimizer, model)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            self.guard.check_gradients(model, 'before_optimizer')

        def on_step_end(self, args, state, control, **kwargs):
            self.guard.window.clear()  # Also covers optimizer steps skipped by AMP.

    class GuardedTrainer(base_trainer):
        def __init__(self, *args, numerics_context=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.numerics_guard = NumericsGuard(self, numerics_context)
            self.add_callback(GuardCallback(self.numerics_guard))

        def training_step(self, model, inputs, *args, **kwargs):
            metadata = inputs.pop(DEBUG_BATCH_KEY, None)
            guard = self.numerics_guard
            guard.start_microbatch(metadata, inputs)
            guard.in_training_step = True
            try:
                loss = super().training_step(model, inputs, *args, **kwargs)
                guard.check_loss(loss, phase='train')
                guard.check_gradients(model, 'after_backward')
                return loss
            finally:
                guard.in_training_step = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            metadata = inputs.pop(DEBUG_BATCH_KEY, None)
            diagnostic_inputs = dict(inputs)
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            self.numerics_guard.check_loss(loss,
                phase='train' if self.numerics_guard.in_training_step else 'eval', metadata=metadata, inputs=diagnostic_inputs)
            return result

        def _clip_grad_norm(self, model):
            self.numerics_guard.check_gradients(model, 'before_gradient_clipping')
            norm = super()._clip_grad_norm(model)
            self.numerics_guard.check_scalar(norm, 'gradient_norm')
            return norm

        def _get_grad_norm(self, model, grad_norm=None):
            norm = super()._get_grad_norm(model, grad_norm=grad_norm)
            self.numerics_guard.check_scalar(norm, 'gradient_norm')
            return norm

        def train(self, *args, **kwargs):
            try:
                return super().train(*args, **kwargs)
            finally:
                self.numerics_guard.close()

    return GuardedTrainer
