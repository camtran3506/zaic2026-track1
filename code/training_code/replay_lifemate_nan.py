#!/usr/bin/env python3
"""Probe saved LifeMate inputs on a fresh base + LoRA, without optimizer updates.

Uses the repository's loader settings and numerics guard. This is not an exact
reproduction of step 91: the input dump does not include step-90 weights or RNG.
Run each backend in a separate process (Unsloth patches global framework code).
"""
import argparse
import contextlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import traceback


class ReplayComplete(Exception):
    """All selected backward passes finished; intercept optimizer.step()."""


def assistant_logits_loss(model, inputs):
    """HF 5.4 Qwen3.5 batch=1: exact masked causal CE using selected logits.

    Keep every input token in the transformer. Only project the hidden states
    immediately preceding supervised labels through the vocabulary head. This
    matches the finite-valued dense objective, but exercises a different head/
    loss execution path, which matters when investigating numerical failures.
    """
    import torch
    from torch.nn import functional as F
    labels = inputs['labels']
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError('Selected logits require batch size 1')
    positions = labels[0, 1:].ne(-100).nonzero(as_tuple=False).flatten()
    if not positions.numel():
        raise ValueError('No supervised causal targets')
    model_inputs = {key: value for key, value in inputs.items() if key != 'labels'}
    outputs = model(**model_inputs, labels=None, logits_to_keep=positions, use_cache=False)
    targets = labels[:, 1:].index_select(1, positions)
    logits = outputs.logits.float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss, outputs


def load_rows(batch_path, failure, selection):
    # Framework imports must remain after Unsloth in main().
    import torch
    from safetensors.torch import load_file
    tensors = load_file(str(batch_path))
    batches = failure['microbatches']
    if selection == 'last':
        batches = batches[-1:]
    rows = []
    for batch in batches:
        prefix = f"microbatch_{batch['microbatch']:03d}"
        values = {key: tensors[prefix+'.'+key] for key in ('input_ids', 'attention_mask', 'labels')}
        ids, mask, labels = (values[key] for key in ('input_ids', 'attention_mask', 'labels'))
        if ids.ndim != 2 or ids.shape != labels.shape or ids.shape != mask.shape:
            raise ValueError(f'{prefix}: invalid tensor shapes')
        if any(value.dtype != torch.int64 for value in values.values()):
            raise ValueError(f'{prefix}: expected int64 tensors')
        if ids.shape[0] != 1 or len(batch['samples']) != 1:
            raise ValueError('This probe preserves the reported batch-size=1 run only')
        supervised = labels.ne(-100)
        if (not bool(labels[:, 1:].ne(-100).any()) or bool(ids.lt(0).any())
                or not bool(mask.eq(1).all()) or not bool(labels[supervised].eq(ids[supervised]).all())):
            raise ValueError(f'{prefix}: invalid IDs, labels or attention mask')
        if list(ids.shape) != batch['shapes']['input_ids'] or int(supervised.sum()) != batch['supervised_tokens']:
            raise ValueError(f'{prefix}: input tensors disagree with failure.json')
        from lifemate.numerics import DEBUG_BATCH_KEY
        row = {key: value[0].tolist() for key, value in values.items()}
        row[DEBUG_BATCH_KEY] = dict(batch['samples'][0], original_microbatch=batch['microbatch'])
        rows.append(row)
    if not rows:
        raise ValueError('No microbatches selected')
    return rows


def use_reference_linear_attention():
    """Disable optional Qwen3.5 kernels before constructing a HF 5.4.0 model."""
    import transformers
    if transformers.__version__ != '5.4.0':
        raise ValueError('--reference-linear-attention is verified for Transformers 5.4.0 only')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    for name in ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
                 'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated'):
        setattr(q, name, None)
    q.is_fast_path_available = False

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app-dir', type=Path, default=Path(__file__).resolve().parents[1], help='Repository code/ folder')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--failure', type=Path, required=True)
    p.add_argument('--batches', type=Path, help='Default: batches.safetensors next to failure.json')
    p.add_argument('--backend', choices=('unsloth', 'transformers'), required=True)
    p.add_argument('--selection', choices=('last', 'window'), default='last')
    p.add_argument('--output-dir', type=Path, required=True, help='New empty probe directory')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--cuda-visible-devices', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0],
                   help='One CUDA GPU index or UUID, selected before framework imports (default first inherited GPU or 0)')
    p.add_argument('--assistant-logits-only', action='store_true',
                   help='HF 5.4 Qwen3.5 only: compute vocabulary logits at supervised causal positions to save VRAM')
    p.add_argument('--reference-linear-attention', action='store_true', help='HF-only kernel isolation; slower')
    p.add_argument('--anomaly', action='store_true', help='Extra autograd debugging; changes runtime overhead')
    args = p.parse_args(argv)
    app = args.app_dir.resolve()
    if not (app/'lifemate/numerics.py').is_file():
        p.error('--app-dir must point to code/ containing lifemate/numerics.py')
    sys.path.insert(0, str(app))
    from lifemate.gpu import configure_cuda_device, single_cuda_device
    try:
        args.cuda_visible_devices = single_cuda_device(args.cuda_visible_devices)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))
    if args.device == 'cpu':
        args.cuda_visible_devices = '-1'
    configure_cuda_device(args)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    failure = json.loads(args.failure.read_text(encoding='utf-8'))
    if config['quantization'] != 'none' or config['batch_size'] != 1:
        p.error('This probe supports the uploaded unquantized batch-size=1 configuration')
    if args.reference_linear_attention and args.backend != 'transformers':
        p.error('--reference-linear-attention requires --backend transformers')
    if args.assistant_logits_only and args.backend != 'transformers':
        p.error('--assistant-logits-only requires --backend transformers')
    if args.device == 'cpu' and args.backend == 'unsloth':
        p.error('Unsloth requires CUDA')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('--output-dir must be empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
        logging.FileHandler(args.output_dir/'probe.log', encoding='utf-8')])
    result = {'status': 'starting', 'backend': args.backend, 'selection': args.selection,
              'cuda_visible_devices': args.cuda_visible_devices, 'assistant_logits_only': args.assistant_logits_only,
              'reference_linear_attention': args.reference_linear_attention, 'anomaly': args.anomaly,
              'fresh_base_and_fresh_lora': True, 'optimizer_update_performed': False,
              'original_completed_optimizer_steps': failure['completed_optimizer_steps'],
              'original_config': str(args.config.resolve()), 'original_failure': str(args.failure.resolve()),
              'limitation': 'No step-90 adapter/RNG in the dump. A passing fresh-base probe does not exonerate this runtime.'}
    code = 1
    trainer = stopper = None
    try:
        from lifemate.llm import is_qwen35, lora_targets, model_loader
        from lifemate.gpu import model_device_map
        fast = None
        if args.backend == 'unsloth':
            from unsloth import FastLanguageModel, FastVisionModel
            fast = FastVisionModel if is_qwen35(config['base_model']) else FastLanguageModel
        import torch
        import transformers
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import SequentialSampler
        from lifemate.numerics import guarded_trainer_class, DEBUG_BATCH_KEY
        from lifemate.sft import AssistantCollator
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; no model was loaded')
        if args.device == 'cuda' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('This configuration expects CUDA BF16 support')
        if config['training_dtype'] != 'torch.bfloat16':
            raise ValueError('This probe expects the uploaded BF16 configuration')
        if args.assistant_logits_only and (transformers.__version__ != '5.4.0' or not is_qwen35(config['base_model'])):
            raise ValueError('--assistant-logits-only is tested for Transformers 5.4.0 + Qwen3.5 only')
        if args.device == 'cuda':
            if torch.cuda.device_count() != 1:
                raise RuntimeError('Probe requires exactly one visible GPU; restart with --cuda-visible-devices GPU-UUID')
            result['gpu'] = {'name': torch.cuda.get_device_name(0),
                             'total_memory_gib': torch.cuda.get_device_properties(0).total_memory / 2**30}
        # CPU mode exists for tiny-model tests; full 4B CPU replay is not advised.
        dtype = torch.bfloat16 if args.device == 'cuda' else torch.float32
        set_seed(config['seed'])
        if args.reference_linear_attention:
            use_reference_linear_attention()
        rows = load_rows(args.batches or args.failure.with_name('batches.safetensors'), failure, args.selection)
        result['samples'] = [row[DEBUG_BATCH_KEY] for row in rows]
        result['input_lengths'] = [len(row['input_ids']) for row in rows]
        result['supervised_tokens'] = [sum(x != -100 for x in row['labels']) for row in rows]
        result['versions'] = {}
        for package in ('torch', 'transformers', 'peft', 'accelerate', 'unsloth', 'unsloth-zoo',
                        'triton', 'flash-linear-attention', 'fla-core', 'causal-conv1d'):
            try:
                result['versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        placement = model_device_map(args.device)
        if fast:
            model, _ = fast.from_pretrained(model_name=config['base_model'],
                max_seq_length=config['max_length'], dtype=dtype, load_in_4bit=False,
                local_files_only=True, trust_remote_code=False, fast_inference=False,
                device_map=placement, **({'auto_model': model_loader(config['base_model'])}
                    if is_qwen35(config['base_model']) else {'use_exact_model_name': True}))
            model = fast.get_peft_model(model, r=config['lora_r'], lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'], bias='none', target_modules=lora_targets(config['base_model']),
                use_gradient_checkpointing='unsloth', random_state=config['seed'], use_rslora=False,
                loftq_config=None, **({'finetune_vision_layers': False} if is_qwen35(config['base_model']) else {}))
        else:
            model = model_loader(config['base_model']).from_pretrained(config['base_model'],
                local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
                attn_implementation='sdpa', device_map=placement)
            model = get_peft_model(model, LoraConfig(r=config['lora_r'], lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'], bias='none', task_type='CAUSAL_LM',
                target_modules=lora_targets(config['base_model'])))
        model.config.use_cache = False
        if hasattr(model.config, 'text_config'):
            model.config.text_config.use_cache = False
        result['embedding_rows'] = model.get_input_embeddings().num_embeddings
        for row in rows:
            if max(row['input_ids']) >= result['embedding_rows'] or len(row['input_ids']) > config['max_length']:
                raise ValueError('Saved input exceeds this model vocabulary or configured context length')
        result['trainable_parameters'] = sum(x.numel() for x in model.parameters() if x.requires_grad)
        result['linear_kernels'] = {}
        for name, module in model.named_modules():
            if hasattr(module, 'chunk_gated_delta_rule'):
                result['linear_kernels'][name] = {
                    attr: getattr(getattr(module, attr, None), '__module__', None)
                    for attr in ('chunk_gated_delta_rule', 'causal_conv1d_fn')}
        logging.info('Probe fresh base + LoRA: %s; rows=%d; lengths=%s; no optimizer updates',
                     args.backend, len(rows), result['input_lengths'])
        class SelectedLogitsTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                loss, outputs = assistant_logits_loss(model, inputs)
                return (loss, outputs) if return_outputs else loss
        guarded = guarded_trainer_class(SelectedLogitsTrainer if args.assistant_logits_only else Trainer)
        class SequentialTrainer(guarded):
            def _get_train_sampler(self, *a, **kw):
                return SequentialSampler(self.train_dataset)
        class StopBeforeUpdate(TrainerCallback):
            handle = None
            def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
                seen = set()
                while getattr(optimizer, 'optimizer', None) is not None and id(optimizer) not in seen:
                    seen.add(id(optimizer)); optimizer = optimizer.optimizer
                def stop(*a, **kw):
                    raise ReplayComplete('All selected gradients passed; stopped before optimizer update')
                self.handle = optimizer.register_step_pre_hook(stop)
            def close(self):
                if self.handle is not None:
                    self.handle.remove()
        training_args = TrainingArguments(output_dir=str(args.output_dir), max_steps=1,
            per_device_train_batch_size=1, gradient_accumulation_steps=len(rows),
            learning_rate=0., weight_decay=0., max_grad_norm=1., optim='adamw_torch',
            bf16=args.device == 'cuda', fp16=False, use_cpu=args.device == 'cpu',
            gradient_checkpointing=args.backend != 'unsloth',
            gradient_checkpointing_kwargs={'use_reentrant': False},
            logging_steps=1, logging_nan_inf_filter=False, eval_strategy='no', save_strategy='no',
            report_to='none', seed=config['seed'], data_seed=config['seed'],
            dataloader_num_workers=0, remove_unused_columns=False, disable_tqdm=True)
        trainer = SequentialTrainer(model=model, args=training_args, train_dataset=rows,
            data_collator=AssistantCollator(config['runtime']['generation']['pad_token_id']),
            numerics_context={'probe': True, 'fresh_base_and_fresh_lora': True,
                              'original_failure': str(args.failure.resolve())})
        if args.assistant_logits_only:
            # The conditional-generation HF 5.4 model uses per-microbatch mean
            # loss, then Trainer divides by accumulation. Preserve that policy.
            trainer.model_accepts_loss_kwargs = False
        stopper = StopBeforeUpdate()
        # Register after GuardedTrainer's callback, so its optimizer barrier runs first.
        trainer.add_callback(stopper)
        with torch.autograd.detect_anomaly(check_nan=True) if args.anomaly else contextlib.nullcontext():
            trainer.train()
        raise RuntimeError('Probe did not intercept optimizer.step; inspect probe.log')
    except ReplayComplete as exc:
        result.update(status='finite_gradients', message=str(exc)); code = 0
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc))
        result['traceback'] = traceback.format_exc()
        if type(exc).__name__ == 'NonFiniteTrainingError':
            result['status'] = 'nonfinite_detected'; code = 2
        logging.exception('Probe stopped')
    finally:
        if stopper is not None:
            stopper.close()
        if trainer is not None:
            result['completed_optimizer_steps'] = trainer.state.global_step
            result['loss_history'] = trainer.state.log_history
        (args.output_dir/'probe_result.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'result': str(args.output_dir/'probe_result.json')}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())