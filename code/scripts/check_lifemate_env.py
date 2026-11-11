#!/usr/bin/env python3
"""Isolated import/backend checks for LifeMate. Does not install packages/load weights."""
import argparse
import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys
import textwrap

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--app-dir', type=Path, default=Path.cwd(), help='Directory containing main.py and finetune.py')
p.add_argument('--split-dir', type=Path, action='append', default=[], help='Directory containing the split-specific zaic_tools; repeat for train/public_test')
p.add_argument('--probe-toolenv', action='store_true', help='Construct one independent diagnostic session per split; validate only, never env.call')
p.add_argument('--only-toolenv', action='store_true', help='Skip model/package checks and check only ToolEnv')
p.add_argument('--timeout', type=int, default=180, help='Timeout seconds per isolated check')
a = p.parse_args()
root = a.app_dir.expanduser().resolve()
if not (root / 'main.py').is_file() or not (root / 'lifemate').is_dir():
    p.error('--app-dir must point to the LifeMate code directory')
if a.timeout <= 0:
    p.error('--timeout must be positive')
print('Python:', sys.executable, flush=True)
print('Prefix:', sys.prefix, flush=True)
print('Python version:', sys.version.split()[0], flush=True)
print('App:', root, flush=True)
for pkg in ['torch', 'transformers', 'peft', 'unsloth', 'unsloth-zoo', 'accelerate', 'sherpa-onnx', 'soundfile', 'flash-linear-attention', 'fla-core', 'causal-conv1d']:
    try:
        print('  ', pkg, metadata.version(pkg), flush=True)
    except metadata.PackageNotFoundError:
        print('  ', pkg, 'NOT INSTALLED', flush=True)
try:
    if metadata.version('transformers') != '5.4.0':
        print('[WARN] Repo currently pins Transformers 5.4.0; another version requires model-level verification.', flush=True)
except metadata.PackageNotFoundError:
    pass
child_env = os.environ.copy()
child_env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
child_env['PYTHONPATH'] = str(root) + os.pathsep + child_env.get('PYTHONPATH', '')
failed = []
warned = []


def check(name, code=None, command=None, optional=False, show_output=False):
    args = command if command is not None else [sys.executable, '-c', textwrap.dedent(code)]
    try:
        result = subprocess.run(args, cwd=root, env=child_env, capture_output=True, text=True, errors='replace', timeout=a.timeout)
        output = '\n'.join(part.strip() for part in [result.stdout, result.stderr] if part.strip())
        ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        output, ok = f'Timed out after {a.timeout}s', False
    level = 'PASS' if ok else 'WARN' if optional else 'FAIL'
    print(f'[{level}] {name}', flush=True)
    if ok and show_output and output:
        print(output, flush=True)
    if not ok:
        print(output or 'Child process failed without output', flush=True)
        (warned if optional else failed).append(name)


if not a.only_toolenv:
    check('Installed dependency consistency', command=[sys.executable, '-m', 'pip', 'check'])
    check('ASR imports and WAV I/O', r'''
    import numpy as np, soundfile as sf, sherpa_onnx
    from pathlib import Path
    import tempfile
    from lifemate.asr import read_audio, LocalASR
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'check.wav'
        sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
        samples, rate = read_audio(path)
        assert rate == 16000 and len(samples) == 1600
    ''')
    check('Standalone Transformers/PEFT Qwen3.5 inference imports', r'''
    import torch, transformers, peft, accelerate, safetensors, tokenizers
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from lifemate.models import LocalLLM
    from lifemate.decoding import NativeToolStoppingCriteria
    from lifemate.function_calling import native_chat
    from lifemate.tool_execution import precheck_call, attempt_env_call
    ''')
    check('CUDA and BF16 actual matrix multiplication', r'''
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable; default GPU pipeline cannot run'
    assert torch.cuda.is_bf16_supported(), 'GPU does not support default BF16 pipeline'
    x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    assert torch.isfinite(y).all().item() and (y == 32).all().item()
    ''')
    check('Unsloth training imports (Unsloth imported FIRST)', r'''
    from unsloth import FastLanguageModel, FastVisionModel
    import torch
    from transformers import Trainer, TrainingArguments, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from lifemate.sft import AssistantCollator, SupervisedDataset
    from lifemate.evaluation import StateToolEvaluator
    import train_lora, merge_lora
    ''')
    check('Optional FLA kernels import', r'''
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    ''', optional=True)
    check('Optional causal-conv1d binary import', r'''
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    ''', optional=True)
    check('Optional bitsandbytes import', 'import bitsandbytes', optional=True)
    for name in ['main.py', 'prepare_sft.py', 'train_lora.py', 'finetune.py', 'merge_lora.py']:
        if not (root / name).is_file():
            failed.append(name)
            print(f'[FAIL] Missing entrypoint {name}', flush=True)
            continue
        check(f'Entrypoint {name} --help', command=[sys.executable, str(root/name), '--help'])
for split in a.split_dir:
    split = split.expanduser().resolve()
    label = 'Official ToolEnv instance API' if a.probe_toolenv else 'Official ToolEnv imports'
    check(f'{label}: {split}', code=f'''
import sys, inspect, json
from copy import deepcopy
from pathlib import Path
split = Path({str(split)!r})
assert split.is_dir(), 'Split directory does not exist'
sys.path.insert(0, str(split))
import zaic_tools
from zaic_tools import ToolEnv
origin = Path(zaic_tools.__file__).resolve()
assert origin.is_relative_to(split), f'Wrong zaic_tools imported: {{origin}}'
assert callable(ToolEnv), 'ToolEnv must be a callable class or factory'
print('ToolEnv type:', 'class' if inspect.isclass(ToolEnv) else 'factory/callable')
print('Source:', origin)
if {a.probe_toolenv!r}:
    manifest = split / (split.name + '.jsonl')
    assert manifest.is_file(), f'Missing public sample manifest: {{manifest}}'
    with manifest.open(encoding='utf-8') as stream:
        item = next((json.loads(line) for line in stream if line.strip()), None)
    assert item and isinstance(item.get('id'), str), 'No usable public sample ID'
    # One instance in this independent diagnostic run. No env.call or get_state.
    env = ToolEnv(split.name, item['id'], root=str(split.parent))
    for method in ['validate', 'call', 'get_tool_calls', 'get_state_hash']:
        assert callable(getattr(env, method, None)), f'Missing instance API: {{method}}'
    before = deepcopy(env.get_tool_calls())
    assert isinstance(before, list), 'get_tool_calls must return a list'
    response = env.validate('__lifemate_diagnostic_missing_tool__', {{}})
    assert isinstance(response, dict) and response.get('ok') is False, f'Unexpected validation response: {{response!r}}'
    assert env.get_tool_calls() == before, 'validate unexpectedly changed the call log'
    state_hash = env.get_state_hash()
    assert isinstance(state_hash, str) and state_hash, 'get_state_hash must return a nonempty string'
    print('Instance API passed; schema rejection checked; no application calls executed.')
else:
    print('Instance API not checked; use --probe-toolenv to verify factory/delegated methods.')
''', show_output=True)
if not a.split_dir:
    print('[SKIP] Official ToolEnv not checked; add --split-dir for each split.', flush=True)
print(f'\nResult: {len(failed)} FAIL; {len(warned)} optional WARN', flush=True)
print('These checks do not certify model weights, adapters, VRAM capacity, CUDA kernel execution, or actual dataset rollouts.', flush=True)
sys.exit(1 if failed else 0)