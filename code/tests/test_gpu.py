"""GPU masks must be set before CUDA imports and survive subprocess boundaries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lifemate.gpu import add_cuda_device_argument, configure_cuda_device

APP = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('inherited,explicit,expected', [
    (None, None, '0'), ('2,1,0', None, '2'), ('3', None, '3'),
    ('', None, ''), ('-1', None, '-1'), ('0,1', '2', '2'),
    ('0', 'GPU-1234-abcd', 'GPU-1234-abcd'),
])
def test_one_gpu_selection(monkeypatch, inherited, explicit, expected):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    if inherited is not None:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', inherited)
    parser = argparse.ArgumentParser()
    add_cuda_device_argument(parser)
    args = parser.parse_args([] if explicit is None else ['--cuda-visible-devices', explicit])
    configure_cuda_device(args)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == expected


@pytest.mark.parametrize('script,required', [
    ('main.py', ['--data-root', 'data']),
    ('training_code/finetune.py', ['--data-root', 'data', '--base-model', 'model', '--output-dir', 'run']),
    ('training_code/train_lora.py', ['--base-model', 'model', '--sft-dir', 'sft', '--output-dir', 'run']),
    ('training_code/prepare_sft.py', ['--data-root', 'data', '--output-dir', 'run']),
])
def test_cli_rejects_multiple_gpus_before_loading_models(script, required):
    result = subprocess.run([sys.executable, APP / script, *required,
                             '--cuda-visible-devices', '0,1'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Choose one GPU' in result.stderr
    assert 'Traceback' not in result.stderr


def test_training_selects_gpu_before_importing_unsloth():
    probe = '''
import builtins, os, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'unsloth':
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
        print('GPU selected before Unsloth')
        raise SystemExit(0)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['train_lora.py', '--base-model', 'model', '--sft-dir', 'sft',
            '--output-dir', 'run', '--cuda-visible-devices', '2']
runpy.run_path('training_code/train_lora.py', run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', probe], cwd=APP, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'GPU selected before Unsloth' in result.stdout


def test_unsloth_receives_explicit_gpu_map_and_does_not_retry_oom(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import torch
    import train_lora
    base = tmp_path/'base'; base.mkdir()
    oom = torch.cuda.OutOfMemoryError('simulated training VRAM exhaustion')
    loader = Mock(side_effect=oom)
    fast = SimpleNamespace(from_pretrained=loader)
    monkeypatch.setitem(sys.modules, 'unsloth', SimpleNamespace(FastLanguageModel=fast, FastVisionModel=fast))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(train_lora, 'training_dtype', lambda torch: torch.float32)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    monkeypatch.setattr(sys, 'argv', ['train_lora.py', '--base-model', str(base), '--sft-dir', str(tmp_path/'sft'),
        '--output-dir', str(tmp_path/'training'), '--backend', 'unsloth'])
    with pytest.raises(torch.cuda.OutOfMemoryError) as error:
        train_lora.main()
    assert error.value is oom and loader.call_count == 1
    assert loader.call_args.kwargs['device_map'] == {'': 0}  # Physical GPU 2 is local 0.
    assert loader.call_args.kwargs['fast_inference'] is False


@pytest.mark.parametrize('pipeline', ['inference', 'finetune'])
def test_child_processes_inherit_single_gpu(monkeypatch, tmp_path, pipeline):
    import main
    import finetune
    observed = []

    class Child:
        stdout = ()

        def __init__(self, command, **kwargs):
            # Start a real fresh process with the same environment as the child.
            mask = subprocess.check_output(
                [sys.executable, '-c', 'import os; print(os.environ["CUDA_VISIBLE_DEVICES"])'],
                env=kwargs.get('env'), text=True).strip()
            observed.append((command, mask))

        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self): return 0

    # Capture Popen without intercepting check_output's own use of Popen.
    from types import SimpleNamespace
    fake_subprocess = SimpleNamespace(Popen=Child, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,1,0')
    if pipeline == 'inference':
        monkeypatch.setattr(main, 'subprocess', fake_subprocess)
        monkeypatch.setattr(sys, 'argv', ['main.py', '--stage', 'full', '--data-root', 'data',
            '--asr-model', 'asr', '--llm-model', 'llm', '--output', str(tmp_path / 'out.jsonl')])
        main.main()
        assert [cmd[cmd.index('--stage') + 1] for cmd, _ in observed] == ['asr', 'llm']
    else:
        monkeypatch.setattr(finetune, 'subprocess', fake_subprocess)
        base = tmp_path / 'base'
        base.mkdir()
        monkeypatch.setattr(sys, 'argv', ['finetune.py', '--data-root', 'data', '--asr-model', 'asr',
            '--base-model', str(base), '--output-dir', str(tmp_path / 'run')])
        finetune.main()
        assert [Path(cmd[1]).name for cmd, _ in observed] == ['prepare_sft.py', 'train_lora.py']
        prepare = observed[0][0]
        assert prepare[prepare.index('--asr-transcript-ratio') + 1] == '0.5'
        assert prepare[prepare.index('--transcript-mode') + 1] == 'dual'
        assert prepare[prepare.index('--asr-model') + 1] == 'asr'
        assert prepare[prepare.index('--rename-variants') + 1] == '0'
        assert prepare[prepare.index('--distractor-probability') + 1] == '0.0'
        assert prepare[prepare.index('--heldout-apps') + 1] == 'none'
        assert prepare[prepare.index('--validation-fraction') + 1] == '0.0'
        assert prepare[prepare.index('--tokenizer') + 1] == str(base.resolve())
        assert prepare[prepare.index('--max-length') + 1] == '16384'
        assert '--asr-transcript-ratio' not in observed[1][0]
    assert [mask for _, mask in observed] == ['2', '2']


def test_finetune_rejects_holdouts_before_creating_run(monkeypatch, tmp_path, capsys):
    import finetune
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['finetune.py', '--data-root', 'data',
        '--base-model', 'unused-model', '--output-dir', str(output), '--asr-transcript-ratio', '0',
        '--heldout-apps', 'auto'])
    with pytest.raises(SystemExit) as exc:
        finetune.main()
    assert exc.value.code == 2
    assert 'Training uses all samples' in capsys.readouterr().err
    assert not output.exists()


def test_server_forwards_gpu_argument(tmp_path):
    fake_python = tmp_path / 'python'
    fake_python.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake_python.chmod(0o755)
    result = subprocess.check_output(['bash', APP / 'run_server.sh', '--cuda-visible-devices', '2'],
        env={**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH']}, text=True)
    args = json.loads(result)
    assert args[args.index('--cuda-visible-devices') + 1] == '2'
