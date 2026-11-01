"""Two-process orchestration, durable transcripts and cache-only LLM inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lifemate.transcripts import (asr_identity, cache_paths, dataset_identity,
                                  load_transcripts, prepare_transcripts, read_jsonl)


@pytest.fixture
def dataset(tmp_path):
    split = tmp_path / 'data' / 'public_test'
    split.mkdir(parents=True)
    rows = []
    for i in range(3):
        (split / f'{i}.wav').write_bytes(b'audio fixture')
        rows.append({'id': f'public_{i}', 'audio': f'{i}.wav',
            'context': {'installed_apps': {'home': {'tools': ['home.read']}}},
            'history': [{'role': 'user', 'audio': 'history.wav'},
                        {'role': 'assistant', 'text': 'Earlier answer'}]})
    (split / 'history.wav').write_bytes(b'history fixture')
    (split / 'public_test.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (split / 'public_test_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': [
        {'name': 'home.read', 'kind': 'read', 'parameters': {'type': 'object', 'properties': {}}}]}]}))
    return split, rows


class ASR:
    def __init__(self): self.paths = []
    def transcribe(self, path):
        self.paths.append(path.name)
        return 'recognized:' + path.name


def test_cache_current_and_history_is_complete_and_reusable(dataset, tmp_path):
    split, items = dataset
    cache = tmp_path / 'transcripts.jsonl'
    identity = dataset_identity(items, split, 'public_test')
    asr = ASR()
    prepare_transcripts(items, split, cache, identity, {'model': 'medium'}, lambda: asr)
    rows, metadata = load_transcripts(cache, identity)
    assert rows[0]['transcript'] == 'recognized:0.wav'
    assert rows[0]['history'] == [{'role': 'user', 'text': 'recognized:history.wav'},
                                  {'role': 'assistant', 'text': 'Earlier answer'}]
    assert metadata['status'] == 'complete'
    assert len(rows) == 3
    def must_not_load():
        raise AssertionError('Completed cache must not reload ASR')
    prepare_transcripts(items, split, cache, identity, {'model': 'medium'}, must_not_load)
    with pytest.raises(ValueError, match='ASR model/settings changed'):
        prepare_transcripts(items, split, cache, identity, {'model': 'small'}, must_not_load)
    (split / '0.wav').write_bytes(b'a different audio')
    changed = dataset_identity(items, split, 'public_test')
    with pytest.raises(ValueError, match='dataset/audio/selection changed'):
        load_transcripts(cache, changed)
    with cache.open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='checksum'):
        load_transcripts(cache, identity)


def test_public_request_uses_asr_and_full_history_is_saved(dataset, tmp_path):
    split, items = dataset
    items[0]['audio_text'] = 'must not use test request transcript'
    items[0]['history'].append({'role': 'assistant', 'audio': 'history.wav'})
    items[0]['history'] += [{'role': 'assistant', 'text': str(i) + 'x'*1800} for i in range(15)]
    identity = dataset_identity(items, split, 'public_test')
    cache = tmp_path / 'all-text.jsonl'
    prepare_transcripts(items, split, cache, identity, {}, ASR)
    row = load_transcripts(cache, identity)[0][0]
    assert row['transcript'] == 'recognized:0.wav'
    assert len(row['history']) == 18
    assert row['history'][2] == {'role': 'assistant', 'text': 'recognized:history.wav'}
    assert len(row['history'][-1]['text']) > 1700
    with pytest.raises(ValueError, match='forbidden'):
        dataset_identity(items, split, 'public_test', use_train_transcript=True)


def test_asr_failure_preserves_prefix_and_resumes_without_empty_prediction(dataset, tmp_path):
    split, items = dataset
    cache = tmp_path / 'transcripts.jsonl'
    identity = dataset_identity(items, split, 'public_test')
    class FailingASR(ASR):
        def transcribe(self, path):
            if path.name == '1.wav': raise RuntimeError('fixture OOM')
            return super().transcribe(path)
    with pytest.raises(RuntimeError, match='fixture OOM'):
        prepare_transcripts(items, split, cache, identity, {}, FailingASR)
    assert not cache.exists()
    assert len(read_jsonl(cache_paths(cache)[1])) == 1
    with pytest.raises(FileNotFoundError):
        load_transcripts(cache, identity)
    resumed = ASR()
    prepare_transcripts(items, split, cache, identity, {}, lambda: resumed)
    assert '0.wav' not in resumed.paths
    assert [r['id'] for r in load_transcripts(cache, identity)[0]] == identity['sample_ids']
    assert read_jsonl(cache.with_suffix('.errors.jsonl'))[0]['id'] == 'public_1'


def test_full_cli_exits_asr_process_before_loading_llm_and_llm_only_reuses_cache(dataset, tmp_path):
    split, items = dataset
    app = tmp_path / 'app'
    app.mkdir()
    source = Path(__file__).resolve().parents[1]
    shutil.copy2(source / 'main.py', app / 'main.py')
    shutil.copytree(source / 'lifemate', app / 'lifemate', ignore=shutil.ignore_patterns('__pycache__'))
    (app / 'lifemate/models.py').write_text('''import atexit,json,os,sys
from pathlib import Path
def event(name, **data):
    with open(os.environ['LIFEMATE_TEST_EVENTS'], 'a') as f:
        f.write(json.dumps(dict(event=name,pid=os.getpid(),**data))+'\\n')
class LocalASR:
    def __init__(self,*args,**kwargs):
        assert sys.argv[sys.argv.index('--stage')+1]=='asr'
        event('asr_load')
        atexit.register(lambda:event('asr_exit'))
    def transcribe(self,path): return 'recognized:'+path.name
class LocalLLM:
    def __init__(self,*args,**kwargs):
        assert sys.argv[sys.argv.index('--stage')+1]=='llm'
        event('llm_load')
    def generate(self,system,user):
        payload=json.loads(user)
        assert payload['request_transcript'].startswith('recognized:')
        assert payload['conversation_history'][0]['text']=='recognized:history.wav'
        return '{"action":"done"}' if payload['previous_actions_and_results'] else '{"action":"call","name":"home.read","arguments":{}}'
''')
    (split / 'zaic_tools.py').write_text('''class ToolEnv:
    def __init__(self,split,id,root): self.calls=[]
    def validate(self,name,args): return {'ok':True}
    def call(self,name,args):
        self.calls.append({'name':name,'arguments':args}); return {'ok':True}
    def get_tool_calls(self): return self.calls
    def get_state_hash(self): return 'fixture-state'
''')
    model = tmp_path / 'asr-model'
    model.mkdir()
    for name in ('encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt'):
        (model / name).write_bytes(b'fixture checkpoint')
    events = tmp_path / 'events.jsonl'
    output = tmp_path / 'submission.jsonl'
    cache = tmp_path / 'transcripts.jsonl'
    env = {**os.environ, 'LIFEMATE_TEST_EVENTS': str(events), 'CUDA_VISIBLE_DEVICES': '',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
    command = [sys.executable, app / 'main.py', '--data-root', split.parent, '--asr-model', model,
        '--llm-model', tmp_path / 'llm-fixture', '--transcript-cache', cache,
        '--device', 'cpu', '--output', output]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = read_jsonl(events)
    assert [e['event'] for e in recorded] == ['asr_load', 'asr_exit', 'llm_load']
    assert recorded[0]['pid'] != recorded[-1]['pid']
    assert len(read_jsonl(output)) == len(items)
    assert all(len(row['tool_calls']) == 1 for row in read_jsonl(output))
    assert all('asr_seconds' in row and 'total_sample_seconds' in row for row in read_jsonl(output.with_suffix('.diagnostics.jsonl')))
    assert cache.with_suffix('.asr.log').is_file() and output.with_suffix('.llm.log').is_file()
    # LLM-only does not even require the ASR checkpoint to exist.
    shutil.rmtree(model)
    llm_only = [sys.executable, app / 'main.py', '--stage', 'llm', '--data-root', split.parent,
        '--llm-model', tmp_path / 'llm-fixture', '--transcript-cache', cache, '--device', 'cpu', '--output', output]
    result = subprocess.run(llm_only, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [e['event'] for e in read_jsonl(events)].count('asr_load') == 1


def test_agent_default_is_unlimited_and_uses_only_prepared_history():
    from lifemate.agent import AgentConfig, LifeMateAgent
    from test_baseline import CATALOG, FakeEnv, item
    assert AgentConfig().sample_seconds == float('inf')
    class NoAudio:
        def transcribe(self, path): raise AssertionError('LLM stage tried to run ASR')
    class Planner:
        def generate(self, system, user):
            assert json.loads(user)['request_transcript'] == 'cached request'
            return '{"action":"done"}'
    prepared = {'id': 'train_0001', 'transcript': 'cached request', 'history': []}
    result = LifeMateAgent(NoAudio(), Planner(), CATALOG, AgentConfig()).run(
        item(), FakeEnv(), Path('/unused'), 'train', prepared_transcript=prepared)
    assert result['transcript'] == 'cached request'


def test_repo_cli_defaults_to_single_gpu_and_unlimited_time(monkeypatch):
    import main
    monkeypatch.setattr(sys, 'argv', ['main.py', '--data-root', '/unused'])
    args = main.parse_args()
    assert args.llm_device_map == 'single'
    assert args.sample_seconds == float('inf')
    assert args.stage == 'full'


def test_llm_stage_rejects_whisper_cache_before_loading_llm(dataset, tmp_path, monkeypatch):
    import main
    split, items = dataset
    cache = tmp_path / 'legacy.jsonl'
    identity = dataset_identity(items, split, 'public_test')
    prepare_transcripts(items, split, cache, identity, {'backend': 'faster-whisper'}, ASR)
    def must_not_load(*args):
        raise AssertionError('The LLM must not load when ASR provenance is wrong')
    monkeypatch.setattr(main, 'run_llm_stage', must_not_load)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--stage', 'llm', '--data-root', str(split),
        '--transcript-cache', str(cache), '--llm-model', str(tmp_path / 'llm')])
    with pytest.raises(ValueError, match='Transcript ASR source is not GIPformer'):
        main.main()


def test_balanced_llm_routes_input_without_moving_entire_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lifemate.models import LocalLLM
    model = Mock()
    model.hf_device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 1, 'lm_head': 1}
    model.get_input_embeddings.return_value = SimpleNamespace(weight=SimpleNamespace(device=torch.device('cpu')))
    model.to.side_effect = AssertionError('Do not move a sharded model wholesale')
    loader = Mock(return_value=model)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', loader)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', Mock())
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda index: (14*1024**3, 15*1024**3))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'simulated T4')
    llm = LocalLLM(tmp_path, device_map='balanced', dtype='float16')
    assert loader.call_args.kwargs['device_map'] == 'balanced'
    assert loader.call_args.kwargs['max_memory'] == {0: 6*1024**3, 1: 6*1024**3}
    assert llm.device == torch.device('cpu')  # mocked embedding placement used for input routing
    model.to.assert_not_called()
    model.hf_device_map = {'': 0}
    with pytest.raises(RuntimeError, match='multiple GPUs'):
        LocalLLM(tmp_path, device_map='balanced')
