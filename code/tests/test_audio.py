"""GIPformer audio handling and ONNX runtime contract (no model downloads)."""
from types import SimpleNamespace
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import soundfile as sf
from lifemate.asr import ASR_BACKEND, ASR_REPO_ID, LocalASR, asr_files, read_audio
from lifemate.transcripts import asr_identity


def write_models(path, precision='int8'):
    suffix = '.int8.onnx' if precision == 'int8' else '.onnx'
    for name in ['tokens.txt', *(part + suffix for part in ('encoder', 'decoder', 'joiner'))]:
        (path / name).write_bytes(b'fixture model')


def test_audio_is_float32_mono_with_original_rate_for_runtime_resampling(tmp_path):
    rate = 8000
    samples = 0.4 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate)
    path = tmp_path / 'speech.wav'
    sf.write(path, np.column_stack([samples, np.zeros_like(samples)]), rate, subtype='PCM_16')
    audio, actual_rate = read_audio(path)
    assert actual_rate == rate and audio.shape == (4000,)
    assert audio.dtype == np.float32 and audio.flags.c_contiguous
    assert np.max(np.abs(audio)) == pytest.approx(0.2, abs=0.001)


@pytest.mark.parametrize('samples', [[], [np.nan], [np.inf], [1.5]])
def test_empty_nonfinite_or_out_of_range_audio_fails(tmp_path, samples):
    path = tmp_path / 'invalid.wav'
    sf.write(path, np.array(samples), 16000, subtype='FLOAT')
    with pytest.raises(ValueError, match='Empty or invalid audio'):
        read_audio(path)


@pytest.mark.parametrize('precision,suffix', [('int8', '.int8.onnx'), ('fp32', '.onnx')])
def test_onnx_cpu_recognizer_receives_model_files_and_preserves_cached_text(tmp_path, monkeypatch, precision, suffix):
    import sherpa_onnx
    write_models(tmp_path, precision)
    audio = tmp_path / 'speech.wav'
    sf.write(audio, np.zeros(4000, dtype=np.float32), 8000)
    accepted, decoded, options = [], [], {}
    class Stream:
        result = SimpleNamespace(text=' BẬT đèn, lúc 08:30! ')
        def accept_waveform(self, rate, samples): accepted.append((rate, samples))
    recognizer = SimpleNamespace(create_stream=Stream, decode_streams=lambda streams: decoded.extend(streams))
    def create(**kwargs):
        options.update(kwargs)
        return recognizer
    monkeypatch.setattr(sherpa_onnx.OfflineRecognizer, 'from_transducer', create)
    asr = LocalASR(tmp_path, precision=precision, num_threads=2)
    assert asr.transcribe(audio) == ' BẬT đèn, lúc 08:30! '
    assert asr.transcribe(audio) == ' BẬT đèn, lúc 08:30! '
    assert len(accepted) == len(decoded) == 1
    assert accepted[0][0] == 8000 and accepted[0][1].dtype == np.float32
    assert options['provider'] == 'cpu' and options['num_threads'] == 2
    assert options['sample_rate'] == 16000 and options['feature_dim'] == 80
    assert options['decoding_method'] == 'greedy_search'
    for part in ('encoder', 'decoder', 'joiner'):
        assert options[part] == str(tmp_path / (part + suffix))
    assert options['tokens'] == str(tmp_path / 'tokens.txt')


def test_missing_precision_files_do_not_fall_back_or_accept_whisper(tmp_path):
    write_models(tmp_path, 'fp32')
    (tmp_path / 'model.bin').write_bytes(b'legacy Whisper')
    with pytest.raises(FileNotFoundError, match='encoder.int8.onnx'):
        asr_files(tmp_path)
    (tmp_path / 'encoder.onnx').write_bytes(b'')
    with pytest.raises(FileNotFoundError, match='encoder.onnx'):
        asr_files(tmp_path, 'fp32')
    with pytest.raises(ValueError, match='precision'):
        asr_files(tmp_path, 'float16')
    with pytest.raises(ValueError, match='threads'):
        LocalASR(tmp_path, num_threads=0)
    with pytest.raises(FileNotFoundError):
        read_audio(tmp_path / 'missing.wav')


def test_asr_identity_tracks_selected_weights_settings_and_runtime(tmp_path):
    write_models(tmp_path)
    first = asr_identity(tmp_path, 'int8', 2)
    assert first['repo_id'] == ASR_REPO_ID and first['backend'] == ASR_BACKEND
    assert first['device'] == 'cpu' and first['precision'] == 'int8'
    assert 'sherpa_onnx_version' in first and 'soundfile_version' in first
    assert asr_identity(tmp_path, 'int8', 3) != first
    (tmp_path / 'encoder.int8.onnx').write_bytes(b'replaced selected weights')
    assert asr_identity(tmp_path, 'int8', 2) != first
    baseline = asr_identity(tmp_path, 'int8', 2)
    (tmp_path / 'irrelevant-readme.txt').write_text('does not affect inference')
    assert asr_identity(tmp_path, 'int8', 2) == baseline


def test_asr_cli_runs_without_torch_and_forces_cpu(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    write_models(model)
    split = tmp_path / 'train'
    split.mkdir()
    (split / 'current.wav').write_bytes(b'fixture audio; model is stubbed')
    (split / 'train.jsonl').write_text(json.dumps({'id': 'train_1', 'audio': 'current.wav', 'history': []}) + '\n')
    cache = tmp_path / 'transcripts.jsonl'
    script = '''import builtins, os, runpy, sys, types
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'ctranslate2', 'faster_whisper'):
        raise AssertionError('ASR must not import ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
class ASR:
    def __init__(self, path, **kwargs):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs == {'precision': 'int8', 'num_threads': 2}
    def transcribe(self, path): return 'BẬT đèn, lúc 08:30!'
sys.modules['lifemate.models'] = types.SimpleNamespace(LocalASR=ASR)
sys.argv = ['main.py', *sys.argv[1:]]
runpy.run_path('main.py', run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', script, '--stage', 'asr', '--data-root', str(split),
                             '--split', 'train', '--asr-model', str(model), '--asr-threads', '2',
                             '--transcript-cache', str(cache)],
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '2'},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(cache.read_text())['transcript'] == 'BẬT đèn, lúc 08:30!'
    metadata = json.loads(cache.with_suffix('.manifest.json').read_text())['asr']
    assert metadata['repo_id'] == ASR_REPO_ID and metadata['device'] == 'cpu'
    assert metadata['num_threads'] == 2
