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
def test_onnx_cpu_recognizer_receives_model_files_and_normalizes_cached_text(tmp_path, monkeypatch, precision, suffix):
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
    assert asr.transcribe(audio) == 'Bật đèn, lúc 08:30!'
    assert asr.transcribe(audio) == 'Bật đèn, lúc 08:30!'
    assert len(accepted) == len(decoded) == 1
    assert accepted[0][0] == 8000 and accepted[0][1].dtype == np.float32
    assert options['provider'] == 'cpu' and options['num_threads'] == 2
    assert options['sample_rate'] == 16000 and options['feature_dim'] == 80
    assert options['decoding_method'] == 'modified_beam_search'
    assert options['max_active_paths'] == 4
    for part in ('encoder', 'decoder', 'joiner'):
        assert options[part] == str(tmp_path / (part + suffix))
    assert options['tokens'] == str(tmp_path / 'tokens.txt')
    # Explicit greedy_search remains selectable for ablations.
    options.clear()
    asr_greedy = LocalASR(tmp_path, precision=precision, num_threads=2, decoding_method='greedy_search')
    assert asr_greedy.decoding_method == 'greedy_search'
    assert options['decoding_method'] == 'greedy_search'


def test_transcribe_batch_batches_offline_audio_and_populates_cache(tmp_path, monkeypatch):
    import sherpa_onnx
    write_models(tmp_path, 'int8')
    audios = []
    for i in range(5):
        audio = tmp_path / f'speech_{i}.wav'
        sf.write(audio, np.zeros(2000, dtype=np.float32), 16000)
        audios.append(audio)

    decode_calls = []

    class MockStream:
        def __init__(self, text):
            self.result = SimpleNamespace(text=text)

        def accept_waveform(self, rate, samples):
            pass

    stream_idx = 0

    def create_stream():
        nonlocal stream_idx
        stream = MockStream(f' câu số {stream_idx}! ')
        stream_idx += 1
        return stream

    def decode_streams(streams):
        decode_calls.append(list(streams))

    recognizer = SimpleNamespace(create_stream=create_stream, decode_streams=decode_streams)
    monkeypatch.setattr(sherpa_onnx.OfflineRecognizer, 'from_transducer', lambda **kw: recognizer)

    asr = LocalASR(tmp_path, precision='int8', num_threads=2)
    assert asr.transcribe_batch([]) == []
    with pytest.raises(ValueError, match="batch_size"):
        asr.transcribe_batch(audios, batch_size=0)

    # Batch decode 5 audios with batch_size=2 -> 3 chunks of sizes [2, 2, 1]
    results = asr.transcribe_batch(audios, batch_size=2)
    assert len(results) == 5
    assert len(decode_calls) == 3
    assert [len(call) for call in decode_calls] == [2, 2, 1]
    assert results == [f'Câu số {i}!' for i in range(5)]

    # Subsequent single-stream transcribe calls hit cache without invoking decode_streams
    decode_calls.clear()
    for i, audio in enumerate(audios):
        assert asr.transcribe(audio) == f'Câu số {i}!'
    assert len(decode_calls) == 0


def test_pretranscribe_sft_partition_batches_audio_and_history(tmp_path):
    from lifemate.sft import pretranscribe_sft_partition
    split_dir = tmp_path / 'train'
    split_dir.mkdir()
    for name in ['req1.wav', 'req2.wav', 'hist1.wav']:
        (split_dir / name).write_bytes(b'mock audio')

    batched_paths = []

    class MockASR:
        def transcribe_batch(self, paths, batch_size=16):
            batched_paths.extend(paths)
            return [p.name for p in paths]

    records = [
        {'id': 's1', 'audio': 'req1.wav', 'history': [{'role': 'user', 'audio': 'hist1.wav'}]},
        {'id': 's2', 'audio': 'req2.wav', 'history': [{'role': 'user', 'text': 'already text', 'audio': 'hist2.wav'}]},
    ]
    # For train partition, if sources only has s1 as 'asr', s2 uses audio_text so req2.wav is not transcribed
    sources = {'s1': 'asr', 's2': 'audio_text'}
    pretranscribe_sft_partition(records, split_dir, MockASR(), 'train', sources, batch_size=2)
    assert [p.name for p in batched_paths] == ['req1.wav', 'hist1.wav']

    # For validation partition, all sample audios are needed for rollout
    batched_paths.clear()
    pretranscribe_sft_partition(records, split_dir, MockASR(), 'validation', sources, batch_size=2)
    assert [p.name for p in batched_paths] == ['req1.wav', 'hist1.wav', 'req2.wav']


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
    assert json.loads(cache.read_text())['transcript'] == 'Bật đèn, lúc 08:30!'
    metadata = json.loads(cache.with_suffix('.manifest.json').read_text())['asr']
    assert metadata['repo_id'] == ASR_REPO_ID and metadata['device'] == 'cpu'
    assert metadata['num_threads'] == 2
