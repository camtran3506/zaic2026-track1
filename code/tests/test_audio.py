"""Exercise the real faster-whisper/PyAV audio decoder without model weights."""
import wave

import numpy as np
import pytest


def test_whisper_decodes_and_resamples_stereo_wav(tmp_path):
    decode_audio = pytest.importorskip('faster_whisper.audio').decode_audio
    path = tmp_path / 'speech.wav'
    rate = 8000
    samples = (2000 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate)).astype('<i2')
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.column_stack([samples, samples]).tobytes())
    # Calls av.open(metadata_errors='ignore') inside faster-whisper, then resamples.
    audio = decode_audio(str(path), sampling_rate=16000)
    assert audio.shape == (8000,)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert 0.01 < np.max(np.abs(audio)) < 1
