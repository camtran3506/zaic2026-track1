"""Artifact capability checks without requiring a model or CUDA dependencies."""
import json
from types import SimpleNamespace

import pytest

from lifemate.local_time import LOCAL_TIME_POLICY
from lifemate.runtime import RUNTIME_MANIFEST, load_runtime_manifest, make_runtime_manifest


@pytest.fixture
def temporal_artifact(tmp_path, monkeypatch):
    from lifemate import runtime

    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_text('{}')
    (base / 'model.safetensors').write_bytes(b'identity-only fixture')
    tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(to_str=lambda: '{"model":{}}'),
        chat_template='fixture template', special_tokens_map={},
        eos_token_id=1, pad_token_id=0,
    )
    artifact = tmp_path / 'adapter'
    artifact.mkdir()
    manifest = make_runtime_manifest(base, tokenizer)
    (artifact / RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    # The feature being checked is the recorded helper contract, not HF generation.
    monkeypatch.setattr(runtime, 'transformers_generation_config', lambda policy: None)
    return base, artifact, tokenizer, manifest


def test_local_time_capability_is_saved_and_loaded(temporal_artifact):
    base, artifact, tokenizer, manifest = temporal_artifact
    assert manifest['local_time_policy'] == LOCAL_TIME_POLICY
    assert load_runtime_manifest(base, tokenizer, artifact) == manifest


@pytest.mark.parametrize('policy', [None, 'local-time-tools-unknown'])
def test_old_or_different_helper_training_is_not_silently_accepted(temporal_artifact, policy):
    base, artifact, tokenizer, manifest = temporal_artifact
    if policy is None:
        manifest.pop('local_time_policy')
    else:
        manifest['local_time_policy'] = policy
    (artifact / RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Local time tool policy.*regenerate SFT and retrain'):
        load_runtime_manifest(base, tokenizer, artifact)


def test_manifest_free_legacy_adapter_warns_helper_training_is_unverified(temporal_artifact, caplog):
    base, artifact, tokenizer, _ = temporal_artifact
    (artifact / RUNTIME_MANIFEST).unlink()
    assert load_runtime_manifest(base, tokenizer, artifact) is None
    assert 'identity cannot be verified' in caplog.text
    assert 'local time tool training is unverified' in caplog.text
