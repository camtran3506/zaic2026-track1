"""Exercise model selection through the shell launcher and the real CLI parser."""
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest

APP = Path(__file__).resolve().parents[1]


@pytest.fixture
def server(tmp_path):
    app = tmp_path / "app with spaces"
    app.mkdir()
    script = app / "run_server.sh"
    shutil.copy2(APP / "run_server.sh", script)
    shim = tmp_path / "python"
    shim.write_text(f"""#!{sys.executable}
import json, sys
sys.path.insert(0, {str(APP)!r})
import main
start = next(i for i, value in enumerate(sys.argv) if value.endswith('/main.py'))
sys.argv = sys.argv[start:]
args = main.parse_args()
print(json.dumps({{'asr': str(args.asr_model), 'llm': str(args.llm_model),
                  'adapter': str(args.llm_adapter), 'gpu': args.cuda_visible_devices}}))
""")
    shim.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "ASR_MODEL": "/stale/phowhisper-small-ct2", "LLM_MODEL": "/stale/base-llm"}
    return app, script, env


@pytest.mark.parametrize("extra", [
    [],
    ["--asr-model", "/stale/cli-asr", "--llm-model", "/stale/cli-llm"],
    ["--asr-model=/stale/cli-asr", "--llm-model=/stale/cli-llm"],
    ["--asr-m=/stale/cli-asr", "--llm-mo=/stale/cli-llm"],
])
def test_launcher_uses_configured_folders_despite_stale_environment_and_cli(server, extra, tmp_path):
    app, script, env = server
    # An unrelated CWD and spaces in the checkout path must not select another model.
    result = subprocess.run(["bash", script, *extra, "--cuda-visible-devices", "2",
                             "--llm-adapter", "/trained adapter"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)
    assert args["asr"] == str(app / "models/asr")
    assert args["llm"] == str(app / "models/llm")
    assert args["adapter"] == "/trained adapter"
    assert args["gpu"] == "2"
    assert f"[run_server] ASR_MODEL={app / 'models/asr'}" in result.stderr


def test_editing_launcher_folder_is_respected(server):
    app, script, env = server
    text = script.read_text().replace('ASR_MODEL="${SCRIPT_DIR}/models/asr"',
                                      'ASR_MODEL="${SCRIPT_DIR}/weights/new asr"')
    text = text.replace('LLM_MODEL="${SCRIPT_DIR}/models/llm"',
                        'LLM_MODEL="${SCRIPT_DIR}/weights/my base"')
    script.write_text(text)
    result = subprocess.run(["bash", script], env=env, capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)
    assert args["asr"] == str(app / "weights/new asr")
    assert args["llm"] == str(app / "weights/my base")


def test_download_default_matches_launcher_and_only_asr_preserves_llm(server, monkeypatch):
    import huggingface_hub
    app, _, _ = server
    scripts = app / "scripts"
    scripts.mkdir()
    script = scripts / "download_models.py"
    shutil.copy2(APP / "scripts/download_models.py", script)
    llm = app / "models/llm"
    llm.mkdir(parents=True)
    weights = llm / "model.safetensors"
    weights.write_bytes(b"existing base used to train adapter")
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        for name in (
            "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx",
            "encoder.onnx", "decoder.onnx", "joiner.onnx",
            "tokens.txt", "bpe.model", "model.pt",
        ):
            (Path(kwargs["local_dir"]) / name).write_bytes(b"downloaded ASR")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.chdir(app.parent)  # Download location is independent of CWD.
    monkeypatch.setattr(sys, "argv", [str(script), "--only", "asr"])
    runpy.run_path(str(script), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "g-group-ai-lab/gipformer1.5-68M-rnnt"
    for required in ("encoder.int8.onnx", "encoder.onnx", "bpe.model", "model.pt", "tokens.txt"):
        assert required in calls[0]["allow_patterns"]
    assert calls[0]["local_dir"] == str(app / "models/asr")
    assert weights.read_bytes() == b"existing base used to train adapter"


def test_full_server_pipeline_passes_configured_models_to_both_children(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    for name in ("main.py", "run_server.sh"):
        shutil.copy2(APP / name, app / name)
    shutil.copytree(APP / "lifemate", app / "lifemate", ignore=shutil.ignore_patterns("__pycache__"))
    # A local adapter stub exercises real process/argument/cache wiring, not ASR quality.
    (app / "lifemate/models.py").write_text('''import json, os
from pathlib import Path
def event(kind, path, adapter=None):
    with open(os.environ['MODEL_PATH_EVENTS'], 'a') as out:
        out.write(json.dumps(dict(kind=kind, path=str(path), adapter=str(adapter), pid=os.getpid()))+'\\n')
class LocalASR:
    def __init__(self, path, **kwargs): event('asr', path)
    def transcribe(self, path): return 'bật đèn'
class LocalLLM:
    def __init__(self, path, **kwargs): event('llm', path, kwargs.get('adapter_path'))
    def generate(self, system, user):
        assert json.loads(user)['request_transcript'] == 'Bật đèn.'
        return '{"action":"done"}'
''')
    asr = app / "models/asr"
    asr.mkdir(parents=True)
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (asr / name).write_bytes(b"fixture checkpoint")
    split = tmp_path / "dataset/public_test"
    split.mkdir(parents=True)
    (split / "audio.wav").write_bytes(b"fixture audio")
    (split / "public_test.jsonl").write_text(json.dumps({
        "id": "public_1", "audio": "audio.wav", "context": {}, "history": []}) + "\n")
    (split / "public_test_tools_catalog.json").write_text('{"apps": []}')
    (split / "zaic_tools.py").write_text('''class ToolEnv:
    def __init__(self, *args, **kwargs): pass
    def get_tool_calls(self): return []
    def get_state_hash(self): return 'fixture-state'
''')
    events = tmp_path / "events.jsonl"
    adapter = tmp_path / "trained adapter"
    output = tmp_path / "submission.jsonl"
    cache = tmp_path / "transcripts.jsonl"
    env = {**os.environ, "ASR_MODEL": "/old/phowhisper-small-ct2", "LLM_MODEL": "/old/llm",
           "CUDA_VISIBLE_DEVICES": "", "MODEL_PATH_EVENTS": str(events), "OMP_NUM_THREADS": "1",
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    command = ["bash", app / "run_server.sh", "--device", "cpu",
        "--data-root", split.parent, "--output", output, "--transcript-cache", cache,
        "--llm-adapter", adapter, "--asr-model", "/old/cli-asr", "--llm-model", "/old/cli-llm"]
    result = subprocess.run(command,
        env=env, capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = [json.loads(line) for line in events.read_text().splitlines()]
    assert [e["kind"] for e in recorded] == ["asr", "llm"]
    assert [e["path"] for e in recorded] == [str(asr), str(app / "models/llm")]
    assert recorded[0]["pid"] != recorded[1]["pid"]
    assert recorded[1]["adapter"] == str(adapter)
    assert json.loads(output.read_text())["state_hash"] == "fixture-state"
    assert json.loads(cache.with_suffix(".manifest.json").read_text())["asr"]["model_path"] == str(asr)
    assert "phowhisper-small-ct2" not in result.stderr
    assert "Starting stage=full" in result.stderr
    assert "Starting stage=asr" in result.stdout
    assert "Starting stage=llm" in result.stdout
    # Replacing a checkpoint in the same configured folder must not reuse old text.
    (asr / "encoder.int8.onnx").write_bytes(b"a different ASR checkpoint")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert result.returncode != 0
    assert "ASR model/settings changed" in result.stdout + result.stderr
    assert len(events.read_text().splitlines()) == 2  # No extra model was loaded.
    # A missing configured folder is an error; do not search a previous model path.
    shutil.rmtree(asr)
    fallback = tmp_path / "previous-model"
    fallback.mkdir()
    env["ASR_MODEL"] = str(fallback)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert result.returncode != 0
    assert f"FileNotFoundError: {asr}" in result.stdout + result.stderr
    assert len(events.read_text().splitlines()) == 2
