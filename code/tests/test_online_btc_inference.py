"""Tests for single-process BTC-compliant online inference, predict.sh, timing script, and notebook parity."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_fixture_split(root: Path, split_name: str = "private_test") -> tuple[Path, list[dict]]:
    split = root / split_name
    (split / "audios").mkdir(parents=True, exist_ok=True)
    (split / "audios" / "s1.wav").write_bytes(b"RIFF0001")
    (split / "audios" / "s2.wav").write_bytes(b"RIFF0002")

    # Different installed_apps schemas to prove original sequential order is preserved.
    items = [
        {
            "id": f"{split_name}_001",
            "audio": "audios/s1.wav",
            "context": {"installed_apps": {"notes": {"tools": ["notes.create"]}}},
        },
        {
            "id": f"{split_name}_002",
            "audio": "audios/s2.wav",
            "context": {"installed_apps": {"clock": {"tools": ["clock.set_alarm"]}, "notes": {"tools": ["notes.create"]}}},
        },
        {
            "id": f"{split_name}_003",
            "audio": "audios/s1.wav",
            "context": {"installed_apps": {"notes": {"tools": ["notes.create"]}}},
        },
    ]
    write_jsonl(split / f"{split_name}.jsonl", items)
    (split / f"{split_name}_tools_catalog.json").write_text(
        json.dumps(
            {
                "apps": [
                    {
                        "app_name": "notes",
                        "tools": [{
                            "name": "notes.create",
                            "kind": "write",
                            "description": "Create note",
                            "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
                        }],
                    },
                    {
                        "app_name": "clock",
                        "tools": [{
                            "name": "clock.set_alarm",
                            "kind": "write",
                            "description": "Set alarm",
                            "parameters": {"type": "object", "properties": {"time": {"type": "string"}}, "required": ["time"]},
                        }],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (split / "zaic_tools" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (split / "zaic_tools" / "__init__.py").write_text(
        """import json
from pathlib import Path

CREATED = []

class ToolEnv:
    def __init__(self, split, item_id, root=None):
        self.split = split
        self.item_id = item_id
        self.root = str(root)
        self.calls = []
        CREATED.append(item_id)

    def get_available_tools(self):
        return ['notes.create'] if self.item_id.endswith(('001', '003')) else ['clock.set_alarm', 'notes.create']

    def validate(self, name, args):
        return {'ok': True}

    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': args})
        return {'ok': True, 'tool': name}

    def get_tool_calls(self):
        return list(self.calls)

    def get_state_hash(self):
        return f"hash:{self.item_id}:{len(self.calls)}"
""",
        encoding="utf-8",
    )
    return split, items


def stage_app_with_fake_models(tmp_path: Path, *, with_asr_ft: bool = False) -> tuple[Path, Path, Path]:
    app = tmp_path / "code"
    app.mkdir()
    for fname in ("main.py", "predict.py", "predict.sh", "benchmark_timing.py", "predict_notebook.ipynb"):
        shutil.copy2(APP / fname, app / fname)
    shutil.copytree(APP / "lifemate", app / "lifemate")
    (app / "lifemate" / "models.py").write_text(
        """import json
import os
import time
from pathlib import Path


def _emit(payload: dict) -> None:
    path = os.environ.get('LIFEMATE_ONLINE_EVENTS')
    if path:
        with Path(path).open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\\n')


class LocalASR:
    def __init__(self, path, precision='auto', num_threads=4, **kwargs):
        _emit({
            'event': 'asr_load',
            'pid': os.getpid(),
            'path': str(path),
            'precision': precision,
            'num_threads': num_threads,
            'kwargs': kwargs,
            'hash_seed': os.environ.get('PYTHONHASHSEED'),
        })
        self.closed = False

    def transcribe(self, audio_path, hotwords=None):
        time.sleep(0.03)
        return f'transcribed:{Path(audio_path).name}'

    def close(self):
        self.closed = True


class LocalLLM:
    def __init__(self, path, **kwargs):
        _emit({
            'event': 'llm_load',
            'pid': os.getpid(),
            'path': str(path),
            'seed': kwargs.get('seed'),
            'adapter_path': str(kwargs.get('adapter_path')) if kwargs.get('adapter_path') else None,
        })
        self.step = 0
        self.closed = False

    def generate(self, system, user, thinking_budget=0):
        payload = json.loads(user)
        if not payload['previous_actions_and_results']:
            text = payload['request_transcript']
            if 's2.wav' in text.lower():
                return json.dumps({'action': 'call', 'name': 'clock.set_alarm', 'arguments': {'time': text}})
            return json.dumps({'action': 'call', 'name': 'notes.create', 'arguments': {'title': text}})
        return json.dumps({'action': 'done'})

    def close(self):
        self.closed = True
""",
        encoding="utf-8",
    )
    asr = app / "saved_models" / "asr"
    asr.mkdir(parents=True)
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (asr / name).write_bytes(b"fixture")
    if with_asr_ft:
        asr_ft = app / "saved_models" / "asr_ft"
        asr_ft.mkdir(parents=True)
        for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
            (asr_ft / name).write_bytes(b"fixture-ft")
    llm = app / "saved_models" / "llm"
    llm.mkdir(parents=True)
    (llm / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
    return app, asr, llm


def test_main_defaults_to_single_process_online_in_original_order(tmp_path):
    app, asr, llm = stage_app_with_fake_models(tmp_path)
    data_dir = tmp_path / "data"
    split_dir, items = make_fixture_split(data_dir, "private_test")
    events = tmp_path / "events.jsonl"
    output = tmp_path / "result" / "submission.jsonl"
    env = {
        **os.environ,
        "LIFEMATE_ONLINE_EVENTS": str(events),
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    res = subprocess.run(
        [
            sys.executable,
            str(app / "main.py"),
            "--data-root",
            str(data_dir),
            "--split",
            "auto",
            "--asr-model",
            str(asr),
            "--llm-model",
            str(llm),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stdout + res.stderr

    recorded = read_jsonl(events)
    assert [e["event"] for e in recorded] == ["asr_load", "llm_load"]
    assert recorded[0]["pid"] == recorded[1]["pid"]

    rows = read_jsonl(output)
    assert [r["id"] for r in rows] == [it["id"] for it in items]
    for r in rows:
        assert isinstance(r["time"], int)
        # ASR fixture sleeps 30ms; proves t1 starts before reading/transcribing audio.
        assert r["time"] >= 20
        assert len(r["tool_calls"]) == 1
        assert r["state_hash"].startswith("hash:")

    # Diagnostics file is not dumped alongside submission.jsonl in online mode unless requested.
    assert not output.with_suffix(".diagnostics.jsonl").exists()
    assert "Starting stage=online" in res.stderr


def test_predictor_flushes_line_by_line_before_later_sample_failure(tmp_path, monkeypatch):
    import lifemate.models as runtime_models
    from lifemate.predictor import LifeMatePredictor, PredictorConfig

    data_dir = tmp_path / "data"
    split_dir, items = make_fixture_split(data_dir, "public_test")
    asr = tmp_path / "asr"
    asr.mkdir()
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (asr / name).write_bytes(b"fixture")
    llm = tmp_path / "llm"
    llm.mkdir()
    output = tmp_path / "result" / "submission.jsonl"

    class FakeASR:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio_path, hotwords=None):
            if Path(audio_path).name == "s2.wav":
                # Sample 1 must already be flushed on disk before sample 2 runs!
                flushed = read_jsonl(output)
                assert len(flushed) == 1
                assert flushed[0]["id"] == items[0]["id"]
                raise RuntimeError("boom on sample 2")
            return f"transcribed:{Path(audio_path).name}"

        def close(self):
            pass

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, system, user, thinking_budget=0):
            payload = json.loads(user)
            if not payload["previous_actions_and_results"]:
                return json.dumps({"action": "call", "name": "notes.create", "arguments": {"title": "ok"}})
            return json.dumps({"action": "done"})

        def close(self):
            pass

    monkeypatch.setattr(runtime_models, "LocalASR", FakeASR)
    monkeypatch.setattr(runtime_models, "LocalLLM", FakeLLM)

    cfg = PredictorConfig(
        asr_model=asr,
        llm_model=llm,
        device="cpu",
        fail_fast=True,
    )

    with LifeMatePredictor(cfg) as pred:
        with pytest.raises(RuntimeError, match="boom on sample 2"):
            pred.predict_dataset(data_dir, output, split="auto")

    # Even after sample 2 failed fast, sample 1 remains flushed on disk.
    persisted = read_jsonl(output)
    assert len(persisted) == 1
    assert persisted[0]["id"] == items[0]["id"]


def test_predict_sh_notebook_and_timing_script_exact_parity(tmp_path):
    app, asr, llm = stage_app_with_fake_models(tmp_path, with_asr_ft=True)
    data_dir = tmp_path / "data"
    split_dir, items = make_fixture_split(data_dir, "private_test")

    sub_path = tmp_path / "result" / "submission.jsonl"
    jup_path = tmp_path / "result" / "jupyter_submission.jsonl"
    bench_path = tmp_path / "result" / "bench_submission.jsonl"
    report_path = tmp_path / "result" / "timing_report.json"
    events = tmp_path / "events.jsonl"

    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "DATA_ROOT": str(data_dir),
        "OUTPUT": str(sub_path),
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LIFEMATE_ONLINE_EVENTS": str(events),
    }

    # 1. Run predict.sh with --device cpu override (no model or split flags needed).
    res = subprocess.run(
        ["bash", str(app / "predict.sh"), "--device", "cpu"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "[predict] Model load time:" in res.stdout
    assert "[predict] Split=private_test | Samples=3" in res.stdout

    # Verify asr_ft auto-selection + BTC tuned ASR defaults + seed=42.
    recorded = read_jsonl(events)
    assert recorded[0]["path"].endswith("saved_models/asr_ft")
    assert recorded[0]["kwargs"]["max_active_paths"] == 8
    assert recorded[0]["kwargs"]["hotwords_mode"] == "auto"
    assert recorded[0]["kwargs"]["post_normalize"] is True
    assert recorded[0]["num_threads"] == 8
    assert recorded[0]["hash_seed"] == "42"
    assert recorded[1]["seed"] == 42

    # 2. Run benchmark_timing.py comparing against submission.jsonl.
    bench_res = subprocess.run(
        [
            sys.executable,
            str(app / "benchmark_timing.py"),
            "--data-root",
            str(data_dir),
            "--output",
            str(bench_path),
            "--device",
            "cpu",
            "--compare",
            str(sub_path),
            "--strict-compare",
            "--report-json",
            str(report_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert bench_res.returncode == 0, bench_res.stdout + bench_res.stderr
    assert "[Parity] matches=True" in bench_res.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["num_samples"] == len(items)
    assert report["summary"]["asr_mean_ms"] >= 20.0
    assert len(report["samples"]) == len(items)

    # 3. Execute predict_notebook.ipynb cells sequentially in a subprocess and check exact parity.
    nb = json.loads((app / "predict_notebook.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 4

    nb_runner = "\n\n".join(code_cells)
    nb_env = {
        **env,
        "PYTHONPATH": os.pathsep.join(p for p in [str(app), os.environ.get("PYTHONPATH", "")] if p),
        "LIFEMATE_DATA_ROOT": str(data_dir),
        "LIFEMATE_OUTPUT": str(jup_path),
        "LIFEMATE_DEVICE": "cpu",
    }
    nb_res = subprocess.run(
        [sys.executable, "-c", nb_runner],
        cwd=str(app),
        env=nb_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert nb_res.returncode == 0, nb_res.stdout + nb_res.stderr
    assert "[Step 1/4]" in nb_res.stdout
    assert "[Step 2/4]" in nb_res.stdout
    assert "[Step 3/4]" in nb_res.stdout
    assert '"matches": true' in nb_res.stdout

    sub_rows = read_jsonl(sub_path)
    jup_rows = read_jsonl(jup_path)
    assert [(r["id"], r["tool_calls"], r["state_hash"]) for r in sub_rows] == [
        (r["id"], r["tool_calls"], r["state_hash"]) for r in jup_rows
    ]
    assert all(isinstance(r["time"], int) and r["time"] >= 20 for r in jup_rows)


def test_btc_docker_and_jupyter_artifacts_contract():
    dockerfile = (APP / "Dockerfile").read_text(encoding="utf-8")
    start_jupyter = (APP / "start_jupyter.sh").read_text(encoding="utf-8")
    predict_sh = (APP / "predict.sh").read_text(encoding="utf-8")
    reqs = (APP / "requirements.txt").read_text(encoding="utf-8")

    # CUDA version <= 13.2 check on FROM image tag
    m = re.search(r"FROM\s+[^\s:]+:(\d+)\.(\d+)", dockerfile)
    assert m is not None
    major, minor = int(m.group(1)), int(m.group(2))
    assert (major, minor) <= (13, 2)

    assert "WORKDIR /code" in dockerfile
    assert "ubuntu24.04" in dockerfile
    assert "ENTRYPOINT" not in dockerfile
    assert 'CMD ["/bin/bash", "/code/predict.sh"]' in dockerfile
    assert "jupyterlab==4.4.10" in reqs
    for pin in (
        "torch==2.11.0+cu128",
        "torchaudio==2.11.0+cu128",
        "transformers==5.4.0",
        "peft==0.19.1",
        "accelerate==1.15.0",
        "sherpa_onnx==1.13.8",
        "onnxruntime==1.31.1",
        "numpy==2.5.4",
        "fla-core==0.6.0",
    ):
        assert pin in reqs
    assert "causal_conv1d-1.6.1" in dockerfile
    assert "flash-linear-attention==0.6.0" in dockerfile
    for copied in (
        "main.py",
        "predict.py",
        "benchmark_timing.py",
        "predict_notebook.ipynb",
        "predict.sh",
        "start_jupyter.sh",
        "run_server.sh",
        "requirements.txt",
        "README.md",
        "training_code/README.md",
        "training_code/score_train.py",
    ):
        assert copied in dockerfile or copied.split("/", 1)[0] + "/" in dockerfile
        assert (APP / copied).is_file()
    for copied_dir in ("lifemate/", "training_code/", "saved_models/"):
        assert copied_dir in dockerfile
        assert (APP / copied_dir.rstrip("/")).is_dir()

    assert "--port \"${JUPYTER_PORT:-9777}\"" in start_jupyter or "9777" in start_jupyter
    assert "zac2026" in start_jupyter
    assert "--allow-root" in start_jupyter
    assert "--no-browser" in start_jupyter
    assert "predict.py" in predict_sh
