"""Ensure the standalone Kaggle notebook embeds current pipeline/training code."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_notebook_sources_and_training_defaults():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / 'notebooks/LifeMate_Kaggle.ipynb').read_text(encoding='utf-8'))
    sources = [c['source'] for c in notebook['cells'] if c['cell_type'] == 'code']
    embedded = set()
    for source in sources:
        if source.startswith('%%writefile'):
            header, body = source.split('\n', 1)
            relative = header.split('/lifemate_app/', 1)[1]
            assert body == (root / 'code' / relative).read_text().rstrip() + '\n'
            compile(body, relative, 'exec')
            embedded.add(relative)
        else:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    if any(isinstance(t, ast.Name) and t.id in {'download_script', 'hardware_script', 'toolenv_check_script'} for t in node.targets):
                        compile(node.value.value, 'subprocess-script', 'exec')
    assert {'train_lora.py', 'prepare_sft.py', 'finetune.py', 'merge_lora.py', 'lifemate/sft.py'} <= embedded
    assert 'TRAIN_BACKEND = "unsloth"' in sources[0]
    assert 'TRAIN_LLM = True' in sources[0]
    assert any('command.extend(["--llm-adapter", ADAPTER_PATH])' in s for s in sources)


@pytest.mark.parametrize('missing', [False, True])
def test_toolenv_preflight_checks_factory_instance(tmp_path, missing):
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / 'notebooks/LifeMate_Kaggle.ipynb').read_text())
    scripts = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code' or cell['source'].startswith('%%'):
            continue
        for node in ast.walk(ast.parse(cell['source'])):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'toolenv_check_script' for t in node.targets):
                scripts.append(ast.literal_eval(node.value))
    assert len(scripts) == 1
    split_dir = tmp_path / 'mounted-split'
    split_dir.mkdir()
    (split_dir / 'public_test.jsonl').touch()
    # Organizer ToolEnv may be a factory, with the API present only on its result.
    methods = ['validate', 'call', 'get_tool_calls', 'get_state_hash']
    if missing:
        methods.remove('validate')
    (split_dir / 'zaic_tools.py').write_text(
        'from types import SimpleNamespace\n'
        'from pathlib import Path\n'
        'def forbidden(*args): raise AssertionError("Preflight must not execute tools")\n'
        'def ToolEnv(split, sample_id, root):\n'
        '    assert split == "public_test" and sample_id == "public_1"\n'
        '    assert (Path(root) / split).is_dir()\n'
        f'    return SimpleNamespace(**{{name: forbidden for name in {methods!r}}})\n')
    result = subprocess.run([sys.executable, '-c', scripts[0], split_dir, 'public_test', 'public_1'],
        cwd=root / 'code', env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True)
    if missing:
        assert result.returncode != 0
        assert 'missing callable API methods: validate' in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert 'ToolEnv instance public API OK' in result.stdout
