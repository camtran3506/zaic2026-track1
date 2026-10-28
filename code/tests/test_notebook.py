"""Ensure the standalone Kaggle notebook embeds current pipeline/training code."""
import ast
import json
from pathlib import Path


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
                    if any(isinstance(t, ast.Name) and t.id in {'download_script', 'hardware_script'} for t in node.targets):
                        compile(node.value.value, 'subprocess-script', 'exec')
    assert {'train_lora.py', 'prepare_sft.py', 'finetune.py', 'merge_lora.py', 'lifemate/sft.py'} <= embedded
    assert 'TRAIN_BACKEND = "unsloth"' in sources[0]
    assert 'TRAIN_LLM = True' in sources[0]
    assert any('command.extend(["--llm-adapter", ADAPTER_PATH])' in s for s in sources)
