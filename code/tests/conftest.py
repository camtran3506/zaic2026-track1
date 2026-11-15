"""Ensure pytest imports both code/ and code/training_code/ modules cleanly."""
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = CODE_DIR / "training_code"

for path in (TRAINING_DIR, CODE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
