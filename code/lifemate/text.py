"""Text normalization utilities for Vietnamese ASR transcripts and LLM prompts."""
from __future__ import annotations

import unicodedata


def normalize_text(text: str, normalize: bool = True) -> str:
    """Normalize Vietnamese text using the standard competition WER & LLM rules.

    Rules:
    1. Unicode NFC composition (canonical precomposed Vietnamese characters).
    2. Lowercasing (removes arbitrary ASR casing variations).
    3. Punctuation removal (replaces Unicode category 'P*' characters with spaces
       to avoid accidental word concatenations).
    4. Whitespace collapsing (strips leading/trailing spaces and collapses internal whitespace).
    """
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    if normalize:
        text = text.lower()
        text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())
