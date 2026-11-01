"""Transcript normalization shared by ASR, training text and spoken history."""
import unicodedata
import re


TEXT_NORMALIZATION_VERSION = "nfc_lower_preserve_numeric_v2"

# Keep numeric structure without guessing locale, date order or ASR corrections.
# A separator belongs to a number only when followed by another digit, so final
# sentence punctuation is not swallowed. Also retain signs and percentages.
_NUMERIC = re.compile(r"(?<!\w)[+-]?\d+(?:[/:.,-]\d+)*(?:\s*%)?(?!\w)")


def normalize_transcript(text: str) -> str:
    """LLM input: normalize prose but retain written times/dates/numeric values.

    Preserve e.g. 08:30, 29/10/2026, 2026-10-29, 3.14, 1.000,50, -5 and 20%.
    Do not validate or reinterpret these values, or infer punctuation from words.
    """
    text = unicodedata.normalize("NFC", text).lower()
    parts = []
    end = 0
    for match in _NUMERIC.finditer(text):
        parts.append("".join(" " if unicodedata.category(c).startswith("P") else c
                             for c in text[end:match.start()]))
        parts.append(match.group())
        end = match.end()
    parts.append("".join(" " if unicodedata.category(c).startswith("P") else c
                         for c in text[end:]))
    return " ".join("".join(parts).split())


def history_turn_text(turn: dict) -> str | None:
    """Normalize spoken text; preserve assistant/system text. None means absent.

    An existing punctuation-only transcript normalizes to an empty string, but
    must not trigger another ASR pass as if the source text were missing.
    """
    role = str(turn.get("role", turn.get("speaker", "user")))
    for key in ("audio_text", "text", "content", "transcript"):
        text = turn.get(key)
        if isinstance(text, str) and text.strip():
            if key in ("audio_text", "transcript") or role.lower() in ("user", "human"):
                return normalize_transcript(text)
            return text
    return None
