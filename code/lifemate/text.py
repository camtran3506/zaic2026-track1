"""Shared utterance normalization for ASR, SFT, history and inference."""
import unicodedata


TEXT_NORMALIZATION = "nfc-whitespace-lower-initial-terminal-v1"


def normalize_transcript(text: str) -> str:
    """Normalize one entire turn without truncation or internal sentence splitting.

    Existing punctuation/numeric separators are retained. Only the first letter
    of the turn is capitalized, even when quotes or digits precede it. An ending
    period is added only when terminal punctuation is absent, checking before
    closing quotes/brackets too. Empty/whitespace-only text remains empty.
    """
    if not isinstance(text, str):
        raise TypeError("Transcript must be a string")
    text = " ".join(unicodedata.normalize("NFC", text).lower().split())
    if not text:
        return ""
    for index, char in enumerate(text):
        if char.isalpha():
            upper = char.upper()
            # Unicode capitalization can expand one letter (e.g. ß -> SS).
            # Keep only the initial letter uppercase so repeated calls are stable.
            text = text[:index] + upper[0] + upper[1:].lower() + text[index + 1:]
            break
    if not text.rstrip('\"\'”’»)]}').endswith((".", "?", "!", "…")):
        text += "."
    return unicodedata.normalize("NFC", text)
