"""Shared verbatim history handling for ASR caches, SFT and inference."""


def history_turn_text(turn):
    for key in ("audio_text", "text", "content", "transcript"):
        text = turn.get(key)
        if isinstance(text, str) and text.strip():
            return text
    return None


def history_entries(history):
    """Yield text/audio for every role without applying prompt limits."""
    if not isinstance(history, list):
        return
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", turn.get("speaker", "user")))
        audio = turn.get("audio", turn.get("audio_path"))
        yield role, history_turn_text(turn), audio if isinstance(audio, str) else None


def resolve_history(history, transcribe):
    """Resolve audio-only turns for all roles and retain full, nonempty text."""
    result = []
    for role, text, audio in history_entries(history):
        if text is None and audio:
            text = transcribe(audio)
        if text:
            result.append({"role": role, "text": text})
    return result


def prompt_history(history):
    """Filter first, then select twelve turns and apply the existing text limit."""
    turns = [turn for turn in history if turn["text"]]
    return [{"role": turn["role"], "text": turn["text"][:1700]} for turn in turns[-12:]]
