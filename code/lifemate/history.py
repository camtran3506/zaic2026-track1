"""Shared verbatim history handling for ASR caches, SFT and inference."""
from copy import deepcopy


def history_tool_pairs(turn):
    """Preserve paired structured assistant calls/results for the planner."""
    if turn.get("role", turn.get("speaker")) != "assistant":
        return []
    calls, results = turn.get("tool_calls"), turn.get("tool_results")
    if not isinstance(calls, list) or not isinstance(results, list) or len(calls) != len(results):
        return []
    return [(call, result) for call, result in zip(calls, results)
            if isinstance(call, dict) and isinstance(call.get("name"), str)
            and isinstance(call.get("arguments"), dict)]


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
    for turn in history if isinstance(history, list) else []:
        if not isinstance(turn, dict):
            continue
        role, text, audio = next(history_entries([turn]))
        if text is None and audio:
            text = transcribe(audio)
        pairs = history_tool_pairs(turn)
        if text or pairs:
            resolved = {"role": role, "text": text or ""}
            if pairs:
                resolved["tool_calls"] = deepcopy([call for call, _ in pairs])
                resolved["tool_results"] = deepcopy([value for _, value in pairs])
            result.append(resolved)
    return result


def prompt_history(history):
    """Retain every usable turn and its complete text and paired tool evidence."""
    turns = [turn for turn in history if turn["text"] or history_tool_pairs(turn)]
    result = []
    for turn in turns:
        row = {"role": turn["role"], "text": turn["text"]}
        pairs = history_tool_pairs(turn)
        if pairs:
            row["tool_calls"] = deepcopy([call for call, _ in pairs])
            row["tool_results"] = deepcopy([value for _, value in pairs])
        result.append(row)
    return result
