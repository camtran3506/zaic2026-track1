"""Deterministic Short Checked Reasoning synthesis and code-level verification.

Synthesizes concise one-line Vietnamese reasoning grounded exclusively in
visible prompt evidence (context, ITN candidates, date_reference, history,
and prior tool observations) and the current target decision, then verifies
all numbers and entity IDs before attaching them as SFT `reasoning_content`.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from .batching import BATCH_TOOL, observation_children


REASONING_POLICY = "deterministic-short-checked-reasoning-v1"
DEFAULT_SHORT_REASONING = False
DEFAULT_REASONING_WEIGHT = 0.25
DEFAULT_REASONING_STEPWISE_RATIO = 0.5
_MAX_REASONING_CHARS = 280
_HEADER_COUNT_RE = re.compile(
    r"^(?:Bước \d+:\s*(?:Gộp \d+ lệnh)?|Đã xử lý \d+ bước(?:\s*\([^)]*\))?;\s*|Gộp \d+ lệnh)"
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])")
_ID_TOKEN_RE = re.compile(r'"([^"\n]{1,64})"')
_FORBIDDEN_MARKERS = ("<think", "</think", "<tool_call", "</tool_call")


def _clean_line(text: str) -> str:
    cleaned = " ".join(str(text).split())
    for marker in ("<think>", "</think>", "<tool_call>", "</tool_call>"):
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


def _parse_json_maybe(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _scalar_strings(value, out: set[str]) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _scalar_strings(v, out)
    elif isinstance(value, list):
        for item in value:
            _scalar_strings(item, out)
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        out.add(str(value))
        if isinstance(value, float) and value.is_integer():
            out.add(str(int(value)))
    elif isinstance(value, str):
        out.add(value)
        for m in _NUMBER_RE.findall(value):
            out.add(m)


def _entity_label(record: dict) -> str | None:
    for key in ("name", "title", "label", "alias", "location"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return " ".join(val.split())[:40]
    return None


def _collect_id_provenance(value, source: str, out: dict[str, tuple[str, str | None]]) -> None:
    if isinstance(value, dict):
        label = _entity_label(value)
        for key, child in value.items():
            if (key == "id" or key.endswith("_id")) and isinstance(child, str) and child:
                out.setdefault(child, (source, label))
            _collect_id_provenance(child, source, out)
    elif isinstance(value, list):
        for item in value:
            _collect_id_provenance(item, source, out)


def _index_id_sources(payload: dict) -> dict[str, tuple[str, str | None]]:
    sources: dict[str, tuple[str, str | None]] = {}
    observations = payload.get("previous_actions_and_results") or []
    for obs in observation_children(observations):
        if "result" not in obs:
            continue
        parsed = _parse_json_maybe(obs.get("result"))
        _collect_id_provenance(parsed, obs.get("name", "tool"), sources)
    for turn in payload.get("conversation_history") or []:
        if not isinstance(turn, dict):
            continue
        calls = turn.get("tool_calls") or []
        results = turn.get("tool_results") or []
        for call, res in zip(calls, results):
            src = call.get("name") if isinstance(call, dict) and call.get("name") else "history"
            if isinstance(res, dict) and res.get("ok") is True and res.get("done") is not False:
                _collect_id_provenance(res, src, sources)
    ctx = payload.get("user_context") or {}
    if isinstance(ctx, dict):
        _collect_id_provenance(ctx.get("collections"), "user_context", sources)
        for key, val in ctx.items():
            if key not in {"installed_apps", "collections", "date_reference", "apps", "current_time", "timezone"}:
                if isinstance(val, (dict, list)):
                    _collect_id_provenance(val, "user_context", sources)
    return sources


def _index_scalar_sources(payload: dict) -> dict[str, str]:
    sources: dict[str, str] = {}
    for cand in payload.get("itn_candidates") or []:
        if isinstance(cand, dict) and "value" in cand and cand.get("source_text"):
            sources.setdefault(str(cand["value"]), f"ITN '{cand['source_text']}'")
    for hist_entry in payload.get("itn_history_candidates") or []:
        if isinstance(hist_entry, dict):
            for cand in hist_entry.get("candidates") or []:
                if isinstance(cand, dict) and "value" in cand and cand.get("source_text"):
                    sources.setdefault(str(cand["value"]), f"ITN history '{cand['source_text']}'")
    ctx = payload.get("user_context") or {}
    if isinstance(ctx, dict):
        date_ref = ctx.get("date_reference")
        if isinstance(date_ref, dict):
            for entry in date_ref.get("days") or []:
                if isinstance(entry, str) and len(entry) >= 10:
                    iso_day, _, desc = entry.partition(" ")
                    label = desc.strip() or iso_day
                    sources.setdefault(iso_day, f"date_reference '{label}'")
            for rel_label, iso_day in (date_ref.get("relative_dates") or {}).items():
                if isinstance(iso_day, str):
                    sources.setdefault(iso_day, f"date_reference '{rel_label}'")
            for week_label, days in (date_ref.get("weeks") or {}).items():
                if isinstance(days, dict):
                    for day_label, iso_day in days.items():
                        if isinstance(iso_day, str):
                            sources.setdefault(iso_day, f"date_reference '{day_label} {week_label}'")
        cur = ctx.get("current_time")
        if isinstance(cur, str) and len(cur) >= 10:
            sources.setdefault(cur[:10], "current_time")
    for obs in observation_children(payload.get("previous_actions_and_results") or []):
        if "result" not in obs:
            continue
        parsed = _parse_json_maybe(obs.get("result"))
        obs_vals: set[str] = set()
        _scalar_strings(parsed, obs_vals)
        src_name = obs.get("name", "tool")
        for val in obs_vals:
            sources.setdefault(val, f"từ {src_name}")
    return sources


def _argument_notes(
    name: str,
    arguments: dict,
    schemas: dict[str, dict],
    id_sources: dict[str, tuple[str, str | None]],
    scalar_sources: dict[str, str],
) -> list[str]:
    if not isinstance(arguments, dict):
        return []
    props = (schemas.get(name, {}).get("parameters") or {}).get("properties") or {}
    notes: list[str] = []
    for key, val in arguments.items():
        if len(notes) >= 3:
            break
        prop = props.get(key, {}) if isinstance(props.get(key), dict) else {}
        is_xref = bool(prop.get("x-ref")) or key == "id" or key.endswith("_id")
        if is_xref and isinstance(val, str) and val in id_sources:
            src, label = id_sources[val]
            label_part = f" ({label})" if label and label not in val else ""
            notes.append(f'{key}="{val}"{label_part} từ {src}')
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            val_str = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
            src = scalar_sources.get(val_str)
            notes.append(f"{key}={val_str} ({src})" if src else f"{key}={val_str}")
            continue
        if isinstance(val, str) and val.strip():
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?", val):
                src = scalar_sources.get(val) or scalar_sources.get(val[:10])
                notes.append(f"{key}={val} ({src})" if src else f"{key}={val}")
            elif val in id_sources:
                src, label = id_sources[val]
                notes.append(f'{key}="{val}" (từ {src})')
            elif val in scalar_sources and (len(val) <= 32 or key in {"phone", "email"}):
                src = scalar_sources.get(val)
                notes.append(f'{key}="{val}" ({src})')
            elif key == "query" and len(val) <= 40:
                notes.append(f'query="{val}"')
    return notes


def verify_reasoning_line(
    line: str,
    payload: dict,
    target: dict,
    *,
    observed_ids: dict[str, set[str]] | None = None,
) -> bool:
    """Check that `line` is single-line, bounded, and cites only grounded numbers/IDs."""
    if not isinstance(line, str) or not line.strip() or "\n" in line or "\r" in line:
        return False
    if len(line) > _MAX_REASONING_CHARS:
        return False
    lowered = line.casefold()
    if any(marker in lowered for marker in _FORBIDDEN_MARKERS):
        return False

    allowed_values: set[str] = set()
    for field in (
        "request_transcript",
        "itn_candidates",
        "itn_history_candidates",
        "user_context",
        "conversation_history",
        "previous_actions_and_results",
    ):
        _scalar_strings(payload.get(field), allowed_values)
    _scalar_strings(target.get("arguments"), allowed_values)
    for ids in (observed_ids or {}).values():
        allowed_values.update(ids)

    body = _HEADER_COUNT_RE.sub("", line, count=1)
    for num in _NUMBER_RE.findall(body):
        if num not in allowed_values:
            return False
    for quoted in _ID_TOKEN_RE.findall(body):
        if re.fullmatch(r"[a-z]+_\w+", quoted) and quoted not in allowed_values:
            return False
    return True


def synthesize_checked_reasoning(
    payload: dict,
    target: dict,
    step_idx: int = 0,
    *,
    observed_ids: dict[str, set[str]] | None = None,
) -> str:
    """Synthesize a deterministic 1-line checked reasoning string for one SFT step."""
    step_num = max(1, int(step_idx) + 1)
    observations = payload.get("previous_actions_and_results") or []
    flat_obs = observation_children(observations)
    schemas = {
        s["name"]: s
        for s in ((payload.get("available_tools") or {}).get("tool_schemas") or [])
        if isinstance(s, dict) and s.get("name")
    }
    id_sources = _index_id_sources(payload)
    scalar_sources = _index_scalar_sources(payload)

    action = target.get("action")
    tail_rejected = observations[-1] if observations and "result" not in observations[-1] else None

    if tail_rejected is not None:
        stage = tail_rejected.get("stage", "feedback")
        failed_name = tail_rejected.get("name", "lệnh trước")
        if action == "done":
            candidate = _clean_line(
                f"Phản hồi {stage} từ {failed_name}: không thể thực hiện hợp lệ -> kết thúc."
            )
        else:
            t_name = target.get("name", "")
            t_args = target.get("arguments") or {}
            notes = _argument_notes(t_name, t_args, schemas, id_sources, scalar_sources)
            note_suffix = f" ({'; '.join(notes)})" if notes else ""
            if stage == "xref_guard":
                candidate = _clean_line(
                    f"Phản hồi xref_guard từ {failed_name}: cần ID từ bước đọc trước -> gọi {t_name}{note_suffix}."
                )
            elif stage == "validate":
                candidate = _clean_line(
                    f"Phản hồi validate từ {failed_name}: sửa tham số theo schema{note_suffix} -> gọi {t_name}."
                )
            else:
                candidate = _clean_line(
                    f"Phản hồi {stage} từ {failed_name}: điều chỉnh kế hoạch{note_suffix} -> gọi {t_name}."
                )
    elif action == "done":
        env_calls = [
            o["name"] for o in flat_obs
            if "result" in o and isinstance(o.get("name"), str) and not o["name"].startswith("local_")
        ]
        if env_calls:
            shown = ", ".join(env_calls[:5]) + ("..." if len(env_calls) > 5 else "")
            candidate = _clean_line(
                f"Đã xử lý {len(env_calls)} bước ({shown}); không còn hành động nào khác -> kết thúc."
            )
        elif any("result" in o for o in flat_obs):
            candidate = _clean_line("Đã kiểm tra qua helper và không cần gọi công cụ ứng dụng -> kết thúc.")
        else:
            candidate = _clean_line("Không có hành động ứng dụng cần thực hiện -> kết thúc.")
    else:
        name = str(target.get("name", ""))
        args = target.get("arguments") or {}
        if name == BATCH_TOOL:
            child_calls = [c for c in args.get("calls", []) if isinstance(c, dict) and c.get("name")]
            child_names = [c["name"] for c in child_calls]
            notes: list[str] = []
            for child in child_calls:
                notes.extend(
                    _argument_notes(child["name"], child.get("arguments") or {}, schemas, id_sources, scalar_sources)
                )
                if len(notes) >= 2:
                    break
            note_str = f"; {'; '.join(notes[:2])}" if notes else ""
            candidate = _clean_line(
                f"Bước {step_num}: Gộp {len(child_names)} lệnh độc lập ({', '.join(child_names)}){note_str} -> gọi {BATCH_TOOL}."
            )
        elif name == "sys.ask_missing_info":
            target_tool = args.get("tool_name", "")
            params = args.get("params") or []
            param_str = ", ".join(str(p) for p in params) if isinstance(params, list) else str(params)
            candidate = _clean_line(
                f"Bước {step_num}: Thiếu thông tin bắt buộc ({param_str}) cho {target_tool} -> gọi sys.ask_missing_info."
            )
        elif name.startswith("local_"):
            task = args.get("task")
            detail = f"task={task}" if task else (args.get("operation") or args.get("kind") or "dữ liệu nguồn")
            candidate = _clean_line(
                f"Bước {step_num}: Chuẩn hóa/kiểm chứng {detail} từ dữ liệu đã có -> gọi {name}."
            )
        else:
            notes = _argument_notes(name, args, schemas, id_sources, scalar_sources)
            env_prev = [
                o["name"] for o in flat_obs
                if "result" in o and isinstance(o.get("name"), str) and not o["name"].startswith("local_")
            ]
            prefix_note = f"Sau {env_prev[-1]}; " if env_prev else ""
            if notes:
                detail = "; ".join(notes)
            elif schemas.get(name, {}).get("kind") == "read":
                detail = f"Tra cứu dữ liệu qua {name}"
            else:
                detail = f"Thực thi {name} theo yêu cầu"
            candidate = _clean_line(f"Bước {step_num}: {prefix_note}{detail} -> gọi {name}.")

    if len(candidate) > _MAX_REASONING_CHARS or not verify_reasoning_line(
        candidate, payload, target, observed_ids=observed_ids
    ):
        if action == "done":
            return "Không còn hành động nào khác -> kết thúc."
        return _clean_line(f"Bước {step_num}: gọi {target.get('name', 'action')}.")
    return candidate


def attach_reasoning_to_trajectory(steps: list[dict], *, step_idx_offset: int = 0) -> list[dict]:
    """Attach deterministic verified `reasoning_content` onto each SFT step copy."""
    enriched: list[dict] = []
    for idx, step in enumerate(steps):
        row = deepcopy(step)
        payload = json.loads(row["messages"][1]["content"])
        target = json.loads(row["messages"][-1]["content"])
        reason = synthesize_checked_reasoning(payload, target, step_idx_offset + idx)
        row["reasoning"] = reason
        row["messages"][-1]["reasoning_content"] = reason
        enriched.append(row)
    return enriched


def select_reasoning_stepwise_rows(
    grouped_steps: list[dict],
    *,
    ratio: float = DEFAULT_REASONING_STEPWISE_RATIO,
    seed: int = 42,
) -> list[dict]:
    """Select intermediate `<tool_call>` turns to emit as step-wise reasoning SFT rows.

    In Qwen3.5's chat template, `<think>` blocks on earlier assistant turns before
    the last `user` message (`automatic_execution_feedback`) are elided during
    `full_trajectory=True` rendering. Emitting selected intermediate turns as
    standalone step-wise rows ensures `<think>{reasoning}</think>` before `<tool_call>`
    is supervised as well as `done`.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("reasoning_stepwise_ratio must be between 0 and 1")
    if ratio == 0.0 or len(grouped_steps) <= 1:
        return []
    selected: list[dict] = []
    # Only steps strictly before steps[-1] lack `<think>` supervision in collapse_trajectory.
    for idx, step in enumerate(grouped_steps[:-1]):
        if step.get("train", True) is False:
            continue
        aug_key = json.dumps(step.get("augmentation"), sort_keys=True, ensure_ascii=False)
        key = f"{seed}:{step.get('sample_id')}:{aug_key}:{idx}".encode("utf-8")
        bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0x100000000
        if bucket >= ratio:
            continue
        row = deepcopy(step)
        row["full_trajectory"] = False
        row["reasoning_stepwise"] = True
        selected.append(row)
    return selected
