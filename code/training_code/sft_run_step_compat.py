"""Reproducibility compatibility profile for `prepare_sft.py --sft-profile run_sft_step`.

Keeps runtime inference guards and fallback hints in `lifemate/` untouched while
letting `training_code/prepare_sft.py` replay SFT trajectories with the exact
Nov-13 (`79fdce3`) guard/execution semantics used to train `saved_models/llm_ft`.
"""
from __future__ import annotations

from contextlib import contextmanager
from difflib import get_close_matches
import json
import re

import lifemate.execution_policy as execution_policy
import lifemate.missing_info_guard as missing_info_guard
import lifemate.recovery as recovery
import lifemate.sft as sft
import lifemate.tool_execution as tool_execution
from lifemate.eval_trace import exception_details

DEFAULT_SFT_PROFILE = "run_sft_step"
SFT_PROFILES = ("run_sft_step", "current")


def legacy_check_tool_access(name, catalog, enabled, local_names=()):
    """Exact `79fdce3` exact-name access check without alternative-read fallback hints."""
    if isinstance(name, str) and name in local_names:
        return {"allowed": True, "status": "allowed_local_helper", "scope": "local"}
    if isinstance(name, str) and name in enabled:
        return {"allowed": True, "status": "allowed_env_tool", "scope": "env"}
    status = (
        "invalid_tool_name" if not isinstance(name, str)
        else "tool_not_allowed" if name in catalog
        else "unknown_tool"
    )
    exposed = sorted(set(enabled) | set(local_names))
    suggestions = get_close_matches(name, exposed, n=3, cutoff=0.45) if isinstance(name, str) else []
    return {
        "allowed": False,
        "status": status,
        "scope": "none",
        "suggested_tools": suggestions,
        "instruction": (
            "Tool không tồn tại hoặc không được sample cho phép. Chọn đúng tên trong tool_schemas; "
            "gợi ý chỉ để đối chiếu, không tự thay tool hoặc đoán mục tiêu."
        ),
    }


def legacy_entity_value(target, argument, prop, text, context, history, observations, tools):
    """Exact `79fdce3` visible-collection lookup before phone/sms/email/account fallbacks."""
    collection = prop.get("x-ref")
    projection = "id"
    if not collection and argument in {"phone", "phone_number", "email"} and target in missing_info_guard._CONTACT_TOOLS:
        collection, projection = "contacts.contacts", "email" if argument == "email" else "phone"
    if not isinstance(collection, str) or not text or missing_info_guard._UNCERTAIN.search(text):
        return None
    candidates, ids, matches = [], {}, {}
    for row in missing_info_guard._records(collection, context, history, observations, tools):
        value = row.get(projection, row.get(argument))
        if not missing_info_guard._valid_value(value, prop):
            continue
        labels = [row.get(k) for k in ("name", "title", "label", "relation")]
        labels += row.get("aliases", []) if isinstance(row.get("aliases"), list) else []
        labels = [v for v in labels if isinstance(v, str) and missing_info_guard._fold(v)]
        identity = json.dumps(value, sort_keys=True)
        candidates.append((identity, labels))
        explicit_id = (
            projection == "id"
            and isinstance(value, str)
            and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text)
        )
        if explicit_id:
            ids[identity] = value
        named = [v for v in labels if missing_info_guard._mentioned(v, text)]
        if named:
            matches[identity] = (value, named)
    if len(ids) == 1:
        return next(iter(ids.values())), {"source": "explicit_visible_id", "collection": collection}
    if ids:
        return None
    if len(matches) != 1:
        return None
    identity, (value, labels) = next(iter(matches.items()))
    # "Lộc" remains ambiguous when the visible list also contains "Anh Lộc".
    if not any(
        not any(other != identity and any(missing_info_guard._mentioned(label, v) for v in names) for other, names in candidates)
        for label in labels
    ):
        return None
    return value, {"source": "visible_collection", "collection": collection}


def legacy_precheck_call(
    name,
    arguments,
    catalog,
    enabled,
    monitor,
    local_names=(),
    *,
    transcript="",
    context=None,
    history=(),
    observations=(),
):
    """Exact `79fdce3` precheck before runtime missing-required/stagnant/travel loop guards."""
    access = legacy_check_tool_access(name, catalog, enabled, local_names)
    if not access["allowed"]:
        return {
            "stage": "precheck",
            "name": name,
            "arguments": arguments,
            "error": access["instruction"],
            "access": access,
        }
    if not isinstance(arguments, dict) or execution_policy._canonical(arguments) is None:
        return {
            "stage": "precheck",
            "name": name,
            "arguments": arguments,
            "error": "arguments must be a finite JSON object",
        }
    if access["scope"] == "local":
        return None
    target = arguments.get("tool_name")
    if name == "sys.ask_missing_info" and (not isinstance(target, str) or target not in enabled or target == name):
        return {
            "stage": "precheck",
            "name": name,
            "arguments": arguments,
            "error": "Missing-info tool_name must be another available application tool.",
        }
    if name == "sys.ask_missing_info":
        feedback = missing_info_guard.check_missing_info(
            arguments,
            enabled,
            transcript=transcript,
            context=context,
            history=history,
            observations=observations,
        )
        if feedback:
            return {
                "stage": "missing_info_guard",
                "name": name,
                "arguments": arguments,
                "error": feedback["instruction"],
                "feedback": feedback,
            }
    feedback = monitor.check(name, arguments, enabled[name].get("kind", "unknown"))
    if feedback:
        return {
            "stage": "loop_guard",
            "name": name,
            "arguments": arguments,
            "error": feedback["instruction"],
            "feedback": feedback,
        }
    return None


def legacy_attempt_env_call(
    env,
    name,
    arguments,
    kind,
    monitor,
    call_index,
    *,
    emit=None,
    spec=None,
    observed_ids=None,
    **_ignored_runtime_context,
):
    """Exact `79fdce3` env boundary without dynamic search-fallback or xref hints."""
    emit = emit or (lambda event, **data: None)
    emit("validate_start", name=name, arguments=arguments)
    try:
        validation = env.validate(name, arguments)
    except Exception as exc:
        details = exception_details(exc)
        monitor.record_technical_failure(name, arguments, details, stage="validate")
        emit("validate_error", name=name, arguments=arguments, exception=details)
        return {
            "executed": False,
            "feedback": {
                "error": str(exc),
                "stage": "validate",
                "name": name,
                "arguments": arguments,
                "exception": details,
                "origin": "validation_exception",
            },
        }
    accepted = isinstance(validation, dict) and validation.get("ok") is True
    emit("validate_result", name=name, arguments=arguments, result=validation, accepted=accepted)
    if not accepted:
        schema_rejected = isinstance(validation, dict) and validation.get("ok") is False
        if schema_rejected:
            monitor.record_validation_failure(name, arguments, validation)
        else:
            monitor.record_technical_failure(name, arguments, validation, stage="validate_protocol")
        return {
            "executed": False,
            "feedback": {
                "error": f"Schema invalid: {validation!r}" if schema_rejected else f"Invalid validation response: {validation!r}",
                "stage": "validate",
                "name": name,
                "arguments": arguments,
                "validation": validation,
                "origin": "schema_rejection" if schema_rejected else "validation_protocol",
            },
        }
    if observed_ids is not None and spec is not None:
        xref_error = tool_execution.check_observed_xref(arguments, spec, observed_ids)
        if xref_error is not None:
            emit("xref_guard_rejected", name=name, arguments=arguments, validation=xref_error)
            return {
                "executed": False,
                "feedback": {
                    "error": xref_error["error"],
                    "stage": "xref_guard",
                    "name": name,
                    "arguments": arguments,
                    "validation": xref_error,
                    "origin": "observed_id_guard",
                },
            }
    emit("call_start", name=name, arguments=arguments, call_index=call_index)
    execution_exception = None
    try:
        result = env.call(name, arguments)
    except Exception as exc:
        execution_exception = exception_details(exc)
        result = {"execution_exception": execution_exception}
        emit("call_error", name=name, arguments=arguments, call_index=call_index,
             exception=execution_exception, agent_result=result)
    else:
        emit("call_result", name=name, arguments=arguments, call_index=call_index, result=result)
    tool_execution.update_observed_ids(name, arguments, result, kind, observed_ids)
    verification = monitor.record_result(
        name,
        arguments,
        result,
        kind,
        call_index=call_index,
        execution_exception=execution_exception,
    )
    emit("execution_verified", name=name, verification=verification)
    if verification.get("contract_violation"):
        emit("execution_contract_violation", name=name, arguments=arguments, result=result)
    return {
        "executed": True,
        "result": result,
        "verification": verification,
        "automatic_execution_feedback": monitor.automatic_feedback(kind, verification, call_index),
    }


@contextmanager
def use_sft_replay_profile(profile: str = DEFAULT_SFT_PROFILE):
    """Temporarily bind legacy `79fdce3` SFT-replay hooks within the active process."""
    if profile not in SFT_PROFILES:
        raise ValueError(f"Unknown SFT profile {profile!r}; expected one of {SFT_PROFILES}")
    if profile == "current":
        yield
        return

    patches = (
        (execution_policy, "check_tool_access", legacy_check_tool_access),
        (tool_execution, "check_tool_access", legacy_check_tool_access),
        (missing_info_guard, "_entity_value", legacy_entity_value),
        (tool_execution, "precheck_call", legacy_precheck_call),
        (sft, "precheck_call", legacy_precheck_call),
        (recovery, "precheck_call", legacy_precheck_call),
        (tool_execution, "attempt_env_call", legacy_attempt_env_call),
        (sft, "attempt_env_call", legacy_attempt_env_call),
        (recovery, "attempt_env_call", legacy_attempt_env_call),
    )
    saved = [(mod, attr, getattr(mod, attr)) for mod, attr, _ in patches]
    try:
        for mod, attr, replacement in patches:
            setattr(mod, attr, replacement)
        yield
    finally:
        for mod, attr, original in reversed(saved):
            setattr(mod, attr, original)
