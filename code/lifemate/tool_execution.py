"""One validated ToolEnv execution boundary shared by inference and SFT replay.

Automatic guards are Python middleware, not model-callable helper tools. This
module uses only the provided env's validate/call methods; it never creates an
instance, reads state, or edits/reconstructs the official call log.
"""
from .eval_trace import exception_details
from .execution_policy import check_tool_access, _canonical
from .history import collect_ids


def prepare_call_arguments(name, arguments, spec, context=None, transcript=""):
    """Clean optional empty filters on reads and unmentioned default profile city on weather."""
    if not isinstance(arguments, dict) or not isinstance(spec, dict):
        return arguments
    cleaned = dict(arguments)
    if spec.get("kind") == "read":
        required = set((spec.get("parameters") or {}).get("required") or [])
        cleaned = {
            k: v for k, v in cleaned.items()
            if k in required or v not in (None, "", [], {})
        }
    if name in {"weather.get_current", "weather.get_forecast"} and isinstance(cleaned.get("location"), str):
        profile = context.get("user_profile") if isinstance(context, dict) else None
        city = profile.get("city") if isinstance(profile, dict) else None
        if (isinstance(city, str) and city.strip()
                and cleaned["location"].casefold().strip() == city.casefold().strip()
                and city.casefold() not in (transcript or "").casefold()):
            cleaned.pop("location", None)
    return cleaned


def _check_xref_node(value, schema, observed_ids, key_name=""):
    if not isinstance(schema, dict):
        return None
    ref = schema.get("x-ref")
    if ref and value is not None:
        ref_app = str(ref).split(".", 1)[0]
        allowed = observed_ids.get(ref_app, set())
        if not isinstance(value, str) or value not in allowed:
            return {"ok": False, "error": f"{key_name or 'id'} must come from an observed {ref} ID; call a read tool first"}
    if isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            for key, child in value.items():
                if key in props:
                    err = _check_xref_node(child, props[key], observed_ids, key)
                    if err is not None:
                        return err
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for child in value:
            err = _check_xref_node(child, schema["items"], observed_ids, key_name)
            if err is not None:
                return err
    return None


def check_observed_xref(arguments, spec, observed_ids):
    """Verify any present x-ref argument came from visible history, context, or prior reads."""
    if observed_ids is None or not isinstance(arguments, dict) or not isinstance(spec, dict):
        return None
    return _check_xref_node(arguments, spec.get("parameters"), observed_ids)


def precheck_call(name, arguments, catalog, enabled, monitor, local_names=(), *,
                  transcript='', context=None, history=(), observations=()):
    access = check_tool_access(name, catalog, enabled, local_names)
    if not access["allowed"]:
        return {"stage": "precheck", "name": name, "arguments": arguments,
                "error": access["instruction"], "access": access}
    if not isinstance(arguments, dict) or _canonical(arguments) is None:
        return {"stage": "precheck", "name": name, "arguments": arguments,
                "error": "arguments must be a finite JSON object"}
    if access["scope"] == "local":
        return None
    target = arguments.get("tool_name")
    if name == "sys.ask_missing_info" and (not isinstance(target, str) or target not in enabled or target == name):
        return {"stage": "precheck", "name": name, "arguments": arguments,
                "error": "Missing-info tool_name must be another available application tool."}
    if name == "sys.ask_missing_info":
        from .missing_info_guard import check_missing_info
        feedback = check_missing_info(arguments, enabled, transcript=transcript, context=context,
                                      history=history, observations=observations)
        if feedback:
            return {"stage": "missing_info_guard", "name": name, "arguments": arguments,
                    "error": feedback['instruction'], "feedback": feedback}
    feedback = monitor.check(name, arguments, enabled[name].get("kind", "unknown"))
    if feedback:
        return {"stage": "loop_guard", "name": name, "arguments": arguments,
                "error": feedback["instruction"], "feedback": feedback}
    return None


def update_observed_ids(name, arguments, result, kind, observed_ids):
    """Share successful-result ID provenance with environment and recorded replay."""
    if (observed_ids is not None and isinstance(result, dict)
            and result.get('ok') is True and result.get('done') is not False):
        app = name.split('.', 1)[0]
        collect_ids(result, observed_ids, app)
        if kind == 'write':
            collect_ids(arguments, observed_ids, app)


def attempt_env_call(env, name, arguments, kind, monitor, call_index, *, emit=None, spec=None, observed_ids=None):
    """Caller must precheck first. Execution ALWAYS requires explicit ok:true."""
    emit = emit or (lambda event, **data: None)
    emit("validate_start", name=name, arguments=arguments)
    try:
        validation = env.validate(name, arguments)
    except Exception as exc:
        details = exception_details(exc)
        monitor.record_technical_failure(name, arguments, details, stage="validate")
        emit("validate_error", name=name, arguments=arguments, exception=details)
        return {"executed": False, "feedback": {
            "error": str(exc), "stage": "validate", "name": name, "arguments": arguments,
            "exception": details, "origin": "validation_exception"}}
    accepted = isinstance(validation, dict) and validation.get("ok") is True
    emit("validate_result", name=name, arguments=arguments, result=validation, accepted=accepted)
    if not accepted:
        schema_rejected = isinstance(validation, dict) and validation.get("ok") is False
        if schema_rejected:
            monitor.record_validation_failure(name, arguments, validation)
        else:
            monitor.record_technical_failure(name, arguments, validation, stage="validate_protocol")
        return {"executed": False, "feedback": {
            "error": f"Schema invalid: {validation!r}" if schema_rejected else f"Invalid validation response: {validation!r}",
            "stage": "validate", "name": name, "arguments": arguments, "validation": validation,
            "origin": "schema_rejection" if schema_rejected else "validation_protocol"}}
    if observed_ids is not None and spec is not None:
        xref_error = check_observed_xref(arguments, spec, observed_ids)
        if xref_error is not None:
            emit("xref_guard_rejected", name=name, arguments=arguments, validation=xref_error)
            return {"executed": False, "feedback": {
                "error": xref_error["error"], "stage": "xref_guard", "name": name,
                "arguments": arguments, "validation": xref_error, "origin": "observed_id_guard"}}
    emit("call_start", name=name, arguments=arguments, call_index=call_index)
    execution_exception = None
    try:
        result = env.call(name, arguments)
    except Exception as exc:
        execution_exception = exception_details(exc)
        # This envelope is agent-generated, not an organizer ok:false response.
        result = {"execution_exception": execution_exception}
        emit("call_error", name=name, arguments=arguments, call_index=call_index,
             exception=execution_exception, agent_result=result)
    else:
        emit("call_result", name=name, arguments=arguments, call_index=call_index, result=result)
    update_observed_ids(name, arguments, result, kind, observed_ids)
    verification = monitor.record_result(name, arguments, result, kind,
                                         call_index=call_index, execution_exception=execution_exception)
    emit("execution_verified", name=name, verification=verification)
    if verification.get("contract_violation"):
        emit("execution_contract_violation", name=name, arguments=arguments, result=result)
    return {"executed": True, "result": result, "verification": verification,
            "automatic_execution_feedback": monitor.automatic_feedback(kind, verification, call_index)}


def result_observation(name, arguments, outcome, serialize):
    observation = {"name": name, "arguments": arguments, "result": serialize(outcome["result"]),
                   "automatic_execution_feedback": outcome["automatic_execution_feedback"]}
    if outcome["verification"]["status"] in ("failure", "empty", "unknown"):
        observation["verification"] = outcome["verification"]
    return observation
