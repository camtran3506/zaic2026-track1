"""One validated ToolEnv execution boundary shared by inference and SFT replay.

Automatic guards are Python middleware, not model-callable helper tools. This
module uses only the provided env's validate/call methods; it never creates an
instance, reads state, or edits/reconstructs the official call log.
"""
from .eval_trace import exception_details
from .execution_policy import check_tool_access, _canonical


def precheck_call(name, arguments, catalog, enabled, monitor, local_names=()):
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
    feedback = monitor.check(name, arguments, enabled[name].get("kind", "unknown"))
    if feedback:
        return {"stage": "loop_guard", "name": name, "arguments": arguments,
                "error": feedback["instruction"], "feedback": feedback}
    return None


def attempt_env_call(env, name, arguments, kind, monitor, call_index, *, emit=None):
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
