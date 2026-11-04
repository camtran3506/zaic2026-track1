"""Specialists-as-tools registry: bounded Python functions, no nested LLM/env."""
from . import local_checks, local_knowledge, local_time

LOCAL_HELPER_POLICY = "python-specialists-itn-v2"
DEFAULT_MAX_LOCAL_CALLS = 8
_PREFIXES = ("local_time.", "local_context.", "local_entity.", "local_text.",
             "local_checks.", "local_schema.")


def helper_tool_schemas():
    return {**local_time.local_tool_schemas(), **local_knowledge.schemas(), **local_checks.schemas()}


def with_local_tools(env_tools):
    if any(name.startswith(_PREFIXES) for name in env_tools):
        raise ValueError("ToolEnv catalog collides with reserved local helper namespace")
    return {**env_tools, **helper_tool_schemas()}


def execute_helper(name, arguments, context, history, env_tools, observations=None):
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("Local helper name must be a string and arguments a JSON object")
    if not isinstance(context, dict) or not isinstance(history, list) or not isinstance(env_tools, dict):
        raise ValueError("Local helper context/history/tools have invalid types")
    schemas = helper_tool_schemas()
    if name not in schemas:
        raise ValueError(f"Unknown local helper: {name}")
    validation = local_checks.inspect_call(name, arguments, schemas)
    if validation["status"] == "invalid":
        raise ValueError(f"Invalid helper arguments: {validation['errors']}")
    if name.startswith("local_time."):
        return local_time.execute_local_tool(name, arguments, context)
    if name.startswith(("local_context.", "local_entity.")):
        # Deliberately exclude current observations, model-created candidate lists
        # and the environment itself from the entity specialist's capabilities.
        return local_knowledge.execute(name, arguments, context, history, env_tools)
    return local_checks.execute(name, arguments, env_tools)
