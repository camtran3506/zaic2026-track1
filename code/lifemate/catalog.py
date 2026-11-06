"""Catalog parsing. Never accesses any *_environment/ file."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from .schema_prompt import compact_tool_schema


def load_catalog(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tools: dict[str, dict] = {}
    for app in payload.get("apps", []):
        for spec in app.get("tools", []):
            name = spec.get("name")
            if isinstance(name, str) and name:
                tools[name] = spec
    # The competition makes this meta tool universally available. Its schema
    # is supplied here only if missing from the catalog; env.validate decides.
    tools.setdefault("sys.ask_missing_info", {
        "name": "sys.ask_missing_info",
        "kind": "meta",
        "description": "Yêu cầu thông tin còn thiếu cho công cụ dự định gọi",
        "parameters": {"type": "object", "properties": {
            "tool_name": {"type": "string"},
            "params": {"type": "array", "items": {"type": "string"}},
        }, "required": ["tool_name", "params"]},
    })
    return tools


def allowed_tools(catalog: dict[str, dict], installed_apps: Any) -> dict[str, dict]:
    """Only the tools exposed to the sample, plus sys.ask_missing_info."""
    explicit: set[str] = set()
    allowed_apps: set[str] = set()
    if isinstance(installed_apps, dict):
        for app, info in installed_apps.items():
            allowed_apps.add(str(app))
            if isinstance(info, dict) and "tools" in info:
                for entry in info["tools"] or []:
                    name = entry if isinstance(entry, str) else entry.get("name") if isinstance(entry, dict) else None
                    if name:
                        explicit.add(str(name))
            else:
                explicit.update(name for name in catalog if name.startswith(f"{app}."))
    elif isinstance(installed_apps, list):
        for item in installed_apps:
            if isinstance(item, str):
                allowed_apps.add(item)
                explicit.update(name for name in catalog if name.startswith(f"{item}."))
            elif isinstance(item, dict):
                app = item.get("id", item.get("name"))
                if not app:
                    continue
                allowed_apps.add(str(app))
                if "tools" in item:
                    for entry in item["tools"] or []:
                        name = entry if isinstance(entry, str) else entry.get("name") if isinstance(entry, dict) else None
                        if name:
                            explicit.add(str(name))
                else:
                    explicit.update(name for name in catalog if name.startswith(f"{app}."))
    result = {name: catalog[name] for name in sorted(explicit) if name in catalog and name.split(".")[0] in allowed_apps}
    result["sys.ask_missing_info"] = catalog["sys.ask_missing_info"]
    return result


def prompt_catalog(allowed: dict[str, dict]) -> dict:
    """Expose every allowed tool; compact only redundant prompt metadata."""
    return {
        "tool_index": [{"name": name, "kind": allowed[name].get("kind", ""),
                        "description": deepcopy(allowed[name].get("description", ""))}
                       for name in sorted(allowed)],
        "tool_schemas": [compact_tool_schema(allowed[name]) for name in sorted(allowed)],
    }
