"""Catalog parsing. Never accesses any *_environment/ file."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


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


def _tokens(s: str) -> set[str]:
    norm = unicodedata.normalize("NFD", s.casefold())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return {t for t in re.findall(r"[a-z0-9_]+", norm) if len(t) > 1}


def rank_tools(allowed: dict[str, dict], query: str, last_tool: str | None = None) -> list[str]:
    q = _tokens(query)
    last_app = last_tool.split(".")[0] if last_tool else None
    ranks = []
    for name, spec in allowed.items():
        n = _tokens(name.replace("_", " ").replace(".", " "))
        d = _tokens(spec.get("description", ""))
        props = _tokens(" ".join(spec.get("parameters", {}).get("properties", {}).keys()))
        score = 4 * len(q & n) + 1.6 * len(q & d) + 0.7 * len(q & props)
        if last_app and name.startswith(f"{last_app}."):
            score += 2
        if spec.get("kind") == "read":
            score += 0.3
        if name == "sys.ask_missing_info":
            score += 0.4
        ranks.append((-score, name))
    return [name for _, name in sorted(ranks)]


def compact_schema(spec: dict) -> dict:
    params = spec.get("parameters", {})
    return {
        "name": spec["name"],
        "kind": spec.get("kind", ""),
        "description": str(spec.get("description", ""))[:250],
        "parameters": params,
        "returns": str(spec.get("returns", ""))[:220],
    }


def prompt_catalog(allowed: dict[str, dict], query: str, last_tool: str | None, max_detailed: int,
                   inspected_tool: str | None = None) -> dict:
    ordered = rank_tools(allowed, query, last_tool)
    detailed = set(ordered[:max_detailed])
    detailed.add("sys.ask_missing_info")
    if inspected_tool in allowed:
        detailed.add(inspected_tool)
    # Read tools from selected apps help discover IDs before attempting writes.
    for name in ordered[:max(1, max_detailed // 2)]:
        app = name.split(".")[0]
        reads = [x for x in ordered if x.startswith(f"{app}.") and allowed[x].get("kind") == "read"]
        for read in reads[:2]:
            if len(detailed) < max_detailed + 8:
                detailed.add(read)
    return {
        "tool_index": [{"name": n, "kind": allowed[n].get("kind", ""),
                        "description": str(allowed[n].get("description", ""))[:105]} for n in sorted(allowed)],
        "tool_schemas": [compact_schema(allowed[n]) for n in ordered if n in detailed],
    }
