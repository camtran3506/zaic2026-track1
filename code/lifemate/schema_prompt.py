"""Conservative prompt-only schema compaction; official catalogs stay untouched.

Cache serialized copies, never shared mutable dicts. Unknown metadata, constraints,
business descriptions and result shapes are preserved, including extension keys.
"""
from copy import deepcopy
from functools import lru_cache
import json


SCHEMA_PROMPT_POLICY = 'redundant-schema-metadata-v1'
_INTEGER_DESCRIPTION = 'Số nguyên, không dấu chấm hay phẩy, ví dụ 2'


def _compact_parameters(schema):
    if not isinstance(schema, dict):
        return
    if schema.get('type') == 'integer' and schema.get('description') == _INTEGER_DESCRIPTION:
        del schema['description']
    # Only traverse JSON Schema nodes. Property names and literal enum/default/
    # example data may themselves contain words such as app or description.
    for keyword in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'):
        values = schema.get(keyword)
        if isinstance(values, dict):
            for child in values.values():
                _compact_parameters(child)
    for keyword in ('items', 'additionalProperties', 'contains', 'not', 'if', 'then', 'else'):
        child = schema.get(keyword)
        if isinstance(child, list):
            for entry in child:
                _compact_parameters(entry)
        else:
            _compact_parameters(child)
    for keyword in ('allOf', 'anyOf', 'oneOf', 'prefixItems'):
        children = schema.get(keyword)
        if isinstance(children, list):
            for child in children:
                _compact_parameters(child)


@lru_cache(maxsize=512)
def _compiled(policy, serialized):
    tool = json.loads(serialized)
    if tool.get('app') == tool.get('name', '').partition('.')[0]:
        tool.pop('app', None)
    if tool.get('returns') == '':
        tool.pop('returns', None)
    _compact_parameters(tool.get('parameters'))
    return json.dumps(tool, ensure_ascii=False, separators=(',', ':'))


def compact_tool_schema(tool):
    """Return an independent prompt view, preserving names/types/all constraints."""
    if not isinstance(tool, dict):
        return deepcopy(tool)
    source = json.dumps(tool, ensure_ascii=False, separators=(',', ':'))
    return json.loads(_compiled(SCHEMA_PROMPT_POLICY, source))
