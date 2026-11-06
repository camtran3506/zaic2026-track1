"""TRAIN-only schema augmentation and strict app-disjoint diagnostic splitting.

No ToolEnv access. Aliases are applied only after canonical gold replay; they
never change the organisers' package, environment, call log or final-state hash.
"""
from copy import deepcopy
import hashlib
import json
import random
import re

from .catalog import allowed_tools
from .context import MAX_CONTEXT_TOKENS
from .sft import split_records
from .schema_prompt import compact_tool_schema

POLICY = "train-schema-augmentation-app-holdout-v1"
GROUPS = {
    "communication": ["contacts", "phone", "sms", "email", "chat_ott", "video_meeting"],
    "planning": ["calendar", "reminders", "alarm", "tasks", "medication", "household_chores"],
    "home": ["smart_home", "home_security", "door_lock", "home_automation", "appliances", "home_energy", "robot_vacuum"],
    "finance": ["banking", "ewallet", "bill_payment", "loyalty", "subscriptions", "exchange_rate"],
    "travel": ["maps", "ride_hailing", "bike_rental", "flights", "package_tracking"],
    "information": ["weather", "news", "gold_price", "fuel_price", "lottery", "exchange_rate"],
    "media": ["music", "tv", "podcast_radio", "news"],
    "shopping": ["food_delivery", "shopping_list", "loyalty", "package_tracking"],
    "learning": ["learning", "library"], "health": ["fitness", "medication"],
}
_KNOWLEDGE = {"local_entity.resolve", "local_context.lookup"}


def add_arguments(parser, *, enabled=False):
    parser.add_argument("--rename-variants", type=int, default=2 if enabled else 0,
                        help="Additional app/tool-name variants per TRAIN sample; local_*/sys names stay fixed")
    parser.add_argument("--distractor-probability", type=float, default=.5 if enabled else 0.,
                        help="Probability of adding 1-3 unrelated TRAIN-catalog apps to an augmented variant")
    parser.add_argument("--heldout-apps", default="auto" if enabled else "none",
                        help="Comma-separated app IDs, auto (one feasible app), or none")
    parser.add_argument("--heldout-fraction", type=float, default=.1,
                        help="Target fraction of source IDs exposed to the auto-selected held-out app")
    parser.add_argument("--app-groups-file", type=str,
                        help="Optional JSON object group_name -> list of TRAIN app IDs, extending functional groups")
    parser.add_argument("--heldout-groups", default="", help="Comma-separated group names to hold out instead of auto app selection")


def validate_options(args):
    if not 0 <= args.rename_variants <= 8:
        raise ValueError("rename-variants must be between 0 and 8")
    if not 0 <= args.distractor_probability <= 1 or not 0 < args.heldout_fraction < 1:
        raise ValueError("Invalid distractor probability or heldout fraction")


def app_ids(catalog):
    return {name.split('.')[0] for name in catalog if '.' in name and not name.startswith(('sys.', 'local_'))}


def namespace_refs(value, apps):
    """Recognise namespace IDs, not generic words such as the email data type."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return {app for app in apps if re.search(r'(?<![\w])' + re.escape(app) + r'\.', text)}


def exposed_apps(record, catalog):
    apps = app_ids(catalog)
    enabled = allowed_tools(catalog, record.get('context', {}).get('installed_apps', {}))
    names = {name.split('.')[0] for name in enabled if not name.startswith('sys.')}
    return names | namespace_refs([list(enabled.values()), record.get('context', {}), record.get('history', []),
                                    record.get('audio_text', ''),
                                    record.get('label', {}).get('tool_calls', [])], apps)


def target_apps(record, apps):
    """Use structured call targets, not app-like strings inside user data."""
    targets = set()
    for call in record.get('label', {}).get('tool_calls', []):
        name = call.get('name', '')
        if name == 'sys.ask_missing_info':
            name = call.get('arguments', {}).get('tool_name', '')
        if isinstance(name, str) and '.' in name and name.split('.')[0] in apps:
            targets.add(name.split('.')[0])
    return targets


def load_groups(path, catalog):
    groups = {k: set(v) for k, v in GROUPS.items()}
    if path:
        from pathlib import Path
        custom = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(custom, dict):
            raise ValueError('app-groups-file must be a JSON object')
        for name, values in custom.items():
            if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
                raise ValueError(f'Invalid app group: {name}')
            if set(values) - app_ids(catalog):
                raise ValueError(f'Unknown TRAIN app in group {name}')
            groups[name] = set(values)
    return groups


def split_generalization(records, catalog, args, groups):
    validate_options(args)
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate sample IDs')
    records = sorted(records, key=lambda r: str(r['id']))
    apps = app_ids(catalog)
    exposure = {r['id']: exposed_apps(r, catalog) for r in records}
    # Global helper/system instructions must not expose a claimed unseen app.
    from .agent import SYSTEM_PROMPT
    from .function_calling import NATIVE_SYSTEM_PROMPT
    from .local_helpers import helper_tool_schemas
    global_refs = namespace_refs([SYSTEM_PROMPT, NATIVE_SYSTEM_PROMPT, helper_tool_schemas()], apps)
    selected = set() if args.heldout_apps in ('auto', 'none') else {v.strip() for v in args.heldout_apps.split(',') if v.strip()}
    for name in (v.strip() for v in args.heldout_groups.split(',') if v.strip()):
        if name not in groups:
            raise ValueError(f'Unknown held-out group: {name}')
        present = groups[name] & apps
        if not present:
            raise ValueError(f'Held-out group has no apps in the TRAIN catalog: {name}')
        selected |= present
    if selected - apps or selected & global_refs:
        raise ValueError(f'Held-out app is unknown or exposed by global instructions/helpers: {sorted((selected-apps) | (selected & global_refs))}')
    status = 'off'
    if args.heldout_apps == 'auto' and not args.heldout_groups:
        candidates = []
        for app in sorted(apps - global_refs):
            affected = [r for r in records if app in exposure[r['id']]]
            targeted = any(app in target_apps(r, apps) for r in affected)
            if targeted and 0 < len(affected) <= len(records)-2:
                tie = hashlib.sha256(f'{args.seed}:{app}'.encode()).hexdigest()
                candidates.append((abs(len(affected)/len(records)-args.heldout_fraction), tie, app))
        if candidates:
            selected = {min(candidates)[2]}
        else:
            status = 'unavailable'  # Tiny/single-app fixtures cannot establish an unseen-app benchmark.
    heldout = [r for r in records if exposure[r['id']] & selected]
    remaining = [r for r in records if not exposure[r['id']] & selected]
    if selected:
        if not heldout or len(remaining) < 2:
            raise ValueError('Held-out apps leave no diagnostic samples or insufficient train/validation data')
        if not any(target_apps(r, selected) for r in heldout):
            raise ValueError('Held-out apps have no supervised tool targets')
        status = 'enabled'
    train, validation = split_records(remaining, args.validation_fraction, args.seed)
    info = {'policy': POLICY, 'status': status, 'apps': sorted(selected),
            'requested_apps': args.heldout_apps, 'requested_groups': args.heldout_groups,
            'target_fraction': args.heldout_fraction,
            'exposed_apps_by_id': {k: sorted(v) for k, v in exposure.items()}}
    return {'train': train, 'validation': validation, 'app_heldout': heldout}, info


def assert_no_heldout(example, apps):
    if not apps:
        return
    if namespace_refs(example['messages'], apps):
        raise ValueError(f"{example['sample_id']}: held-out namespace leaked into TRAIN prompt/target")
    meta = example.get('augmentation', {})
    origins = set(meta.get('source_apps', [])) | set(meta.get('distractor_apps', [])) | set(meta.get('app_aliases', {}))
    if origins & set(apps) or namespace_refs(meta.get('tool_aliases', {}), apps):
        raise ValueError(f"{example['sample_id']}: alias/distractor conceals a held-out app")


def validate_split_manifest(manifest, catalog):
    """Recompute exposure from unchanged TRAIN input; do not trust ID lists alone."""
    from pathlib import Path
    from .sft import file_sha256
    info = manifest.get('app_holdout', {})
    if info.get('status') != 'enabled':
        return
    if info.get('policy') != POLICY or not info.get('apps') or set(info['apps']) - app_ids(catalog):
        raise ValueError('Invalid app-held-out policy/apps')
    source = Path(manifest['source'])
    if source.name != 'train.jsonl' or any(p.endswith('_environment') for p in source.parts):
        raise ValueError('App-held-out audit requires the public TRAIN manifest')
    if file_sha256(source) != manifest['source_sha256']:
        raise ValueError('Source TRAIN data changed after preparation')
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    exposure = {r['id']: exposed_apps(r, catalog) for r in records}
    if {k: sorted(v) for k, v in exposure.items()} != info.get('exposed_apps_by_id'):
        raise ValueError('App-held-out exposure audit differs from source')
    selected = set(info['apps'])
    expected = {key for key, values in exposure.items() if values & selected}
    partitions = manifest['partitions']
    actual = set(partitions.get('app_heldout', {}).get('sample_ids', []))
    if not expected or expected != actual or partitions['app_heldout']['skipped_samples']:
        raise ValueError('App-held-out evaluation must contain every exposed source ID')
    seen = set()
    for name in ('train', 'validation', 'app_heldout'):
        ids = set(partitions[name]['sample_ids'])
        if seen & ids or not ids <= exposure.keys():
            raise ValueError('App-held-out partitions overlap or contain unknown IDs')
        if name != 'app_heldout' and any(exposure[key] & selected for key in ids):
            raise ValueError('Held-out app leaked into train/regular validation')
        seen |= ids


def _refs(text, tools, apps):
    # Schema/program references only. Never apply this to free-form user text,
    # argument values, names, IDs, email addresses or app response messages.
    for old in sorted(tools, key=len, reverse=True):
        text = re.sub(r'(?<![\w])' + re.escape(old) + r'(?![\w.])', tools[old], text)
    for old, new in apps.items():
        text = re.sub(r'(?<![\w])' + re.escape(old) + r'\.', new + '.', text)
    return text


def _schema(value, tools, apps, key=''):
    if isinstance(value, list):
        return [_schema(v, tools, apps, key) for v in value]
    if isinstance(value, dict):
        result = {k: _schema(v, tools, apps, k) for k, v in value.items()}
        # The organizer's schemas also carry a top-level `app` ID. Rename
        # that metadata without touching similarly named argument data.
        if not key and isinstance(value.get('app'), str):
            result['app'] = apps.get(value['app'], value['app'])
        return result
    if isinstance(value, str) and key not in {'enum', 'const', 'default', 'example', 'examples', 'type', 'format', 'pattern', 'required'}:
        return _refs(value, tools, apps)
    return value


def _program_refs(value, tools, apps, key=''):
    if isinstance(value, list):
        return [_program_refs(v, tools, apps, key) for v in value]
    if isinstance(value, dict):
        return {k: _program_refs(v, tools, apps, k) for k, v in value.items()}
    if isinstance(value, str) and key in {'tool', 'tool_name', 'target_tool', 'collection', 'suggested_tools'}:
        return tools.get(value, _refs(value, {}, apps))
    return value


def _collection_keys(value, apps):
    if isinstance(value, list):
        return [_collection_keys(v, apps) for v in value]
    if isinstance(value, dict):
        return {_refs(k, {}, apps): _collection_keys(v, apps) for k, v in value.items()}
    return value


def _call(call, tools):
    result = deepcopy(call)
    original = result.get('name')
    result['name'] = tools.get(original, original)
    if original in {'sys.ask_missing_info', 'local_schema.inspect'}:
        key = 'tool_name' if original == 'sys.ask_missing_info' else 'target_tool'
        args = result.get('arguments', {})
        if key in args:
            args[key] = tools.get(args[key], args[key])
    return result


class AppAugmenter:
    def __init__(self, catalog, train_records, heldout_apps, groups, *, seed=42,
                 rename_variants=2, distractor_probability=.5, tokenizer=None, max_length=MAX_CONTEXT_TOKENS):
        self.catalog, self.groups, self.seed = catalog, groups, seed
        self.heldout = set(heldout_apps)
        self.variants, self.probability = rename_variants, distractor_probability
        self.tokenizer, self.max_length = tokenizer, max_length
        seen = set().union(*(exposed_apps(r, catalog) for r in train_records)) if train_records else set()
        self.pool = {a for a in seen - self.heldout if not namespace_refs(
            [spec for name, spec in catalog.items() if name.startswith(a+'.')], self.heldout)}
        self.stats = {'variants_kept': 0, 'variants_dropped_token_budget': 0,
                      'generated_knowledge_demos_removed': 0, 'distractor_variants': 0,
                      'variants_skipped_literal_tool_reference': 0, 'variants_skipped_no_apps': 0}

    def augment(self, record, examples):
        if not self.variants and not self.probability:
            return []
        if self.tokenizer is None:
            raise ValueError('Augmentation requires --tokenizer to verify the exact SFT token budget')
        variants = []
        targets = record.get('label', {}).get('tool_calls', [])
        intent = target_apps(record, app_ids(self.catalog))
        active_groups = {g for g, values in self.groups.items() if values & intent}
        related = set().union(*(self.groups[g] for g in active_groups)) if active_groups else set()
        installed = exposed_apps(record, self.catalog)
        candidates = sorted(a for a in self.pool - installed - related
                            if intent and active_groups and any(a in v for v in self.groups.values()))
        for index in range(max(self.variants, 1)):
            digest = hashlib.sha256(f'{self.seed}:{record["id"]}:{index}'.encode()).hexdigest()
            rng = random.Random(digest)
            extras = rng.sample(candidates, rng.randint(1, min(3, len(candidates)))) if candidates and rng.random() < self.probability else []
            if not self.variants and not extras:
                continue
            renamed = bool(self.variants)
            app_map, tool_map = {}, {}
            if renamed:
                if not installed and not extras:
                    self.stats['variants_skipped_no_apps'] += 1
                    continue
                # A literal API name in free-form user text cannot safely be
                # distinguished from data to preserve; skip instead of editing it.
                literal = [json.loads(e['messages'][1]['content'])['request_transcript'] for e in examples]
                literal += [turn.get('text', '') for turn in json.loads(examples[0]['messages'][1]['content'])['conversation_history']]
                literal += [call.get('arguments', {}) for call in targets if call.get('name') != 'sys.ask_missing_info']
                for turn in json.loads(examples[0]['messages'][1]['content'])['conversation_history']:
                    literal += [call.get('arguments', {}) for call in turn.get('tool_calls', []) if call.get('name') != 'sys.ask_missing_info']
                if namespace_refs(literal, app_ids(self.catalog)):
                    self.stats['variants_skipped_literal_tool_reference'] += 1
                    continue
                for app in sorted(installed | set(extras)):
                    app_map[app] = 'app_' + hashlib.sha256(f'{digest}:{app}'.encode()).hexdigest()[:12]
                for name in self.catalog:
                    app = name.split('.')[0]
                    if app in app_map:
                        tool_map[name] = app_map[app] + '.tool_' + hashlib.sha256(f'{digest}:{name}'.encode()).hexdigest()[:12]
            rows, removed = [], 0
            for example in examples:
                row = deepcopy(example)
                target = json.loads(row['messages'][-1]['content'])
                if renamed and target.get('name') in _KNOWLEDGE:
                    removed += 1
                    continue
                payload = json.loads(row['messages'][1]['content'])
                specs = payload['available_tools']['tool_schemas']
                specs += [deepcopy(spec) for name, spec in sorted(self.catalog.items()) if name.split('.')[0] in extras]
                schemas = [compact_tool_schema(_schema(spec, tool_map, app_map)) for spec in specs]
                payload['available_tools'] = {
                    'tool_index': [{'name': s['name'], 'kind': s.get('kind', ''), 'description': s.get('description', '')} for s in sorted(schemas, key=lambda s: s['name'])],
                    'tool_schemas': sorted(schemas, key=lambda s: s['name'])}
                for turn in payload['conversation_history']:
                    if 'tool_calls' in turn:
                        turn['tool_calls'] = [_call(call, tool_map) for call in turn['tool_calls']]
                payload['user_context'] = _collection_keys(payload['user_context'], app_map)
                observations = []
                for obs in payload['previous_actions_and_results']:
                    if renamed and obs.get('name') in _KNOWLEDGE:
                        continue
                    converted = _call(obs, tool_map) if 'name' in obs else deepcopy(obs)
                    for key in ('verification', 'automatic_execution_feedback', 'feedback', 'access'):
                        if key in converted:
                            converted[key] = _program_refs(converted[key], tool_map, app_map)
                    observations.append(converted)
                payload['previous_actions_and_results'] = observations
                row['messages'][0]['content'] = _refs(row['messages'][0]['content'], tool_map, app_map)
                row['messages'][1]['content'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
                row['messages'][-1]['content'] = json.dumps(_call(target, tool_map) if target.get('action') == 'call' else target,
                                                           ensure_ascii=False, separators=(',', ':'))
                row['step'] = len(rows)
                row['augmentation'] = {'policy': POLICY, 'variant': index+1,
                    'app_aliases': app_map, 'tool_aliases': tool_map, 'distractor_apps': extras,
                    'source_apps': sorted(installed)}
                assert_no_heldout(row, self.heldout)
                # Verify target visibility and the unchanged parameter schema.
                from .local_checks import inspect_call
                updated = json.loads(row['messages'][-1]['content'])
                if updated['action'] == 'call' and inspect_call(updated['name'], updated['arguments'],
                        {s['name']: s for s in schemas})['status'] == 'invalid':
                    raise ValueError('Augmented target violates its exposed schema')
                rows.append(row)
            from .llm import chat_token_ids
            if any(len(chat_token_ids(self.tokenizer, row['messages'], add_generation_prompt=False)) > self.max_length for row in rows):
                self.stats['variants_dropped_token_budget'] += 1
                continue
            self.stats['variants_kept'] += 1
            self.stats['generated_knowledge_demos_removed'] += removed
            self.stats['distractor_variants'] += bool(extras)
            variants.extend(rows)
        return variants
