"""Bounded, model-local cache of sample-free system/tool prefix snapshots.

Admission in auto mode requires the same prefix in two distinct samples. A unique
schema retains no GPU snapshot beyond its ordinary per-sample cache. No disk cache,
extra models, environment calls or cross-sample request/history state is involved.
"""
from collections import OrderedDict
import hashlib
import json
import math
import weakref


DEFAULT_SCHEMA_CACHE_MB = 256


def cache_budget_mb(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Schema cache memory budget must be positive and finite')
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str, separators=(',', ':'))


def _model_signature(model):
    """Detect ordinary weight updates, adapter switches, placement and patches.

    Call clear() explicitly after unsupported raw-storage/.data mutations, which
    bypass PyTorch version counters. Training callbacks also destroy this pool
    before backward() and construct a fresh wrapper for each evaluation.
    """
    parameters = tuple((name, id(p), p.data_ptr(), p._version, str(p.dtype), str(p.device), tuple(p.shape))
                       for name, p in model.named_parameters())
    modules = tuple((name, module.training, id(getattr(module.forward, '__func__', module.forward)),
                     repr(getattr(module, 'active_adapters', None)),
                     bool(getattr(module, '_disable_adapters', False)),
                     id(getattr(module, '_compiled_call_impl', None)))
                    for name, module in model.named_modules())
    config = getattr(model, 'config', None)
    attention = [getattr(config, '_attn_implementation', None),
                 getattr(getattr(config, 'text_config', None), '_attn_implementation', None)]
    config = config.to_dict() if callable(getattr(config, 'to_dict', None)) else vars(config) if config is not None else {}
    return id(model), parameters, modules, _json(config), _json(attention)


def snapshot_bytes(snapshot):
    """Count retained tensor storages once, including conv/recurrent states."""
    import torch
    seen, storages = set(), {}

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[str(value.device), storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    visit(vars(snapshot))
    return sum(storages.values())


class SchemaCache:
    def __init__(self, mode='auto', max_mb=DEFAULT_SCHEMA_CACHE_MB, *, max_entries=8):
        if mode not in {'auto', 'off', 'shared'}:
            raise ValueError('schema_cache must be auto, off or shared')
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('Schema cache entry limit must be positive')
        self.mode, self.max_bytes = mode, int(cache_budget_mb(max_mb)*1024**2)
        self.max_entries = max_entries
        self.entries, self.seen, self.rendered = OrderedDict(), OrderedDict(), OrderedDict()
        self.signature = None
        self._model_ref = None
        self.prefix_source = None
        self.sample = 0
        self.total_hits = self.total_builds = self.invalidations = 0
        self.begin_sample()

    def begin_sample(self):
        self.sample += 1
        self.stats = dict(hits=0, misses=0, builds=0, evictions=0, skipped_unique=0,
                          skipped_oversize=0, reused_tokens=0, prefix_tokens=0)

    def clear(self):
        self.entries.clear()
        self.seen.clear()
        self.rendered.clear()
        self.signature = None
        self._model_ref = None
        self.prefix_source = None

    def bind(self, model):
        signature = _model_signature(model)
        changed = self.signature is not None and (self._model_ref() is not model or self.signature != signature)
        if changed:
            self.clear()
            self.invalidations += 1
        self.signature = signature
        self._model_ref = weakref.ref(model)
        return changed

    def report(self):
        return {**self.stats, 'mode': self.mode, 'entries': len(self.entries),
                'bytes': sum(entry[2] for entry in self.entries.values()),
                'max_bytes': self.max_bytes, 'total_hits': self.total_hits,
                'total_builds': self.total_builds, 'invalidations': self.invalidations}

    def prefix_ids(self, tokenizer, messages, full_ids, *, head=None):
        """Render ONLY native system/tools, then verify the full input token prefix."""
        self.prefix_source = None
        if self.mode == 'off' or getattr(tokenizer, 'lifemate_native_fc', False) is not True:
            return ()
        from .function_calling import template_inputs
        from .schema_prompt import SCHEMA_PROMPT_POLICY
        if head is None:
            chat, kwargs = template_inputs(tokenizer, messages)
            if not chat or chat[0].get('role') != 'system' or 'tools' not in kwargs:
                return ()
            head = chat[:1], kwargs
        chat, kwargs = head
        template = tokenizer.chat_template
        source = _json([SCHEMA_PROMPT_POLICY, chat, kwargs, template, len(tokenizer)])
        key = id(tokenizer), hashlib.sha256(source.encode()).digest()
        candidate = self.rendered.pop(key, None)

        def matches(ids):
            return bool(ids and len(ids) < len(full_ids) and tuple(full_ids[:len(ids)]) == ids)

        if candidate is None or not matches(candidate):
            # Qwen's template requires a user turn even when rendering a system
            # prefix. Remove the entire dummy user envelope before tokenizing;
            # never put its content or tokens into a GPU snapshot.
            anchor = 'LifeMate schema boundary 8b0f755c'
            try:
                rendered = tokenizer.apply_chat_template(chat+[{'role': 'user', 'content': anchor}],
                    tokenize=False, add_generation_prompt=False, **kwargs)
            except Exception:
                return ()  # Keep the per-sample cache on unsupported templates.
            boundary = rendered.rfind('<|im_start|>user\n'+anchor+'<|im_end|>')
            if boundary < 0:
                return ()
            text = rendered[:boundary]
            # End at a closed special token; the following newline belongs to
            # the request suffix, avoiding independently-tokenized BPE merges.
            text = text.rstrip('\n')
            if not text.endswith('<|im_end|>'):
                return ()
            candidate = tuple(tokenizer(text, add_special_tokens=False, truncation=False,
                                        return_token_type_ids=False)['input_ids'])
        if not matches(candidate):
            return ()
        self.rendered[key] = candidate
        self.prefix_source = key
        if len(self.rendered) > 32:
            self.rendered.popitem(last=False)
        self.stats['prefix_tokens'] = len(candidate)
        return candidate

    def snapshot_for(self, prefix_ids, build):
        """Return immutable snapshot or None; caller must clone before extending."""
        if self.mode == 'off' or not prefix_ids:
            return None
        tokens = tuple(prefix_ids)
        key = hashlib.sha256(_json([self.prefix_source, tokens]).encode()).digest()
        entry = self.entries.pop(key, None)
        if entry is not None and entry[0] == tokens:
            self.entries[key] = entry
            self.stats['hits'] += 1
            self.total_hits += 1
            self.stats['reused_tokens'] += len(tokens)
            return entry[1]
        self.stats['misses'] += 1
        previous_sample, oversized = self.seen.pop(key, (self.sample, False))
        self.seen[key] = previous_sample, oversized
        if len(self.seen) > 256:
            self.seen.popitem(last=False)
        if oversized:
            self.stats['skipped_oversize'] += 1
            return None
        if self.mode == 'auto' and previous_sample == self.sample:
            self.stats['skipped_unique'] += 1
            return None
        snapshot = build()
        self.stats['builds'] += 1
        self.total_builds += 1
        size = snapshot_bytes(snapshot)
        if size > self.max_bytes:
            self.seen[key] = previous_sample, True
            self.stats['skipped_oversize'] += 1
        else:
            while self.entries and (len(self.entries) >= self.max_entries
                    or sum(e[2] for e in self.entries.values())+size > self.max_bytes):
                self.entries.popitem(last=False)
                self.stats['evictions'] += 1
            self.entries[key] = tokens, snapshot, size
        return snapshot
