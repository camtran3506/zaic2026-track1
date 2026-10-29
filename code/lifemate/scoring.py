"""State-gated tool-call score using the supplied challenge rules.

Fuzzy text is tokenized on whitespace after lowercasing and NFC normalization.
The organisers' tokenizer/replay remains authoritative for official scores.
"""
import unicodedata
from collections import Counter


def token_f1(a, b):
    aa = unicodedata.normalize("NFC", str(a).lower()).split()
    bb = unicodedata.normalize("NFC", str(b).lower()).split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    common = sum((Counter(aa) & Counter(bb)).values())
    return 2 * common / (len(aa) + len(bb))


def identical(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def calls_match(pred, gold, catalog):
    if pred.get("name") != gold.get("name"):
        return False
    p, g = pred.get("arguments", {}), gold.get("arguments", {})
    if not isinstance(p, dict) or not isinstance(g, dict) or p.keys() != g.keys():
        return False
    props = catalog.get(pred["name"], {}).get("parameters", {}).get("properties", {})
    return all(token_f1(p[k], g[k]) > 0.6 if props.get(k, {}).get("x-match") == "fuzzy"
               else identical(p[k], g[k]) for k in p)


def max_matches(preds, golds, catalog):
    edges = [[j for j, gold in enumerate(golds) if calls_match(pred, gold, catalog)]
             for pred in preds]
    gold_to_pred = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in gold_to_pred or augment(gold_to_pred[j], seen):
                gold_to_pred[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(preds)))


def call_f1(preds, golds, catalog):
    if not golds:
        return float(not preds), 0
    matched = max_matches(preds, golds, catalog)
    return (2 * matched / (len(preds) + len(golds)) if matched else 0.0), matched


def state_tool_score(preds, golds, predicted_hash, gold_hash, catalog):
    # The empty-gold rule takes precedence, per the supplied specification.
    if not golds:
        return float(not preds)
    if predicted_hash != gold_hash:
        return 0.0
    return call_f1(preds, golds, catalog)[0]
