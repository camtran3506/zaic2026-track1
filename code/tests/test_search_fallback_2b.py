"""Priority 2B zero-retraining search fallback & doomed-retry guard tests."""
from copy import deepcopy
import json
from pathlib import Path

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.tool_execution import _extract_landmark_queries, build_search_fallback, prepare_call_arguments
from test_baseline import CATALOG, StubASR, StubLLM, item


SEARCH_CATALOG = {
    **CATALOG,
    "maps.search_place": {
        "name": "maps.search_place", "kind": "read", "description": "Tìm địa điểm",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "near": {"type": "string"}}, "required": ["query"]},
    },
    "event_tickets.search": {
        "name": "event_tickets.search", "kind": "read", "description": "Tìm vé sự kiện",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []},
    },
    "ecommerce.search": {
        "name": "ecommerce.search", "kind": "read", "description": "Tìm sản phẩm",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "max_price": {"type": "integer"}}, "required": ["query"]},
    },
    "recipes.search": {
        "name": "recipes.search", "kind": "read", "description": "Tìm công thức",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "servings": {"type": "integer"}}, "required": ["query"]},
    },
    "contacts.search": {
        "name": "contacts.search", "kind": "read", "description": "Tìm liên hệ",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []},
    },
    "notes.search": {
        "name": "notes.search", "kind": "read", "description": "Tìm ghi chú",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []},
    },
    "notes.append": {
        "name": "notes.append", "kind": "write", "description": "Thêm ghi chú",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    "stocks.get_quote": {
        "name": "stocks.get_quote", "kind": "read", "description": "Tra giá cổ phiếu",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    },
    "stocks.get_portfolio": {
        "name": "stocks.get_portfolio", "kind": "read", "description": "Xem danh mục cổ phiếu",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


class SearchEnv:
    def __init__(self, responder):
        self.calls = []
        self.validations = []
        self._responder = responder

    def validate(self, name, args):
        self.validations.append({"name": name, "arguments": deepcopy(args)})
        required = set(SEARCH_CATALOG[name]["parameters"].get("required", []))
        missing = required - set(args)
        if missing:
            return {"ok": False, "error": f"missing {sorted(missing)}"}
        return {"ok": True}

    def call(self, name, args):
        self.calls.append({"name": name, "arguments": deepcopy(args)})
        return self._responder(name, args, self.calls)

    def get_tool_calls(self):
        return self.calls


def _entry(apps, sample_id="train_0001"):
    row = item()
    row["id"] = sample_id
    row["context"]["installed_apps"] = apps
    return row


def test_landmark_extraction_preserves_diacritics_and_stops_at_particles():
    t1 = "Tối ni chị hẹn với cree vill hold theo cái đối diện quán mộc lan à em tìm rồi được rồi."
    assert _extract_landmark_queries(t1)[:2] == ["quán mộc lan", "mộc lan"]
    # Diacritics must distinguish "Quán Đỏ" (proper noun) from "đó" (demonstrative).
    t2 = "Tìm chỗ gần quán Đỏ nha bạn ơi."
    assert _extract_landmark_queries(t2) == ["quán Đỏ", "Đỏ"]
    # Generic phrase without a proper name after the noun must not trigger a bogus landmark.
    t3 = "Tìm ở quán đó đi nhé."
    assert _extract_landmark_queries(t3) == []


def test_maps_dead_token_and_landmark_fallback_public_0619():
    transcript = "Tối ni chị hẹn với cree vill hold theo cái đối diện quán mộc lan à em tìm rồi."
    entry = _entry({"maps": {"tools": ["maps.search_place"]}})

    def responder(name, args, calls):
        q = args.get("query", "")
        if "mộc lan" in q.lower() or q == "*":
            return {"ok": True, "places": [{"id": "p_01", "name": "Quán Mộc Lan"}]}
        return {"ok": True, "places": []}

    outputs = [
        json.dumps({"action": "call", "name": "maps.search_place", "arguments": {"query": "Cree Village"}}),
        # Doomed subset + hallucinated `near` must redirect to Vietnamese landmark "quán mộc lan" without `near`.
        json.dumps({"action": "call", "name": "maps.search_place", "arguments": {"query": "Cree", "near": "Lê Đình Dương"}}),
        json.dumps({"action": "done"}),
    ]
    env = SearchEnv(responder)
    planner = StubLLM(outputs)
    seen_prompts = []

    class RecordingLLM:
        def generate(self, system, user):
            seen_prompts.append(json.loads(user))
            return planner.generate(system, user)

    stats = LifeMateAgent(StubASR(), RecordingLLM(), SEARCH_CATALOG, AgentConfig()).run(
        entry, env, Path("/unused"), "public_test",
        prepared_transcript={"id": entry["id"], "transcript": transcript, "history": []},
    )
    assert stats["executed"] == 2 and stats["invalid_proposals"] == 0
    assert env.calls == [
        {"name": "maps.search_place", "arguments": {"query": "Cree Village"}},
        {"name": "maps.search_place", "arguments": {"query": "quán mộc lan"}},
    ]
    obs1 = seen_prompts[1]["previous_actions_and_results"][0]
    fb = obs1["verification"]["search_fallback"]
    assert fb["failed_query"] == "Cree Village"
    assert fb["alternative_queries"][:2] == ["quán mộc lan", "mộc lan"]
    assert fb["fallback_call"] == {"name": "maps.search_place", "arguments": {"query": "*"}}


def test_maps_wildcard_fallback_when_landmark_also_empty():
    transcript = "Hẹn chỗ đối diện quán mộc lan à em."
    entry = _entry({"maps": {"tools": ["maps.search_place"]}})

    def responder(name, args, calls):
        if args.get("query") == "*":
            return {"ok": True, "places": [{"id": "p_99", "name": "Green Hill"}]}
        return {"ok": True, "places": []}

    outputs = [
        json.dumps({"action": "call", "name": "maps.search_place", "arguments": {"query": "Cree Village"}}),
        # Redirects to landmark "quán mộc lan" (empty)
        json.dumps({"action": "call", "name": "maps.search_place", "arguments": {"query": "Cree"}}),
        # Now both Cree Village and mộc lan are dead -> skips "mộc lan" and goes straight to {"query": "*"}!
        json.dumps({"action": "call", "name": "maps.search_place", "arguments": {"query": "Village", "near": "Lê Đình Dương"}}),
        json.dumps({"action": "done"}),
    ]
    env = SearchEnv(responder)
    stats = LifeMateAgent(StubASR(), StubLLM(outputs), SEARCH_CATALOG, AgentConfig()).run(
        entry, env, Path("/unused"), "public_test",
        prepared_transcript={"id": entry["id"], "transcript": transcript, "history": []},
    )
    assert stats["invalid_proposals"] == 0
    assert env.calls == [
        {"name": "maps.search_place", "arguments": {"query": "Cree Village"}},
        {"name": "maps.search_place", "arguments": {"query": "quán mộc lan"}},
        {"name": "maps.search_place", "arguments": {"query": "*"}},
    ]


def test_query_expansion_after_empty_redirected_public_0336_0505():
    # Case 1: event_tickets.search (optional `query`) -> redirects expansion to {}
    entry_ev = _entry({"event_tickets": {"tools": ["event_tickets.search"]}}, "ev")
    ev_env = SearchEnv(lambda n, a, c: {"ok": True, "events": [{"id": "ev_1", "name": "Charles"}] if not a else []})
    stats_ev = LifeMateAgent(
        StubASR(),
        StubLLM([
            json.dumps({"action": "call", "name": "event_tickets.search", "arguments": {"query": "Chals"}}),
            json.dumps({"action": "call", "name": "event_tickets.search", "arguments": {"query": "Chals tối thứ sáu tuần sau"}}),
            json.dumps({"action": "done"}),
        ]),
        SEARCH_CATALOG, AgentConfig(),
    ).run(entry_ev, ev_env, Path("/unused"), "public_test",
          prepared_transcript={"id": "ev", "transcript": "Lấy hai vé đêm nhạc chals tối thứ sáu tuần sau.", "history": []})
    assert stats_ev["invalid_proposals"] == 0
    assert ev_env.calls == [
        {"name": "event_tickets.search", "arguments": {"query": "Chals"}},
        {"name": "event_tickets.search", "arguments": {}},
    ]

    # Case 2: ecommerce.search (required `query`) -> redirects expansion to {"query": "*"}
    entry_ec = _entry({"ecommerce": {"tools": ["ecommerce.search"]}}, "ec")
    ec_env = SearchEnv(lambda n, a, c: {"ok": True, "products": [{"id": "pr_1", "name": "Chuột Logitech"}] if a.get("query") == "*" else []})
    stats_ec = LifeMateAgent(
        StubASR(),
        StubLLM([
            json.dumps({"action": "call", "name": "ecommerce.search", "arguments": {"query": "Tuột Luitech"}}),
            json.dumps({"action": "call", "name": "ecommerce.search", "arguments": {"query": "Tuột Luitech gió"}}),
            json.dumps({"action": "done"}),
        ]),
        SEARCH_CATALOG, AgentConfig(),
    ).run(entry_ec, ec_env, Path("/unused"), "public_test",
          prepared_transcript={"id": "ec", "transcript": "Cho anh hai con tuột luitech vào gió.", "history": []})
    assert stats_ec["invalid_proposals"] == 0
    assert ec_env.calls == [
        {"name": "ecommerce.search", "arguments": {"query": "Tuột Luitech"}},
        {"name": "ecommerce.search", "arguments": {"query": "*"}},
    ]


def test_filter_relaxation_is_not_blocked_or_redirected():
    # If the previous empty search had max_price=100000 and the retry drops max_price,
    # it may succeed and MUST NOT be treated as doomed.
    prior_obs = [{
        "name": "ecommerce.search",
        "arguments": {"query": "Chuột", "max_price": 100000},
        "result": json.dumps({"ok": True, "products": []}),
        "verification": {"status": "empty"},
        "automatic_execution_feedback": {"repeat_policy": "reuse_read_until_state_change"},
    }]
    next_args = prepare_call_arguments(
        "ecommerce.search", {"query": "Chuột không dây"},
        SEARCH_CATALOG["ecommerce.search"], {}, "Tìm chuột không dây", prior_obs,
    )
    assert next_args == {"query": "Chuột không dây"}


def test_honorific_stripping_in_search_fallback_public_0188():
    fb = build_search_fallback(
        "contacts.search", {"query": "Cô giáo Ly"}, SEARCH_CATALOG["contacts.search"],
        transcript="Em chuyển cho cô ly ba trăm rưỡi qua ví nhé.",
    )
    assert fb is not None
    assert "Ly" in fb["alternative_queries"]
    assert fb["fallback_call"] == {"name": "contacts.search", "arguments": {}}


def test_repeated_empty_read_surfaces_search_fallback_public_0565():
    entry = _entry({"recipes": {"tools": ["recipes.search"]}}, "rc")
    env = SearchEnv(lambda n, a, c: {"ok": True, "recipes": [{"id": "r1"}] if a.get("query") == "*" else []})
    events = []
    stats = LifeMateAgent(
        StubASR(),
        StubLLM([
            json.dumps({"action": "call", "name": "recipes.search", "arguments": {"query": "Clue Tuís"}}),
            # Exact repeat -> must hit LoopMonitor (not rewritten) and surface search_fallback!
            json.dumps({"action": "call", "name": "recipes.search", "arguments": {"query": "Clue Tuís"}}),
            json.dumps({"action": "call", "name": "recipes.search", "arguments": {"query": "*"}}),
            json.dumps({"action": "done"}),
        ]),
        SEARCH_CATALOG, AgentConfig(),
    ).run(entry, env, Path("/unused"), "public_test",
          prepared_transcript={"id": "rc", "transcript": "Bà xã muốn nấu món của clue tuis.", "history": []},
          trace=events.append)
    assert stats["invalid_proposals"] == 1 and stats["executed"] == 2
    assert env.calls == [
        {"name": "recipes.search", "arguments": {"query": "Clue Tuís"}},
        {"name": "recipes.search", "arguments": {"query": "*"}},
    ]
    rejected = next(e for e in events if e["event"] == "decision_rejected")
    assert rejected["feedback"]["reason"] == "repeated_empty_read"
    assert rejected["feedback"]["search_fallback"]["fallback_call"] == {"name": "recipes.search", "arguments": {"query": "*"}}


def test_stocks_get_quote_case_variant_and_portfolio_fallback_public_0665():
    entry = _entry({"stocks": {"tools": ["stocks.get_quote", "stocks.get_portfolio"]}}, "st")

    def responder(name, args, calls):
        if name == "stocks.get_portfolio":
            return {"ok": True, "holdings": [{"symbol": "SRI", "name": "Sunrise", "price": 25000}]}
        return {"ok": True, "done": False, "message": f"không tìm thấy mã {args.get('symbol')!r}"}

    events = []
    env = SearchEnv(responder)
    stats = LifeMateAgent(
        StubASR(),
        StubLLM([
            json.dumps({"action": "call", "name": "stocks.get_quote", "arguments": {"symbol": "SunRI"}}),
            # Case-only variation "SUNRI" -> normalized to "SunRI" so LoopMonitor stops it without env.call!
            json.dumps({"action": "call", "name": "stocks.get_quote", "arguments": {"symbol": "SUNRI"}}),
            json.dumps({"action": "call", "name": "stocks.get_portfolio", "arguments": {}}),
            json.dumps({"action": "done"}),
        ]),
        SEARCH_CATALOG, AgentConfig(),
    ).run(entry, env, Path("/unused"), "public_test",
          prepared_transcript={"id": "st", "transcript": "Cổ phiếu sunri của cô giờ bao nhiêu con.", "history": []},
          trace=events.append)
    assert stats["invalid_proposals"] == 1 and stats["executed"] == 2
    assert env.calls == [
        {"name": "stocks.get_quote", "arguments": {"symbol": "SunRI"}},
        {"name": "stocks.get_portfolio", "arguments": {}},
    ]
    rejected = next(e for e in events if e["event"] == "decision_rejected")
    assert rejected["feedback"]["reason"] == "repeated_failed_read"
    assert rejected["feedback"]["search_fallback"]["fallback_call"] == {"name": "stocks.get_portfolio", "arguments": {}}


def test_distinct_queries_and_write_epoch_reset_preserved():
    entry = _entry({"notes": {"tools": ["notes.search", "notes.append"]}}, "n1")
    env = SearchEnv(lambda n, a, c: {"ok": True, "done": True} if n == "notes.append" else {"ok": True, "notes": []})
    outputs = [
        json.dumps({"action": "call", "name": "notes.search", "arguments": {"query": "Alpha"}}),
        json.dumps({"action": "call", "name": "notes.search", "arguments": {"query": "Beta"}}),
        json.dumps({"action": "call", "name": "notes.append", "arguments": {"text": "Alpha Beta expanded"}}),
        # After a state-changing write, previously empty query expansion is fresh in the new epoch.
        json.dumps({"action": "call", "name": "notes.search", "arguments": {"query": "Alpha Beta"}}),
        json.dumps({"action": "done"}),
    ]
    stats = LifeMateAgent(StubASR(), StubLLM(outputs), SEARCH_CATALOG, AgentConfig()).run(
        entry, env, Path("/unused"), "train",
        prepared_transcript={"id": "n1", "transcript": "Tìm ghi chú.", "history": []},
    )
    assert stats["invalid_proposals"] == 0 and stats["executed"] == 4
    assert env.calls == [
        {"name": "notes.search", "arguments": {"query": "Alpha"}},
        {"name": "notes.search", "arguments": {"query": "Beta"}},
        {"name": "notes.append", "arguments": {"text": "Alpha Beta expanded"}},
        {"name": "notes.search", "arguments": {"query": "Alpha Beta"}},
    ]


def test_generic_companion_list_discovery_for_unseen_domain():
    from lifemate.tool_execution import build_search_fallback

    get_quote_spec = {
        "kind": "read",
        "description": "Tra cứu giá tài sản theo mã symbol.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    }
    list_assets_spec = {
        "kind": "read",
        "description": "Liệt kê tất cả mã tài sản hiện có.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
    enabled = {
        "crypto.get_quote": get_quote_spec,
        "crypto.list_assets": list_assets_spec,
    }
    fb = build_search_fallback(
        "crypto.get_quote",
        {"symbol": "UNKNOWN"},
        get_quote_spec,
        transcript="Xem giá coin UNKNOWN.",
        enabled=enabled,
        observations=[],
    )
    assert fb is not None
    assert fb["failed_value"] == "UNKNOWN"
    assert fb["failed_symbol"] == "UNKNOWN"
    assert fb["fallback_call"] == {"name": "crypto.list_assets", "arguments": {}}


def test_dynamic_token_or_mode_from_description_blocks_subset_retry():
    from lifemate.tool_execution import _search_match_mode, prepare_call_arguments

    docs_spec = {
        "kind": "read",
        "description": "Tìm tài liệu theo từ khóa, chỉ cần một từ khớp với truy vấn.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    docs_and_spec = {
        **docs_spec,
        "description": "Tìm tài liệu khớp chính xác tất cả từ khóa.",
    }
    observations = [
        {
            "name": "docs.search",
            "arguments": {"query": "Alpha Beta"},
            "status": "empty",
            "result": {"ok": True, "items": []},
        }
    ]
    assert _search_match_mode("docs.search", docs_spec) == "token_or"
    assert _search_match_mode("docs.search", docs_and_spec) == "substring"
    redirected = prepare_call_arguments(
        "docs.search",
        {"query": "Alpha"},
        docs_spec,
        transcript="Tìm tài liệu Alpha Beta.",
        observations=observations,
    )
    assert redirected == {"query": "*"}
    preserved = prepare_call_arguments(
        "docs.search",
        {"query": "Alpha"},
        docs_and_spec,
        transcript="Tìm tài liệu Alpha Beta.",
        observations=observations,
    )
    assert preserved == {"query": "Alpha"}


def test_generic_optional_filter_relaxation_preserves_non_filter_params():
    from lifemate.tool_execution import build_search_fallback

    catalog_spec = {
        "kind": "read",
        "description": "Tìm sản phẩm trong danh mục.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    }
    fb = build_search_fallback(
        "catalog.search",
        {"query": "Gamma Widget", "category": "office", "limit": 5},
        catalog_spec,
        transcript="Tìm Gamma Widget.",
        enabled={"catalog.search": catalog_spec},
        observations=[],
    )
    assert fb is not None
    assert fb["failed_query"] == "Gamma Widget"
    assert fb["relaxed_call"] == {
        "name": "catalog.search",
        "arguments": {"query": "Gamma Widget", "limit": 5},
    }

