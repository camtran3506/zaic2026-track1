"""Unit tests for non-conflicting Tier 1 guard and fallback improvements (P1.1-P1.6)."""
import json
from pathlib import Path

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.execution_policy import (
    DEFAULT_EMPTY_READ_LIMIT,
    EXECUTION_POLICY,
    LoopMonitor,
    check_tool_access,
)
from lifemate.itn import ITN_POLICY
from lifemate.local_assist import ASSIST_POLICY
from lifemate.local_helpers import LOCAL_HELPER_POLICY
from lifemate.local_time import LOCAL_TIME_POLICY
from lifemate.missing_info_guard import check_missing_info
from lifemate.reasoning import REASONING_POLICY
from lifemate.schema_prompt import SCHEMA_PROMPT_POLICY
from lifemate.text import TEXT_NORMALIZATION
from lifemate.tool_execution import (
    attempt_env_call,
    build_search_fallback,
    precheck_call,
    prepare_call_arguments,
)
from test_python_helpers_integration import DONE, Env, Planner, call, prepared


TIER1_CATALOG = {
    "flights.search": {
        "name": "flights.search",
        "kind": "read",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
                "passengers": {"type": "integer"},
            },
            "required": ["origin", "destination", "date"],
        },
        "returns": "flights[]",
    },
    "flights.book": {
        "name": "flights.book",
        "kind": "write",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_id": {"type": "string", "x-ref": "flights.flights"},
                "passengers": {"type": "integer"},
            },
            "required": ["flight_id", "passengers"],
        },
    },
    "train_bus_tickets.search": {
        "name": "train_bus_tickets.search",
        "kind": "read",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["origin", "destination", "date"],
        },
        "returns": "trips[]",
    },
    "maps.search_place": {
        "name": "maps.search_place",
        "kind": "read",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "near": {"type": "string"},
            },
            "required": ["query"],
        },
        "returns": "places[]",
    },
    "parking.find": {
        "name": "parking.find",
        "kind": "read",
        "parameters": {
            "type": "object",
            "properties": {"near": {"type": "string"}},
            "required": ["near"],
        },
        "returns": "lots[]",
    },
    "parking.reserve": {
        "name": "parking.reserve",
        "kind": "write",
        "parameters": {
            "type": "object",
            "properties": {
                "lot_id": {"type": "string", "x-ref": "parking.lots"},
                "from_time": {"type": "string"},
                "hours": {"type": "integer"},
            },
            "required": ["lot_id", "from_time", "hours"],
        },
    },
    "contacts.search": {
        "name": "contacts.search",
        "kind": "read",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        "returns": "contacts[]",
    },
    "phone.recent_calls": {
        "name": "phone.recent_calls",
        "kind": "read",
        "parameters": {"type": "object", "properties": {"missed_only": {"type": "boolean"}}},
        "returns": "calls[]",
    },
    "phone.call": {
        "name": "phone.call",
        "kind": "write",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                "phone_number": {"type": "string"},
            },
            "required": [],
            "x-one-of": [["contact_id"], ["phone_number"]],
        },
    },
    "chat_ott.list_chats": {
        "name": "chat_ott.list_chats",
        "kind": "read",
        "parameters": {"type": "object", "properties": {}},
        "returns": "chats[]",
    },
    "banking.get_balance": {
        "name": "banking.get_balance",
        "kind": "read",
        "parameters": {"type": "object", "properties": {}},
        "returns": "accounts[]",
    },
    "banking.open_savings": {
        "name": "banking.open_savings",
        "kind": "write",
        "parameters": {
            "type": "object",
            "properties": {
                "from_account_id": {"type": "string", "x-ref": "banking.accounts"},
                "amount": {"type": "integer"},
                "term_months": {"type": "integer"},
            },
            "required": ["from_account_id", "amount", "term_months"],
        },
    },
    "tasks.complete": {
        "name": "tasks.complete",
        "kind": "write",
        "parameters": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    "sys.ask_missing_info": {
        "name": "sys.ask_missing_info",
        "kind": "control",
        "parameters": {
            "type": "object",
            "properties": {"tool_name": {"type": "string"}, "params": {"type": "array"}},
            "required": ["tool_name", "params"],
        },
    },
}


def test_flight_iata_normalization_and_contradiction_rewrite():
    spec = TIER1_CATALOG["flights.search"]
    norm = prepare_call_arguments(
        "flights.search",
        {"origin": "TP.HCM", "destination": "Hà Nội", "date": "2026-12-01"},
        spec,
        transcript="Đặt vé đi Hà Nội ngày 1 tháng 12",
    )
    assert norm == {"origin": "SGN", "destination": "HAN", "date": "2026-12-01"}

    # public_0213: DAD contradicted by "đi đà lạt" (and "sân bay nội bài" belongs to origin HAN, not dest)
    rewritten_dli = prepare_call_arguments(
        "flights.search",
        {"origin": "HAN", "destination": "DAD", "date": "2026-10-01"},
        spec,
        transcript="Đặt giúp chị hai vé chuyến rv157 đi đà lạt thứ năm tuần sau nhé rồi giữ chỗ ở sân bay nội bài.",
    )
    assert rewritten_dli["destination"] == "DLI"
    assert rewritten_dli["origin"] == "HAN"

    # public_0252: HUI contradicted by "đi quy nhơn"
    rewritten_uih = prepare_call_arguments(
        "flights.search",
        {"origin": "HAN", "destination": "HUI", "date": "2026-10-09"},
        spec,
        transcript="Đặt cho em hai vé chuyến rv23253 đi quy nhơn hôm mùng chín tháng mười.",
    )
    assert rewritten_uih["destination"] == "UIH"

    # Preserves valid city-supported IATA code (never breaks multi-reference SGN prompt)
    kept_sgn = prepare_call_arguments(
        "flights.search",
        {"origin": "HAN", "destination": "SGN", "date": "2026-10-22"},
        spec,
        transcript="Cái hội nghị trong Sài Gòn ấy, đặt vé bay từ Hà Nội vào hôm đó. Xong gọi xe ra sân bay Tân Sơn Nhất.",
    )
    assert kept_sgn == {"origin": "HAN", "destination": "SGN", "date": "2026-10-22"}

    # Never IATA-izes train_bus_tickets.search or mistakes "Vĩnh Long" for Vinh (VII)
    bus_args = prepare_call_arguments(
        "train_bus_tickets.search",
        {"origin": "Hà Nội", "destination": "Đà Nẵng", "date": "2026-10-10"},
        TIER1_CATALOG["train_bus_tickets.search"],
        transcript="Tìm vé tàu từ Hà Nội đi Đà Nẵng.",
    )
    assert bus_args["origin"] == "Hà Nội" and bus_args["destination"] == "Đà Nẵng"
    vinh_long = prepare_call_arguments(
        "flights.search",
        {"origin": "HAN", "destination": "VCA", "date": "2026-10-10"},
        spec,
        transcript="Tìm vé đi Vĩnh Long ghé Cần Thơ.",
    )
    assert vinh_long["destination"] == "VCA"


def test_travel_search_fallback_for_explicit_spoken_date_and_next_week():
    spec = TIER1_CATALOG["flights.search"]
    # public_0061 spoken Vietnamese date mismatch: "ngày hai mươi tháng mười một" vs searched 2026-11-21
    fb_explicit = build_search_fallback(
        "flights.search",
        {"origin": "HAN", "destination": "SGN", "date": "2026-11-21"},
        spec,
        transcript="Mua cho tôi ba vé đi sài gòn ngày hai mươi tháng mười một giá mỗi vé dưới một triệu.",
        context={"current_time": "2026-11-10T08:10:00+07:00"},
    )
    assert fb_explicit is not None
    assert fb_explicit["suggested_dates"][0] == "2026-11-20"
    assert "2026-11-20" in fb_explicit["instruction"]

    # public_0287 "thứ bảy tuần sau" searched in current week (2026-10-03 -> +7d = 2026-10-10)
    fb_next_week = build_search_fallback(
        "flights.search",
        {"origin": "HAN", "destination": "DAD", "date": "2026-10-03"},
        spec,
        transcript="Sáng thứ bảy tuần sau đặt vé máy bay đi đà nẵng.",
        context={"current_time": "2026-10-01T09:00:00+07:00"},
    )
    assert fb_next_week is not None
    assert "2026-10-10" in fb_next_week["suggested_dates"]
    assert "2026-10-10" in fb_next_week["instruction"]

    # Works for train_bus_tickets.search too
    bus_fb = build_search_fallback(
        "train_bus_tickets.search",
        {"origin": "Hà Nội", "destination": "Đà Nẵng", "date": "2026-10-03"},
        TIER1_CATALOG["train_bus_tickets.search"],
        transcript="Chủ nhật tuần sau đi Đà Nẵng",
        context={"current_time": "2026-10-01T09:00:00+07:00"},
    )
    assert bus_fb is not None and "2026-10-10" in bus_fb["suggested_dates"]


    # public_0445 "thứ năm tới" relative to Friday 2026-11-13 -> upcoming Thursday 2026-11-19
    bus_upcoming = build_search_fallback(
        "train_bus_tickets.search",
        {"origin": "Hà Nội", "destination": "Hải Phòng", "date": "2026-11-13"},
        TIER1_CATALOG["train_bus_tickets.search"],
        transcript="Vé xe hải phòng thứ năm tới mấy giờ tới nơi.",
        context={"current_time": "2026-11-13T18:10:00+07:00"},
    )
    assert bus_upcoming is not None and "2026-11-19" in bus_upcoming["suggested_dates"]


def test_route_date_scan_limit_and_stagnant_maps_loop_preserve_empty_read_limit():
    # Conflict preservation check: DEFAULT_EMPTY_READ_LIMIT stays 0
    assert DEFAULT_EMPTY_READ_LIMIT == 0

    monitor = LoopMonitor()
    empty_obs = [
        {"name": "flights.search", "arguments": {"origin": "HAN", "destination": "SGN", "date": d},
         "scope": "env", "result": json.dumps({"ok": True, "flights": []})}
        for d in ("2026-11-21", "2026-11-22", "2026-11-23")
    ]
    blocked = precheck_call(
        "flights.search",
        {"origin": "HAN", "destination": "SGN", "date": "2026-11-24"},
        TIER1_CATALOG,
        TIER1_CATALOG,
        monitor,
        observations=empty_obs,
    )
    assert blocked is not None and blocked["stage"] == "loop_guard"
    assert blocked["feedback"]["reason"] == "route_date_scan_limit"

    # Different route is NOT blocked, and an epoch-advancing write resets the counter
    other_route = precheck_call(
        "flights.search",
        {"origin": "HAN", "destination": "DAD", "date": "2026-11-24"},
        TIER1_CATALOG,
        TIER1_CATALOG,
        monitor,
        observations=empty_obs,
    )
    assert other_route is None
    reset_obs = empty_obs + [
        {"name": "tasks.complete", "arguments": {"task_id": "t1"}, "scope": "env",
         "result": json.dumps({"ok": True, "done": True}),
         "automatic_execution_feedback": {"repeat_policy": "allowed"}},
    ]
    assert precheck_call(
        "flights.search",
        {"origin": "HAN", "destination": "SGN", "date": "2026-11-24"},
        TIER1_CATALOG,
        TIER1_CATALOG,
        monitor,
        observations=reset_obs,
    ) is None

    # public_0574 stagnant maps.search_place loop with repeated non-matching IDs + profile hint
    maps_obs = [
        {"name": "maps.search_place", "arguments": {"query": q}, "scope": "env",
         "result": json.dumps({"ok": True, "places": [
             {"id": "place_80", "name": "Phở Hoà Pasteur", "address": "260C Pasteur, Quận 3, TP.HCM"}
         ]}, ensure_ascii=False)}
        for q in ("tiệm tạp hóa Thảo", "tạp hóa Thảo")
    ]
    ctx = {"user_profile": {"work_address": "Tiệm tạp hóa Thảo, 203 Lê Văn Sỹ, Quận 3"}}
    stagnant = precheck_call(
        "maps.search_place",
        {"query": "tạp hóa"},
        TIER1_CATALOG,
        TIER1_CATALOG,
        LoopMonitor(),
        context=ctx,
        observations=maps_obs,
    )
    assert stagnant is not None and stagnant["stage"] == "loop_guard"
    assert stagnant["feedback"]["reason"] == "repeated_search_results"
    assert "work_address" in stagnant["error"]


def test_contacts_routing_phone_calls_and_banking_account_missing_info():
    # public_0511: contacts not enabled, phone.recent_calls + chat_ott.list_chats are enabled
    enabled_no_contacts = {k: v for k, v in TIER1_CATALOG.items() if not k.startswith("contacts.")}
    access = check_tool_access("contacts.search", TIER1_CATALOG, enabled_no_contacts)
    assert access["allowed"] is False and access["status"] == "tool_not_allowed"
    assert set(access["suggested_tools"]) <= set(enabled_no_contacts)
    assert "phone.recent_calls" in access["suggested_tools"]
    assert "phone.recent_calls" in access["instruction"]

    # missing_info_guard resolves phone_number from observed phone.recent_calls when contacts absent
    obs = [{"name": "phone.recent_calls", "arguments": {}, "scope": "env",
            "result": json.dumps({"ok": True, "calls": [
                {"id": "call_22", "phone_number": "09126743080", "name": "Mẹ", "missed": True},
            ]}, ensure_ascii=False)}]
    fb_phone = check_missing_info(
        {"tool_name": "phone.call", "params": ["phone_number"]},
        enabled_no_contacts,
        transcript="Gọi điện cho mẹ nha.",
        observations=obs,
    )
    assert fb_phone is not None
    assert fb_phone["known_arguments"]["phone_number"] == "09126743080"

    # public_0529: "bên thanh toán" resolves uniquely to "Tài khoản thanh toán" (acc_2807)
    hist_529 = [{"role": "assistant", "text": "Dạ tài khoản thanh toán còn 24.6 triệu.",
                 "tool_calls": [{"name": "banking.get_balance", "arguments": {}}],
                 "tool_results": [{"ok": True, "accounts": [
                     {"id": "acc_2807", "name": "Tài khoản thanh toán", "balance": 24600000},
                     {"id": "acc_555", "name": "Tài khoản tiết kiệm", "balance": 50000000},
                 ]}]}]
    fb_bank = check_missing_info(
        {"tool_name": "banking.open_savings", "params": ["from_account_id"]},
        TIER1_CATALOG,
        transcript="Ờ tiền bên thanh toán đâu có đủ gửi hai cái tôi gửi một cái thôi.",
        history=hist_529,
    )
    assert fb_bank is not None
    assert fb_bank["known_arguments"]["from_account_id"] == "acc_2807"

    # Ambiguous two accounts with "thanh toán" still refuses safely (None)
    hist_ambig = [{"role": "assistant", "text": "...",
                   "tool_calls": [{"name": "banking.get_balance", "arguments": {}}],
                   "tool_results": [{"ok": True, "accounts": [
                       {"id": "acc_1", "name": "Tài khoản thanh toán VNĐ"},
                       {"id": "acc_2", "name": "Tài khoản thanh toán USD"},
                   ]}]}]
    assert check_missing_info(
        {"tool_name": "banking.open_savings", "params": ["from_account_id"]},
        TIER1_CATALOG,
        transcript="Gửi từ bên thanh toán nhé.",
        history=hist_ambig,
    ) is None


def test_xref_empty_collection_guidance_and_parking_near_hints():
    class DummyEnv(Env):
        def validate(self, name, args):
            return {"ok": True}

        def call(self, name, args):
            return {"ok": True, "done": False}

    # P1.5: xref_guard after empty flights.search tells agent not to retry or hallucinate IDs
    obs_empty_flights = [
        {"name": "flights.search", "arguments": {"origin": "SGN", "destination": "HAN", "date": "2026-12-01"},
         "scope": "env", "result": json.dumps({"ok": True, "flights": []}),
         "verification": {"status": "empty"}},
    ]
    outcome_empty_ref = attempt_env_call(
        DummyEnv(),
        "flights.book",
        {"flight_id": "fl_999", "passengers": 1},
        "write",
        LoopMonitor(),
        1,
        spec=TIER1_CATALOG["flights.book"],
        observed_ids={"flights": set()},
        observations=obs_empty_flights,
    )
    assert outcome_empty_ref["executed"] is False
    assert outcome_empty_ref["feedback"]["stage"] == "xref_guard"
    assert "rỗng []" in outcome_empty_ref["feedback"]["error"]

    # P1.6: parking.find without `near` rejected in precheck with clear `near` hint
    missing_near = precheck_call(
        "parking.find",
        {"query": "Nội Bài"},
        TIER1_CATALOG,
        TIER1_CATALOG,
        LoopMonitor(),
    )
    assert missing_near is not None and missing_near["stage"] == "precheck"
    assert missing_near["feedback"]["required_param"] == "near"
    assert "near" in missing_near["error"] and "query" in missing_near["error"]

    # P1.6: parking.reserve before parking.find directs model to parking.find({"near": ...}) dynamically
    outcome_parking_ref = attempt_env_call(
        DummyEnv(),
        "parking.reserve",
        {"lot_id": "khu_b_05", "from_time": "2026-10-01T09:00:00+07:00", "hours": 5},
        "write",
        LoopMonitor(),
        0,
        spec=TIER1_CATALOG["parking.reserve"],
        observed_ids={"parking": set()},
        enabled=TIER1_CATALOG,
        observations=[],
    )
    assert outcome_parking_ref["executed"] is False
    assert outcome_parking_ref["feedback"]["stage"] == "xref_guard"
    assert "parking.find" in outcome_parking_ref["feedback"]["error"] and "near" in outcome_parking_ref["feedback"]["error"]

    # Generic schema/xref guards also work for any other app (e.g. flights.book or sms tools)
    outcome_flights_ref = attempt_env_call(
        DummyEnv(),
        "flights.book",
        {"flight_id": "fl_guess", "passengers": 1},
        "write",
        LoopMonitor(),
        0,
        spec=TIER1_CATALOG["flights.book"],
        observed_ids={"flights": set()},
        enabled=TIER1_CATALOG,
        observations=[],
    )
    assert outcome_flights_ref["executed"] is False
    assert "flights.search" in outcome_flights_ref["feedback"]["error"]


def test_generalized_diacritics_international_prepositions_filters_and_contacts_fallback():
    flight_spec = TIER1_CATALOG["flights.search"]

    # 1. Diacritic check blocks "Vĩnh Yên" / "Vịnh Hạ Long" without brittle word blacklist
    for place_text in ("Tìm vé đi Vĩnh Yên ghé Cần Thơ.", "Tìm vé ra Vịnh Hạ Long ghé Đà Nẵng."):
        kept = prepare_call_arguments(
            "flights.search",
            {"origin": "HAN", "destination": "DAD", "date": "2026-10-10"},
            flight_spec,
            transcript=place_text,
        )
        assert kept["destination"] == "DAD"

    # 2. International IATA normalization + multi-airport city preservation ("HND" for Tokyo)
    intl = prepare_call_arguments(
        "flights.search",
        {"origin": "TP.HCM", "destination": "Đài Bắc", "date": "2026-11-20"},
        flight_spec,
        transcript="Đặt vé từ Sài Gòn đi Đài Bắc.",
    )
    assert intl["origin"] == "SGN" and intl["destination"] == "TPE"
    hnd_kept = prepare_call_arguments(
        "flights.search",
        {"origin": "HAN", "destination": "HND", "date": "2026-11-20"},
        flight_spec,
        transcript="Đặt vé từ Hà Nội đi Tokyo.",
    )
    assert hnd_kept["destination"] == "HND"

    # 3. Dec -> Jan year rollover & prepositional "tới" exclusion in travel fallback
    fb_rollover = build_search_fallback(
        "flights.search",
        {"origin": "SGN", "destination": "HAN", "date": "2026-12-29"},
        flight_spec,
        transcript="Tìm vé đi Hà Nội ngày mùng năm tháng một.",
        context={"current_time": "2026-12-28T09:00:00+07:00"},
    )
    assert fb_rollover is not None and "2027-01-05" in fb_rollover["suggested_dates"]

    for prep_text in ("Tìm vé từ thứ hai tới thứ tư.", "Chuyến bay thứ năm tới Đà Nẵng lúc mấy giờ."):
        fb_prep = build_search_fallback(
            "flights.search",
            {"origin": "HAN", "destination": "DAD", "date": "2026-11-13"},
            flight_spec,
            transcript=prep_text,
            context={"current_time": "2026-11-13T18:10:00+07:00"},
        )
        assert fb_prep is None

    # 4. _check_stagnant_search_loop respects non-query filters (`near`)
    filtered_maps_obs = [
        {"name": "maps.search_place", "arguments": {"query": "cà phê", "near": n}, "scope": "env",
         "result": json.dumps({"ok": True, "places": [{"id": "place_1", "name": "Highlands"}]}, ensure_ascii=False)}
        for n in ("Quận 1", "Quận 3")
    ]
    assert precheck_call(
        "maps.search_place",
        {"query": "cà phê", "near": "Bình Thạnh"},
        TIER1_CATALOG,
        TIER1_CATALOG,
        LoopMonitor(),
        observations=filtered_maps_obs,
    ) is None

    # 5. Banking verb "thanh toán hóa đơn" must NOT falsely match "Tài khoản thanh toán"
    hist_verb = [{"role": "assistant", "text": "...",
                  "tool_calls": [{"name": "banking.get_balance", "arguments": {}}],
                  "tool_results": [{"ok": True, "accounts": [
                      {"id": "acc_pay", "name": "Tài khoản thanh toán", "balance": 1000000},
                      {"id": "acc_sav", "name": "Tài khoản tiết kiệm", "balance": 5000000},
                  ]}]}]
    assert check_missing_info(
        {"tool_name": "banking.open_savings", "params": ["from_account_id"]},
        TIER1_CATALOG,
        transcript="Thanh toán xong thì mở giúp tôi sổ tiết kiệm nhé.",
        history=hist_verb,
    ) is None

    # 6. phone.calls fallback works even when contacts.contacts has unrelated entries, and respects contacts ambiguity
    unrelated_contacts_obs = [
        {"name": "contacts.search", "arguments": {}, "scope": "env",
         "result": json.dumps({"ok": True, "contacts": [{"id": "c_1", "name": "Anh Hùng", "phone": "0901111111"}]}, ensure_ascii=False)},
        {"name": "phone.recent_calls", "arguments": {}, "scope": "env",
         "result": json.dumps({"ok": True, "calls": [{"id": "call_1", "name": "Mẹ", "phone_number": "09126743080"}]}, ensure_ascii=False)},
    ]
    fb_unrelated = check_missing_info(
        {"tool_name": "phone.call", "params": ["phone_number"]},
        TIER1_CATALOG,
        transcript="Gọi điện cho mẹ nhé.",
        observations=unrelated_contacts_obs,
    )
    assert fb_unrelated is not None and fb_unrelated["known_arguments"]["phone_number"] == "09126743080"

    ambig_contacts_obs = [
        {"name": "contacts.search", "arguments": {}, "scope": "env",
         "result": json.dumps({"ok": True, "contacts": [
             {"id": "c_1", "name": "Chị Lan", "phone": "0901"},
             {"id": "c_2", "name": "Cô Lan", "phone": "0902"},
         ]}, ensure_ascii=False)},
        {"name": "phone.recent_calls", "arguments": {}, "scope": "env",
         "result": json.dumps({"ok": True, "calls": [{"id": "call_1", "name": "Lan", "phone_number": "0903"}]}, ensure_ascii=False)},
    ]
    assert check_missing_info(
        {"tool_name": "phone.call", "params": ["phone_number"]},
        TIER1_CATALOG,
        transcript="Gọi điện cho Lan nhé.",
        observations=ambig_contacts_obs,
    ) is None

    # Generic access routing includes sms.read_recent (real tool) + works when sms disabled
    sms_catalog = {**TIER1_CATALOG, "sms.read_recent": {"name": "sms.read_recent", "kind": "read", "parameters": {"type": "object", "properties": {}}}}
    enabled_no_sms = {k: v for k, v in sms_catalog.items() if not k.startswith("sms.")}
    sms_access = check_tool_access("sms.read_recent", sms_catalog, enabled_no_sms)
    assert "chat_ott.list_chats" in sms_access["suggested_tools"]


def test_agent_public_0213_destination_rewrite_before_env_call():
    class FlightEnv(Env):
        def validate(self, name, args):
            return {"ok": True}

        def call(self, name, args):
            self.calls.append({"name": name, "arguments": dict(args)})
            return {"ok": True, "flights": [{"id": "fl_01", "origin": args["origin"], "destination": args["destination"]}]}

    env = FlightEnv()
    planner = Planner([
        call("flights.search", origin="HAN", destination="DAD", date="2026-10-08"),
        DONE,
    ])
    report = LifeMateAgent(None, planner, TIER1_CATALOG, AgentConfig()).run(
        {**prepared(), "id": "public_0213_sim",
         "context": {"installed_apps": {"flights": {"tools": ["flights.search", "flights.book"]}}}},
        env,
        Path("/unused"),
        "public_test",
        {**prepared(), "id": "public_0213_sim",
         "transcript": "Em ơi đặt giúp chị hai vé chuyến rv157 đi đà lạt thứ năm tuần sau nhé rồi giữ chỗ ở sân bay nội bài."},
    )
    assert report["executed"] == 1
    assert env.calls == [{"name": "flights.search", "arguments": {"origin": "HAN", "destination": "DLI", "date": "2026-10-08"}}]


def test_agent_batch_precheck_shares_route_date_scan_guard():
    class EmptyFlightEnv(Env):
        def validate(self, name, args):
            return {"ok": True}

        def call(self, name, args):
            self.calls.append({"name": name, "arguments": dict(args)})
            return {"ok": True, "flights": []}

    env = EmptyFlightEnv()
    planner = Planner([
        call("flights.search", origin="HAN", destination="SGN", date="2026-11-21"),
        call("flights.search", origin="HAN", destination="SGN", date="2026-11-22"),
        call("flights.search", origin="HAN", destination="SGN", date="2026-11-23"),
        {"action": "call", "name": "sys.batch", "arguments": {"calls": [
            {"name": "flights.search", "arguments": {"origin": "HAN", "destination": "SGN", "date": "2026-11-24"}},
        ]}},
        DONE,
    ])
    report = LifeMateAgent(None, planner, TIER1_CATALOG, AgentConfig()).run(
        {**prepared(), "id": "public_batch_sim",
         "context": {"installed_apps": {"flights": {"tools": ["flights.search"]}}}},
        env,
        Path("/unused"),
        "public_test",
        {**prepared(), "id": "public_batch_sim", "transcript": "Tìm vé Hà Nội Sài Gòn."},
    )
    assert report["executed"] == 3 and report["invalid_proposals"] == 1
    assert len(env.calls) == 3


def test_trained_adapter_manifest_policies_unchanged():
    manifest_path = Path(__file__).resolve().parents[1] / "result" / "run_sft_traj" / "training" / "adapter" / "lifemate_runtime.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["text_normalization"] == TEXT_NORMALIZATION
    assert manifest["local_time_policy"] == LOCAL_TIME_POLICY
    assert manifest["local_helper_policy"] == LOCAL_HELPER_POLICY
    assert manifest["assist_policy"] == ASSIST_POLICY
    assert manifest["schema_prompt_policy"] == SCHEMA_PROMPT_POLICY
    assert manifest["itn_policy"] == ITN_POLICY
    assert manifest["execution_policy"] == EXECUTION_POLICY
    assert manifest["reasoning_policy"] == REASONING_POLICY
