"""Tests for P0/P1 improvements: full-trajectory SFT, fuzzy weighting, date_reference,
context.apps, dual transcripts, recovery injection, and pre-execution guards."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.catalog import load_catalog, prompt_catalog
from lifemate.execution_policy import LoopMonitor
from lifemate.function_calling import native_chat
from lifemate.local_time import date_reference
from lifemate.prompt_session import fixed_prompt_payload
from lifemate.recovery import build_recovery_trajectory
from lifemate.sft import AssistantCollator, SupervisedDataset, collapse_trajectory, encode_example, replay_sample
from test_baseline import CATALOG, FakeEnv, StubASR, StubLLM, item
from test_prompt_session import bpe_tokenizer


def test_load_catalog_preserves_app_metadata_in_context_not_schemas(tmp_path):
    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(json.dumps({
        "apps": [{
            "id": "ewallet",
            "name_vi": "Ví điện tử",
            "description": "Thanh toán QR và chuyển tiền",
            "tools": [
                {
                    "name": "ewallet.get_balance",
                    "kind": "read",
                    "description": "Xem số dư ví",
                    "app": "ewallet",
                    "returns": "",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }]
    }, ensure_ascii=False), encoding="utf-8")
    catalog = load_catalog(cat_path)
    assert catalog["ewallet.get_balance"]["_app"] == {
        "id": "ewallet",
        "name": "Ví điện tử",
        "description": "Thanh toán QR và chuyển tiền",
    }
    tools, payload = fixed_prompt_payload(
        "Kiểm tra bên ví",
        {"current_time": "2026-10-13T09:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"},
        [],
        catalog,
    )
    assert payload["user_context"]["apps"] == {"ewallet": "Ví điện tử: Thanh toán QR và chuyển tiền"}
    assert "date_reference" in payload["user_context"]
    assert payload["user_context"]["date_reference"]["days"][1].startswith("2026-10-13")
    assert all("_app" not in spec for spec in tools["tool_schemas"])


def test_full_trajectory_fuzzy_weight_and_train_false_mask(bpe_tokenizer):
    catalog = deepcopy(CATALOG)
    catalog["notes.create"] = {
        "name": "notes.create",
        "kind": "write",
        "description": "Tạo ghi chú",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "x-match": "fuzzy"},
                "priority": {"type": "integer"},
            },
            "required": ["title"],
        },
    }
    entry = item()
    entry["context"]["installed_apps"]["notes"] = {"tools": ["notes.create"]}
    entry["history"] = [
        {
            "role": "assistant",
            "text": "Ghi chú cũ",
            "tool_calls": [{"name": "smart_home.get_status", "arguments": {}}],
            "tool_results": [{"ok": True, "devices": []}],
        }
    ]
    entry["audio_text"] = "Tạo ghi chú mua sữa"
    entry["label"] = {
        "tool_calls": [
            {"name": "smart_home.get_status", "arguments": {}},
            {"name": "notes.create", "arguments": {"title": "mua sữa ngay", "priority": 1}},
        ]
    }
    steps = replay_sample(
        entry, catalog, type("CatEnv", (FakeEnv,), {"validate": lambda self, n, a: {"ok": True}})(),
        Path("/unused"), include_date_helpers=False,
    )
    assert len(steps) == 3
    steps[0]["train"] = False  # Simulate failed first attempt masked out
    traj = collapse_trajectory(steps)
    assert traj["full_trajectory"] is True
    assert traj["assistant_train_mask"] == [False, True, True]

    row = encode_example(traj, bpe_tokenizer, 16384, fuzzy_weight=0.3)
    weights = row["loss_weights"]
    labels = row["labels"]
    assert len(weights) == len(labels) == len(row["input_ids"])
    # Ensure step 0 (train=False) and history turns are masked to -100
    assert any(l != -100 and w == pytest.approx(0.3) for l, w in zip(labels, weights))
    assert any(l != -100 and w == pytest.approx(1.0) for l, w in zip(labels, weights))
    assert all(w == 0.0 for l, w in zip(labels, weights) if l == -100)

    batch = AssistantCollator(bpe_tokenizer.pad_token_id)([row])
    assert batch["loss_weights"].shape == batch["labels"].shape


def test_real_qwen35_tokenizer_full_trajectory_and_fuzzy_weights():
    tok_dir = Path(__file__).resolve().parents[1] / "saved_models" / "llm"
    if not tok_dir.is_dir():
        pytest.skip("Local Qwen3.5 tokenizer weights absent")
    from transformers import AutoTokenizer
    from lifemate.llm import configure_tokenizer

    tok = configure_tokenizer(AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True), tok_dir)
    catalog = deepcopy(CATALOG)
    catalog["notes.create"] = {
        "name": "notes.create",
        "kind": "write",
        "description": "Tạo ghi chú",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "x-match": "fuzzy"},
                "priority": {"type": "integer"},
            },
            "required": ["title"],
        },
    }
    entry = item()
    entry["context"]["installed_apps"]["notes"] = {"tools": ["notes.create"]}
    entry["audio_text"] = "Tạo ghi chú họp nhóm"
    entry["label"] = {
        "tool_calls": [
            {"name": "notes.create", "arguments": {"title": "họp nhóm sáng thứ hai", "priority": 2}},
        ]
    }
    steps = replay_sample(
        entry, catalog, type("CatEnv", (FakeEnv,), {"validate": lambda self, n, a: {"ok": True}})(),
        Path("/unused"), include_date_helpers=False,
    )
    traj = collapse_trajectory(steps)
    row = encode_example(traj, tok, 16384, fuzzy_weight=0.3)
    fuzzy_tokens = [tok.decode([tid]) for tid, lbl, w in zip(row["input_ids"], row["labels"], row["loss_weights"]) if lbl != -100 and w < 1.0]
    exact_tokens = [tok.decode([tid]) for tid, lbl, w in zip(row["input_ids"], row["labels"], row["loss_weights"]) if lbl != -100 and w == 1.0]
    assert "họp" in "".join(fuzzy_tokens)
    assert "notes.create" in "".join(exact_tokens)
    assert "done" in "".join(exact_tokens)


def test_pre_execution_guards_xref_distinct_reads_and_filters():
    catalog = {
        **CATALOG,
        "contacts.search": {
            "name": "contacts.search", "kind": "read", "description": "Tìm liên hệ",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}, "tag": {"type": "string"}}, "required": []},
        },
        "weather.get_current": {
            "name": "weather.get_current", "kind": "read", "description": "Thời tiết",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        },
    }

    class Env(FakeEnv):
        def validate(self, name, args):
            return {"ok": True}

        def call(self, name, args):
            self.calls.append({"name": name, "arguments": deepcopy(args)})
            if name == "contacts.search":
                return {"ok": True, "contacts": []}
            if name == "smart_home.get_status":
                return {"ok": True, "devices": [{"id": "light_living", "name": "đèn phòng khách"}]}
            return {"ok": True}

    entry = item()
    entry["context"]["installed_apps"] = {
        "smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set"]},
        "contacts": {"tools": ["contacts.search"]},
        "weather": {"tools": ["weather.get_current"]},
    }
    entry["context"]["user_profile"] = {"city": "Huế"}
    outputs = [
        # 1. Guessed x-ref before get_status -> blocked BEFORE env.call
        json.dumps({"action": "call", "name": "smart_home.light_set", "arguments": {"device_id": "fake_id", "power": "on"}}),
        # 2. Empty optional filter should be cleaned before validate/call
        json.dumps({"action": "call", "name": "contacts.search", "arguments": {"query": "A", "tag": ""}}),
        # 3. Second empty search -> allowed
        json.dumps({"action": "call", "name": "contacts.search", "arguments": {"query": "B"}}),
        # 4. Third distinct empty search -> allowed without an empty-read cap
        json.dumps({"action": "call", "name": "contacts.search", "arguments": {"query": "C"}}),
        # 5. Unmentioned default profile city -> popped from weather.get_current
        json.dumps({"action": "call", "name": "weather.get_current", "arguments": {"location": "Huế"}}),
        # 6. Observe device ID cleanly then mutate
        json.dumps({"action": "call", "name": "smart_home.get_status", "arguments": {}}),
        json.dumps({"action": "call", "name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "on"}}),
        '{"action":"done"}',
    ]
    env = Env()
    prepared = {"id": entry["id"], "transcript": "Thời tiết hôm nay sao nhỉ", "history": []}
    stats = LifeMateAgent(StubASR(), StubLLM(outputs), catalog, AgentConfig(max_invalid=5)).run(
        entry, env, Path("/unused"), "train", prepared_transcript=prepared
    )
    assert stats["invalid_proposals"] == 1  # Only the guessed x-ref ID is blocked.
    assert [c["name"] for c in env.calls] == [
        "contacts.search", "contacts.search", "contacts.search", "weather.get_current",
        "smart_home.get_status", "smart_home.light_set"
    ]
    assert env.calls[0]["arguments"] == {"query": "A"}  # empty tag stripped
    assert env.calls[3]["arguments"] == {}  # default Huế popped
    assert env.calls[5]["arguments"] == {"device_id": "light_living", "power": "on"}


def test_cross_app_contact_id_observed_in_history_passes_xref():
    catalog = {
        **CATALOG,
        "sms.send": {
            "name": "sms.send", "kind": "write", "description": "Gửi SMS",
            "parameters": {"type": "object", "properties": {
                "contact_id": {"type": "string", "x-ref": "contacts.contacts"}, "content": {"type": "string"}}},
        },
    }
    entry = item()
    entry["context"]["installed_apps"] = {"sms": {"tools": ["sms.send"]}}
    entry["history"] = [{
        "role": "assistant", "text": "Có tin nhắn",
        "tool_calls": [{"name": "sms.read_recent", "arguments": {}}],
        "tool_results": [{"ok": True, "messages": [{"id": "m1", "contact_id": "c_778"}]}],
    }]
    env = FakeEnv()
    env.validate = lambda name, args: {"ok": True}
    outputs = [
        json.dumps({"action": "call", "name": "sms.send", "arguments": {"contact_id": "c_778", "content": "ok"}}),
        '{"action":"done"}',
    ]
    stats = LifeMateAgent(StubASR(), StubLLM(outputs), catalog, AgentConfig()).run(
        entry, env, Path("/unused"), "train", prepared_transcript={"id": entry["id"], "transcript": "nhắn", "history": entry["history"]}
    )
    assert stats["invalid_proposals"] == 0
    assert env.calls == [{"name": "sms.send", "arguments": {"contact_id": "c_778", "content": "ok"}}]


@pytest.mark.parametrize('prompt_source', [None, 'asr'])
def test_build_recovery_trajectory_honorific_masks_wrong_search(prompt_source):
    catalog = {
        "sys.ask_missing_info": CATALOG["sys.ask_missing_info"],
        "contacts.search": {
            "name": "contacts.search", "kind": "read", "description": "Tìm liên hệ",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
        "sms.send": {
            "name": "sms.send", "kind": "write", "description": "Gửi tin",
            "parameters": {"type": "object", "properties": {"contact_id": {"type": "string"}, "content": {"type": "string"}}},
        },
    }

    class ContactEnv:
        def __init__(self):
            self.calls = []

        def validate(self, name, args):
            return {"ok": True}

        def call(self, name, args):
            self.calls.append({"name": name, "arguments": deepcopy(args)})
            if name == "contacts.search":
                q = args.get("query", "")
                if q == "Dũng":
                    return {"ok": True, "contacts": [{"id": "c_1", "name": "Dũng"}]}
                return {"ok": True, "contacts": []}
            return {"ok": True, "done": True}

    record = {
        "id": "train_rec",
        "audio_text": "Nhắn cho anh Dũng là mai họp",
        "context": {"installed_apps": {"contacts": {"tools": ["contacts.search"]}, "sms": {"tools": ["sms.send"]}}},
        "history": [],
        "label": {"tool_calls": [
            {"name": "contacts.search", "arguments": {"query": "Dũng"}},
            {"name": "sms.send", "arguments": {"contact_id": "c_1", "content": "mai họp"}},
        ]},
    }
    prepared = {"id": "train_rec", "transcript": "nhắn cho anh dũng là mai họp", "history": [], "transcript_source": "asr"}
    if prompt_source is not None:
        prepared['prompt_transcript_source'] = prompt_source
    steps = replay_sample(record, catalog, ContactEnv(), Path("/unused"), prepared_transcript=prepared, include_date_helpers=False)
    rec = build_recovery_trajectory(record, prepared, steps, catalog, ContactEnv)
    assert rec is not None
    rec_steps, meta = rec
    assert meta == {"kind": "honorific", "wrong": "anh Dũng", "recover": {"query": "Dũng"}, "gold_query": "Dũng"}
    assert [s.get("train", True) for s in rec_steps] == [False, True, True, True]
    traj = collapse_trajectory(rec_steps)
    assert traj["assistant_train_mask"] == [False, True, True, True]
    assert all(json.loads(s['messages'][1]['content']).get('request_transcript_source') == prompt_source
               for s in rec_steps)
    assert json.loads(traj['messages'][1]['content']).get('request_transcript_source') == prompt_source


def test_weighted_loss_trainer_matches_full_vocab_loss_and_gradients(tmp_path):
    from transformers import Trainer, TrainingArguments
    from lifemate.numerics import guarded_trainer_class
    from test_prefix_cache import tiny_model
    import train_lora

    model_full = tiny_model(conditional=True, vocab_size=32)
    model_sparse = deepcopy(model_full)
    model_full.train()
    model_sparse.train()

    input_ids = torch.tensor([[2, 5, 8, 11, 14, 17, 20, 1]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 8, 11, -100, 17, 20, 1]])
    loss_weights = torch.tensor([[0.0, 0.0, 1.0, 0.3, 0.0, 0.3, 1.0, 1.0]])

    # Explicit full-vocab reference loss + gradients
    out = model_full(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = out.logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    ce = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shift_labels)
    ref_loss = (ce * loss_weights[:, 1:]).sum() / shift_labels.ne(-100).sum().float()
    ref_loss.backward()

    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        per_device_train_batch_size=1,
        max_steps=1,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer_cls = guarded_trainer_class(train_lora.make_weighted_trainer_class(Trainer, "transformers"))
    trainer = trainer_cls(model=model_sparse, args=args)
    sparse_loss = trainer.compute_loss(
        model_sparse,
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "loss_weights": loss_weights},
    )
    sparse_loss.backward()

    torch.testing.assert_close(sparse_loss, ref_loss, rtol=1e-5, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(model_full.named_parameters(), model_sparse.named_parameters()):
        assert n1 == n2
        if p1.grad is None:
            assert p2.grad is None
        else:
            torch.testing.assert_close(p2.grad, p1.grad, rtol=1e-4, atol=1e-4)

