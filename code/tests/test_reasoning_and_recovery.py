"""Tests for guard self-correction SFT, optional checked reasoning, and context budgets."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lifemate.agent import SYSTEM_PROMPT, AgentConfig, LifeMateAgent, build_step_prompt, parse_decision
from lifemate.context import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_THINKING_BUDGET,
    MAX_CONTEXT_TOKENS,
    adaptive_thinking_budget,
)
from lifemate.decoding import (
    DecisionJsonStoppingCriteria,
    NativeToolStoppingCriteria,
    ThinkingBudgetLogitsProcessor,
    complete_decision_json,
)
from lifemate.function_calling import native_chat, visible_output
from lifemate.llm import render_prompt
from lifemate.models import LocalLLM
from lifemate.prompt_session import InferenceSession
from lifemate.reasoning import (
    attach_reasoning_to_trajectory,
    select_reasoning_stepwise_rows,
    synthesize_checked_reasoning,
    verify_reasoning_line,
)
from lifemate.recovery import build_guard_recovery_examples, sample_selected_for_guard_recovery
from lifemate.sft import AssistantCollator, SupervisedDataset, collapse_trajectory, encode_example, replay_sample
from test_baseline import CATALOG, FakeEnv, StubASR, item
from test_prefix_cache import tiny_model
from test_prompt_session import bpe_tokenizer


# ---------------------------------------------------------------------------
# Idea 1: Self-Correction / agent_feedback SFT Trajectories
# ---------------------------------------------------------------------------

def _contacts_sms_catalog():
    return {
        **CATALOG,
        "contacts.search": {
            "name": "contacts.search",
            "kind": "read",
            "description": "Tìm liên hệ",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": [],
            },
        },
        "sms.send": {
            "name": "sms.send",
            "kind": "write",
            "description": "Gửi tin nhắn",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                    "content": {"type": "string"},
                },
                "required": ["contact_id", "content"],
            },
        },
        "ewallet.transfer": {
            "name": "ewallet.transfer",
            "kind": "write",
            "description": "Chuyển tiền",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                    "amount": {"type": "integer"},
                },
                "required": ["contact_id", "amount"],
            },
        },
    }


@pytest.mark.parametrize('prompt_source', [None, 'audio_text', 'asr'])
def test_guard_recovery_builds_xref_feedback_and_direct_read_target(bpe_tokenizer, prompt_source):
    catalog = _contacts_sms_catalog()

    class Env:
        def __init__(self):
            self.calls = []

        def validate(self, name, args):
            spec = catalog[name]["parameters"]
            missing = set(spec.get("required", [])) - set(args)
            return {"ok": not missing, "error": "missing required" if missing else ""}

        def call(self, name, args):
            self.calls.append({"name": name, "arguments": deepcopy(args)})
            if name == "contacts.search":
                return {"ok": True, "contacts": [{"id": "c_08", "name": "Cô Ly"}]}
            return {"ok": True, "done": True}

    record = {
        "id": "train_xref",
        "audio_text": "Nhắn Cô Ly tối nay họp nhé",
        "context": {
            "installed_apps": {
                "contacts": {"tools": ["contacts.search"]},
                "sms": {"tools": ["sms.send"]},
            }
        },
        "history": [],
        "label": {
            "tool_calls": [
                {"name": "contacts.search", "arguments": {"query": "Cô Ly"}},
                {"name": "sms.send", "arguments": {"contact_id": "c_08", "content": "tối nay họp nhé"}},
            ]
        },
    }
    prepared = {"id": "train_xref", "transcript": "nhắn cô ly tối nay họp nhé", "history": [], "transcript_source": "audio_text"}
    if prompt_source is not None:
        prepared['transcript_source'] = prepared['prompt_transcript_source'] = prompt_source
    steps = replay_sample(record, catalog, Env(), Path("/unused"), prepared_transcript=prepared, include_date_helpers=False)

    examples = build_guard_recovery_examples(record, prepared, steps, catalog, Env, ratio=1.0, seed=42)
    assert len(examples) == 1
    ex = examples[0]
    assert ex["full_trajectory"] is False
    assert ex["recovery"]["kind"] == "guard_xref"
    assert ex["recovery"]["stage"] == "xref_guard"
    assert ex["recovery"]["failed_tool"] == "sms.send"
    assert ex["recovery"]["target_tool"] == "contacts.search"

    payload = json.loads(ex["messages"][1]["content"])
    assert payload.get('request_transcript_source') == prompt_source
    if prompt_source is not None:
        assert payload['request_transcript_note']
    obs = payload["previous_actions_and_results"]
    assert len(obs) == 1
    assert obs[0]["stage"] == "xref_guard"
    assert "result" not in obs[0]
    assert obs[0]["arguments"]["contact_id"] == "Cô Ly"
    assert json.loads(ex["messages"][2]["content"]) == {
        "action": "call",
        "name": "contacts.search",
        "arguments": {"query": "Cô Ly"},
    }

    # Verify native_chat emits a user `agent_feedback` turn with no fake failed assistant turn.
    chat, _ = native_chat(ex["messages"])
    user_feedback = json.loads(chat[-2]["content"])
    assert "agent_feedback" in user_feedback
    assert user_feedback["agent_feedback"]["stage"] == "xref_guard"
    assert chat[-3]["role"] == "user"  # Request user turn, NOT an assistant turn

    encoded = encode_example(ex, bpe_tokenizer, 16384, fuzzy_weight=0.3, reasoning_weight=0.25)
    assert any(lbl != -100 for lbl in encoded["labels"])


def test_guard_recovery_builds_validate_feedback_and_fixed_write_target():
    catalog = _contacts_sms_catalog()

    class Env(FakeEnv):
        def validate(self, name, args):
            spec = catalog[name]["parameters"]
            missing = set(spec.get("required", [])) - set(args)
            return {"ok": not missing, "error": f"missing {sorted(missing)}" if missing else ""}

        def call(self, name, args):
            self.calls.append({"name": name, "arguments": deepcopy(args)})
            return {"ok": True, "done": True}

    record = {
        "id": "train_val",
        "audio_text": "Chuyển cho c_01 50000 đồng",
        "context": {
            "collections": {"contacts": [{"id": "c_01", "name": "Lan"}]},
            "installed_apps": {"ewallet": {"tools": ["ewallet.transfer"]}},
        },
        "history": [],
        "label": {
            "tool_calls": [
                {"name": "ewallet.transfer", "arguments": {"contact_id": "c_01", "amount": 50000}},
            ]
        },
    }
    prepared = {"id": "train_val", "transcript": "chuyển cho c_01 50000 đồng", "history": [], "transcript_source": "audio_text"}
    steps = replay_sample(record, catalog, Env(), Path("/unused"), prepared_transcript=prepared, include_date_helpers=False)

    examples = build_guard_recovery_examples(record, prepared, steps, catalog, Env, ratio=1.0, seed=42)
    assert len(examples) == 1
    ex = examples[0]
    assert ex["recovery"]["kind"] == "guard_validate"
    assert ex["recovery"]["stage"] == "validate"
    obs = json.loads(ex["messages"][1]["content"])["previous_actions_and_results"]
    assert obs[-1]["stage"] == "validate"
    assert "amount" not in obs[-1]["arguments"]
    assert json.loads(ex["messages"][2]["content"]) == {
        "action": "call",
        "name": "ewallet.transfer",
        "arguments": {"contact_id": "c_01", "amount": 50000},
    }


def test_guard_recovery_hash_ratio_selection_is_deterministic():
    selected = [sample_selected_for_guard_recovery(f"train_{i:04d}", ratio=0.2, seed=42) for i in range(500)]
    assert 0.14 <= (sum(selected) / len(selected)) <= 0.26
    assert selected == [sample_selected_for_guard_recovery(f"train_{i:04d}", ratio=0.2, seed=42) for i in range(500)]
    assert not any(sample_selected_for_guard_recovery(f"train_{i}", ratio=0.0, seed=42) for i in range(50))
    assert all(sample_selected_for_guard_recovery(f"train_{i}", ratio=1.0, seed=42) for i in range(50))


# ---------------------------------------------------------------------------
# Idea 2: Deterministic Short Checked Reasoning + SFT Weighted Loss
# ---------------------------------------------------------------------------

def test_synthesize_and_verify_short_checked_reasoning_provenance(bpe_tokenizer):
    catalog = _contacts_sms_catalog()
    payload = {
        "request_transcript": "Chuyển 350000 đồng cho Cô Ly vào ngày mai.",
        "itn_candidates": [{"kind": "money", "source_text": "ba trăm rưỡi đồng", "value": 350000}],
        "itn_history_candidates": [],
        "user_context": {
            "date_reference": {"relative_dates": {"ngày mai": "2026-11-12"}},
        },
        "conversation_history": [],
        "available_tools": {"tool_schemas": list(catalog.values())},
        "previous_actions_and_results": [
            {
                "name": "contacts.search",
                "arguments": {"query": "Cô Ly"},
                "result": json.dumps({"ok": True, "contacts": [{"id": "c_08", "name": "Cô Ly"}]}),
            }
        ],
    }
    target = {
        "action": "call",
        "name": "ewallet.transfer",
        "arguments": {"contact_id": "c_08", "amount": 350000},
    }
    line = synthesize_checked_reasoning(payload, target, step_idx=1)
    assert "\n" not in line
    assert 'contact_id="c_08"' in line
    assert "contacts.search" in line
    assert "amount=350000" in line
    assert "ITN" in line
    assert verify_reasoning_line(line, payload, target)

    # Hallucinated numbers or IDs must fail verification.
    assert not verify_reasoning_line('Bước 2: chuyển 999999 cho "c_99" -> gọi ewallet.transfer.', payload, target)
    assert not verify_reasoning_line("dòng 1\ndòng 2", payload, target)
    assert not verify_reasoning_line("rò rỉ </think> thẻ", payload, target)


def test_reasoning_sft_weights_and_stepwise_selection(bpe_tokenizer):
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
    entry["audio_text"] = "Tạo ghi chú mua sữa ưu tiên 2"
    entry["label"] = {
        "tool_calls": [
            {"name": "smart_home.get_status", "arguments": {}},
            {"name": "notes.create", "arguments": {"title": "mua sữa tươi", "priority": 2}},
        ]
    }

    class NoteEnv(FakeEnv):
        def validate(self, name, args):
            return {"ok": True}

    steps = replay_sample(entry, catalog, NoteEnv(), Path("/unused"), include_date_helpers=False)
    enriched = attach_reasoning_to_trajectory(steps)
    assert all(row.get("reasoning") for row in enriched)
    assert all(row["messages"][-1]["reasoning_content"] == row["reasoning"] for row in enriched)

    # Intermediate call steps selected as step-wise rows retain `enable_thinking=True` alignment.
    stepwise = select_reasoning_stepwise_rows(enriched, ratio=1.0, seed=42)
    assert len(stepwise) == 2
    assert all(not r["full_trajectory"] and r["reasoning_stepwise"] for r in stepwise)

    # Full trajectory keeps reasoning on `steps[-1]` and downweights reasoning tokens cleanly.
    traj = collapse_trajectory(enriched)
    assert traj["reasoning"] == enriched[-1]["reasoning"]

    # Check single step with BOTH fuzzy arg and reasoning weight.
    write_row = encode_example(stepwise[1], bpe_tokenizer, 16384, fuzzy_weight=0.3, reasoning_weight=0.25)
    weights = write_row["loss_weights"]
    labels = write_row["labels"]
    assert any(l != -100 and w == pytest.approx(0.25) for l, w in zip(labels, weights))
    assert any(l != -100 and w == pytest.approx(0.3) for l, w in zip(labels, weights))
    assert any(l != -100 and w == pytest.approx(1.0) for l, w in zip(labels, weights))
    assert all(w == 0.0 for l, w in zip(labels, weights) if l == -100)

    # Verify default (unweighted) encode_example also isolates `<think>\n` prefix cleanly!
    plain_write = encode_example(stepwise[1], bpe_tokenizer, 16384)
    _, expected_prefix_ids = render_prompt(bpe_tokenizer, stepwise[1]["messages"][:-1], enable_thinking=True)
    assert plain_write["labels"][:len(expected_prefix_ids)] == [-100] * len(expected_prefix_ids)
    assert all(lbl != -100 for lbl in plain_write["labels"][len(expected_prefix_ids):])

    # SupervisedDataset and AssistantCollator handle mixed batch lengths & loss_weights.
    ds = SupervisedDataset([traj, stepwise[1]], bpe_tokenizer, 16384, fuzzy_weight=0.3, reasoning_weight=0.25)
    batch = AssistantCollator(bpe_tokenizer.pad_token_id)([ds[0], ds[1]])
    assert batch["loss_weights"].shape == batch["labels"].shape == batch["input_ids"].shape


def test_full_trajectory_preserves_think_prefix_on_every_step_for_ab_parity(bpe_tokenizer):
    def _contiguous_supervised_ranges(labels):
        ranges = []
        start = None
        for idx, lbl in enumerate(labels):
            if lbl != -100 and start is None:
                start = idx
            elif lbl == -100 and start is not None:
                ranges.append((start, idx))
                start = None
        if start is not None:
            ranges.append((start, len(labels)))
        return ranges

    catalog = _contacts_sms_catalog()
    catalog["sms.send"]["parameters"]["properties"]["content"]["x-match"] = "fuzzy"
    entry = item()
    entry["context"]["installed_apps"]["contacts"] = {"tools": ["contacts.search"]}
    entry["context"]["installed_apps"]["sms"] = {"tools": ["sms.send"]}
    entry["audio_text"] = "Nhắn cho anh Dũng là mai họp"
    entry["label"] = {
        "tool_calls": [
            {"name": "contacts.search", "arguments": {"query": "Dũng"}},
            {"name": "sms.send", "arguments": {"contact_id": "c_08", "content": "mai họp"}},
        ]
    }

    class ContactSmsEnv(FakeEnv):
        def validate(self, name, args):
            return {"ok": True}

        def call(self, name, args):
            if name == "contacts.search":
                return {"ok": True, "contacts": [{"id": "c_08", "name": "Dũng"}]}
            return {"ok": True, "sent": True}

    steps = replay_sample(entry, catalog, ContactSmsEnv(), Path("/unused"), include_date_helpers=False)
    assert len(steps) == 3  # contacts.search -> sms.send -> done

    empty_think_ids = bpe_tokenizer("<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    open_think_ids = bpe_tokenizer("<|im_start|>assistant\n<think>\n", add_special_tokens=False)["input_ids"]

    # 1) Default full-trajectory (without short reasoning): every step has empty <think> block before target
    # and matches step-wise SFT target tokens & fuzzy loss_weights 1:1.
    traj = collapse_trajectory(steps)
    row = encode_example(traj, bpe_tokenizer, 16384, fuzzy_weight=0.3)
    token_spans = _contiguous_supervised_ranges(row["labels"])
    assert len(token_spans) == 3
    for idx, (step_row_in, (ts, te)) in enumerate(zip(steps, token_spans)):
        assert ts >= len(empty_think_ids)
        assert row["input_ids"][ts - len(empty_think_ids):ts] == empty_think_ids, idx
        assert all(lbl == -100 for lbl in row["labels"][ts - len(empty_think_ids):ts])
        assert all(w == 0.0 for w in row["loss_weights"][ts - len(empty_think_ids):ts])

        step_encoded = encode_example(step_row_in, bpe_tokenizer, 16384, fuzzy_weight=0.3)
        step_target_ids = [tid for tid, lbl in zip(step_encoded["input_ids"], step_encoded["labels"]) if lbl != -100]
        step_target_weights = [w for w, lbl in zip(step_encoded["loss_weights"], step_encoded["labels"]) if lbl != -100]
        assert row["input_ids"][ts:te] == step_target_ids
        assert row["loss_weights"][ts:te] == pytest.approx(step_target_weights)

    # 2) Full-trajectory WITH short reasoning: every intermediate & final step keeps its reasoning_content
    # and matches step-wise SFT target tokens, reasoning weights (0.25), and fuzzy weights (0.3) 1:1.
    enriched = attach_reasoning_to_trajectory(steps)
    traj_r = collapse_trajectory(enriched)
    assert traj_r["assistant_reasonings"] == [r["reasoning"] for r in enriched]
    row_r = encode_example(traj_r, bpe_tokenizer, 16384, fuzzy_weight=0.3, reasoning_weight=0.25)
    token_spans_r = _contiguous_supervised_ranges(row_r["labels"])
    assert len(token_spans_r) == 3
    for idx, (step_row_in, (ts, te)) in enumerate(zip(enriched, token_spans_r)):
        assert ts >= len(open_think_ids)
        assert row_r["input_ids"][ts - len(open_think_ids):ts] == open_think_ids, idx
        assert all(lbl == -100 for lbl in row_r["labels"][ts - len(open_think_ids):ts])
        assert all(w == 0.0 for w in row_r["loss_weights"][ts - len(open_think_ids):ts])

        step_encoded = encode_example(step_row_in, bpe_tokenizer, 16384, fuzzy_weight=0.3, reasoning_weight=0.25)
        step_target_ids = [tid for tid, lbl in zip(step_encoded["input_ids"], step_encoded["labels"]) if lbl != -100]
        step_target_weights = [w for w, lbl in zip(step_encoded["loss_weights"], step_encoded["labels"]) if lbl != -100]
        assert row_r["input_ids"][ts:te] == step_target_ids
        assert row_r["loss_weights"][ts:te] == pytest.approx(step_target_weights)
        assert any(w == pytest.approx(0.25) for w in step_target_weights)
        assert any(w == pytest.approx(1.0) for w in step_target_weights)
        if idx == 1:  # sms.send has fuzzy `content`
            assert any(w == pytest.approx(0.3) for w in step_target_weights)


# ---------------------------------------------------------------------------
# Default 16000 / 384 budgets and optional adaptive thinking
# ---------------------------------------------------------------------------

def test_default_budgets_disable_thinking_and_adaptive_context_clamp():
    assert MAX_CONTEXT_TOKENS == 16384
    assert DEFAULT_MAX_NEW_TOKENS == 384
    assert DEFAULT_MAX_INPUT_TOKENS == 16000
    assert DEFAULT_THINKING_BUDGET == 0
    assert AgentConfig().thinking_budget == 0
    assert adaptive_thinking_budget(16000, DEFAULT_MAX_NEW_TOKENS, DEFAULT_THINKING_BUDGET) == (384, 0)

    assert adaptive_thinking_budget(10000, 512, 256) == (512, 256)
    assert adaptive_thinking_budget(15872, 512, 256) == (512, 256)
    # Tight output budget (384 -> Config B fallback): keeps 192 for tool call, caps think at 192.
    assert adaptive_thinking_budget(15000, 384, 256) == (384, 192)
    # Ultra-long prompt near 16384: degrades thinking budget to 0 cleanly.
    assert adaptive_thinking_budget(16192, 512, 256) == (192, 0)
    assert adaptive_thinking_budget(10000, 512, 0) == (512, 0)


def test_thinking_opened_stopping_and_visible_output_ignore_unclosed_think():
    assert visible_output('Bước 1: "trích dẫn" {"action":"done"}', thinking_opened=True) == ""
    out = visible_output('Bước 1: tìm ID "c_1"\n</think>\n\n<tool_call><function=a.b></function></tool_call>', thinking_opened=True)
    assert out == '<tool_call><function=a.b></function></tool_call>'
    assert complete_decision_json('Suy nghĩ {"action":"done"}', thinking_opened=True) is None
    assert complete_decision_json('Suy nghĩ </think>\n\n{"action":"done"}', thinking_opened=True) == '{"action":"done"}'


def test_thinking_budget_logits_processor_forces_close_tag_at_budget():
    from transformers import LogitsProcessor, LogitsProcessorList, Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokens = {
        "[UNK]": 0, "[EOS]": 1, "prompt": 2, "t1": 3, "t2": 4, "t3": 5, "t4": 6,
        "\n": 7, "</think>": 8, "\n\n": 9, "<tool_call>": 10, "<function=h.r>": 11,
        "</function>": 12, "</tool_call>": 13,
    }
    backend = Tokenizer(models.WordLevel(tokens, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[EOS]", eos_token="[EOS]",
        additional_special_tokens=["</think>", "<tool_call>", "</tool_call>"],
    )
    tokenizer.encode_orig = tokenizer.encode
    # WordLevel + WhitespaceSplit strips newlines unless mapped via custom encode or BPE; use explicit encode mock cleanly.
    class NewlineTok:
        unk_token_id = 0
        eos_token_id = 1
        pad_token_id = 1
        def convert_tokens_to_ids(self, t): return tokens.get(t, 0)
        def encode(self, s, add_special_tokens=False):
            return [tokens["\n"], tokens["</think>"], tokens["\n\n"]] if s == "\n</think>\n\n" else []
        def decode(self, ids, skip_special_tokens=False):
            rev = {v: k for k, v in tokens.items()}
            return "".join(rev[int(i)] for i in ids)

    ntok = NewlineTok()
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, eos_token_id=1, pad_token_id=1,
    )).eval()

    class AlwaysThinkThenCall(LogitsProcessor):
        def __init__(self, prompt_len):
            self.prompt_len = prompt_len
            self.after_close = [10, 11, 12, 13, 1]
            self.idx = 0

        def __call__(self, input_ids, scores):
            scores.fill_(float("-inf"))
            gen = input_ids[0, self.prompt_len:].tolist()
            if 8 in gen and gen[-1] == 9:
                self.after_mode = True
            if getattr(self, "after_mode", False):
                tok_id = self.after_close[min(self.idx, len(self.after_close) - 1)]
                self.idx += 1
            else:
                tok_id = 3  # Keeps looping thinking token `t1`
            scores[:, tok_id] = 0.0
            return scores

    prompt = torch.tensor([[2]])
    budget_proc = ThinkingBudgetLogitsProcessor(ntok, prompt.shape[1], max_thinking_tokens=3)
    stop = NativeToolStoppingCriteria(ntok, prompt.shape[1], fast_check=False, thinking_opened=True)
    out = model.generate(
        prompt, attention_mask=torch.ones_like(prompt), do_sample=False, max_new_tokens=20,
        pad_token_id=1,
        logits_processor=LogitsProcessorList([AlwaysThinkThenCall(prompt.shape[1]), budget_proc]),
        stopping_criteria=[stop],
    )
    gen = out[0, prompt.shape[1]:].tolist()
    # 3 thinking tokens [3, 3, 3], then forced close [7, 8, 9], then <tool_call> [10, 11, 12, 13]!
    assert gen == [3, 3, 3, 7, 8, 9, 10, 11, 12, 13]
    assert stop.completed_output == "<tool_call><function=h.r></function></tool_call>"


def test_agent_and_local_llm_adaptive_time_and_prefix_cache_across_turns(bpe_tokenizer, monkeypatch):
    from tokenizers import decoders
    tok = bpe_tokenizer
    tok.backend_tokenizer.decoder = decoders.ByteLevel()
    tok.clean_up_tokenization_spaces = False
    tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>", "<tool_call>", "</tool_call>"]})
    model = tiny_model(conditional=True, vocab_size=len(tok))
    llm = LocalLLM.from_loaded_model(
        model, tok, max_input_tokens=15872, max_new_tokens=512,
        prefix_cache="auto", prompt_cache="auto",
    )

    seen_input_ids = []
    seen_processors = []

    def fake_generate(**kwargs):
        seen_input_ids.append(kwargs["input_ids"][0].tolist())
        seen_processors.append(tuple(type(p).__name__ for p in kwargs.get('logits_processor', ())))
        end_think = tok.convert_tokens_to_ids("</think>")
        if 'ThinkingBudgetLogitsProcessor' in seen_processors[-1]:
            suffix = tok("step 1\n", add_special_tokens=False)["input_ids"] + [end_think] + tok('\n\n{"action":"done"}', add_special_tokens=False)["input_ids"]
        else:
            suffix = tok('{"action":"done"}', add_special_tokens=False)["input_ids"]
        seq = kwargs["input_ids"]
        criteria = kwargs.get("stopping_criteria")
        for tid in suffix:
            seq = torch.cat([seq, torch.tensor([[tid]])], dim=1)
            if criteria and criteria(seq, None):
                break
        return seq

    monkeypatch.setattr(model, "generate", fake_generate)

    session = InferenceSession()
    session.prepare("Bật đèn", {}, [], CATALOG)

    # Turn 0: thinking_budget=256 -> opens `<think>\n` + attaches ThinkingBudgetLogitsProcessor
    p0 = session.prompt([])
    res0 = llm.generate(SYSTEM_PROMPT, p0, thinking_budget=256)
    _, ref_on_ids = render_prompt(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": str(p0)}], enable_thinking=True)
    assert seen_input_ids[-1] == ref_on_ids
    assert seen_processors[-1] == ('ThinkingBudgetLogitsProcessor',)
    assert "<think>" in res0 and "</think>" in res0
    assert parse_decision(res0) == {"action": "done"}

    # Turn 1: budget 0 closes the prompt's think block and guards the output prefix.
    obs1 = [{"name": "smart_home.get_status", "arguments": {}, "result": "{}"}]
    p1 = session.prompt(obs1)
    res1 = llm.generate(SYSTEM_PROMPT, p1, thinking_budget=0)
    _, ref_off_ids = render_prompt(tok, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": str(p1)}], enable_thinking=False)
    assert seen_input_ids[-1] == ref_off_ids
    assert seen_processors[-1] == ('DirectActionLogitsProcessor',)
    assert parse_decision(res1) == {"action": "done"}
    # CPU & GPU prefix caches keep hitting across dynamic thinking switch!
    assert session.report()["suffix_renders"] == 1
    assert session.report()["full_renders"] == 1
    assert llm.prefix_cache_stats()["hits"] == 1
    assert llm.prefix_cache_stats()["fallbacks"] == 0


@pytest.mark.parametrize("thinking_budget", [None, 256])
def test_agent_if_time_allows_toggles_thinking_budget(monkeypatch, thinking_budget):
    import lifemate.agent as agent_mod

    now = [0.0]
    monkeypatch.setattr(agent_mod.time, "monotonic", lambda: now[0])
    recorded_budgets = []

    class RecordingLLM:
        def __init__(self):
            self.outputs = iter([
                '{"action":"call","name":"smart_home.get_status","arguments":{}}',
                '{"action":"done"}',
            ])

        def generate(self, system, user, *, thinking_budget=0):
            recorded_budgets.append(thinking_budget)
            # Advance clock so turn 1 is below reserve_seconds + thinking_reserve_seconds (1 + 2 = 3s).
            now[0] = 8.0
            return next(self.outputs)

    options = {} if thinking_budget is None else {"thinking_budget": thinking_budget}
    cfg = AgentConfig(sample_seconds=10.0, reserve_seconds=1.0, thinking_reserve_seconds=2.0, **options)
    report = LifeMateAgent(StubASR(), RecordingLLM(), CATALOG, cfg).run(
        item(), FakeEnv(), Path("/unused"), "public_test",
        prepared_transcript={"id": item()["id"], "transcript": "Bật đèn", "history": []},
    )
    assert recorded_budgets == [thinking_budget or 0, 0]
    assert report["thinking_turns"] == int(bool(thinking_budget))
    assert report["stop_reason"] == "done"


@pytest.mark.parametrize("short_reasoning", [False, True])
def test_prepare_sft_emits_guard_recovery_and_reasoning_stepwise_rows(tmp_path, bpe_tokenizer, monkeypatch, short_reasoning):
    import sys
    import prepare_sft

    split = tmp_path / "data" / "train"
    split.mkdir(parents=True)
    catalog = _contacts_sms_catalog()
    (split / "train_tools_catalog.json").write_text(json.dumps({"apps": [{"tools": list(catalog.values())}]}))
    record = {
        "id": "train_0001",
        "audio_text": "Nhắn Cô Ly tối mai họp nhé",
        "context": {
            "installed_apps": {
                "contacts": {"tools": ["contacts.search"]},
                "sms": {"tools": ["sms.send"]},
            }
        },
        "history": [],
        "label": {
            "tool_calls": [
                {"name": "contacts.search", "arguments": {"query": "Cô Ly"}},
                {"name": "sms.send", "arguments": {"contact_id": "c_08", "content": "tối mai họp nhé"}},
            ]
        },
    }
    (split / "train.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    (split / "zaic_tools.py").write_text(
        """class ToolEnv:
    def __init__(self, split, id, root): pass
    def validate(self, name, args):
        if name == 'sms.send' and ('contact_id' not in args or 'content' not in args):
            return {'ok': False, 'error': 'missing'}
        return {'ok': True}
    def call(self, name, args):
        if name == 'contacts.search':
            return {'ok': True, 'contacts': [{'id': 'c_08', 'name': 'Cô Ly'}]}
        return {'ok': True, 'done': True}
    def get_tool_calls(self): return []
    def get_state_hash(self): return 'h'
"""
    )
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    bpe_tokenizer.save_pretrained(tok_dir)
    (tok_dir / "config.json").write_text('{"model_type":"qwen3_5"}')

    out_dir = tmp_path / "sft_out"
    monkeypatch.delitem(sys.modules, "zaic_tools", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        [
            "prepare_sft.py",
            "--data-root", str(split.parent),
            "--output-dir", str(out_dir),
            "--asr-transcript-ratio", "0",
            "--tokenizer", str(tok_dir),
            "--guard-recovery-ratio", "1.0",
            "--no-step-wise-sft",
            *(["--short-reasoning"] if short_reasoning else ["--no-short-reasoning"]),
            "--reasoning-stepwise-ratio", "1.0",
        ],
    )
    prepare_sft.main()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    train_meta = manifest["partitions"]["train"]
    assert manifest["batching"]["max_new_tokens"] == 384
    assert manifest["sft_format"]["full_trajectory"] is True
    assert manifest["sft_format"]["short_reasoning"] is short_reasoning
    assert manifest["sft_format"]["inject_guard_recovery"] is True
    assert train_meta["guard_recovery_examples"] == 1
    assert train_meta["reasoning_stepwise_examples"] == (2 if short_reasoning else 0)

    rows = [json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines() if line.strip()]
    expected_rows = 4 if short_reasoning else 2
    assert len(rows) == expected_rows  # full trajectory + optional reasoning steps + guard recovery
    if not short_reasoning:
        assert all("reasoning_content" not in message for row in rows for message in row["messages"])
    ds = SupervisedDataset(rows, bpe_tokenizer, 16384, fuzzy_weight=0.3, reasoning_weight=0.25)
    assert len(ds) == expected_rows


def test_prepare_sft_default_stepwise_filters_untrained_recovery_steps_and_preserves_think_prefix(
    tmp_path, bpe_tokenizer, monkeypatch
):
    import sys
    import prepare_sft

    split = tmp_path / "data" / "train"
    split.mkdir(parents=True)
    catalog = _contacts_sms_catalog()
    (split / "train_tools_catalog.json").write_text(json.dumps({"apps": [{"tools": list(catalog.values())}]}))
    record = {
        "id": "train_0001",
        "audio_text": "Nhắn cho anh Dũng là mai họp",
        "context": {
            "installed_apps": {
                "contacts": {"tools": ["contacts.search"]},
                "sms": {"tools": ["sms.send"]},
            }
        },
        "history": [],
        "label": {
            "tool_calls": [
                {"name": "contacts.search", "arguments": {"query": "Dũng"}},
                {"name": "sms.send", "arguments": {"contact_id": "c_08", "content": "mai họp"}},
            ]
        },
    }
    (split / "train.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    (split / "zaic_tools.py").write_text(
        """class ToolEnv:
    def __init__(self, split, id, root): pass
    def validate(self, name, args):
        return {'ok': True}
    def call(self, name, args):
        if name == 'contacts.search':
            if args.get('query') == 'Dũng':
                return {'ok': True, 'contacts': [{'id': 'c_08', 'name': 'Dũng'}]}
            return {'ok': True, 'contacts': []}
        return {'ok': True, 'done': True}
    def get_tool_calls(self): return []
    def get_state_hash(self): return 'h'
"""
    )
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    bpe_tokenizer.save_pretrained(tok_dir)
    (tok_dir / "config.json").write_text('{"model_type":"qwen3_5"}')

    out_dir = tmp_path / "sft_stepwise_default"
    monkeypatch.delitem(sys.modules, "zaic_tools", raising=False)
    monkeypatch.setattr(
        sys, "argv",
        [
            "prepare_sft.py",
            "--data-root", str(split.parent),
            "--output-dir", str(out_dir),
            "--asr-transcript-ratio", "0",
            "--tokenizer", str(tok_dir),
            "--guard-recovery-ratio", "0.0",
        ],
    )
    prepare_sft.main()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    train_meta = manifest["partitions"]["train"]
    assert manifest["sft_format"]["full_trajectory"] is False
    assert manifest["sft_format"]["short_reasoning"] is False
    assert manifest["sft_format"]["inject_recovery"] is True
    assert train_meta["recovery_trajectories"] == 1

    rows = [json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines() if line.strip()]
    assert all(not row.get("full_trajectory") for row in rows)
    assert all(row.get("train", True) is not False for row in rows)
    assert all("reasoning_content" not in message for row in rows for message in row["messages"])
    assert len(rows) == 6  # 3 canonical steps + 3 trainable recovery steps (step 0 with train=False omitted)

    ds = SupervisedDataset(rows, bpe_tokenizer, 16384, fuzzy_weight=0.3)
    assert len(ds) == 6
    for row in rows:
        chat, tools = native_chat(row["messages"])
        text = bpe_tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False, tools=tools, enable_thinking=False
        )
        assert text.count("<think>\n\n</think>\n\n") >= 1
        assert text.endswith("<|im_end|>\n") or text.endswith("<|im_end|>")

