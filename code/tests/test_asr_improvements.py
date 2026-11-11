"""Tests for ASR P0/P1 improvements: FP32/beam defaults, batched cache, asr_ft selection,
selective train_asr.jsonl target prep, and --filter-damaged-asr."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

APP = Path(__file__).resolve().parents[1]
REPO_ROOT = APP.parent

from lifemate.agent import AgentConfig  # noqa: E402
from lifemate.asr import LocalASR, asr_files, resolve_asr_precision  # noqa: E402
from lifemate.asr_targets import (  # noqa: E402
    build_asr_target,
    format_human_target,
    load_vocab_chars,
    model_style,
    needs_target_normalization,
    target_v3,
)
from lifemate.sft import is_asr_argument_damaged  # noqa: E402
from lifemate.transcripts import asr_identity, dataset_identity, load_transcripts, prepare_transcripts  # noqa: E402


def _write_onnx(folder: Path, precision: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    suffix = ".int8.onnx" if precision == "int8" else ".onnx"
    for name in ("tokens.txt", *(p + suffix for p in ("encoder", "decoder", "joiner"))):
        (folder / name).write_bytes(b"fixture-onnx")


def test_auto_precision_prefers_fp32_while_supporting_int8_only(tmp_path):
    int8_dir = tmp_path / "asr_int8"
    _write_onnx(int8_dir, "int8")
    assert resolve_asr_precision(int8_dir, "auto") == "int8"

    both_dir = tmp_path / "asr_both"
    _write_onnx(both_dir, "int8")
    _write_onnx(both_dir, "fp32")
    assert resolve_asr_precision(both_dir, "auto") == "fp32"
    assert asr_files(both_dir, "auto")["encoder"].name == "encoder.onnx"
    assert asr_files(both_dir, "int8")["encoder"].name == "encoder.int8.onnx"

    # Explicit fp32 must never silently fall back to int8.
    with pytest.raises(FileNotFoundError, match="encoder.onnx"):
        asr_files(int8_dir, "fp32")


def test_asr_identity_tracks_decoding_method_and_beam_size(tmp_path):
    _write_onnx(tmp_path, "fp32")
    beam4 = asr_identity(tmp_path, "auto", 2)
    beam8 = asr_identity(tmp_path, "auto", 2, decoding_method="modified_beam_search", max_active_paths=8)
    greedy = asr_identity(tmp_path, "auto", 2, decoding_method="greedy_search")
    assert beam4["precision"] == "fp32"
    assert beam4["decoding_method"] == "modified_beam_search" and beam4["max_active_paths"] == 4
    assert beam4 != beam8 and beam4 != greedy and greedy["max_active_paths"] == 1


def test_prepare_transcripts_uses_batched_decode_and_recovers_partial_errors(tmp_path):
    split = tmp_path / "data"
    split.mkdir()
    items = []
    for i in range(4):
        (split / f"{i}.wav").write_bytes(b"audio")
        items.append({"id": f"s_{i}", "audio": f"{i}.wav", "history": [{"role": "user", "audio": "h.wav"}]})
    (split / "h.wav").write_bytes(b"hist")
    identity = dataset_identity(items, split, "public_test")
    batch_calls = []

    class BatchASR:
        def transcribe_batch(self, paths, batch_size=16):
            batch_calls.append(([p.name for p in paths], batch_size))
            return [f"text {p.name}" for p in paths]

        def transcribe(self, path):
            return f"text {path.name}"

    cache = tmp_path / "cache.jsonl"
    prepare_transcripts(items, split, cache, identity, {}, BatchASR, batch_size=2)
    rows, _ = load_transcripts(cache, identity)
    assert len(rows) == 4
    assert len(batch_calls) == 2
    assert all(bs == 2 for _, bs in batch_calls)
    assert rows[0]["transcript"] == "Text 0.wav."
    assert rows[0]["history"] == [{"role": "user", "text": "Text h.wav."}]

    # Failure inside one item in a chunk must still flush preceding items and attribute error ID.
    class FailOnOne(BatchASR):
        def transcribe_batch(self, paths, batch_size=16):
            raise RuntimeError("boom in batch")

        def transcribe(self, path):
            if path.name == "2.wav":
                raise RuntimeError("corrupt wav 2")
            return super().transcribe(path)

    bad_cache = tmp_path / "bad_cache.jsonl"
    with pytest.raises(RuntimeError, match="corrupt wav 2"):
        prepare_transcripts(items, split, bad_cache, identity, {}, FailOnOne, batch_size=3)
    partial_rows = [json.loads(line) for line in bad_cache.with_suffix(".partial.jsonl").read_text().splitlines()]
    assert [r["id"] for r in partial_rows] == ["s_0", "s_1"]
    assert json.loads(bad_cache.with_suffix(".errors.jsonl").read_text())["id"] == "s_2"


def test_run_server_prefers_asr_ft_when_present_while_keeping_asr_base(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    script = app / "run_server.sh"
    shutil.copy2(APP / "run_server.sh", script)
    shim = tmp_path / "python"
    shim.write_text(f"""#!{sys.executable}
import json, sys
sys.path.insert(0, {str(APP)!r})
import main
start = next(i for i, v in enumerate(sys.argv) if v.endswith('/main.py'))
sys.argv = sys.argv[start:]
args = main.parse_args()
print(json.dumps({{'asr': str(args.asr_model)}}))
""")
    shim.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "ASR_MODEL": "/stale/asr"}
    (app / "models/asr").mkdir(parents=True)
    (app / "models/asr/tokens.txt").write_text("a 1\n")

    out_base = json.loads(subprocess.check_output(["bash", script], env=env, text=True))
    assert out_base["asr"] == str(app / "models/asr")

    (app / "models/asr_ft").mkdir(parents=True)
    (app / "models/asr_ft/tokens.txt").write_text("a 1\n")
    out_ft = json.loads(subprocess.check_output(["bash", script], env=env, text=True))
    assert out_ft["asr"] == str(app / "models/asr_ft")


def test_download_models_can_stage_both_asr_and_asr_ft(tmp_path, monkeypatch):
    import huggingface_hub
    app = tmp_path / "app"
    (app / "scripts").mkdir(parents=True)
    script = app / "scripts/download_models.py"
    shutil.copy2(APP / "scripts/download_models.py", script)

    def fake_snapshot(**kwargs):
        dest = Path(kwargs["local_dir"])
        if kwargs["repo_id"] == "g-group-ai-lab/gipformer1.5-68M-rnnt":
            for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx",
                         "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx",
                         "tokens.txt", "bpe.model", "model.pt"):
                (dest / name).write_bytes(b"base")
        else:
            sub = dest / "asr-gipformer"
            sub.mkdir(parents=True, exist_ok=True)
            for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
                (sub / name).write_bytes(b"finetuned")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--only", "asr", "--models-dir", str(app / "models"), "--asr-ft-repo", "example/release",
    ])
    runpy.run_path(str(script), run_name="__main__")
    assert (app / "models/asr/model.pt").is_file() and (app / "models/asr/encoder.onnx").is_file()
    assert (app / "models/asr_ft/encoder.onnx").read_bytes() == b"finetuned"


def test_selective_asr_targets_and_alphanumeric_patterns():
    assert model_style("HOÀ KHOÁ HUỶ THUỶ QUÝ") == "HÒA KHÓA HỦY THỦY QUÝ"
    assert not needs_target_normalization("Nếu mai mưa thì dời chạy bộ một tiếng, nhắn Sếp Dũng.")
    assert needs_target_normalization("Thế xoá lịch uống Omega-3 đi")
    assert needs_target_normalization("Ghi dùm cái trận S+ đang đá")

    # Human-verified main audio_text keeps its exact wording regardless of hypothesis.
    target, meta = build_asr_target(
        "Ơi, chuyển giúp mình năm mươi ngàn nhé!",
        hypothesis="VỜI CHUYỀN GIÚP MÌNH NĂM MƯƠI NGHÌN NHÉ",
        is_history=False,
        max_history_wer=0.1,
    )
    assert meta["mode"] == "human_verified"
    assert target == "ƠI CHUYỂN GIÚP MÌNH NĂM MƯƠI NGÀN NHÉ"

    # Human-verified audio_text applies identical symbol transforms (&, %, +, @, °) as history audio:
    assert format_human_target("Giao hàng & lắp đặt giảm mười phần trăm") == "GIAO HÀNG VÀ LẮP ĐẶT GIẢM MƯỜI PHẦN TRĂM"
    assert format_human_target("Trận S+ đang đá", hypothesis="TRẬN S CỘNG ĐANG ĐÁ") == "TRẬN S CỘNG ĐANG ĐÁ"
    assert format_human_target("Thời tiết ba mươi độ C") == "THỜI TIẾT BA MƯƠI ĐỘ C"

    # Leftover main rows & history rows expand digits/symbols and align to hypothesis.
    assert target_v3("Xăng RON95 hôm nay?", "XĂNG RON CHÍN LĂM HÔM NAY") == "XĂNG RON CHÍN LĂM HÔM NAY"
    assert target_v3("Lớp 8A3 đi bảo tàng", "LỚP TÁM A BA ĐI BẢO TÀNG") == "LỚP TÁM A BA ĐI BẢO TÀNG"
    assert target_v3("Góc Phố 24h", "GÓC PHỐ HAI TƯ GIỜ") == "GÓC PHỐ HAI TƯ GIỜ"
    assert target_v3("Kênh 101", "KÊNH MỘT KHÔNG MỘT") == "KÊNH MỘT KHÔNG MỘT"
    assert target_v3("S+ Sport", "S CỘNG SPORT") == "S CỘNG SPORT"
    assert target_v3("Chuyển 2.500.000đ lúc 7:30 ngày 15/11",
                     "CHUYỂN HAI TRIỆU NĂM TRĂM NGHÌN ĐỒNG LÚC BẢY GIỜ RƯỠI NGÀY MƯỜI LĂM THÁNG MƯỜI MỘT") is not None

    # Noisy-label WER filtering applies ONLY to unverified history rows, never human-verified main text.
    hist_drop, hmeta = build_asr_target(
        "Mở tivi kênh HTV7 đi.", hypothesis="TẮT QUẠT PHÒNG NGỦ", is_history=True, max_history_wer=0.5,
    )
    assert hist_drop is None and hmeta["reason"] == "high_history_wer"


def test_prepare_gipformer_sft_selective_policy_and_auto_hypotheses(tmp_path, monkeypatch):
    from scripts.asr_finetune import prepare_gipformer_sft as prep

    root = tmp_path / "data"
    train_dir = root / "train"
    (train_dir / "audio").mkdir(parents=True)
    for name in ("m1.wav", "m2.wav", "h1.wav", "h2.wav"):
        sf.write(train_dir / "audio" / name, np.zeros(1600, dtype=np.float32), 16000)

    jsonl = tmp_path / "train_asr.jsonl"
    rows = [
        {
            "id": "train_0001",
            "audio": "audio/m1.wav",
            "audio_text": "Nếu ví đủ thì chuyển năm mươi ngàn cho Hoà.",
            "history": [
                {"role": "user", "audio": "audio/h1.wav", "audio_text": "Bật kênh VTV6 lúc 7:30."},
                {"role": "assistant", "text": "Đã bật."},
            ],
        },
        {
            "id": "train_0002",
            "audio": "audio/m2.wav",
            "audio_text": "Kiểm tra lớp 8A3 giúp mình.",  # missed digit example in main audio_text
            "history": [
                {"role": "user", "audio": "audio/h2.wav", "audio_text": "Bật đèn phòng khách."},
            ],
        },
    ]
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    model_dir = tmp_path / "models/asr"
    _write_onnx(model_dir, "fp32")
    # Reuse repo tokens.txt if present for real vocab check.
    repo_tokens = APP / "models/asr/tokens.txt"
    if repo_tokens.is_file():
        shutil.copy2(repo_tokens, model_dir / "tokens.txt")

    transcribed_batches = []

    class StubLocalASR:
        def __init__(self, *a, **kw): pass

        def transcribe_batch(self, paths, batch_size=16):
            transcribed_batches.append([p.name for p in paths])
            out = []
            for p in paths:
                if p.name == "h1.wav":
                    out.append("BẬT KÊNH VTV SÁU LÚC BẢY GIỜ RƯỠI")
                elif p.name == "h2.wav":
                    out.append("CÂU NÀY SAI HOÀN TOÀN")
                else:
                    out.append("KIỂM TRA LỚP TÁM A BA GIÚP MÌNH")
            return out

    import lifemate.asr as asr_mod
    monkeypatch.setattr(asr_mod, "LocalASR", StubLocalASR)

    out_dir = tmp_path / "prepared"
    manifest = prep.prepare_asr_dataset(
        input_jsonl=jsonl,
        data_root=root,
        output_dir=out_dir,
        model_dir=model_dir,
        val_fraction=0.0,
        generate_hypotheses=True,
        max_history_wer=0.5,
        final=True,
    )
    # m1.wav (human-verified clean audio_text) must NEVER be sent through hypothesis generation or filtered!
    all_batched = [name for batch in transcribed_batches for name in batch]
    assert "m1.wav" not in all_batched
    assert set(all_batched) == {"m2.wav", "h1.wav", "h2.wav"}
    assert manifest["main_human_verified"] == 1
    assert manifest["main_leftover_normalized"] == 1
    assert manifest["history_kept"] == 1
    assert manifest["skipped_history"] == 1
    assert manifest["skipped_main"] == 0

    trained = [json.loads(line) for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    by_audio = {r["audio"]: r["target_text"] for r in trained}
    assert by_audio["audio/m1.wav"] == "NẾU VÍ ĐỦ THÌ CHUYỂN NĂM MƯƠI NGÀN CHO HÒA"
    assert by_audio["audio/m2.wav"] == "KIỂM TRA LỚP TÁM A BA GIÚP MÌNH"
    assert by_audio["audio/h1.wav"] == "BẬT KÊNH VTV SÁU LÚC BẢY GIỜ RƯỠI"


def test_real_train_asr_jsonl_selective_coverage():
    real_jsonl = REPO_ROOT / "train_asr.jsonl"
    real_tokens = APP / "models/asr/tokens.txt"
    if not real_jsonl.is_file() or not real_tokens.is_file():
        pytest.skip("Workspace train_asr.jsonl or tokens.txt not present")
    vocab = load_vocab_chars(real_tokens)
    human_cnt = leftover_cnt = hist_cnt = 0
    seen_hist = set()
    for line in real_jsonl.open(encoding="utf-8"):
        row = json.loads(line)
        t, meta = build_asr_target(row["audio_text"], is_history=False, vocab_chars=vocab)
        assert t is not None, (row["id"], meta)
        if meta["mode"] == "human_verified":
            human_cnt += 1
        else:
            leftover_cnt += 1
        for turn in row.get("history") or []:
            if turn.get("role") == "user" and "audio" in turn and "audio_text" in turn:
                a = turn["audio"]
                if a in seen_hist:
                    continue
                seen_hist.add(a)
                ht, _ = build_asr_target(turn["audio_text"], is_history=True, vocab_chars=vocab)
                assert ht is not None, (row["id"], a, turn["audio_text"])
                hist_cnt += 1
    assert human_cnt >= 1530 and leftover_cnt <= 15
    assert hist_cnt == 3491


def test_filter_damaged_asr_in_prepare_sft(tmp_path, monkeypatch):
    import lifemate.models
    import lifemate.transcripts
    import prepare_sft
    from test_baseline import FakeEnv

    catalog = {
        "contacts.search": {
            "name": "contacts.search", "kind": "read",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
        "sms.send": {
            "name": "sms.send", "kind": "write",
            "parameters": {"type": "object", "properties": {
                "recipient": {"type": "string"},
                "content": {"type": "string", "x-match": "fuzzy"},
            }, "required": ["recipient", "content"]},
        },
    }
    record_damaged = {
        "id": "train_0", "audio": "train_0.wav",
        "audio_text": "Nhắn cho Minh Anh nội dung là chào buổi sáng.",
        "context": {"installed_apps": {"contacts": {"tools": ["contacts.search"]}, "sms": {"tools": ["sms.send"]}}},
        "history": [],
        "label": {"tool_calls": [
            {"name": "contacts.search", "arguments": {"query": "Minh Anh"}},
            {"name": "sms.send", "arguments": {"recipient": "Minh Anh", "content": "chào buổi sáng"}},
        ]},
    }
    record_fuzzy_diff = {
        "id": "train_1", "audio": "train_1.wav",
        "audio_text": "Nhắn cho Lan nội dung là hẹn mai gặp nhé.",
        "context": {"installed_apps": {"contacts": {"tools": ["contacts.search"]}, "sms": {"tools": ["sms.send"]}}},
        "history": [],
        "label": {"tool_calls": [
            {"name": "contacts.search", "arguments": {"query": "Lan"}},
            {"name": "sms.send", "arguments": {"recipient": "Lan", "content": "hẹn mai gặp nhé"}},
        ]},
    }
    assert is_asr_argument_damaged(record_damaged, {"transcript": "Nhắn cho mình nội dung là chào buổi sáng.", "history": []}, catalog)
    # Fuzzy field difference ("content") must NOT trigger damaged filter when exact-match ("Lan") survives.
    assert not is_asr_argument_damaged(record_fuzzy_diff, {"transcript": "Nhắn cho Lan nội dung là hẹn mai gặp nha.", "history": []}, catalog)

    src = tmp_path / "train"
    src.mkdir()
    for i in range(2):
        (src / f"train_{i}.wav").write_bytes(b"audio")
    (src / "train_tools_catalog.json").write_text(json.dumps({"apps": [{"tools": list(catalog.values())}]}))
    (src / "train.jsonl").write_text(
        json.dumps(record_damaged, ensure_ascii=False) + "\n"
        + json.dumps(record_fuzzy_diff, ensure_ascii=False) + "\n"
    )

    class StubASR:
        def __init__(self, *a, **kw): pass
        def transcribe(self, path):
            if path.name == "train_0.wav":
                return "Nhắn cho mình nội dung là chào buổi sáng."
            return "Nhắn cho Lan nội dung là hẹn mai gặp nha."

    class Env(FakeEnv):
        def __init__(self, *a, **kw): super().__init__()
        def validate(self, name, args): return {"ok": True}
        def get_state_hash(self): return "state"

    monkeypatch.setattr(lifemate.models, "LocalASR", StubASR)
    monkeypatch.setattr(lifemate.transcripts, "asr_identity", lambda *a, **kw: {"model": "stub"})
    monkeypatch.setitem(sys.modules, "zaic_tools", SimpleNamespace(ToolEnv=Env))

    out = tmp_path / "sft_out"
    monkeypatch.setattr(sys, "argv", [
        "prepare_sft.py", "--data-root", str(src), "--output-dir", str(out),
        "--asr-model", "stub-asr", "--filter-damaged-asr", "--no-inject-recovery",
    ])
    prepare_sft.main()
    manifest = json.loads((out / "manifest.json").read_text())
    info = manifest["partitions"]["train"]
    assert info["asr_damaged_filtered"] == 1
    assert info["transcript_source_counts"] == {"audio_text": 2, "asr": 1}


def test_export_gipformer_refuses_to_overwrite_original_dir(tmp_path):
    from scripts.asr_finetune.export_gipformer import export_finetuned_onnx

    orig = tmp_path / "models/asr"
    orig.mkdir(parents=True)
    with pytest.raises(ValueError, match="Refusing to overwrite original base ASR folder"):
        export_finetuned_onnx(best=tmp_path / "best.pt", original_dir=orig, output_dir=orig)


def test_train_gipformer_resolve_training_rows_selective_bpe_check():
    from scripts.asr_finetune.train_gipformer import resolve_training_rows

    class StubSP:
        def unk_id(self):
            return 1

        def encode(self, text, out_type=int):
            if "BAD" in text:
                return [1, 2]
            return [2, 3, 4]

    rows = [
        {"audio": "m1.wav", "text": "Bật đèn đi.", "target_text": "BẬT ĐÈN ĐI", "source": "main", "human_verified": True},
        {"audio": "h1.wav", "text": "Kênh 9", "source": "history", "human_verified": False},
        {"audio": "h2.wav", "text": "Câu hỏng", "source": "history", "human_verified": False},
    ]
    kept, skipped = resolve_training_rows(
        rows,
        StubSP(),
        vocab_size=100,
        hypotheses={"h1.wav": "KÊNH CHÍN", "h2.wav": "KHÁC HẲN HOÀN TOÀN"},
        max_history_wer=0.5,
        is_train=True,
    )
    assert skipped == 1
    assert [r["audio"] for r in kept] == ["m1.wav", "h1.wav"]
    assert kept[1]["target_text"] == "KÊNH CHÍN"


def test_train_gipformer_cli_uses_cuda_visible_devices_and_soundfile():
    import inspect
    import subprocess
    from scripts.asr_finetune import train_gipformer

    src = inspect.getsource(train_gipformer.main)
    assert "torchaudio.load" not in src
    assert "read_audio(" in src

    proc = subprocess.run(
        [sys.executable, str(APP / "scripts/asr_finetune/train_gipformer.py"), "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--cuda-visible-devices" in proc.stdout
