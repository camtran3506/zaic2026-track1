"""Unit tests for contextual BPE hotwords (1A), beam CLI options (1B), and post-ASR normalization (2A)."""
from argparse import Namespace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifemate.asr import LocalASR, asr_extra_options  # noqa: E402
from lifemate.asr_biasing import (  # noqa: E402
    collect_batch_audio_hotwords,
    ensure_bpe_vocab,
    extract_sample_hotwords,
    history_asr_for_item,
    post_normalize_asr_text,
    repair_sample_entities,
    transcribe_with_item,
)
from lifemate.transcripts import asr_identity, dataset_identity, load_transcripts, prepare_transcripts  # noqa: E402


def _make_stub_asr_dir(model_dir: Path, *, with_bpe_vocab: bool = True) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx"):
        (model_dir / name).write_bytes(b"stub")
    (model_dir / "tokens.txt").write_text(
        "<blk> 0\n▁ 1\nA 2\nB 3\nC 4\nE 5\nG 6\nH 7\nI 8\nK 9\nL 10\nM 11\nN 12\n"
        "O 13\nP 14\nR 15\nS 16\nT 17\nU 18\nV 19\nX 20\nY 21\nÀ 22\nÁ 23\nÂ 24\n"
        "Ă 25\nÊ 26\nÔ 27\nƠ 28\nƯ 29\nĐ 30\nẤ 31\nẮ 32\nỘ 33\nẢ 34\n",
        encoding="utf-8",
    )
    if with_bpe_vocab:
        (model_dir / "bpe.vocab").write_text("<unk>\t0.0\n▁TÀU\t-1.0\n", encoding="utf-8")


def test_asr_extra_options_omits_defaults_and_tracks_enabled_flags():
    default_args = Namespace(
        asr_decoding_method="modified_beam_search",
        asr_beam_size=4,
        asr_hotwords="off",
        asr_hotwords_score=1.5,
        asr_post_normalize=False,
    )
    assert asr_extra_options(default_args) == {}

    tuned_args = Namespace(
        asr_decoding_method="modified_beam_search",
        asr_beam_size=8,
        asr_hotwords="auto",
        asr_hotwords_score=1.5,
        asr_post_normalize=True,
    )
    assert asr_extra_options(tuned_args) == {
        "max_active_paths": 8,
        "hotwords_mode": "auto",
        "post_normalize": True,
    }

    custom_score = Namespace(
        asr_decoding_method="modified_beam_search",
        asr_beam_size=8,
        asr_hotwords="auto",
        asr_hotwords_score=1.8,
        asr_post_normalize=False,
    )
    assert asr_extra_options(custom_score) == {
        "max_active_paths": 8,
        "hotwords_mode": "auto",
        "hotwords_score": 1.8,
    }


def test_asr_identity_tracks_hotwords_and_post_normalize_cleanly(tmp_path):
    _make_stub_asr_dir(tmp_path)
    base = asr_identity(tmp_path, "fp32", 2)
    assert "hotwords_mode" not in base
    assert "hotwords_score" not in base
    assert "post_normalize" not in base

    tuned = asr_identity(
        tmp_path,
        "fp32",
        2,
        max_active_paths=8,
        hotwords_mode="auto",
        hotwords_score=1.5,
        post_normalize=True,
    )
    assert tuned["max_active_paths"] == 8
    assert tuned["hotwords_mode"] == "auto"
    assert tuned["hotwords_score"] == 1.5
    assert tuned["post_normalize"] is True


def test_extract_sample_hotwords_sanitizes_and_avoids_future_leakage():
    sample = {
        "context": {
            "user_profile": {
                "name": "Anh Khoa",
                "home_address": "25 Láng Hạ, Đống Đa, Hà Nội",
                "work_address": "Tòa nhà Kim Cương, Nam Từ Liêm, Hà Nội",
            },
            "installed_apps": {"public_transit": True, "train_bus_tickets": True, "gold_price": True},
        },
        "history": [
            {"role": "user", "audio": "h0.wav"},
            {
                "role": "assistant",
                "text": "Dạ có tàu NA3 lúc chín giờ, quán 'Cà phê Mộc Lam' và Tom Baker.",
                "tool_results": [{"ok": True, "operator": "Tàu SE3", "lot_name": "Bãi xe Lê Duẩn"}],
            },
            {"role": "user", "audio": "h2.wav"},
            {
                "role": "assistant",
                "text": "Bạn đặt bàn tại Skyline Rooftop Bar nhé.",
                "tool_results": [{"name": "Northwind Trading"}],
            },
        ],
    }
    vocab = set("ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ ")

    turn0_hw = extract_sample_hotwords(sample, up_to_turn=0, vocab_chars=vocab)
    assert "ANH KHOA" in turn0_hw
    assert "VÀNG SJC" not in turn0_hw
    assert "TÀU S E BA" not in turn0_hw
    assert "TOM BAKER" not in turn0_hw
    assert "SKYLINE ROOFTOP BAR" not in turn0_hw

    # Structured context collections feed hotwords generically without hardcoded app hacks
    sample_with_ctx = {
        **sample,
        "context": {
            **sample["context"],
            "collections": {"contacts.contacts": [{"name": "Phúc Long Coffee"}]},
        },
    }
    assert "PHÚC LONG COFFEE" in extract_sample_hotwords(sample_with_ctx, up_to_turn=0, vocab_chars=vocab)

    turn2_hw = extract_sample_hotwords(sample, up_to_turn=2, vocab_chars=vocab)
    assert "N A BA" in turn2_hw
    assert "CÀ PHÊ MỘC LAM" in turn2_hw
    assert "TOM BAKER" in turn2_hw
    assert "S E BA" in turn2_hw
    assert "SKYLINE ROOFTOP BAR" not in turn2_hw
    assert "NORTHWIND TRADING" not in turn2_hw

    main_hw = extract_sample_hotwords(sample, up_to_turn=None, vocab_chars=vocab)
    assert "SKYLINE ROOFTOP BAR" in main_hw
    assert "NORTHWIND TRADING" in main_hw
    assert all(not any(ch.isdigit() for ch in phrase) for phrase in main_hw.split(" / "))


def test_post_normalize_asr_text_codes_and_negative_guards():
    assert post_normalize_asr_text("Đặt vé tàu s e năm và tàu s e ba.") == "Đặt vé tàu se5 và tàu se3."
    assert post_normalize_asr_text("Lấy tàu n a ba đi sớm.") == "Lấy tàu na3 đi sớm."
    assert post_normalize_asr_text("Chuyến rv hai mốt và rv một năm bảy.") == "Chuyến rv21 và rv157."
    assert post_normalize_asr_text("Bật vtv sáu, h t v bảy, vov ba, sctv chín.") == "Bật vtv6, htv7, vov3, sctv9."
    assert post_normalize_asr_text("Mở vov một không thì đọc thư.") == "Mở vov1 không thì đọc thư."
    assert post_normalize_asr_text("Xăng ron chín mươi lăm và lít e năm.") == "Xăng ron 95 và lít e5."
    assert post_normalize_asr_text("Tuyến r mười sáu với cà phê g bảy.") == "Tuyến r16 với cà phê g7."
    assert post_normalize_asr_text("In giấy a bốn, check meo giúp anh.") == "In giấy a4, check mail giúp anh."
    assert post_normalize_asr_text("Kiểm tra giá vàng s j c hôm nay.") == "Kiểm tra giá vàng sjc hôm nay."

    # Negative / false-positive guards must remain untouched
    unchanged = [
        "Xem vtv hai tiếng rồi ngủ.",
        "Bật vtv một lúc thôi.",
        "Chị ăn trưa với julia meo xong rồi.",
        "Đi bộ năm ba bước quanh sân.",
        "Mã đặt chỗ rv hai ba hai năm ba của tôi.",
        "Kho hàng mới ra lô hàng với da lô này.",
    ]
    for sentence in unchanged:
        assert post_normalize_asr_text(sentence) == sentence

    # Idempotence
    once = post_normalize_asr_text("Đặt tàu s e năm mở vov một không thì coi vtv sáu.")
    assert post_normalize_asr_text(once) == once


def test_ensure_bpe_vocab_behaviour(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ensure_bpe_vocab(empty, tmp_path / "out") is None

    direct = tmp_path / "direct"
    _make_stub_asr_dir(direct, with_bpe_vocab=True)
    assert ensure_bpe_vocab(direct, tmp_path / "out") == direct / "bpe.vocab"

    real_bpe = ROOT / "models" / "asr_ft" / "bpe.model"
    if real_bpe.is_file():
        real_copy_dir = tmp_path / "from_model"
        real_copy_dir.mkdir()
        (real_copy_dir / "bpe.model").write_bytes(real_bpe.read_bytes())
        gen = ensure_bpe_vocab(real_copy_dir, tmp_path / "gen_vocab")
        assert gen is not None and gen.is_file()
        first_line = gen.read_text(encoding="utf-8").splitlines()[0]
        assert "\t" in first_line


def test_local_asr_hotwords_cache_and_prepare_transcripts_end_to_end(tmp_path, monkeypatch):
    model_dir = tmp_path / "asr"
    _make_stub_asr_dir(model_dir, with_bpe_vocab=True)
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    for name in ("h0.wav", "main.wav"):
        sf.write(split_dir / name, np.zeros(800, dtype=np.float32), 16000)

    recorded_init = {}
    recorded_streams = []

    class FakeStream:
        def __init__(self, hotwords=""):
            self.hotwords = hotwords
            self.result = SimpleNamespace(
                text="MỞ VOV MỘT KHÔNG THÌ ĐẶT TÀU S E NĂM" if hotwords else "XĂNG RON CHÍN MƯƠI LĂM"
            )

        def accept_waveform(self, sr, samples):
            pass

    class FakeRecognizer:
        @classmethod
        def from_transducer(cls, **kwargs):
            recorded_init.update(kwargs)
            return cls()

        def create_stream(self, hotwords=""):
            recorded_streams.append(hotwords)
            return FakeStream(hotwords)

        def decode_streams(self, streams):
            pass

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(OfflineRecognizer=FakeRecognizer))

    asr = LocalASR(
        model_dir,
        precision="fp32",
        num_threads=2,
        max_active_paths=8,
        hotwords_mode="auto",
        hotwords_score=1.5,
        post_normalize=True,
    )
    assert recorded_init["modeling_unit"] == "bpe"
    assert recorded_init["max_active_paths"] == 8
    assert recorded_init["hotwords_score"] == 1.5
    assert asr.supports_hotwords is True

    # Same wav path with vs without hotwords must not collide in cache
    t_plain = asr.transcribe(split_dir / "main.wav", hotwords="")
    t_bias = asr.transcribe(split_dir / "main.wav", hotwords="S E NĂM")
    assert t_plain == "Xăng ron 95."
    assert t_bias == "Mở vov1 không thì đặt tàu se5."
    asr.close()

    recorded_streams.clear()
    items = [{
        "id": "s1",
        "audio": "main.wav",
        "context": {"user_profile": {"name": "Anh Nam"}},
        "history": [
            {"role": "user", "audio": "h0.wav"},
            {"role": "assistant", "text": "Mời anh xem tàu SE5."},
        ],
    }]
    cache = tmp_path / "transcripts.jsonl"
    identity = dataset_identity(items, split_dir, "public_test", False)
    settings = asr_identity(
        model_dir, "fp32", 2, max_active_paths=8, hotwords_mode="auto", hotwords_score=1.5, post_normalize=True
    )
    prepare_transcripts(
        items,
        split_dir,
        cache,
        identity,
        settings,
        lambda: LocalASR(
            model_dir,
            precision="fp32",
            num_threads=2,
            max_active_paths=8,
            hotwords_mode="auto",
            hotwords_score=1.5,
            post_normalize=True,
        ),
        batch_size=8,
    )
    rows, manifest = load_transcripts(cache, identity)
    assert manifest["asr"]["hotwords_mode"] == "auto"
    assert manifest["asr"]["post_normalize"] is True
    assert manifest["asr"]["max_active_paths"] == 8
    assert rows[0]["transcript"] == "Mở vov1 không thì đặt tàu se5."
    assert "ANH NAM" in recorded_streams[0]
    assert "S E NĂM" in recorded_streams[0]
    assert "S E NĂM" not in recorded_streams[1]


def test_extract_sample_hotwords_branch_b1_channels_hyphen_camel_and_no_subphrase_dups():
    sample = {
        "context": {"user_profile": {"name": "Chị Mai"}},
        "history": [
            {"role": "user", "audio": "h0.wav"},
            {
                "role": "assistant",
                "text": (
                    "Đặt tàu S-18 trên RemitStore hoặc ghé ViTech và Skyline Rooftop Bar. "
                    "Mùa này đi Đà Lạt rất mát."
                ),
                "tool_results": [{"ok": True, "channels": ["VTV6", "HTV7"]}],
            },
            {"role": "user", "audio": "h2.wav"},
            {
                "role": "assistant",
                "text": "Tương lai sẽ xem thêm ở ZenCafe.",
                "tool_results": [{"brand": "BrightPath Holdings"}],
            },
        ],
    }
    vocab = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ ")
    hw2 = extract_sample_hotwords(sample, up_to_turn=2, vocab_chars=vocab)
    phrases = hw2.split(" / ")
    assert "S MỘT TÁM" in phrases or any("S" in p and "TÁM" in p for p in phrases)
    assert "S-18" not in hw2
    assert "V T V SÁU" in phrases or "VTVSÁU" in hw2.replace(" ", "")
    assert "REMITSTORE" in phrases
    assert "VITECH" in phrases
    assert "SKYLINE ROOFTOP BAR" in phrases
    # Single sub-words contained in multi-word hotword must not be duplicated into secondary slots
    assert "SKYLINE" not in phrases
    assert "ZENCAFE" not in hw2


def test_post_normalize_asr_text_branch_a1_rules_and_guards():
    assert (
        post_normalize_asr_text("Kiểm tra chuyến rv bảy hai không đi Đà Nẵng giúp anh.")
        == "Kiểm tra chuyến rv720 đi Đà Nẵng giúp anh."
    )
    assert post_normalize_asr_text("Có chuyến rv bảy hai không?") == "Có chuyến rv72 không?"
    assert post_normalize_asr_text("Đặt vé tàu s e năm và tàu n a bảy nhé.") == "Đặt vé tàu se5 và tàu na7 nhé."
    assert post_normalize_asr_text("Bắt tuyến r một với xe buýt r mười sáu rồi sang kênh h t v mười bảy.") == "Bắt tuyến r1 với xe buýt r16 rồi sang kênh htv17."
    assert post_normalize_asr_text("Mua năm ký gạo st hai lăm nha.") == "Mua năm ký gạo st25 nha."
    assert post_normalize_asr_text("Giá xăng e năm hôm nay bao nhiêu.") == "Giá xăng e5 hôm nay bao nhiêu."
    assert post_normalize_asr_text("Lấy ba mươi mốt thì kênh vtv ba mươi đi.") == "Lấy ba mươi mốt thì kênh vtv30 đi."

    # Sample-overfit ASR slips and ambiguous homophones must NOT be rewritten
    for untouched in (
        "Nói tiếng r rà trong lớp.",
        "Xe e năm chỗ ngồi.",
        "Khóa e năm học sinh mới.",
        "Đổi ra được xe máy mới luôn.",
        "Mở kênh h mười bảy xem thời sự.",
        "Mua hai con gà st hai lăm.",
        "Gửi cho phải nan a còng hoặc legel a còng công ty.",
    ):
        assert post_normalize_asr_text(untouched) == untouched


def test_repair_sample_entities_a2_and_transcribe_integration(tmp_path):
    item = {
        "context": {"user_profile": {"name": "Daniel Lee"}},
        "history": [
            {"role": "user", "audio": "h0.wav"},
            {
                "role": "assistant",
                "text": "Hôm nay có lịch tại Pinnacle Design và Tokyo Garden, cùng 'Webinar Marketing' lúc chín giờ.",
                "tool_results": [{"company": "CTCP BrightPath Holdings"}],
            },
            {"role": "user", "audio": "h2.wav"},
            {
                "role": "assistant",
                "text": "Anh cũng có thể thử món trong danh sách, hoặc họp với Hugo Lem về SaoTel.",
                "tool_results": [{"name": "Bánh mì sandwich"}],
            },
        ],
    }
    assert (
        repair_sample_entities("Đặt cảnh báo brigbard holding và briboard holdings.", item, up_to_turn=None)
        == "Đặt cảnh báo BrightPath Holdings và BrightPath Holdings."
    )
    assert (
        repair_sample_entities("Họp ở pencel design với tokyo gaden xong học webina.", item, up_to_turn=None)
        == "Họp ở Pinnacle Design với Tokyo Garden xong học Webinar."
    )
    assert (
        repair_sample_entities("Danel lee mua bánh mì sandwi cùng Hubo lem ở sao tel anh Thành.", item, up_to_turn=None)
        == "Daniel Lee mua bánh mì sandwich cùng Hugo Lem ở SaoTel anh Thành."
    )

    # Open-vocabulary English words must NEVER be falsely corrupted into context words
    guard_item = {
        "context": {},
        "history": [
            {
                "role": "assistant",
                "text": "Đặt lịch với Alex Tanner ở Grill House và Daniel Lee.",
            }
        ],
    }
    for clean_utterance in (
        "Đặt món grilled snakehead tại nhà hàng.",
        "Gọi cho Alex Turner và David Lee.",
        "Nhắn cho Mr Lee giúp anh.",
    ):
        assert repair_sample_entities(clean_utterance, guard_item, up_to_turn=None) == clean_utterance

    # Turn scoping prevents future assistant leakage
    assert (
        repair_sample_entities("Đặt cảnh báo brigbard holding.", item, up_to_turn=0)
        == "Đặt cảnh báo brigbard holding."
    )
    # Ambiguous distractor must not rewrite
    ambiguous_item = {
        "context": {},
        "history": [
            {"role": "assistant", "text": "Liên hệ Ryan Ford hoặc Ryan Fordham trước nhé."},
        ],
    }
    assert repair_sample_entities("Gọi cho Ryan Ford đi.", ambiguous_item) == "Gọi cho Ryan Ford đi."

    # Integration through transcribe_with_item & history_asr_for_item (turn-scoped A2)
    for name in ("h0.wav", "h2.wav", "u.wav"):
        (tmp_path / name).write_bytes(b"")
    stub_asr = SimpleNamespace(
        supports_hotwords=False,
        hotwords_mode="off",
        post_normalize=True,
        vocab_chars=None,
        transcribe=lambda path, **kw: post_normalize_asr_text(
            "Đặt vé tàu s e năm tới tokyo gaden gửi mail."
        ),
    )
    out = transcribe_with_item(stub_asr, tmp_path / "u.wav", item)
    assert out == "Đặt vé tàu se5 tới Tokyo Garden gửi mail."
    hist_asr = history_asr_for_item(stub_asr, item, tmp_path)
    # h0 precedes assistant turn 1 ("Tokyo Garden"), so A2 must not leak turn 1 entities into h0;
    # h2 follows turn 1, so A2 repairs "tokyo gaden" -> "Tokyo Garden".
    assert hist_asr.transcribe(tmp_path / "h0.wav") == "Đặt vé tàu se5 tới tokyo gaden gửi mail."
    assert hist_asr.transcribe(tmp_path / "h2.wav") == out


def test_collect_batch_audio_hotwords_and_prepare_sft_with_latest_asr_and_cache(tmp_path, monkeypatch):
    import finetune
    import lifemate.models
    import prepare_sft
    from test_baseline import FakeEnv

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    for name in ("h0.wav", "h_skip.wav", "h_empty.wav", "main.wav"):
        sf.write(split_dir / name, np.zeros(800, dtype=np.float32), 16000)

    items = [{
        "id": "train_0",
        "audio": "main.wav",
        "audio_text": "Đặt tàu SE5 tại Tokyo Garden ngày mai.",
        "context": {
            "current_time": "2026-10-18T09:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "installed_apps": {"trains": {"tools": ["trains.search"]}},
            "user_profile": {"name": "Anh Nam"},
        },
        "history": [
            {"role": "user", "audio": "h0.wav"},
            {"role": "user", "audio": "h_skip.wav", "audio_text": "Đã có transcript sạch."},
            {"role": "user", "audio": "h_empty.wav", "text": ""},
            {"role": "assistant", "text": "Gợi ý Tokyo Garden hoặc tàu SE5."},
        ],
        "label": {"tool_calls": [{"name": "trains.search", "arguments": {"query": "SE5"}}]},
    }]
    paths, hws = collect_batch_audio_hotwords(items, split_dir, vocab_chars=None)
    assert [p.name for p in paths] == ["main.wav", "h0.wav", "h_empty.wav"]
    assert "SE5" in hws[0] or "S E NĂM" in hws[0]
    assert "TOKYO GARDEN" not in hws[1]

    asr_dir = tmp_path / "asr"
    _make_stub_asr_dir(asr_dir, with_bpe_vocab=True)
    catalog = {
        "trains.search": {
            "name": "trains.search",
            "kind": "read",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        }
    }
    (split_dir / "train_tools_catalog.json").write_text(json.dumps({"apps": [{"tools": list(catalog.values())}]}))
    (split_dir / "train.jsonl").write_text(json.dumps(items[0], ensure_ascii=False) + "\n", encoding="utf-8")

    class FakeStream:
        def __init__(self, hotwords=""):
            self.result = SimpleNamespace(
                text="ĐẶT TÀU S E NĂM TẠI TOKYO GADEN NGÀY MAI" if hotwords else "KHÔNG CÓ HOTWORDS"
            )

        def accept_waveform(self, sr, samples):
            pass

    class FakeRecognizer:
        @classmethod
        def from_transducer(cls, **kwargs):
            return cls()

        def create_stream(self, hotwords=""):
            return FakeStream(hotwords)

        def decode_streams(self, streams):
            pass

    class Env(FakeEnv):
        def __init__(self, *a, **kw):
            super().__init__()

        def validate(self, name, args):
            return {"ok": True}

        def get_state_hash(self):
            return "state"

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(OfflineRecognizer=FakeRecognizer))
    monkeypatch.setitem(sys.modules, "zaic_tools", SimpleNamespace(ToolEnv=Env))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    # 1) Direct live SFT preparation with latest ASR flags (B1 + A1 + A2, beam 8, batch 4)
    out_live = tmp_path / "sft_live"
    monkeypatch.setattr(sys, "argv", [
        "prepare_sft.py", "--data-root", str(split_dir), "--output-dir", str(out_live),
        "--asr-model", str(asr_dir), "--asr-beam-size", "8",
        "--asr-hotwords", "auto", "--asr-hotwords-score", "1.5", "--asr-post-normalize",
        "--asr-batch-size", "4", "--no-inject-recovery",
    ])
    prepare_sft.main()
    live_manifest = json.loads((out_live / "manifest.json").read_text(encoding="utf-8"))
    assert live_manifest["asr"]["hotwords_mode"] == "auto"
    assert live_manifest["asr"]["max_active_paths"] == 8
    assert live_manifest["asr"]["post_normalize"] is True
    assert live_manifest["sft_format"]["include_date_helpers"] is False
    live_rows = [json.loads(line) for line in (out_live / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    asr_rows = [r for r in live_rows if r["transcript_source"] == "asr"]
    assert asr_rows
    prompt_obj = json.loads(asr_rows[0]["messages"][1]["content"])
    assert prompt_obj["request_transcript"] == "Đặt tàu se5 tại tokyo garden ngày mai."
    days = prompt_obj["user_context"]["date_reference"]["days"]
    assert "tuần này" in days[1] and "hôm nay" in days[1]
    assert "tuần sau" in days[2] and "mai" in days[2]

    # 2) Precomputed transcript cache consumption in prepare_sft.py skips LocalASR initialization
    cache_path = tmp_path / "train_transcripts.jsonl"
    identity = dataset_identity(items, split_dir, "train", False)
    settings = asr_identity(asr_dir, "fp32", 2, max_active_paths=8, hotwords_mode="auto", hotwords_score=1.5, post_normalize=True)
    prepare_transcripts(
        items, split_dir, cache_path, identity, settings,
        lambda: LocalASR(asr_dir, precision="fp32", num_threads=2, max_active_paths=8, hotwords_mode="auto", hotwords_score=1.5, post_normalize=True),
        batch_size=4,
    )
    monkeypatch.setattr(lifemate.models, "LocalASR", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("LocalASR must not load when --transcript-cache is supplied")))
    out_cached = tmp_path / "sft_cached"
    monkeypatch.setattr(sys, "argv", [
        "prepare_sft.py", "--data-root", str(split_dir), "--output-dir", str(out_cached),
        "--transcript-cache", str(cache_path), "--no-inject-recovery",
    ])
    prepare_sft.main()
    cached_manifest = json.loads((out_cached / "manifest.json").read_text(encoding="utf-8"))
    assert cached_manifest["asr"]["hotwords_mode"] == "auto"
    assert cached_manifest["asr"]["post_normalize"] is True
    assert cached_manifest["sft_format"]["transcript_cache"] == str(cache_path.resolve())
    cached_rows = [json.loads(line) for line in (out_cached / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cached_rows) == len(live_rows)

    # 3) finetune.py defaults to latest ASR configuration and forwards --transcript-cache / --asr-batch-size
    forwarded = []

    class Child:
        stdout = ()
        def __init__(self, cmd, **kw): forwarded.append(cmd)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def wait(self): return 0

    base_llm = tmp_path / "base_llm"
    base_llm.mkdir()
    monkeypatch.setattr(finetune.subprocess, "Popen", Child)
    monkeypatch.setattr(sys, "argv", [
        "finetune.py", "--data-root", str(split_dir), "--base-model", str(base_llm),
        "--output-dir", str(tmp_path / "ft_run"), "--transcript-cache", str(cache_path), "--asr-batch-size", "8",
    ])
    finetune.main()
    prep_cmd = forwarded[0]
    assert prep_cmd[prep_cmd.index("--asr-beam-size") + 1] == "8"
    assert prep_cmd[prep_cmd.index("--asr-hotwords") + 1] == "auto"
    assert "--asr-post-normalize" in prep_cmd
    assert prep_cmd[prep_cmd.index("--asr-batch-size") + 1] == "8"
    assert prep_cmd[prep_cmd.index("--transcript-cache") + 1] == str(cache_path)
    assert "--include-date-helpers" not in prep_cmd
