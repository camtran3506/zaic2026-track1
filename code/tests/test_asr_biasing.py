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
    ensure_bpe_vocab,
    extract_sample_hotwords,
    post_normalize_asr_text,
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
    assert "VÀNG SJC" in turn0_hw
    assert "TOM BAKER" not in turn0_hw
    assert "SKYLINE ROOFTOP BAR" not in turn0_hw

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
    assert post_normalize_asr_text("Bật vtv sáu, htv bảy, vov ba, sctv chín.") == "Bật vtv6, htv7, vov3, sctv9."
    assert post_normalize_asr_text("Mở vov một không thì đọc thư.") == "Mở vov1 không thì đọc thư."
    assert post_normalize_asr_text("Xăng ron chín mươi lăm và lít e năm.") == "Xăng ron 95 và lít e5."
    assert post_normalize_asr_text("Tuyến r mười sáu với cà phê g bảy.") == "Tuyến r16 với cà phê g7."
    assert post_normalize_asr_text("In giấy a bốn, chuyển s cộng sport.") == "In giấy a4, chuyển s+ sport."
    assert post_normalize_asr_text("Gửi meo qua gia lô, giá vàng ét di xi.") == "Gửi mail qua zalo, giá vàng sjc."

    # Negative / false-positive guards must remain untouched
    unchanged = [
        "Xem vtv hai tiếng rồi ngủ.",
        "Bật vtv một lúc thôi.",
        "Chị ăn trưa với julia meo xong rồi.",
        "Đi bộ năm ba bước quanh sân.",
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
