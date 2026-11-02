"""Literal normalization policy, idempotence and content preservation."""
import unicodedata

import pytest

from lifemate.text import normalize_transcript


@pytest.mark.parametrize('source,expected', [
    ('  BẬT\tĐÈN\nPHÒNG KHÁCH  ', 'Bật đèn phòng khách.'),
    ('  BA\u0323\u0302T ĐÈN  ', 'Bật đèn.'),
    ('TẮT ĐÈN. ĐẶT BÁO THỨC.', 'Tắt đèn. đặt báo thức.'),
    ('BẬT ĐÈN!', 'Bật đèn!'), ('MẤY GIỜ?', 'Mấy giờ?'),
    ('À THÔI...', 'À thôi...'), ('KHOAN…', 'Khoan…'),
    ('“BẬT ĐÈN!”', '“Bật đèn!”'), ('"BẬT ĐÈN"', '"Bật đèn".'),
    ('123 BẬT ĐÈN', '123 Bật đèn.'),
    (' 08:30 NGÀY 29/10/2026, -5%, +2.5 VÀ 1.234,56 ',
     '08:30 Ngày 29/10/2026, -5%, +2.5 và 1.234,56.'),
    ('GỬI A_B@Example.COM', 'Gửi a_b@example.com.'),
    ('LUCASBENETT', 'Lucasbenett.'),  # No word splitting or spelling correction.
    ('ßETA', 'Sseta.'), ('ﬁLE', 'File.'),
    ('TẮT\u00a0\u2003ĐÈN', 'Tắt đèn.'),
    ('', ''), (' \t\n ', ''), ('?!', '?!'), ('...', '...'), ('123', '123.'),
])
def test_utterance_policy(source, expected):
    actual = normalize_transcript(source)
    assert actual == expected
    assert unicodedata.is_normalized('NFC', actual)
    assert normalize_transcript(actual) == actual


def test_no_length_limit_or_sentence_reconstruction():
    source = 'A! ' * 5000 + 'B'
    result = normalize_transcript(source)
    assert result == 'A! ' + 'a! ' * 4999 + 'b.'


def test_nontext_is_rejected():
    with pytest.raises(TypeError, match='string'):
        normalize_transcript(None)
