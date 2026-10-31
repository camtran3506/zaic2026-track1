"""Unit tests for Vietnamese text normalization in LLM inference and data prep."""
import unittest
import unicodedata

from lifemate.text import normalize_text
from lifemate.agent import AgentConfig, LifeMateAgent


class TestNormalization(unittest.TestCase):
    def test_unicode_nfc_composition(self):
        # Decomposed NFD Vietnamese text: 'h', 'o', '\u0301', 'a' (4 code points)
        nfd_text = "ho" + "\u0301" + "a"
        self.assertEqual(len(nfd_text), 4)
        normalized = normalize_text(nfd_text)
        # Should be composed NFC: 'h', 'ó', 'a' (3 code points)
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized, "hóa")
        self.assertEqual(unicodedata.is_normalized("NFC", normalized), True)

    def test_lowercasing(self):
        raw = "Nếu Mai Mưa Thì Nhắn Sếp Dũng Giờ Mới"
        expected = "nếu mai mưa thì nhắn sếp dũng giờ mới"
        self.assertEqual(normalize_text(raw), expected)

    def test_punctuation_stripping(self):
        raw = "Nếu mai mưa, thì hủy lịch đá bóng... À mà thôi; cứ để lịch đó!"
        expected = "nếu mai mưa thì hủy lịch đá bóng à mà thôi cứ để lịch đó"
        self.assertEqual(normalize_text(raw), expected)

    def test_whitespace_collapsing(self):
        raw = "   nếu   mai \t\n  mưa    thì thôi   "
        expected = "nếu mai mưa thì thôi"
        self.assertEqual(normalize_text(raw), expected)

    def test_edge_cases(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("   "), "")
        self.assertEqual(normalize_text("123, 456.789!"), "123 456 789")

    def test_agent_normalization_enabled(self):
        class MockLLM:
            def __init__(self):
                self.last_user_prompt = None

            def generate(self, system_prompt, user_prompt):
                self.last_user_prompt = user_prompt
                return '{"action": "done"}'

        mock_llm = MockLLM()
        catalog = {
            "sys.ask_missing_info": {"name": "sys.ask_missing_info", "kind": "meta"},
            "app.dummy": {"name": "app.dummy", "kind": "read", "parameters": {}},
        }
        item = {
            "id": "sample_01",
            "context": {"installed_apps": {"app": {"tools": ["app.dummy"]}}},
        }
        prepared_transcript = {
            "id": "sample_01",
            "transcript": "Nếu mai mưa, nhắn Sếp Dũng!",
            "history": [{"role": "user", "text": "Hôm qua trời mưa không?"}],
        }

        # Case 1: normalize_input=True (default)
        agent_norm = LifeMateAgent(asr=None, llm=mock_llm, catalog=catalog, cfg=AgentConfig(normalize_input=True))
        result_norm = agent_norm.run(item, None, None, "test", prepared_transcript=prepared_transcript)
        self.assertEqual(result_norm["transcript"], "nếu mai mưa nhắn sếp dũng")
        self.assertIn('"request_transcript":"nếu mai mưa nhắn sếp dũng"', mock_llm.last_user_prompt)
        self.assertIn('"text":"hôm qua trời mưa không"', mock_llm.last_user_prompt)

        # Case 2: normalize_input=False
        agent_raw = LifeMateAgent(asr=None, llm=mock_llm, catalog=catalog, cfg=AgentConfig(normalize_input=False))
        result_raw = agent_raw.run(item, None, None, "test", prepared_transcript=prepared_transcript)
        self.assertEqual(result_raw["transcript"], "Nếu mai mưa, nhắn Sếp Dũng!")
        self.assertIn('"request_transcript":"Nếu mai mưa, nhắn Sếp Dũng!"', mock_llm.last_user_prompt)


if __name__ == "__main__":
    unittest.main()
