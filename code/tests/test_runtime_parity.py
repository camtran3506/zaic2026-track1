"""Regressions for identical history and token prefixes in SFT and inference."""
import json
from types import SimpleNamespace

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.sft import encode_example, prepare_train_transcript, replay_sample
from lifemate.transcripts import dataset_identity, load_transcripts, prepare_transcripts
from test_baseline import CATALOG, FakeEnv, item


def assert_history_parity(tmp_path, history, expected):
    class ASR:
        def transcribe(self, path):
            return '  GIỮ chữ HOA, dấu câu! ' + path.name

    entry = dict(item(), audio='current.wav', audio_text=' BẬT đèn! ',
                 history=history, label={'tool_calls': []})
    for name in ('current.wav', 'assistant.wav', 'system.wav'):
        (tmp_path / name).write_bytes(b'fixture audio; ASR stubbed')
    identity = dataset_identity([entry], tmp_path, 'train', use_train_transcript=True)
    cache = tmp_path / 'transcripts.jsonl'
    prepare_transcripts([entry], tmp_path, cache, identity, {}, ASR)
    cached = load_transcripts(cache, identity)[0][0]
    prepared = prepare_train_transcript(entry, tmp_path, ASR())
    sft_prompt = replay_sample(entry, CATALOG, FakeEnv(), tmp_path,
                               prepared_transcript=prepared)[0]['messages'][1]['content']
    assert json.loads(sft_prompt)['conversation_history'] == expected

    class LLM:
        def generate(self, system, user):
            # Compare the complete prompt, including history-based schema selection.
            assert user == sft_prompt
            return '{"action":"done"}'

    agent = LifeMateAgent(ASR(), LLM(), CATALOG, AgentConfig(use_train_transcript=True))
    agent.run(entry, FakeEnv(), tmp_path, 'train')
    agent.run(entry, FakeEnv(), tmp_path, 'train', prepared_transcript=cached)
    return cached, prepared


def test_audio_only_history_for_all_roles_matches_sft_and_inference(tmp_path):
    history = [{'role': 'assistant', 'audio': 'assistant.wav'},
               {'speaker': 'system', 'audio_path': 'system.wav'},
               {'role': 'user', 'text': ' ĐỪNG sửa, 08:30! '}]
    expected = [{'role': 'assistant', 'text': '  GIỮ chữ HOA, dấu câu! assistant.wav'},
                {'role': 'system', 'text': '  GIỮ chữ HOA, dấu câu! system.wav'},
                {'role': 'user', 'text': ' ĐỪNG sửa, 08:30! '}]
    assert_history_parity(tmp_path, history, expected)


def test_sft_requires_asr_for_audio_only_assistant_history(tmp_path):
    entry = dict(item(), audio_text='Provided train transcript',
                 history=[{'role': 'assistant', 'audio': 'assistant.wav'}])
    with pytest.raises(ValueError, match='History audio needs --asr-model'):
        prepare_train_transcript(entry, tmp_path)


def test_history_is_filtered_before_taking_last_twelve_turns(tmp_path):
    valid = [{'role': 'user', 'text': f' TURN-{i}! '} for i in range(15)]
    valid[-1]['text'] = ' A! ' * 500
    history = valid + [{'role': 'user'}, None, {'role': 'assistant', 'text': ''}]
    expected = [{'role': 'user', 'text': f' TURN-{i}! '} for i in range(3, 14)]
    expected.append({'role': 'user', 'text': valid[-1]['text'][:1700]})
    cached, prepared = assert_history_parity(tmp_path, history, expected)
    assert len(cached['history']) == 15  # Full history remains available on disk.
    assert cached['history'][-1]['text'] == valid[-1]['text']
    assert prepared['history'] == expected


@pytest.mark.parametrize('reload_tokenizer', [False, True])
def test_inference_token_ids_match_sft_prefix_with_bos_postprocessor(tmp_path, reload_tokenizer):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from lifemate.llm import configure_tokenizer
    from lifemate.models import LocalLLM

    backend = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2,
        '<bos>': 3, '<turn>': 4, '</turn>': 5, '<think>': 6, '</think>': 7}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single='<bos> $A', special_tokens=[('<bos>', 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
        pad_token='[PAD]', eos_token='[EOS]', bos_token='<bos>',
        additional_special_tokens=['<turn>', '</turn>', '<think>', '</think>'])
    tok.chat_template = """{{ bos_token }}{% for m in messages %}{{ '<turn>' + m['role'] + '\n' }}{% if m['role'] == 'assistant' and not enable_thinking %}{{ '<think></think>\n' }}{% endif %}{{ m['content'] + '</turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<turn>assistant\n' }}{% if not enable_thinking %}{{ '<think></think>\n' }}{% endif %}{% endif %}"""
    (tmp_path / 'config.json').write_text('{"model_type":"qwen3_5"}')
    configure_tokenizer(tok, tmp_path)
    messages = [{'role': 'system', 'content': 'Chỉ xuất JSON.'},
                {'role': 'user', 'content': ' BẬT đèn, lúc 08:30! '},
                {'role': 'assistant', 'content': '{"action":"done"}'}]
    row = encode_example({'sample_id': 'train_1', 'step': 0, 'messages': messages}, tok, 512)
    prefix_length = row['labels'].count(-100)
    expected = row['input_ids'][:prefix_length]
    assert expected.count(tok.bos_token_id) == 1
    train_prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                            **tok.lifemate_template_kwargs)
    if reload_tokenizer:
        tok.save_pretrained(tmp_path / 'adapter')
        tok = configure_tokenizer(AutoTokenizer.from_pretrained(tmp_path / 'adapter',
                                                               local_files_only=True), tmp_path)
    inference_prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                                **tok.lifemate_template_kwargs)
    assert train_prompt == inference_prompt

    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def generate(self, input_ids, **kwargs):
            assert input_ids[0].tolist() == expected
            return torch.cat([input_ids, torch.tensor([[tok.eos_token_id]])], dim=1)

    llm = LocalLLM.from_loaded_model(Model(), tok, max_input_tokens=512, max_new_tokens=2)
    llm.generate(messages[0]['content'], messages[1]['content'])
