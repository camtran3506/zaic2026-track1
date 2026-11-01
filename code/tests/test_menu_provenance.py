"""get_menu returns menu[], while place_order refers to item_id (train_0335)."""
from copy import deepcopy

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.sft import replay_sample
from test_baseline import CATALOG as BASE_CATALOG


CATALOG = {
    'sys.ask_missing_info': BASE_CATALOG['sys.ask_missing_info'],
    'food_delivery.list_restaurants': {'name': 'food_delivery.list_restaurants',
        'kind': 'read', 'parameters': {'type': 'object'}, 'returns': 'restaurants[]'},
    'food_delivery.get_menu': {'name': 'food_delivery.get_menu', 'kind': 'read',
        'parameters': {'type': 'object', 'properties': {
            'restaurant_id': {'type': 'string', 'x-ref': 'food_delivery.restaurants'}}},
        'returns': 'menu[]'},
    'food_delivery.place_order': {'name': 'food_delivery.place_order', 'kind': 'write',
        'parameters': {'type': 'object', 'properties': {
            'restaurant_id': {'type': 'string', 'x-ref': 'food_delivery.restaurants'},
            'items': {'type': 'array', 'items': {'type': 'object', 'properties': {
                'item_id': {'type': 'string'}, 'quantity': {'type': 'integer'}}}}}}},
    'chat_ott.list_chats': {'name': 'chat_ott.list_chats', 'kind': 'read',
                            'parameters': {'type': 'object'}},
}
MENU = {'ok': True, 'menu': [{'id': 'rest_01_m2', 'name': 'Bún bò giò', 'price': 45000}]}
CALLS = [
    {'name': 'chat_ott.list_chats', 'arguments': {}},
    {'name': 'food_delivery.list_restaurants', 'arguments': {}},
    {'name': 'food_delivery.get_menu', 'arguments': {'restaurant_id': 'rest_01'}},
    {'name': 'food_delivery.place_order', 'arguments': {'restaurant_id': 'rest_01',
        'items': [{'item_id': 'rest_01_m2', 'quantity': 3}]}},
]


class Env:
    def __init__(self): self.calls = []
    def validate(self, name, args): return {'ok': True}
    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': deepcopy(args)})
        return deepcopy({
            'chat_ott.list_chats': {'ok': True, 'chats': [{'id': 'chat_05', 'members': ['Mẹ', 'Bố', 'Lan']}]},
            'food_delivery.list_restaurants': {'ok': True, 'restaurants': [{'id': 'rest_01'}]},
            'food_delivery.get_menu': MENU,
        }.get(name, {'ok': True}))


def test_menu_gold_replay_and_inference_accept_read_item_without_future_label_leak(tmp_path):
    entry = {'id': 'train_0335', 'audio_text': 'Đặt Bún bò giò cho ba người.',
             'context': {'installed_apps': {app: {'tools': [n for n in CATALOG if n.startswith(app + '.')]}
                                          for app in ('food_delivery', 'chat_ott')}},
             'history': [], 'label': {'tool_calls': deepcopy(CALLS)}}
    env = Env()
    examples = replay_sample(entry, CATALOG, env, tmp_path)
    assert env.calls == CALLS
    assert 'rest_01_m2' not in examples[0]['messages'][1]['content']
    assert any('rest_01_m2' in e['messages'][1]['content'] for e in examples[3:])

    class LLM:
        def __init__(self): self.steps = iter(examples)
        def generate(self, system, user): return next(self.steps)['messages'][-1]['content']
    env = Env()
    stats = LifeMateAgent(None, LLM(), CATALOG, AgentConfig(use_train_transcript=True)).run(
        entry, env, tmp_path, 'train')
    assert env.calls == CALLS
    assert stats['invalid_proposals'] == 0
