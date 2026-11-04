"""Automatic guards use one ToolEnv and share all execution/feedback with SFT."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.execution_policy import EXECUTION_POLICY
from lifemate.function_calling import native_chat
from lifemate.local_helpers import helper_tool_schemas
from lifemate.runtime import RUNTIME_MANIFEST, load_runtime_manifest
from lifemate.sft import replay_sample, encode_example
from test_local_time_runtime import temporal_artifact
from test_native_function_calling import native_tokenizer
from test_python_helpers_integration import CATALOG, Env, Planner, DONE, call, sample, prepared


def run(outputs, env=None, catalog=None, entry=None, prepared_row=None, cfg=None):
    env = env or Env()
    planner, events = Planner(outputs), []
    report = LifeMateAgent(None, planner, catalog or CATALOG, cfg or AgentConfig()).run(
        entry or sample(), env, Path('/unused'), 'public_test', prepared_row or prepared(), trace=events.append)
    return env, report, events, planner


@pytest.mark.parametrize('with_data', [False, True])
def test_duplicate_read_reuses_the_original_result_without_an_env_call_or_fabricated_tool_response(with_data):
    env=Env()
    if with_data: env.notes.append({'text': 'known'})
    read = call('notes.search', query='known')
    env, stats, events, planner = run([read, read, DONE], env)
    assert stats['executed'] == 1 and stats['local_executed'] == 0
    assert len(env.validations) == len(env.get_tool_calls()) == 1
    feedback = next(e for e in events if e['event']=='decision_rejected')['feedback']
    assert feedback['cached_result'] == {'ok': True, 'notes': [{'text': 'known'}] if with_data else []}
    assert feedback['source_call_index'] == 0 and feedback['state_epoch'] == 0
    system, user = planner.requests[-1]
    chat, tools = native_chat([{'role':'system','content':system},{'role':'user','content':user}])
    assert len([turn for turn in chat if turn['role']=='tool']) == 1
    assert len([turn for turn in chat if turn.get('tool_calls')]) == 1
    assert json.loads(chat[-1]['content'])['agent_feedback']['feedback']['cached_result'] == feedback['cached_result']
    assert all('check_tool_access' not in t['function']['name'] for t in tools)
    assert not any(name.startswith('automatic') for name in helper_tool_schemas())


@pytest.mark.parametrize('name,status', [('notes.serch','unknown_tool'),('banking.transfer','tool_not_allowed')])
def test_tool_access_is_checked_by_python_before_env_validation(name,status):
    catalog = {**CATALOG, 'banking.transfer': {'name':'banking.transfer','kind':'write','parameters':{'type':'object'}}}
    env, stats, events, planner = run([call(name), DONE], catalog=catalog)
    assert not env.validations and not env.get_tool_calls()
    feedback = next(e for e in events if e['event']=='decision_rejected')
    assert feedback['stage']=='precheck' and feedback['access']['status']==status
    assert 'banking.transfer' not in feedback['access']['suggested_tools']
    chat,_=native_chat([{'role':'system','content':planner.requests[-1][0]},
                        {'role':'user','content':planner.requests[-1][1]}])
    assert json.loads(chat[-1]['content'])['agent_feedback']['access']['status']==status
    assert stats['local_executed']==0  # No model-selected helper involved.


def test_schema_rejected_write_stays_blocked_after_successful_write_and_new_read_data():
    bad=call('notes.append',text='INVALID')
    env, stats, events, _=run([bad,call('notes.append',text='valid'),call('notes.search',query='valid'),bad,DONE])
    assert stats['invalid_proposals']==2 and stats['executed']==2
    assert len(env.validations)==3
    rejection=[e for e in events if e['event']=='decision_rejected'][-1]
    assert rejection['feedback']['scope']=='sample'
    assert [c['arguments'] for c in env.get_tool_calls()]==[{'text':'valid'},{'query':'valid'}]


def test_schema_rejected_read_stays_blocked_after_write():
    class SchemaEnv(Env):
        def validate(self,name,args):
            if name=='notes.search' and 'query' not in args:
                self.validations.append({'name':name,'arguments':deepcopy(args)})
                return {'ok':False,'error':'missing query'}
            return super().validate(name,args)
    bad=call('notes.search')
    env, stats, events, _=run([bad,call('notes.append',text='valid'),bad,DONE],SchemaEnv())
    assert len(env.validations)==2 and stats['executed']==1
    assert [e for e in events if e['event']=='decision_rejected'][-1]['feedback']['scope']=='sample'


@pytest.mark.parametrize('done', [False,True])
def test_write_noop_preserves_read_cache_and_successful_write_expires_it(done):
    class WriteEnv(Env):
        def call(self,name,args):
            if name=='notes.append' and not done:
                self.calls.append({'name':name,'arguments':deepcopy(args)})
                return {'ok':True,'done':False,'message':'not available'}
            return super().call(name,args)
    read=call('notes.search',query='new')
    env,stats,_,_=run([read,call('notes.append',text='new'),read,DONE],WriteEnv())
    assert stats['executed']==(3 if done else 2)
    assert stats['invalid_proposals']==(0 if done else 1)
    assert len(env.get_tool_calls())==(3 if done else 2)


def test_noop_write_is_blocked_until_a_successful_write_without_hiding_either_call():
    class WriteEnv(Env):
        def call(self,name,args):
            if name=='notes.append' and args['text']=='unavailable':
                self.calls.append({'name':name,'arguments':deepcopy(args)})
                return {'ok':True,'done':False,'message':'unavailable'}
            return super().call(name,args)
    write=call('notes.append',text='unavailable')
    env,stats,events,_=run([write,write,call('notes.append',text='available'),write,DONE],WriteEnv())
    assert stats['executed']==3 and stats['invalid_proposals']==1
    assert len(env.get_tool_calls())==3
    assert next(e for e in events if e['event']=='decision_rejected')['feedback']['reason']=='repeated_no_progress_write'


def test_repeated_successful_writes_preserve_two_loyalty_redemptions():
    name='loyalty.redeem'
    catalog={name:{'name':name,'kind':'write','parameters':{'type':'object','properties':{
        'program':{'type':'string'},'reward_id':{'type':'string'}},'required':['program','reward_id']}},
        'sys.ask_missing_info':CATALOG['sys.ask_missing_info']}
    class LoyaltyEnv(Env):
        def validate(self,tool,args):
            self.validations.append({'name':tool,'arguments':deepcopy(args)})
            return {'ok':True}
        def call(self,tool,args):
            self.calls.append({'name':tool,'arguments':deepcopy(args)})
            return {'ok':True,'done':True,'voucher':len(self.calls)}
    entry={**sample(),'context':{'installed_apps':{'loyalty':{'tools':[name]}}}}
    redemption=call(name,program='ABC',reward_id='reward_1')
    env,stats,_,_=run([redemption,redemption,DONE],LoyaltyEnv(),catalog,entry)
    assert env.get_tool_calls()==[{k:v for k,v in redemption.items() if k!='action'}]*2
    assert stats['executed']==2 and stats['invalid_proposals']==0
    entry['label']={'tool_calls':env.get_tool_calls()}
    replay_env=LoyaltyEnv()
    rows=replay_sample(entry,catalog,replay_env,Path('/unused'),prepared_transcript=prepared())
    assert replay_env.get_tool_calls()==env.get_tool_calls()
    assert len(rows)==3


def test_history_does_not_block_the_first_read_in_the_current_toolenv():
    prior={**prepared(),'history':[{'role':'assistant','text':'Đã đọc.',
        'tool_calls':[{'name':'notes.search','arguments':{'query':'known'}}],
        'tool_results':[{'ok':True,'notes':[{'text':'old'}]}]}]}
    env,stats,_,_=run([call('notes.search',query='known'),DONE],prepared_row=prior)
    assert len(env.get_tool_calls())==1 and stats['invalid_proposals']==0


@pytest.mark.parametrize('phase', ['validate','call'])
def test_technical_errors_have_one_retry_and_are_not_schema_rejections(phase):
    class BrokenEnv(Env):
        def validate(self,name,args):
            if phase=='validate':
                self.validations.append({'name':name,'arguments':deepcopy(args)})
                raise RuntimeError('validator unavailable')
            return super().validate(name,args)
        def call(self,name,args):
            self.calls.append({'name':name,'arguments':deepcopy(args)})
            raise RuntimeError('execution unavailable')
    write=call('notes.append',text='valid')
    env,stats,events,planner=run([write,write,write,DONE],BrokenEnv(),cfg=AgentConfig(max_invalid=5))
    assert len(env.validations)==2
    assert len(env.get_tool_calls())==(2 if phase=='call' else 0)
    blocked=[e for e in events if e['event']=='decision_rejected'][-1]
    assert blocked['feedback']['reason']=='technical_retry_budget'
    assert blocked['feedback']['scope']!='sample'  # Not the schema blacklist.
    if phase=='call':
        observations=json.loads(planner.requests[1][1])['previous_actions_and_results']
        assert 'ok' not in json.loads(observations[-1]['result'])
        assert observations[-1]['verification']['origin']=='agent_exception'


@pytest.mark.parametrize('validation', [None,True,{}, {'ok':1}])
def test_malformed_validator_response_does_not_poison_schema_blacklist(validation):
    class RecoverEnv(Env):
        def validate(self,name,args):
            self.validations.append({'name':name,'arguments':deepcopy(args)})
            return validation if len(self.validations)==1 else {'ok':True}
    write=call('notes.append',text='valid')
    env,stats,_,_=run([write,write,DONE],RecoverEnv())
    assert len(env.validations)==2 and len(env.get_tool_calls())==1
    assert stats['invalid_proposals']==1


def test_sft_and_inference_use_the_same_guards_feedback_and_env_sequence(native_tokenizer):
    calls=[{'name':'notes.search','arguments':{'query':'known'}},
           {'name':'notes.append','arguments':{'text':'new'}}]
    entry={**sample(),'audio_text':'Tìm rồi ghi chú.','label':{'tool_calls':calls}}
    prior={**prepared(),'transcript':'Tìm rồi ghi chú.'}
    training_env=Env(); training_env.notes=[{'text':'known'}]
    rows=replay_sample(entry,CATALOG,training_env,Path('/unused'),prepared_transcript=prior)
    inference_env=Env(); inference_env.notes=[{'text':'known'}]
    env,stats,_,planner=run([{**c,'action':'call'} for c in calls]+[DONE],inference_env,entry=entry,prepared_row=prior)
    for row,(system,user) in zip(rows,planner.requests):
        assert row['messages'][:2]==[{'role':'system','content':system},{'role':'user','content':user}]
    payload=json.loads(rows[1]['messages'][1]['content'])
    feedback=payload['previous_actions_and_results'][0]['automatic_execution_feedback']
    assert feedback['managed_by']=='python' and feedback['repeat_policy']=='reuse_read_until_state_change'
    assert feedback['policy']==EXECUTION_POLICY
    assert training_env.get_tool_calls()==env.get_tool_calls()==calls
    assert training_env.get_state_hash()==env.get_state_hash()
    assert stats['local_executed']==0
    encoded=encode_example(rows[1],native_tokenizer,65536)
    assert -100 in encoded['labels'] and any(label!=-100 for label in encoded['labels'])
    chat,_=native_chat(rows[1]['messages'][:2])
    assert json.loads(chat[-1]['content'])['automatic_execution_feedback']==feedback


def test_sft_refuses_duplicate_read_gold_instead_of_silently_deleting_it():
    read={'name':'notes.search','arguments':{'query':'same'}}
    entry={**sample(),'audio_text':'Tìm ghi chú.','label':{'tool_calls':[read,deepcopy(read)]}}
    env=Env()
    with pytest.raises(ValueError,match='gold blocked by automatic guard'):
        replay_sample(entry,CATALOG,env,Path('/unused'),prepared_transcript=prepared())
    assert env.get_tool_calls()==[read]
    assert len(env.validations)==1


def test_runtime_rejects_artifact_with_missing_automatic_policy(temporal_artifact):
    base,artifact,tokenizer,manifest=temporal_artifact
    assert manifest['execution_policy']==EXECUTION_POLICY
    manifest.pop('execution_policy')
    (artifact/RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Automatic execution policy'):
        load_runtime_manifest(base,tokenizer,artifact)


@pytest.mark.parametrize('target', [None, ['notes.append'], {'name':'notes.append'}, 'missing.tool', 'local_text.normalize'])
def test_missing_info_target_must_be_an_exposed_env_tool_before_validation(target):
    env, stats, events, _=run([call('sys.ask_missing_info',tool_name=target,params=['text']),DONE])
    assert not env.validations and not env.get_tool_calls()
    assert next(e for e in events if e['event']=='decision_rejected')['stage']=='precheck'
    assert stats['invalid_proposals']==1 and stats['local_executed']==0


def test_same_agent_does_not_reuse_read_cache_across_samples():
    read=call('notes.search',query='known')
    planner=Planner([read,DONE,read,DONE])
    agent=LifeMateAgent(None,planner,CATALOG,AgentConfig())
    for text in ['known old','known new']:
        env=Env(); env.notes=[{'text':text}]
        stats=agent.run(sample(),env,Path('/unused'),'public_test',prepared())
        assert len(env.get_tool_calls())==1 and stats['invalid_proposals']==0
        payload=json.loads(planner.requests[-1][1])
        assert json.loads(payload['previous_actions_and_results'][0]['result'])['notes']==[{'text':text}]
