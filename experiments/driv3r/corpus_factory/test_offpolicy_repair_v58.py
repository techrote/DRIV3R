#!/usr/bin/env python3
"""Independent structural/evidence audit of synthetic v58 recovery curriculum.

No trainable policy and no native CyberSand interactions. Verify host-derived
recovery events and relationship to pinned v56 sources without using JS generator.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLITS = ('TRAIN', 'DEV', 'HELDOUT')

def digest(b): return hashlib.sha256(b).hexdigest()
def req(cond, message):
    if not cond: raise AssertionError('V58 independent audit: '+message)
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def check(src, out, modules):
    src,out,modules=map(Path,(src,out,modules))
    parent=json.loads((src/'manifest.json').read_text())
    man=json.loads((out/'manifest.json').read_text())
    req(man['schema']=='driv3r/corpus_factory/offpolicy-repair-v58/1','schema')
    req(man['data_only'] and man['synthetic_not_native_CyberSand'],'misleading native label')
    req(man['source_v56_manifest_sha256']==digest((src/'manifest.json').read_bytes()),'parent identity')
    req(man['source_v56_shards']=={k:v['sha256'] for k,v in parent['files'].items()},'parent shard pins')
    req(man['generator_sha256']==digest((modules/'offpolicy_repair_v58.js').read_bytes()),'generator pin')
    for f,h in man['host_module_sha256'].items():
        req(digest((modules/f).read_bytes())==h,'host source mismatch '+f)
    for f,desc in parent['files'].items():
        b=(src/f).read_bytes()
        req(digest(b)==desc['sha256'] and len(b)==desc['bytes'],'parent shard mismatch '+f)
    for f,desc in man['files'].items():
        b=(out/f).read_bytes()
        req(digest(b)==desc['sha256'] and len(b)==desc['bytes'] and len(b.splitlines())==desc['rows'],'v58 shard mismatch '+f)
    geometry_owners={};totals=Counter()
    for split in SPLITS:
        source=rows(src/(split+('_causal_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_causal_stream.jsonl')))
        prov=rows(src/f'{split}_provenance_EVALUATOR_ONLY.jsonl')
        pairs=rows(src/f'{split}_material_pairs_EVALUATOR_ONLY.jsonl')
        data=rows(out/(split+('_offpolicy_repair_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_offpolicy_repair_stream.jsonl')))
        cert=rows(out/f'{split}_offpolicy_fork_certificates_EVALUATOR_ONLY.jsonl')
        req(len(cert)==sum(len(p['references']) for p in pairs),'fork count')
        expected=[]
        for pair in pairs:
            for q, ref in enumerate(pair['references']):
                p=prov[ref['v56_causal_episode']]
                expected.append((pair,q,ref,p))
                world=p['geometry_sha256']
                req(geometry_owners.setdefault(world,split)==split,'cross-split topology contamination')
        req(len(expected)==len(cert),'lost forks')
        counts=Counter();seen=set()
        for c,(pair,q,ref,p) in zip(cert,expected):
            key=(ref['v56_causal_episode'],ref['source_step']);req(key not in seen,'duplicate source fork');seen.add(key)
            srcsteps=source[key[0]]['steps']; original=srcsteps[key[1]]
            req(c['pair_id']==pair['pair_id'] and c['reference']==q and c['source_step']==key[1],'fork provenance')
            req(c['world_index']==p['world_index'] and c['geometry_sha256']==p['geometry_sha256'] and c['source_seed']==p['source_seed'] and c['arm']==p['arm'],'world lineage')
            req(c['surface']==original['input']['surface'],'surface lineage')
            req(c['original_action']==original['action']==pair['actions'][q] and c['wrong_action']!=c['original_action'],'wrong-action definition')
            req(c['wrong_action'] in original['input']['legal'],'opposite action illegal')
            req(c['instant_observation_sha256']==pair['instant_observation_sha256'],'instantaneous contrast identity')
            req(c['calibration_memory_age_steps']==ref['calibration_memory_age_steps'],'memory witness')
            req(c['intervention_index']==key[1],'intervention location')
            counts['wrong_accepted_forks']+=1
            if c['surface']=='A':
                req(c['terminal_damage'] and not c['nonterminal'] and not c['recovered'],'terminal A classification')
                req(c['trainable_episode_index'] is None and not c['recovery_actions'],'terminal A leaked to training')
                counts['irreversible_wrong_A']+=1
                continue
            req(c['surface']=='B','non-A/B source')
            req(c['nonterminal'] and c['recovered'] and not c['terminal_damage'],'B recoverability')
            req(c['naive_stop_control'] and not c['naive_stop_control']['success'] and c['naive_stop_control']['stage_after']==c['naive_stop_control']['stage_before'],'naive STOP negative control')
            idx=c['trainable_episode_index']
            req(idx==counts['recovered_complete_episodes'] and idx<len(data),'recovery index')
            ep=data[idx]['steps']; j=c['intervention_index']
            req(len(ep)==j+4,'wrong B + exactly three recovery steps')
            for i in range(j):
                req(ep[i]['input']==srcsteps[i]['input'] and ep[i]['action']==srcsteps[i]['action'],'original historical sequence not preserved')
                req(ep[i]['weight']==srcsteps[i]['weight'],'historical weight drift')
            req(ep[j]['input']==original['input'] and ep[j]['action']==c['wrong_action'],'intervention position/observation')
            req(ep[j]['weight']==ep[j]['focus_loss_weight']==0,'wrong intervention not masked')
            actions=[s['action'] for s in ep[j+1:]]
            req(actions==c['recovery_actions']==['ERASE',c['original_action'],'STOP'],'recovery action ordering')
            for i,s in enumerate(ep):
                req(s['action'] in s['input']['legal'],'illegal policy action')
                req('durable' not in s['input'] and 'events_delta' in s['input'],'event-stream regression')
                req('counterfactuals' not in s and 'correct_action' not in s and 'recovery_events' not in s,'evaluator leakage')
                req(s['input']['v']=='synthetic-v54-event-stream','schema drift')
            req([s['focus_loss_weight'] for s in ep[j:]]==[0,6,6,1],'loss weighting')
            req(all(s['weight']==1 for s in ep[j+1:]),'recoveries must be supervised')
            wrong_receipts=[e for e in ep[j+1]['input']['events_delta'] if len(e)>=5 and e[1]=='PAINT_RESULT']
            req(any(e[2]==c['wrong_action'] and e[3]==c['wrong_result'] and e[4]=='B' for e in wrong_receipts),'physical wrong result not delivered')
            repaired_receipts=[e for e in ep[j+3]['input']['events_delta'] if len(e)>=5 and e[1]=='PAINT_RESULT']
            req(any(e[2]==c['original_action'] and e[3]=='B' and e[4]=='B' for e in repaired_receipts),'physical correction not delivered')
            req(c['recovery_events'][0]['action']=='ERASE' and c['recovery_events'][1]['stage_after']>c['recovery_events'][0]['stage_after'],'stage correction')
            req(c['recovery_events'][-1]['done'] and c['recovery_events'][-1]['success'],'terminal success')
            counts['recoverable_wrong_B']+=1
            counts['recovered_complete_episodes']+=1
            counts['demonstration_decisions']+=len(ep)
            counts['zero_weight_forced_wrong_paints']+=1
            counts['naive_stop_successes']+=0
            for a in ('erase','corrected_paint','stop'):counts['host_recovery_'+a]+=1
        counts['source_material_pairs']=len(pairs)
        req(counts==man['splits'][split],f'counts {split}: {counts} vs {man["splits"][split]}')
        req(len(data)==counts['recovered_complete_episodes'],'extra training episodes')
        totals.update(counts)
    return {'status':'PASS','scope':'synthetic-host demonstrations; NOT SLM or native CyberSand',
            'geometries_with_preserved_split':len(geometry_owners),'totals':dict(totals)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--modules',required=True)
    a=p.parse_args();print(json.dumps(check(a.source,a.output,a.modules),indent=2))
