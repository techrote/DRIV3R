#!/usr/bin/env python3
"""v74: TRAIN-only host-legal decision packing; no policy training and no native claims.

Consumes the independently qualified v73 physical synthetic corpus. Exports only
observable pre-action memory, never policy tables, certificates or future receipts.
DEV/HELDOUT are sealed evaluator-only pairs, not training message files.
"""
import argparse
import hashlib
import json
from pathlib import Path

SPLITS = ('TRAIN', 'DEV', 'HELDOUT')
PARENT_SCHEMA = 'driv3r/corpus_factory/endogenous-witness-v73/1'
SCHEMA = 'driv3r/corpus_factory/causal-memory-decision-v74/1'

def compact(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
def sha(x): return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()
def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def check(v, err):
    if not v: raise ValueError('V74 '+err)
def source_names(split):
    prefix = split if split=='TRAIN' else split+'_EVALUATOR_ONLY'
    return (prefix+'_endogenous_witness_stream.jsonl',
            split+'_endogenous_witness_certificates_EVALUATOR_ONLY.jsonl')
def stream_pair(a,b):
    with a.open() as left,b.open() as right:
        while True:
            x=left.readline(); y=right.readline()
            check(bool(x)==bool(y), 'parent rows mismatch')
            if not x: break
            yield json.loads(x),json.loads(y)
def history(steps,t,window=None,remove=()):
    lo = 0 if window is None else max(0,t-window)
    seen=set(); events=[]
    for k in range(lo,t+1):
        for e in steps[k]['input'].get('events_delta',[]):
            if any(str(part).startswith(prefix) for part in e[1:2] for prefix in remove): continue
            key=compact(e)
            if key not in seen:
                seen.add(key);events.append(e)
    return events

def make_input(steps,t,variant='full'):
    s=steps[t]; obs=s['input']; check(s['action'] in obs['legal'], 'illegal teacher action')
    check(obs.get('v')=='synthetic-v73-active-witness','unexpected host evidence class')
    lookback=None if variant in ('full','no_tariff','no_calibration') else (32 if variant=='last32' else 0)
    removes=('WITNESS_PROCEDURE_PASSES',) if variant=='no_tariff' else (
       ('LOCAL_CHANNEL_CHECK_', 'LOCAL_PHYSICAL_COUPON','LOCAL_POSTFAULT_WITNESS_') if variant=='no_calibration' else ())
    current={k:obs[k] for k in ('p','stage','goal','legal','geometry','local','painted','durable')}
    # Current input's limited `recent` field is excluded to prevent redundant
    # history from silently defeating the explicit last32/current-only controls.
    return {'schema':SCHEMA,'source':'synthetic-host-not-native-CyberSand',
            'request':'Choose exactly one next host-legal action token.',
            'current':current,'events_history':history(steps,t,lookback,removes),
            'actions_history':[[i,steps[i]['action']] for i in range(max(0,t-(lookback if lookback is not None else t)),t)] if lookback!=0 else []}

def pivots(c,steps):
    choice=c['decision_step'];critical=c['critical_step'];mode=c['selected']
    p=[('probe_initial',choice)]
    if mode in ('A','BOTH'):
        p.append(('probe_after_A',choice+2*c['passes']['A']))
    p.append(('remote_recovery',critical))
    for _,t in p:
        check(t<len(steps) and steps[t]['weight']==1,'missing supervised pivotal action')
    return p

def validate_parent(root):
    mf=json.loads((root/'manifest.json').read_text())
    check(mf['schema']==PARENT_SCHEMA and mf['synthetic_not_native_CyberSand'],'parent identity')
    for name,meta in mf['files'].items():
        p=root/name
        check(p.is_file() and p.stat().st_size==meta['bytes'] and hash_file(p)==meta['sha256'],'bad parent file '+name)
    return mf

def generate(root,out, variants=('full','last32','current_only','no_tariff','no_calibration')):
    check(not out.exists(),'output already exists')
    mf=validate_parent(root)
    out.mkdir(parents=True)
    output={'schema':SCHEMA,'source_parent_manifest_sha256':hash_file(root/'manifest.json'),
            'generator_sha256':hash_file(Path(__file__)),
            'scope':'observable synthetic-v73 host inputs and exact host legal action targets only',
            'trainable_files':['TRAIN_messages.jsonl'],
            'evaluator_only':['DEV_EVALUATOR_ONLY_pairs.jsonl','HELDOUT_EVALUATOR_ONLY_pairs.jsonl'],
            'warning':'Do not train on any evaluator-only pairs or ablation sidecars. No native CyberSand training or inference implied.',
            'files':{},'splits':{}}
    for split in SPLITS:
        epname,cn=source_names(split)
        target='TRAIN_messages.jsonl' if split=='TRAIN' else split+'_EVALUATOR_ONLY_pairs.jsonl'
        stats={'parent_episodes':0,'pivotal_examples':0,'actions':{},'exact_duplicate_removed':0}
        seen=set(); labels_by_full={}; paired=[]
        with (out/target).open('x') as f:
            for i,(episode,c) in enumerate(stream_pair(root/epname,root/cn)):
                steps=episode['steps'];check(c['split']==split and c['episode']==i,'episode split/provenance')
                check(c['total_legal_actions']==len(steps),'parent accepted decisions mismatch')
                check(sha(json.dumps([[s['input'],s['action'],s['weight']] for s in steps],separators=(',',':'),ensure_ascii=False))==c['episode_sha256'], 'parent episode receipt mismatch')
                for stage,t in pivots(c,steps):
                    inp=make_input(steps,t); gold=steps[t]['action']; key=sha(compact(inp))
                    check(key not in labels_by_full or labels_by_full[key]==gold,'identical full observation has conflicting labels')
                    labels_by_full[key]=gold
                    if (key,gold) in seen:
                        stats['exact_duplicate_removed']+=1;continue
                    seen.add((key,gold));stats['pivotal_examples']+=1
                    stats['actions'][gold]=stats['actions'].get(gold,0)+1
                    row={'messages':[{'role':'user','content':compact(inp)},{'role':'assistant','content':gold}]}
                    if split!='TRAIN':
                        row={'input':inp,'gold_host_action_EVALUATOR_ONLY':gold,'stage':stage,
                             'ablation_inputs_EVALUATOR_ONLY':{v:make_input(steps,t,v) for v in variants if v!='full'}}
                    # Source world, certificate, hidden fault, future action and oracle policy cost never enter model input.
                    f.write(json.dumps(row,separators=(',',':'),ensure_ascii=False)+'\n')
                stats['parent_episodes']+=1
        check(stats['parent_episodes']==mf['splits'][split]['episodes'],'parent split counts')
        stats['source_world_geometry_sha256']=[w['geometry_sha256'] for w in mf['splits'][split]['worlds']]
        output['splits'][split]=stats
        p=out/target;output['files'][target]={'bytes':p.stat().st_size,'sha256':hash_file(p),'rows':stats['pivotal_examples']}
    sets=[set(output['splits'][s]['source_world_geometry_sha256']) for s in SPLITS]
    check(not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'cross-split geometry overlap')
    output['parent_source_hashes']={k:mf[k] for k in ('generator_sha256','host_sha256','parent_v70_sha256','parent_v72_sha256')}
    (out/'manifest.json').write_text(json.dumps(output,indent=2)+'\n')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent_corpus',type=Path);p.add_argument('new_output_directory',type=Path)
    a=p.parse_args();print(json.dumps(generate(a.parent_corpus,a.new_output_directory)['splits'],indent=2))
