#!/usr/bin/env python3
"""Independent v74 audit: consumes parent physical receipts and exported shards.
Does not import the v74 producer, and never trains a model.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

SCHEMA='driv3r/corpus_factory/causal-memory-decision-v74/1'
SPLITS=('TRAIN','DEV','HELDOUT')
def ok(x,msg):
    if not x: raise ValueError('V74 AUDIT '+msg)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v): return json.dumps(v,sort_keys=True,separators=(',', ':'),ensure_ascii=False)
def rows(p):
    with p.open() as f:
        for line in f: yield json.loads(line)
def parent_files(split):
    p=split if split=='TRAIN' else split+'_EVALUATOR_ONLY'
    return (p+'_endogenous_witness_stream.jsonl',split+'_endogenous_witness_certificates_EVALUATOR_ONLY.jsonl')
def build(steps,t,variant='full'):
    s=steps[t]['input'];last=0 if variant=='current_only' else 32 if variant=='last32' else t
    retained=[];seen=set()
    for k in range(t-last,t+1):
        for ev in steps[k]['input'].get('events_delta',[]):
            kind=ev[1]
            if variant=='no_tariff' and kind=='WITNESS_PROCEDURE_PASSES':continue
            if variant=='no_calibration' and (kind.startswith('LOCAL_CHANNEL_CHECK_') or kind.startswith('LOCAL_POSTFAULT_WITNESS_') or kind=='LOCAL_PHYSICAL_COUPON'):continue
            key=canonical(ev)
            if key not in seen:seen.add(key);retained.append(ev)
    return {'schema':SCHEMA,'source':'synthetic-host-not-native-CyberSand','request':'Choose exactly one next host-legal action token.',
        'current':{k:s[k] for k in ('p','stage','goal','legal','geometry','local','painted','durable')},
        'events_history':retained,
        'actions_history':[[k,steps[k]['action']] for k in range(t-last,t)]}
def pivots(c):
    t=c['decision_step'];p=[('probe_initial',t)];
    if c['selected'] in ('A','BOTH'):p.append(('probe_after_A',t+2*c['passes']['A']))
    p.append(('remote_recovery',c['critical_step']));return p

def audit(parent,output):
    pm=json.loads((parent/'manifest.json').read_text());m=json.loads((output/'manifest.json').read_text())
    ok(pm['schema']=='driv3r/corpus_factory/endogenous-witness-v73/1','parent type')
    ok(m['schema']==SCHEMA and m['source_parent_manifest_sha256']==digest(parent/'manifest.json'),'manifest ancestry')
    ok(set(m['trainable_files'])=={'TRAIN_messages.jsonl'},'trainability allowlist')
    for file,meta in m['files'].items():
        p=output/file
        ok(p.stat().st_size==meta['bytes'] and digest(p)==meta['sha256'],'output integrity '+file)
    seenworld=set();summary={}
    for split in SPLITS:
        epname,cn=parent_files(split)
        name='TRAIN_messages.jsonl' if split=='TRAIN' else split+'_EVALUATOR_ONLY_pairs.jsonl'
        actual=list(rows(output/name));expected={};expected_ablations={};pivots_count=0;cur_groups=collections.defaultdict(collections.Counter)
        with (parent/epname).open() as source,(parent/cn).open() as cert:
            for i,(line,crow) in enumerate(zip(source,cert,strict=True)):
                ep=json.loads(line);c=json.loads(crow);steps=ep['steps']
                ok(c['split']==split and c['episode']==i,'provenance')
                ok(c['geometry_sha256'] in m['splits'][split]['source_world_geometry_sha256'],'world provenance')
                ok(c['geometry_sha256'] not in seenworld or c['geometry_sha256'] in m['splits'][split]['source_world_geometry_sha256'],'split overlap')
                seenworld.add(c['geometry_sha256'])
                for stage,t in pivots(c):
                    ok(steps[t]['weight']==1 and steps[t]['action'] in steps[t]['input']['legal'],'accepted pivotal action')
                    inp=build(steps,t);key=canonical(inp);gold=steps[t]['action'];pivots_count+=1
                    ok(key not in expected or expected[key]==gold,'contradictory full-memory labels')
                    expected[key]=gold
                    expected_ablations[key]={v:build(steps,t,v) for v in ('last32','current_only','no_tariff','no_calibration')}
                    cur_groups[canonical(build(steps,t,'current_only'))][gold]+=1
        ok(len(actual)==len(expected)==m['splits'][split]['pivotal_examples'],'export count')
        found={}
        for r in actual:
            if split=='TRAIN':
                ok(set(r)=={'messages'},'TRAIN top-level keys')
                ok([x['role'] for x in r['messages']]==['user','assistant'],'TRAIN message roles')
                inp=json.loads(r['messages'][0]['content']);gold=r['messages'][1]['content']
                ok(set(inp)=={'schema','source','request','current','events_history','actions_history'},'leak into model prompt')
            else:
                ok(set(r)=={'input','gold_host_action_EVALUATOR_ONLY','stage','ablation_inputs_EVALUATOR_ONLY'},'evaluator sidecar keys')
                inp=r['input'];gold=r['gold_host_action_EVALUATOR_ONLY']
                ok(r['ablation_inputs_EVALUATOR_ONLY']==expected_ablations[canonical(inp)],'ablation mismatch with physical source')
                for abname,variant in r['ablation_inputs_EVALUATOR_ONLY'].items():
                    ok(abname in ('last32','current_only','no_tariff','no_calibration'),'invalid ablation')
                    # Parent-reconstruction comparison above establishes the complete event list;
                    # independently check monotone ablations cannot create phantom evidence.
                    orig={canonical(x) for x in inp['events_history']}
                    v={canonical(x) for x in variant['events_history']}
                    ok(v<=orig,'invented event in ablation')
                    if abname=='current_only':ok(not variant['actions_history'],'current only history leak')
                    if abname=='no_tariff':ok(not any(x[1]=='WITNESS_PROCEDURE_PASSES' for x in variant['events_history']),'tariff leak')
            key=canonical(inp);ok(key in expected and expected[key]==gold,'source label or observation mismatch')
            ok(gold in inp['current']['legal'],'export illegal action')
            ok(key not in found,'duplicate label not deduplicated');found[key]=gold
        clashes=sum(len(counts)>1 for counts in cur_groups.values())
        best=sum(max(c.values()) for c in cur_groups.values())
        summary[split]={'episodes':m['splits'][split]['parent_episodes'],'pivots_before_dedup':pivots_count,
                        'retained_examples':len(actual),'unique_current_observations':len(cur_groups),
                        'ambiguous_current_only_groups':clashes,
                        'current_only_majority_upper_bound_on_raw_pivots':round(best/pivots_count,6)}
        ok(m['splits'][split]['parent_episodes']==pm['splits'][split]['episodes'],'split episode count')
        ok(len(actual)==m['files'][name]['rows'],'file row count')
    ok(len(seenworld)==sum(len(m['splits'][s]['source_world_geometry_sha256']) for s in SPLITS),'cross-split geometry leakage')
    return summary

if __name__=='__main__':
    ok(len(sys.argv)==3,'usage: python test_memory_decision_export_v74.py PARENT_V73_DIR V74_DIR')
    print(json.dumps({'result':'PASS','synthetic_only':True,'splits':audit(Path(sys.argv[1]),Path(sys.argv[2]))},indent=2))
