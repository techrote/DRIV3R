#!/usr/bin/env python3
"""DRIV3R Corpus Factory v56: physical-result-certified, counterfactual material-memory curriculum.
DATA-ONLY. Synthetic-host v55 source, not a native CyberSand experiment.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

SCHEMA='driv3r/corpus_factory/physical-memory-causality-v56/1'
SPLITS=('TRAIN','DEV','HELDOUT')

def canonical(x): return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def digest(x): return hashlib.sha256(x).hexdigest()
def insist(ok,message):
    if not ok: raise ValueError('v56: '+message)
def load(path): return [json.loads(line) for line in path.read_text().splitlines()]
def file_entry(path):
    b=path.read_bytes();return {'sha256':digest(b),'bytes':len(b),'rows':len(b.splitlines())}
def inp_name(split): return split+('_focus_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_focus_stream.jsonl')
def out_name(split): return split+('_causal_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_causal_stream.jsonl')

def last_receipt(steps, index, surface):
    """Last delivered observation of PAINT_A's actual material on the current surface."""
    for j in range(index,-1,-1):
        for item in reversed(steps[j]['input']['events_delta']):
            if len(item)==6 and item[1]=='PAINT_RESULT' and item[2]=='PAINT_A' and item[4]==surface:
                return j,item
    raise ValueError('No physical calibration history before protected paint')

def suffix_equivalence(left,li,right,ri):
    k=0
    while k<min(li,ri):
        x=left[li-k-1];y=right[ri-k-1]
        if x['input']!=y['input'] or x['action']!=y['action']: break
        k+=1
    return k

def build(source,out):
    source=Path(source).resolve();out=Path(out).resolve()
    insist(not out.exists(),'refuses to overwrite output')
    origin=json.loads((source/'manifest.json').read_text())
    insist(origin['schema']=='driv3r/corpus_factory/paired-stream-memory-v55/1','requires v55 source')
    for file,entry in origin['files'].items():
        b=(source/file).read_bytes()
        insist(digest(b)==entry['sha256'] and len(b)==entry['bytes'],'invalid source shard '+file)
    out.mkdir(parents=True)
    manifest={'schema':SCHEMA,'date':'2026-11-18','data_only':True,'synthetic_not_native_CyberSand':True,
      'source_v55_manifest_sha256':digest((source/'manifest.json').read_bytes()),
      'source_v55_shard_hashes':{k:v['sha256'] for k,v in sorted(origin['files'].items())},
      'selection':'Only v55 material_calibration pairs; physically certified next-step correct output and progress, prior opposite PAINT_A receipts. Every original episode remains intact.',
      'contract':{
        'model_visible':'Only steps[].input at current time, with past input/action stream where maintained by the agent; never expose paired certificates or evaluator-only provenance.',
        'train':'TRAIN_causal_stream.jsonl contains complete TRAIN episodes, exact original observation/action/weight, and 6x loss on causal critical decision (never unmask source weight=0). It supplements, not replaces, the broad v54/v55 training corpora.',
        'dev':'DEV_EVALUATOR_ONLY_causal_stream.jsonl exclusively for development scoring; never optimize on DEV.',
        'heldout':'HELDOUT_EVALUATOR_ONLY_causal_stream.jsonl sealed final evaluation; neither model training nor hyperparameter selection.',
        'pair_metadata':'*_material_pairs_EVALUATOR_ONLY.jsonl carry future receipts and labels, never included in model input/recurrent state.',
        'primary_metric':'Pair-both-correct fraction on opposite PAINT_A/PAINT_B actions at exactly identical current observation; stratify by last physical receipt age and identical trailing input/action window (1,4,8,16,32).',
        'full_episode_metric':'Autonomous synthetic-host terminal task completion must be measured separately; teacher-forced action accuracy is not closed-loop success.',
        'ablation':'Compare full-history memory against memory reset each step and last 4/8/16/32 previous input+action windows; report same-topology paired failures.',
        'split_isolation':'Retain v54/v55 split/topology lineage across all versions; do not train on DEV/HELDOUT, even via older v53 shards.',
        'safety':'No model training, no simulator writes, no native CyberSand evaluation, no paid compute.'},
      'limitations':['The paired wrong-command branch is inferred from v53 two-law material mapping; it is not separately replayed in this dataset.',
       'Initial synthetic prelude calibrations remain scripted and observations are symbolic, allowing a simple event parser to solve the task.',
       'TRAIN long identical-history windows are sparse; only limited HELDOUT long-memory contrasts exist; do not report zero-shot generalization.',
       'Synthetic-host source-derived focus, not new native CyberSand material physics or an empirical SLM evaluation.',
       'No LFM350/Sprocket training, neural CPU latency/RSS, or immutable Actions shard publication.'],
      'files':{},'splits':{}}
    topology_owners={}
    for split in SPLITS:
        episodes=load(source/inp_name(split)); pairs=load(source/(split+'_pairs_EVALUATOR_ONLY.jsonl'))
        provenance=load(source/(split+'_provenance_EVALUATOR_ONLY.jsonl'))
        insist(len(episodes)==len(provenance)==origin['splits'][split]['selected_full_episodes'],'source size mismatch '+split)
        selected_pairs=[];selected=set();target=defaultdict(set);counts=Counter()
        for pair in pairs:
            if pair['category']!='material_calibration':continue
            refs=pair['references'];insist(len(refs)==2 and sorted(pair['actions'])==['PAINT_A','PAINT_B'],'unexpected material pair')
            actions=[];proof=[]
            for ref in refs:
                i=ref['v55_focus_episode'];j=ref['step'];steps=episodes[i]['steps'];step=steps[j];obs=step['input'];surface=obs['surface']
                insist(step['action'] in ('PAINT_A','PAINT_B') and step['action'] in obs['legal'],'illegal paint action')
                insist(digest(canonical(obs).encode())==pair['instant_observation_sha256'],'not identical instantaneous observation')
                insist(j+1<len(steps),'no physical receipt after material paint')
                recent=steps[j+1]['input']['events_delta']
                correct=[x for x in recent if len(x)==6 and x[1]=='PAINT_RESULT' and x[2]==step['action'] and x[3]==surface and x[4]==surface and x[5]==obs['p']]
                insist(correct,'no actual correct physical paint after decisive command')
                insist(any(x[1] in ('MISSION_COMPLETE','GOAL_REVISION') for x in recent),'paint did not advance mission')
                eidx,receipt=last_receipt(steps,j,surface)
                insist((receipt[3]==surface)==(step['action']=='PAINT_A'),'physical-memory witness inconsistent with selected action')
                proof.append({'source_episode':i,'source_step':j,'surface':surface,'last_PAINT_A_result_material':receipt[3],
                    'last_PAINT_A_result_step':eidx,'calibration_memory_age_steps':j-eidx,
                    'physical_goal_receipt_sha256':digest(canonical(correct[-1]).encode()),
                    'next_step_goal_progress':True,'v55_focus_episode':i})
                actions.append(step['action']); selected.add(i);target[i].add(j)
            insist(actions==pair['actions'] and proof[0]['last_PAINT_A_result_material']!=proof[1]['last_PAINT_A_result_material'],
                   'no causal material-law contrast')
            r0,r1=refs
            insist(r0['world_index']==r1['world_index'] and r0['v55_focus_episode']!=r1['v55_focus_episode'], 'not paired within world')
            eq=suffix_equivalence(episodes[r0['v55_focus_episode']]['steps'],r0['step'],episodes[r1['v55_focus_episode']]['steps'],r1['step'])
            ages=[p['calibration_memory_age_steps'] for p in proof]
            counts['material_pairs']+=1
            for threshold in (1,4,8,16,32):
                counts['equal_history_ge'+str(threshold)]+=int(eq>=threshold)
                counts['both_calibration_age_ge'+str(threshold)]+=int(min(ages)>=threshold)
            selected_pairs.append({'pair_id':pair['pair_id'],'instant_observation_sha256':pair['instant_observation_sha256'],
                'actions':pair['actions'],'category':'physically_certified_material_memory',
                'min_calibration_memory_age_steps':min(ages),
                'identical_past_input_action_suffix_steps':eq,
                'opposite_observed_PAINT_A_materials':True,
                'counterfactual_semantics':'two-law PAINT_A/PAINT_B swap; wrong-command failure inferred, not separately host-replayed',
                'references':proof})
        insist(len(selected_pairs)>0,'no certified pairs '+split)
        mapping={old:new for new,old in enumerate(sorted(selected))}
        stream=out/out_name(split);meta=out/(split+'_provenance_EVALUATOR_ONLY.jsonl')
        pairs_out=out/(split+'_material_pairs_EVALUATOR_ONLY.jsonl')
        decisions=0;zero_weight=0
        with stream.open('x') as fd,meta.open('x') as fm,pairs_out.open('x') as fp:
            for old in sorted(selected):
                original=episodes[old]['steps'];dst=[]
                pr=dict(provenance[old]);topology=pr['geometry_sha256']
                insist(topology not in topology_owners or topology_owners[topology]==split,'topology leaked across splits')
                topology_owners[topology]=split
                for j,s in enumerate(original):
                    item={k:s[k] for k in ('input','action','weight')}
                    insist(item['action'] in item['input']['legal'],'illegal source command')
                    if split=='TRAIN':item['focus_loss_weight']=s['weight']*(6 if j in target[old] else 1)
                    dst.append(item);decisions+=1;zero_weight+=int(s['weight']==0)
                insist(dst[-1]['action']=='STOP','truncated terminal episode')
                fd.write(canonical({'steps':dst})+'\n')
                pr['v56_causal_episode_index']=mapping[old];pr['v55_focus_episode_index']=old
                fm.write(canonical(pr)+'\n')
            for p in selected_pairs:
                p=dict(p);p['references']=[{**r,'v56_causal_episode':mapping[r['v55_focus_episode']]} for r in p['references']]
                fp.write(canonical(p)+'\n')
        for fn in (stream.name,meta.name,pairs_out.name):manifest['files'][fn]=file_entry(out/fn)
        manifest['splits'][split]={'source_episodes':len(episodes),'selected_complete_episodes':len(selected),
          'decisions':decisions,'critical_material_decisions':2*len(selected_pairs),
          'zero_weight_train_repeats_retained_masked':zero_weight if split=='TRAIN' else 0,
          'memoryless_current_input_pair_accuracy_ceiling':0.5,
          'memoryless_current_input_pair_both_correct_ceiling':0.0,
          'metrics':dict(sorted(counts.items()))}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();print(json.dumps(build(a.source,a.output)['splits'],indent=2))
