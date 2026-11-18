#!/usr/bin/env python3
"""DRIV3R v55: balanced counterfactual memory challenge from verified v54 streams.
DATA ONLY. Source is synthetic host replay, not native CyberSand.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

SPLITS = ('TRAIN', 'DEV', 'HELDOUT')
PREFIX = 'driv3r/corpus_factory/paired-stream-memory-v55/1'

def canon(o): return json.dumps(o, ensure_ascii=False, separators=(',', ':'))
def digest(b): return hashlib.sha256(b).hexdigest()
def obs_key(obs): return digest(canon(obs).encode())
def need(b, msg):
    if not b: raise ValueError('v55: '+msg)
def load(path): return [json.loads(l) for l in path.read_text().splitlines()]
def source_names(split):
    return (split+('_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_stream.jsonl'),
            split+'_provenance_EVALUATOR_ONLY.jsonl')
def category(a,b):
    if a.startswith('PAINT_') and b.startswith('PAINT_'): return 'material_calibration'
    if a.startswith('MOVE_') and b.startswith('MOVE_'): return 'navigation_history'
    return 'mixed_action_history'

def build(source, out):
    source=Path(source).resolve(); out=Path(out).resolve()
    need(not out.exists(), 'refuse to overwrite corpus output')
    original=json.loads((source/'manifest.json').read_text())
    need(original['schema']=='driv3r/corpus_factory/stream-memory-v54/1','requires v54 stream source')
    for path, m in original['files'].items():
        contents=(source/path).read_bytes()
        need(digest(contents)==m['sha256'] and len(contents)==m['bytes'], 'source shard hash mismatch '+path)
    out.mkdir(parents=True)
    manifest={'schema':PREFIX,'date':'2026-11-18','data_only':True,
       'synthetic_not_native_CyberSand':True,
       'v54_source_manifest_sha256':digest((source/'manifest.json').read_bytes()),
       'v54_source_files_sha256':{p:m['sha256'] for p,m in original['files'].items()},
       'selection':'one deterministic opposite-label pair per exactly identical instantaneous observation, with same world topology, disjoint episodes, nonzero source weight for TRAIN, and maximum minimum age since last newly delivered event',
       'splits':{},'files':{},
       'contract':{
         'format':'Full ordered v54 source episodes are retained, never individual shuffled critical examples.',
         'train':'TRAIN_focus_stream.jsonl is an additional curriculum subset, not a replacement for v54 TRAIN. Model input is steps[].input only. steps[].focus_loss_weight upweights matched memory decisions; original weight=0 stays masked.',
         'dev':'DEV_EVALUATOR_ONLY_focus_stream.jsonl is for development evaluation only, never optimization.',
         'heldout':'HELDOUT_EVALUATOR_ONLY_focus_stream.jsonl is sealed evaluation; do not tune or train on it.',
         'pairs':'*_pairs_EVALUATOR_ONLY.jsonl contain action labels and decisive indices. NEVER enter them into the model prompt, feature stream or recurrent state.',
         'provenance':'*_provenance_EVALUATOR_ONLY.jsonl are separate, inaccessible to model.',
         'metrics':'Evaluate both members of each pair; report pair-both-correct as primary, per-choice balanced accuracy, 8+-step and 16+-step memory-age tiers, category subgroups and full-episode terminal success separately.',
         'deployment':'Offline teacher-forced labels do not prove closed-loop autonomy; validate using synthetic host first and CyberSand-native ABI only in an independently authorized read/evaluation process.',
         'leakage':'Never pool related world identities across splits or corpus versions. Do not leak selected pair metadata or evaluator labels.'},
       'limitations':['Selection is from existing v54/v53 synthetic episodes, not new host-executed scenarios',
         'Hardness is defined by instantaneous observation collisions; a full symbolic event-memory controller could still solve it',
         'Some opposite navigation labels may be tie-related; evaluate action-regret/goal completion in a real host before treating as causal',
         'Prelude receipts are script-generated; age measured only since latest *streamed* event, not prelude host time',
         'No LFM/Sprocket weight changes or neural CPU latency/RSS; no native CyberSand execution',
         'No immutable Actions release included by this generator']}
    for split in SPLITS:
        input_name, src_side_name=source_names(split)
        episodes=load(source/input_name); provenance=load(source/src_side_name)
        need(len(episodes)==len(provenance)==original['stats'][split]['episodes'], 'episode/source count mismatch')
        groups=defaultdict(lambda:defaultdict(list)); episode_sigs=[]
        for ei,(ep,pr) in enumerate(zip(episodes,provenance)):
            need(set(ep)=={'steps'} and len(ep['steps'])==pr['episode_steps'], 'source episode/provenance mismatch')
            age=-1; sigs=[]
            for si,s in enumerate(ep['steps']):
                obs=s['input']; act=s['action']; w=s['weight']
                need(set(s)=={'input','action','weight'} and act in obs['legal'], 'illegal source action or extraneous fields')
                if obs['events_delta']: age=si
                need(age>=0, 'source missing bootstrap event')
                k=obs_key(obs)
                groups[k][act].append({'episode':ei,'step':si,'age':si-age,'world_index':pr['world_index'],
                                        'source_weight':w,'key':k})
                sigs.append(k)
            need(ep['steps'][-1]['action']=='STOP','non-terminal source episode')
            episode_sigs.append(sigs)
        pairs=[]; chosen=set(); paired_class=Counter(); ages=Counter(); impossible=0
        for key in sorted(groups):
            labels=sorted(groups[key])
            if len(labels)<=1: continue
            need(len(labels)==2, 'multiclass ambiguous instant not supported')
            a,b=labels
            left=[x for x in groups[key][a] if split!='TRAIN' or x['source_weight']>0]
            right=[x for x in groups[key][b] if split!='TRAIN' or x['source_weight']>0]
            candidates=[]
            for x in left:
                for y in right:
                    if x['episode']==y['episode'] or x['world_index']!=y['world_index']: continue
                    # Stable maximizing minimum age, then total age; prefer early episode identifiers to avoid hidden nondeterminism.
                    priority=(-min(x['age'],y['age']),-x['age']-y['age'],x['episode'],y['episode'],x['step'],y['step'])
                    candidates.append((priority,x,y))
            if not candidates: impossible+=1; continue
            _,x,y=min(candidates,key=lambda p:p[0]); kind=category(a,b)
            k_id='v55-'+key[:20]
            age_min=min(x['age'],y['age'])
            pairs.append({'pair_id':k_id,'instant_observation_sha256':key,'category':kind,
                 'actions':[a,b], 'references':[{field:v[field] for field in ('episode','step','age','world_index')} for v in (x,y)],
                 'min_event_age_steps':age_min,'tier_age8':age_min>=8,'tier_age16':age_min>=16,
                 'same_world_topology':True,'opposite_legal_actions':True})
            chosen.update([x['episode'],y['episode']]); paired_class[kind]+=1
            ages['8+']+=int(age_min>=8);ages['16+']+=int(age_min>=16)
        need(impossible==0 and pairs, 'could not pair each opposite-action state')
        chosen=sorted(chosen); remap={old:i for i,old in enumerate(chosen)}
        targets=defaultdict(set)
        for p in pairs:
            for r in p['references']: targets[r['episode']].add(r['step'])
        output_file=split+('_focus_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_focus_stream.jsonl')
        pfile=split+'_pairs_EVALUATOR_ONLY.jsonl'
        prof=split+'_provenance_EVALUATOR_ONLY.jsonl'
        used=Counter()
        with (out/output_file).open('x') as stream, (out/prof).open('x') as meta_out, (out/pfile).open('x') as pair_out:
            for ei in chosen:
                ep=episodes[ei]
                steps=[]
                for si,s in enumerate(ep['steps']):
                    step=dict(s)
                    if split=='TRAIN':
                        # Input is unchanged. Weight remains 0 for repeats, otherwise 4 on matched contrast decisions.
                        step['focus_loss_weight']=(4 if si in targets[ei] else 1) * s['weight']
                        need(not(s['weight']==0 and step['focus_loss_weight']!=0),'unmasked duplicate')
                    steps.append(step)
                    used['decisions']+=1
                    used['focus_decisions']+=int(si in targets[ei])
                    used['zero_weight_repeats']+=int(s['weight']==0)
                stream.write(canon({'steps':steps})+'\n')
                pr=dict(provenance[ei]);pr['v54_source_episode_index']=ei;pr['v55_focus_episode_index']=remap[ei]
                meta_out.write(canon(pr)+'\n')
            for pair in pairs:
                p=dict(pair)
                p['references']=[{**r,'v55_focus_episode':remap[r['episode']]} for r in pair['references']]
                pair_out.write(canon(p)+'\n')
        for name in (output_file,pfile,prof):
            content=(out/name).read_bytes()
            manifest['files'][name]={'sha256':digest(content),'bytes':len(content)}
        manifest['splits'][split]={'source_worlds':original['stats'][split]['worlds'],
            'source_episodes':len(episodes),'selected_full_episodes':len(chosen),
            'pairs':len(pairs),'paired_decisions':2*len(pairs),
            'memoryless_max_correct_on_balanced_paired_decisions':len(pairs),
            'memoryless_balanced_paired_accuracy_ceiling':0.5,
            'pair_both_correct_memoryless_ceiling':0.0,
            'categories':dict(sorted(paired_class.items())),
            'min_event_age_ge8_pairs':ages['8+'],'min_event_age_ge16_pairs':ages['16+'],
            **used}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();m=build(a.source,a.output)
    print(json.dumps(m['splits'],indent=2))
