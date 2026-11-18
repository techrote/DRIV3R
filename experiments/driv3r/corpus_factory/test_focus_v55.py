#!/usr/bin/env python3
"""Independent v55 audit and optional teacher-forced predictions scorer; NOT native gameplay."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import focus_v55 as f

def check(x, msg):
    if not x: raise AssertionError('v55 audit: '+msg)
def records(p): return [json.loads(line) for line in Path(p).read_text().splitlines()]

def audit(source, output):
    source=Path(source);output=Path(output)
    m=json.loads((output/'manifest.json').read_text()); orig=json.loads((source/'manifest.json').read_text())
    check(m['schema']==f.PREFIX,'schema')
    check(m['v54_source_manifest_sha256']==f.digest((source/'manifest.json').read_bytes()),'source manifest digest')
    check(m['v54_source_files_sha256']=={n:v['sha256'] for n,v in orig['files'].items()},'source file identities')
    for path,info in m['files'].items():
        b=(output/path).read_bytes()
        check(f.digest(b)==info['sha256'] and len(b)==info['bytes'],'output shard hash '+path)
    seen_topology={}; ret={}
    for split in f.SPLITS:
        source_stream,source_meta=f.source_names(split)
        old_episodes=records(source/source_stream);old_prov=records(source/source_meta)
        newname=split+('_focus_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_focus_stream.jsonl')
        new_episodes=records(output/newname)
        new_prov=records(output/(split+'_provenance_EVALUATOR_ONLY.jsonl'))
        pairs=records(output/(split+'_pairs_EVALUATOR_ONLY.jsonl'))
        check(len(new_episodes)==len(new_prov)==m['splits'][split]['selected_full_episodes'],'focus episode length')
        check(len(pairs)==m['splits'][split]['pairs'],'pair count')
        lookup={}; grouped=defaultdict(lambda:defaultdict(list))
        for ni,(new,pr) in enumerate(zip(new_episodes,new_prov)):
            old_i=pr['v54_source_episode_index'];old=old_episodes[old_i]
            check(pr['v55_focus_episode_index']==ni,'remap error')
            check({k:v for k,v in pr.items() if k not in ('v54_source_episode_index','v55_focus_episode_index')}==old_prov[old_i],'provenance drift')
            topology=pr['geometry_sha256']
            check(topology not in seen_topology or seen_topology[topology]==split,'world topology leaked across splits')
            seen_topology[topology]=split
            check(len(new['steps'])==len(old['steps']),'episode truncation')
            for j,(s,t) in enumerate(zip(new['steps'],old['steps'])):
                check(s['input']==t['input'] and s['action']==t['action'] and s['weight']==t['weight'],'source action/observation/weight changed')
                check(s['action'] in s['input']['legal'],'illegal action')
                check('pair_id' not in s['input'] and 'focus_loss_weight' not in s['input'],'model-visible metadata leak')
                if split=='TRAIN':
                    check(set(s)=={'input','action','weight','focus_loss_weight'},'train additional field')
                    check(s['focus_loss_weight'] in (0,1,4) and (s['weight']!=0 or s['focus_loss_weight']==0),'train weight anomaly')
                else:check(set(s)=={'input','action','weight'},'eval weight or metadata leak')
                key=f.obs_key(s['input']);grouped[key][s['action']].append((ni,j))
                lookup[(ni,j)]=s
            check(new['steps'][-1]['action']=='STOP','terminal STOP absent')
        used=Counter(); critical=set(); high=Counter(); types=Counter()
        for pair in pairs:
            check(pair['same_world_topology'] and pair['opposite_legal_actions'],'pair assertions')
            check(pair['instant_observation_sha256'] not in used,'duplicate signature pair')
            used[pair['instant_observation_sha256']]+=1
            refs=pair['references']
            check(len(refs)==2 and pair['actions'][0]!=pair['actions'][1],'not binary opposite labels')
            ss=[]
            for ref, expected in zip(refs,pair['actions']):
                ni,j=ref['v55_focus_episode'],ref['step'];step=lookup[(ni,j)]
                check(step['action']==expected,'pair label mismatches stream')
                check(f.obs_key(step['input'])==pair['instant_observation_sha256'],'pair snapshot mismatch')
                check(new_prov[ni]['v54_source_episode_index']==ref['episode'],'pair remap mismatch')
                check(ref['world_index']==new_prov[ni]['world_index'],'pair world mismatch')
                delta_indices=[k for k,x in enumerate(new_episodes[ni]['steps'][:j+1]) if x['input']['events_delta']]
                check(delta_indices and j-delta_indices[-1]==ref['age'],'age claim mismatch')
                check(ref['age']>=0,'negative age')
                ss.append((ni,j,step))
                critical.add((ni,j))
            check(ss[0][0]!=ss[1][0],'pair within same source episode')
            check(ss[0][2]['input']==ss[1][2]['input'],'not truly identical instant')
            check(pair['min_event_age_steps']==min(r['age'] for r in refs),'age min')
            check(pair['tier_age8']==(pair['min_event_age_steps']>=8),'tier8')
            check(pair['tier_age16']==(pair['min_event_age_steps']>=16),'tier16')
            check(pair['category']==f.category(*pair['actions']),'category')
            high['8+']+=int(pair['tier_age8']);high['16+']+=int(pair['tier_age16'])
            types[pair['category']]+=1
        check(len(critical)==2*len(pairs),'reused decision in multiple challenge pairs')
        for (ni,j),step in lookup.items():
            if split=='TRAIN':
                check(step['focus_loss_weight']==step['weight']*(4 if (ni,j) in critical else 1),'bad training upweight')
        counters=m['splits'][split]
        check(counters['paired_decisions']==2*len(pairs),'decision count')
        check(counters['focus_decisions']==len(critical),'focus count')
        check(counters['decisions']==len(lookup),'decision count total')
        check(counters['categories']==dict(sorted(types.items())),'types')
        check(counters['min_event_age_ge8_pairs']==high['8+'] and counters['min_event_age_ge16_pairs']==high['16+'],'ages')
        # Contradiction proof: deterministic stateless policy presented with an identical observation
        # must produce identical outputs, so cannot achieve both correct actions in any pair.
        check(all(p['actions'][0]!=p['actions'][1] for p in pairs),'memoryless contradiction proof')
        ret[split]={'pairs':len(pairs),'full_episodes':len(new_episodes),'decisions':len(lookup),
                    'age8':high['8+'],'age16':high['16+'],'categories':dict(types)}
    return {'audit':'PASS','source':'v54 synthetic event-stream (not native CyberSand)','split_results':ret}

def score_pairs(export, split, predictions):
    """Predictions JSONL: each ordered focus episode is {'actions':[predicted action or null, ...]}.
    Teacher-forced decision metrics only, NOT closed-loop/host success.
    """
    p=Path(export);m=json.loads((p/'manifest.json').read_text())
    check(split in ('DEV','HELDOUT'),'scorer only DEV or HELDOUT')
    ep=records(p/(split+'_EVALUATOR_ONLY_focus_stream.jsonl'))
    pred=records(predictions)
    check(len(ep)==len(pred),'one predictions record required per full focus episode')
    for rec,e in zip(pred,ep):
        check(set(rec)=={'actions'} and isinstance(rec['actions'],list) and len(rec['actions'])==len(e['steps']),'prediction episode length/shape')
        for act,step in zip(rec['actions'],e['steps']):
            check(act is None or act in step['input']['legal'],'predicted action not legal')
    pair=records(p/(split+'_pairs_EVALUATOR_ONLY.jsonl'))
    groups=defaultdict(lambda:Counter())
    for item in pair:
        flags=[]
        for ref,actual in zip(item['references'],item['actions']):
            flags.append(pred[ref['v55_focus_episode']]['actions'][ref['step']]==actual)
        for key in ('ALL',item['category'],'AGE8' if item['tier_age8'] else 'UNDER8','AGE16' if item['tier_age16'] else 'UNDER16'):
            groups[key]['pairs']+=1;groups[key]['correct_actions']+=sum(flags);groups[key]['both_correct']+=all(flags)
    return {'evaluation':'teacher-forced (NOT closed loop)', 'split':split,
            'groups':{k:{**v,'per_decision_accuracy':v['correct_actions']/(2*v['pairs']),
             'pair_both_correct_fraction':v['both_correct']/v['pairs']} for k,v in sorted(groups.items())}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--export',required=True,type=Path)
    p.add_argument('--predictions',type=Path);p.add_argument('--split',choices=('DEV','HELDOUT'))
    a=p.parse_args();print(json.dumps(audit(a.source,a.export),indent=2))
    if a.predictions:print(json.dumps(score_pairs(a.export,a.split,a.predictions),indent=2))
