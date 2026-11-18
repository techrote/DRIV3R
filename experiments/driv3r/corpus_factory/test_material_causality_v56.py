#!/usr/bin/env python3
"""Independent v56 source-to-output and causal-evidence auditor, plus evaluator-only pair scorer.
Does not import the generator; no SLM training or native CyberSand evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

def require(value,message):
    if not value: raise AssertionError('independent v56 audit: '+message)
def read(p): return [json.loads(t) for t in Path(p).read_text().splitlines()]
def sha(b): return hashlib.sha256(b).hexdigest()
def sig(x):return sha(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode())

def audit(source,export):
    source=Path(source);export=Path(export)
    old=json.loads((source/'manifest.json').read_text());m=json.loads((export/'manifest.json').read_text())
    require(m['schema']=='driv3r/corpus_factory/physical-memory-causality-v56/1','schema')
    require(sha((source/'manifest.json').read_bytes())==m['source_v55_manifest_sha256'],'parent SHA')
    require({k:v['sha256'] for k,v in sorted(old['files'].items())}==m['source_v55_shard_hashes'],'parent file identities')
    for fn,desc in m['files'].items():
        b=(export/fn).read_bytes()
        require(sha(b)==desc['sha256'] and len(b)==desc['bytes'] and len(b.splitlines())==desc['rows'],'shard bytes '+fn)
    topology_owners={};report={}
    for split in ('TRAIN','DEV','HELDOUT'):
        stream=split+('_focus_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_focus_stream.jsonl')
        new_stream=split+('_causal_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_causal_stream.jsonl')
        orig=read(source/stream);provenance=read(source/(split+'_provenance_EVALUATOR_ONLY.jsonl'))
        old_pairs=read(source/(split+'_pairs_EVALUATOR_ONLY.jsonl'))
        focus=read(export/new_stream);metadata=read(export/(split+'_provenance_EVALUATOR_ONLY.jsonl'))
        pairs=read(export/(split+'_material_pairs_EVALUATOR_ONLY.jsonl'))
        require(len(focus)==len(metadata)==m['splits'][split]['selected_complete_episodes'],'episode count')
        original_by_id={p['pair_id']:p for p in old_pairs if p['category']=='material_calibration'}
        require(set(p['pair_id'] for p in pairs)==set(original_by_id),'lost/added material pair')
        selected_critical=set();critical_target=set(); stats=Counter()
        for new_idx,(ep,pro) in enumerate(zip(focus,metadata)):
            old_idx=pro['v55_focus_episode_index'];require(pro['v56_causal_episode_index']==new_idx,'provenance remap')
            require({k:v for k,v in pro.items() if k not in ('v56_causal_episode_index',)}==provenance[old_idx],'provenance drift')
            geometry=pro['geometry_sha256']
            require(geometry not in topology_owners or topology_owners[geometry]==split,'cross split topology leak')
            topology_owners[geometry]=split
            old_steps=orig[old_idx]['steps'];steps=ep['steps']
            require(len(old_steps)==len(steps) and steps[-1]['action']=='STOP','non-identical episode length or terminal')
            for j,(before,after) in enumerate(zip(old_steps,steps)):
                stats['decisions']+=1
                require({k:after[k] for k in ('input','action','weight')}=={k:before[k] for k in ('input','action','weight')},'modified original model source')
                require('pair_id' not in after['input'] and 'focus_loss_weight' not in after['input'],'visible evaluator leak')
                require(after['action'] in after['input']['legal'],'illegal action')
                if split=='TRAIN':
                    require(set(after)=={'input','action','weight','focus_loss_weight'},'training schema')
                    require(after['focus_loss_weight'] in (0,1,6),'unusual training focus weight')
                    stats['zero_weight_train_repeats_retained_masked']+=int(after['weight']==0)
                else:require(set(after)=={'input','action','weight'},'DEV/HELDOUT have curriculum metadata')
        for pair in pairs:
            old_pair=original_by_id[pair['pair_id']]
            require(pair['actions']==old_pair['actions'] and pair['actions']==['PAINT_A','PAINT_B'],'action pair labels')
            require(pair['instant_observation_sha256']==old_pair['instant_observation_sha256'],'pair observation signature')
            require(len(pair['references'])==2,'pair has exactly two references')
            witnesses=[];trail=[]
            for ref,oldref,expected in zip(pair['references'],old_pair['references'],pair['actions']):
                ni=ref['v56_causal_episode'];j=ref['source_step'];oi=ref['source_episode']
                require(oi==oldref['v55_focus_episode'] and j==oldref['step'],'pair reference drift')
                require(metadata[ni]['v55_focus_episode_index']==oi,'pair episode remap')
                episode=focus[ni]['steps'];s=episode[j];before=s['input'];surface=before['surface']
                require(s['action']==expected and sig(before)==pair['instant_observation_sha256'],'incorrect model input or label')
                require(j+1<len(episode),'missing followup receipt')
                valid=[event for event in episode[j+1]['input']['events_delta'] if len(event)==6 and event[1]=='PAINT_RESULT' and event[2]==expected and event[3]==surface and event[4]==surface and event[5]==before['p']]
                require(valid and sha(json.dumps(valid[-1],ensure_ascii=False,separators=(',',':')).encode())==ref['physical_goal_receipt_sha256'],'incorrect goal material receipt')
                require(any(event[1] in ('GOAL_REVISION','MISSION_COMPLETE') for event in episode[j+1]['input']['events_delta']),'missing mission success evidence')
                receipt=None
                for idx in range(j,-1,-1):
                    good=[event for event in episode[idx]['input']['events_delta'] if len(event)==6 and event[1]=='PAINT_RESULT' and event[2]=='PAINT_A' and event[4]==surface]
                    if good:
                        receipt=(idx,good[-1]);break
                require(receipt is not None,'missing calibration')
                idx,evt=receipt
                require(ref['last_PAINT_A_result_step']==idx and ref['last_PAINT_A_result_material']==evt[3],'witness differs from actual prior observation')
                require(ref['calibration_memory_age_steps']==j-idx,'receipt age')
                require((evt[3]==surface)==(expected=='PAINT_A'),'wrong label given prior physical evidence')
                require((ni,j) not in selected_critical,'critical decision reused')
                selected_critical.add((ni,j));witnesses.append(evt[3]);trail.append((episode,j))
            require(witnesses[0]!=witnesses[1],'not opposite physical law witness')
            require(trail[0][0][trail[0][1]]['input']==trail[1][0][trail[1][1]]['input'],'not instant equivalent')
            width=0
            steps_a,a=trail[0];steps_b,b=trail[1]
            while width<min(a,b):
                u=steps_a[a-1-width];v=steps_b[b-1-width]
                if u['input']!=v['input'] or u['action']!=v['action']:break
                width+=1
            require(width==pair['identical_past_input_action_suffix_steps'],'equal history window miscalculated')
            require(min(x['calibration_memory_age_steps'] for x in pair['references'])==pair['min_calibration_memory_age_steps'],'min receipt age')
            stats['material_pairs']+=1
            for k in (1,4,8,16,32):
                stats['equal_history_ge'+str(k)]+=int(width>=k)
                stats['both_calibration_age_ge'+str(k)]+=int(pair['min_calibration_memory_age_steps']>=k)
        for ni,ep in enumerate(focus):
            for j,step in enumerate(ep['steps']):
                if split=='TRAIN':
                    require(step['focus_loss_weight']==step['weight']*(6 if (ni,j) in selected_critical else 1),'loss weighting or masked repeat not preserved')
        require(stats['decisions']==m['splits'][split]['decisions'],'decision count')
        require(stats['material_pairs']*2==m['splits'][split]['critical_material_decisions'],'paired decision count')
        for key,val in m['splits'][split]['metrics'].items():require(stats[key]==val,'stat '+key)
        if split=='TRAIN':require(stats['zero_weight_train_repeats_retained_masked']==m['splits'][split]['zero_weight_train_repeats_retained_masked'],'zero-weight count')
        report[split]={'episodes':len(focus),'decisions':stats['decisions'],'pairs':len(pairs),
            'suffix8':stats['equal_history_ge8'],'suffix16':stats['equal_history_ge16'],'suffix32':stats['equal_history_ge32'],
            'physical_age16':stats['both_calibration_age_ge16'],'physical_age32':stats['both_calibration_age_ge32']}
    return {'audit':'PASS','scope':'synthetic-host corpus, not model or native CyberSand qualification','splits':report}

def score(export,split,predictions):
    require(split in ('DEV','HELDOUT'),'score only DEV or HELDOUT')
    export=Path(export)
    episodes=read(export/(split+'_EVALUATOR_ONLY_causal_stream.jsonl'))
    predictions=read(predictions)
    require(len(episodes)==len(predictions),'prediction episode count')
    for ep,pred in zip(episodes,predictions):
        require(set(pred)=={'actions'} and len(pred['actions'])==len(ep['steps']),'prediction episode shape')
        for act,s in zip(pred['actions'],ep['steps']):
            require(act is None or act in s['input']['legal'],'illegal prediction')
    acc=Counter(); groups={}
    for p in read(export/(split+'_material_pairs_EVALUATOR_ONLY.jsonl')):
        hits=[]
        for r,a in zip(p['references'],p['actions']):hits.append(predictions[r['v56_causal_episode']]['actions'][r['source_step']]==a)
        keys=['all']
        for horizon in (4,8,16,32):
            keys.append(('identical_history_ge' if p['identical_past_input_action_suffix_steps']>=horizon else 'identical_history_lt')+str(horizon))
            keys.append(('physical_receipt_age_ge' if p['min_calibration_memory_age_steps']>=horizon else 'physical_receipt_age_lt')+str(horizon))
        for key in keys:
            if key not in groups:groups[key]=Counter()
            groups[key]['pairs']+=1;groups[key]['correct_choices']+=sum(hits);groups[key]['both_correct']+=int(all(hits))
    return {'mode':'teacher-forced only; NOT closed-loop success','split':split,'groups':{
        key:{'pairs':v['pairs'],'paired_action_accuracy':v['correct_choices']/(2*v['pairs']),
        'pair_both_correct_fraction':v['both_correct']/v['pairs']} for key,v in sorted(groups.items())}}

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--source',required=True,type=Path);pa.add_argument('--export',required=True,type=Path)
    pa.add_argument('--predictions',type=Path);pa.add_argument('--split',choices=('DEV','HELDOUT'))
    args=pa.parse_args();print(json.dumps(audit(args.source,args.export),indent=2))
    if args.predictions:
        require(args.split is not None,'split is required with predictions')
        print(json.dumps(score(args.export,args.split,args.predictions),indent=2))
