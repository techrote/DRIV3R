#!/usr/bin/env python3
"""Independent v59 evidence and leak audit; no generator imports or neural scores."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
SPLITS=('TRAIN','DEV','HELDOUT')
def digest(b):return hashlib.sha256(b).hexdigest()
def req(x,s):
    if not x:raise AssertionError('v59 audit '+s)
def records(p):return [json.loads(line) for line in p.read_text().splitlines() if line]
def fname(split,kind):
    if kind=='source':return split+('_causal_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_causal_stream.jsonl')
    if kind=='train':return split+('_receipt_choice_stream.jsonl' if split=='TRAIN' else '_EVALUATOR_ONLY_receipt_choice_stream.jsonl')
    return split+'_'+kind+'_EVALUATOR_ONLY.jsonl'
def audit(source,out,modules):
    source,out,modules=map(Path,(source,out,modules))
    man=json.loads((out/'manifest.json').read_text());src=json.loads((source/'manifest.json').read_text())
    req(man['schema']=='driv3r/corpus_factory/receipt-conditioned-continuation-v59/1','schema')
    req(man['synthetic_not_native_CyberSand'] and man['data_only'],'scope label')
    req(man['source_v56_manifest_sha256']==digest((source/'manifest.json').read_bytes()),'source manifest pin')
    req(man['generator_sha256']==digest((modules/'feedback_choice_v59.js').read_bytes()),'generator source hash')
    req(man['parent_shard_sha256']=={k:v['sha256'] for k,v in src['files'].items()},'source shard pin')
    for f,v in src['files'].items():
        b=(source/f).read_bytes();req(digest(b)==v['sha256'] and len(b)==v['bytes'],'source hash '+f)
    for f,v in man['files'].items():
        b=(out/f).read_bytes();req(digest(b)==v['sha256'] and len(b)==v['bytes'] and len(b.splitlines())==v['rows'],'shard '+f)
    for f,h in man['host_module_sha256'].items():req(digest((modules/f).read_bytes())==h,'host drift '+f)
    geometries={};totals=Counter();contrasts=[]
    for split in SPLITS:
        source_eps=records(source/fname(split,'source'))
        src_pairs=records(source/fname(split,'material_pairs'))
        src_prov=records(source/fname(split,'provenance'))
        eps=records(out/fname(split,'train'))
        side=records(out/(split+'_receipt_choice_certificates_EVALUATOR_ONLY.jsonl'))
        req(len(eps)==len(side),'episode evidence mismatch')
        counts=Counter();seen=set()
        for pair in src_pairs:
            for q,ref in enumerate(pair['references']):
                if ref['surface']!='B':continue
                idx=ref['v56_causal_episode'];j=ref['source_step'];tag=(idx,j)
                req(tag not in seen,'duplicate B reference');seen.add(tag)
                p=src_prov[idx];srcstep=source_eps[idx]['steps'][j]
                req(p['split']==split and srcstep['action']==pair['actions'][q],'source reference mismatch')
                geom=p['geometry_sha256'];req(geometries.setdefault(geom,split)==split,'cross split geometry')
                pair_data=[]
                for arm in ('correct','wrong'):
                    c=side[len(pair_data)+2*(len(seen)-1)]
                    ep=eps[c['trainable_episode_index']]['steps']
                    req(c['branch_type']==arm and c['pair_id']==pair['pair_id'] and c['source_ref']==q,'branch pairing')
                    req(c['source_episode_index']==idx and c['source_step']==j and c['geometry_sha256']==geom,'branch provenance')
                    req(c['source_seed']==p['source_seed'] and c['world_index']==p['world_index'] and c['arm']==p['arm'],'host world provenance')
                    req(c['synthetic_terminal_success'] and c['material_surface']=='B','success/surface')
                    req(c['intervention_action'] in srcstep['input']['legal'],'illegal intervention')
                    req(ep[j]['input']==srcstep['input'] and ep[j]['action']==c['intervention_action'],'fork input')
                    req(ep[j]['weight']==ep[j]['focus_loss_weight']==0,'intervention loss leak')
                    for k,s in enumerate(ep):
                        req(s['input']['v']=='synthetic-v54-event-stream','wrong input protocol')
                        req('durable' not in s['input'] and 'events_delta' in s['input'],'full history leak')
                        req(s['action'] in s['input']['legal'],'illegal action')
                        req(not any(x in s or x in s['input'] for x in ('branch_type','source_ref','correct_action','hidden_law','counterfactuals','recovery_events')),'evaluator metadata leak')
                        if k<j:
                            req(s['input']==source_eps[idx]['steps'][k]['input'] and s['action']==source_eps[idx]['steps'][k]['action'],'historical prefix drift')
                        if k>j:req(s['weight']==1 and s['focus_loss_weight']==6,'critical supervision')
                    delta=ep[j+1]['input']['events_delta']
                    physical=[e for e in delta if len(e)>=5 and e[1]=='PAINT_RESULT' and e[2]==c['intervention_action'] and e[4]=='B']
                    req(physical,'host physical result absent')
                    expected_material='B' if arm=='correct' else 'A'
                    req(physical[-1][3]==expected_material==c['postintervention_physical_result'],'physical result mismatch')
                    if arm=='correct':
                        req(c['intervention_action']==srcstep['action'],'correct arm wrong action')
                        req(len(ep)==j+2 and c['postintervention_actions']==['STOP'],'correct arm actions')
                        req(any(e[1]=='MISSION_COMPLETE' for e in delta),'completion receipt absent')
                    else:
                        req(c['intervention_action']!=srcstep['action'],'wrong arm correct action')
                        req(len(ep)==j+4 and c['postintervention_actions']==['ERASE',srcstep['action'],'STOP'],'wrong arm recovery')
                        req(not any(e[1]=='MISSION_COMPLETE' for e in delta),'wrong arm completion receipt')
                        corrected=[e for e in ep[j+3]['input']['events_delta'] if e[1]=='PAINT_RESULT' and len(e)>=5 and e[3]==e[4]=='B']
                        req(corrected,'host correction receipt missing')
                    counts[arm+'_intervention_episodes']+=1
                    counts['decisions']+=len(ep)
                    counts['forced_paints_zero_weight']+=1
                    counts['supervised_postintervention_decisions']+=len(ep)-j-1
                    counts['host_terminal_successes']+=1
                    counts['episodes']+=1
                    pair_data.append(ep)
                correct,wrong=pair_data
                req(correct[:j]==wrong[:j] or all(a['action']==b['action'] and a['input']==b['input'] for a,b in zip(correct[:j],wrong[:j])),'different fork history')
                req(correct[j]['input']==wrong[j]['input'],'different pre-intervention observation')
                req(correct[j]['action']!=wrong[j]['action'],'no action intervention')
                req(correct[j+1]['action']=='STOP' and wrong[j+1]['action']=='ERASE','no decision contrast')
                contrasts.append((split,pair['pair_id'],q))
        req(counts['correct_intervention_episodes']==counts['wrong_intervention_episodes']==len(seen),'unbalanced arms')
        req(man['splits'][split]['source_physical_pairs']==len(src_pairs),'source pair count')
        req(man['splits'][split]['source_B_goal_decisions']==len(seen),'B count')
        for k,v in counts.items():req(man['splits'][split][k]==v,f'{split} count {k}: {v}')
        req(man['splits'][split]['correct_postaction_STOP']==len(seen),'correct STOP count')
        req(man['splits'][split]['wrong_postaction_ERASE']==len(seen),'wrong ERASE count')
        req(man['splits'][split]['wrong_postaction_corrected_PAINT']==len(seen),'wrong paint count')
        req(man['splits'][split]['wrong_postaction_STOP']==len(seen),'wrong STOP count')
        totals.update(counts)
    req(len(geometries)==man['unique_geometry_worlds'],'geometry count')
    req(totals['decisions']==man['total_decisions'],'decision count')
    return {'audit':'PASS','scope':'synthetic host, teacher continuation; no SLM training',
        'geometry_worlds':len(geometries),'paired_fork_origins':len(contrasts),
        'synthetic_episode_total':totals['episodes'],'decision_total':totals['decisions'],
        'branch_correct':totals['correct_intervention_episodes'],'branch_wrong':totals['wrong_intervention_episodes'],
        'memoryless_fixed_first_action_ceiling':0.5}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--modules',required=True)
    a=p.parse_args();print(json.dumps(audit(a.source,a.output,a.modules),indent=2))
