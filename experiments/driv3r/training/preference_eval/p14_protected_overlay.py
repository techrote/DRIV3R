"""P14: protected-cell overlay stress test of frozen W16 W0 trajectories.

Research-only *proxy*: W0 contains no paint/material/protected-cell ABI. A deterministic
external overlay marks positions as hypothetical protected paint cells. An extra
candidate-action filter can veto entering them; W0 itself remains unchanged and
is still the final host-legal-action authority. Walking into a mask is NOT proof
of a native protected-material write. No human preference labels exist.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import p12_recovery_preference as base

LAYOUTS=('near_origin','forward_band')
POLICIES=('original_late','veto_late','veto_goal','veto_grace16')
FAMILIES=base.FAMILIES
TRAIN=base.DISCOVERY
VALIDATION=base.VALIDATION
HELDOUT=base.HELDOUT
GOAL_NOVEL=48


def protected_cells(seed,layout):
    """Predeclare masks without observing any model actions, world outcomes or test trails."""
    if layout not in LAYOUTS:raise ValueError(layout)
    dy=(int.from_bytes(sha256(f'P14-mask/{seed}/{layout}'.encode()).digest()[:2],'big')%5)-2
    if layout=='near_origin':rects=((11,16,45+dy,50+dy),)
    else:rects=((20,29,43+dy,53+dy),)
    return {(x,y) for a,b,c,d in rects for x in range(a,b+1) for y in range(c,d+1)}


def legal_destinations(obs,history,visited):
    """Mirror only pinned-W0 move selection, never alter source/native host code."""
    x,y,h=obs['x'],obs['y'],obs['heading']
    dirs=((1,0),(0,1),(-1,0),(0,-1))
    arm={}
    for a,rot in (('STRAIGHT',0),('LEFT',-1),('RIGHT',1)):
        d=(h+rot)%4; p=(x+dirs[d][0],y+dirs[d][1])
        if a in obs['legal']:arm[a]=(p[0],p[1],d)
    output={a:arm[a][:2] for a in arm}
    if 'BRANCH' in obs['legal']:
        unexplored=[a for a in ('RIGHT','LEFT','STRAIGHT') if a in arm and arm[a][:2] not in visited]
        selected=(unexplored or [a for a in ('STRAIGHT','LEFT','RIGHT') if a in arm])[0]
        output['BRANCH']=arm[selected][:2]
    if 'RETRACE' in obs['legal']:
        if not history:raise AssertionError('host says RETRACE without history')
        output['RETRACE']=history[-1][:2]
    return output


def records_metrics(rec,protected):
    path=[tuple(p) for p in rec['trajectory']]; intersection=[p for p in path[1:] if p in protected]
    novelty=len(set(path))-1
    return {'novel_positions':novelty,'goal_completed':novelty>=GOAL_NOVEL,
        'protected_contacts':len(intersection),'protected_unique':len(set(intersection)),
        'protected_first_tick':next((i for i,p in enumerate(path[1:]) if p in protected),None),
        'branches':rec['metrics']['branches'],'retraces':rec['metrics']['retraces'],
        'steps':len(rec['events']),'stopped':bool(rec['metrics']['terminated_by_stop']),
        'invalid':rec['metrics']['invalid_raw_choices']}


def rollout(w0,head,seed,family,layout,policy,source):
    mask=protected_cells(seed,layout)
    history=[];visited={(7,48)};last=None;decisions=[];first_goal_tick=None
    def choose(obs):
        nonlocal last,first_goal_tick
        t=obs['tick']
        if t:
            # Validation before extending the prefix: host state is not modified.
            if last is None or (obs['x'],obs['y'])!=last['destination']:
                raise AssertionError('W0 movement diverged from independently predicted destination')
            if last['action']=='RETRACE':history.pop()
            elif last['action']!='STOP':history.append(last['prior'])
            visited.add((obs['x'],obs['y']))
        if len(visited)-1>=GOAL_NOVEL and first_goal_tick is None:first_goal_tick=t
        destinations=legal_destinations(obs,history,visited)
        safe=[a for a in obs['legal'] if a!='STOP' and a in destinations and destinations[a] not in mask]
        with torch.no_grad():
            logits=head(torch.tensor([base.model_features(obs)],dtype=torch.float32))[0].tolist()
        original_moves=[a for a in obs['legal'] if a!='STOP']
        allow=original_moves if policy=='original_late' else safe
        move=base.choice(logits,allow,.7,base.uniform(seed,family,t,'w11-CRN')) if allow else None
        pstop=base.stop_prob(logits,obs['legal'])
        late=t>=124 and base.uniform(seed,family,t,'p10-independent-STOP-CRN-v1')<pstop
        if policy=='original_late':stop=late
        elif policy=='veto_late':stop=late or not safe
        elif policy=='veto_goal':stop=first_goal_tick is not None or not safe
        elif policy=='veto_grace16':stop=(first_goal_tick is not None and t-first_goal_tick>=16) or not safe
        else:raise ValueError(policy)
        action='STOP' if stop else move
        if action not in obs['legal']:raise AssertionError('action not legal in original W0')
        dest=(obs['x'],obs['y']) if action=='STOP' else destinations[action]
        if policy!='original_late' and action!='STOP' and dest in mask:raise AssertionError('protected overlay crossing')
        decisions.append({'tick':t,'legal':list(obs['legal']),'safe_move_count':len(safe),
            'action':action,'destination':list(dest),'goal_complete_at_decision':first_goal_tick is not None,
            'forced_stop':policy!='original_late' and action=='STOP' and not safe,'stop_probability':pstop})
        last={'action':action,'destination':dest,'prior':(obs['x'],obs['y'],obs['heading'])}
        return action
    rec=w0.run(seed,'callback',family,128,selector=choose)
    if not all(a['action']==e['applied'] and e['raw_legal'] for a,e in zip(decisions,rec['events'])):
        raise AssertionError('action legality receipt mismatch')
    if policy=='original_late':
        if rec['trajectory']!=source['trajectory'] or rec['metrics']!=source['metrics']:
            raise AssertionError('unmodified historical W16 baseline mismatch')
    metrics=records_metrics(rec,mask)
    if policy!='original_late' and metrics['protected_contacts']:
        raise AssertionError('protected region changed despite candidate filter')
    return rec,decisions,metrics


def priority_score(m):
    """Transparent *outcome-only* synthetic safety, completion and construction priority.

    No learned aesthetics; this priority is not a human-voted ranking. The STOP
    or action controller may not access this function at runtime.
    """
    return (-int(m['protected_contacts']>0),-min(m['protected_contacts'],128),
        int(m['goal_completed']),m['novel_positions'],m['branches'],-m['steps'])


def rank_features(m):
    # Deliberately exclude layout, seed, model family, and personal/owner ratings.
    return [float(m['protected_contacts']>0),min(m['protected_contacts']/32,1),
        float(m['goal_completed']),m['novel_positions']/128,
        m['branches']/40,m['retraces']/40,m['steps']/128]


def preference_pairs(entries):
    groups=defaultdict(dict)
    for e in entries:groups[(e['family'],e['seed'],e['layout'])][e['policy']]=e
    pairs=[]
    for (family,seed,layout),group in sorted(groups.items()):
        if set(group)!=set(POLICIES):raise AssertionError('incomplete matched policy group')
        for i,one in enumerate(POLICIES):
            for other in POLICIES[i+1:]:
                a=group[one];b=group[other]
                ascore=priority_score(a);bscore=priority_score(b)
                if ascore==bscore:continue
                pairs.append({'family':family,'seed':seed,'layout':layout,'a':one,'b':other,
                    'target_a_preferred':int(ascore>bscore),'source_a_trace':a['trace_sha256'],
                    'source_b_trace':b['trace_sha256'],
                    'x':[u-v for u,v in zip(rank_features(a),rank_features(b))],
                    'human_owner_rating':None,'label_type':'deterministic_safety_completion_proxy'})
    return pairs


def train_ranker(training,validation):
    torch.manual_seed(14016);torch.set_num_threads(1)
    x=torch.tensor([p['x'] for p in training],dtype=torch.float32)
    y=torch.tensor([p['target_a_preferred'] for p in training],dtype=torch.float32)
    vx=torch.tensor([p['x'] for p in validation],dtype=torch.float32)
    vy=torch.tensor([p['target_a_preferred'] for p in validation],dtype=torch.float32)
    if not len(x) or not len(vx):raise ValueError('empty dataset')
    # Linear anti-symmetric preference score: s(a)-s(b), guarantees swapped-pair reversal.
    head=torch.nn.Linear(7,1,bias=False)
    optimizer=torch.optim.Adam(head.parameters(),lr=.04,weight_decay=.0001)
    selected=None;minimum=float('inf');beststep=0
    for t in range(1,501):
        optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(head(x).squeeze(1),y)
        loss.backward();optimizer.step()
        if t%10==0:
            with torch.no_grad():v=float(F.binary_cross_entropy_with_logits(head(vx).squeeze(1),vy))
            if v<minimum:minimum=v;beststep=t;selected={k:v.detach().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(selected);head.eval()
    return head,{'adam_updates':500,'selection_step':beststep,'validation_bce':minimum,
        'train_pairs':len(training),'validation_pairs':len(validation)}


def test_ranker(model,pairs):
    if not pairs:return {'pairs':0}
    x=torch.tensor([p['x'] for p in pairs],dtype=torch.float32)
    y=torch.tensor([p['target_a_preferred'] for p in pairs],dtype=torch.float32)
    with torch.no_grad():prediction=torch.sigmoid(model(x).squeeze(1))
    naive=0
    for p in pairs:
        # Existing simple physical safeguard comparator: protection first, goal second,
        # then position novelty. Does not inspect model probabilities or future moves.
        d=p['x'];value=-100*d[0]-10*d[1]+5*d[2]+d[3]
        naive+=int((value>0)==bool(p['target_a_preferred'])) if value!=0 else 0
    return {'pairs':len(pairs),'accuracy':float(((prediction>=.5)==y).float().mean()),
        'brier':float(((prediction-y)**2).mean()),
        'average_precision':base.average_precision(prediction.tolist(),y.tolist()),
        'naive_priority_correct_at_non_ties':naive,
        'proxy_oracle_correct':len(pairs),'positives':int(y.sum())}


def experiment(source_bytes,out,save_images=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    z,w0,head,cksha=base.open_pinned(source_bytes)
    entries=[];forward_count=0;original_png_matches=0
    seeds=TRAIN+VALIDATION+HELDOUT
    for fam in FAMILIES:
      for seed in seeds:
        source,original_sha,_=base.extract_trace(z,fam,seed)
        for layout in LAYOUTS:
          for policy in POLICIES:
            rec,decisions,m=rollout(w0,head,seed,fam,layout,policy,source)
            canonical=(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n').encode()
            png=w0.png_bytes(rec)
            if policy=='original_late':
                original_png=z.read(f'rollouts/{fam}-{seed}-intact-late.png')
                if png!=original_png:raise AssertionError('baseline PNG changed')
                original_png_matches+=1
            forward_count+=len(decisions)
            stem=f'{fam}-{seed}-{layout}-{policy}'
            if save_images:
                (out/(stem+'.json')).write_bytes(canonical)
                (out/(stem+'.png')).write_bytes(png)
            entries.append({'family':fam,'seed':seed,'layout':layout,'policy':policy,
                'cohort':('train' if seed in TRAIN else 'validation' if seed in VALIDATION else 'heldout'),
                'source_trace_sha256':original_sha,'trace_sha256':base.sha(canonical),
                'png_sha256':base.sha(png),'mask_sha256':base.sha(json.dumps(sorted(protected_cells(seed,layout))).encode()),
                'mask_cells':len(protected_cells(seed,layout)),'forward_passes':len(decisions),**m,
                'all_actions_host_legal':all(e['raw_legal'] for e in rec['events']),
                'forced_stops':sum(d['forced_stop'] for d in decisions)})
    pairs=preference_pairs(entries)
    train=[p for p in pairs if p['seed'] in TRAIN]
    val=[p for p in pairs if p['seed'] in VALIDATION]
    test=[p for p in pairs if p['seed'] in HELDOUT]
    model,training=train_ranker(train,val)
    torch.save({'weights':model.state_dict(),'training':training,'P11_sha256':base.P11_SHA,
        'W16_head_sha256':cksha,'target_type':'synthetic-protection-completion-proxy',
        'human_ratings':0},out/'P14-synthetic-pairwise-ranker.pt')
    grouped={}
    for policy in POLICIES:
        arr=[e for e in entries if e['cohort']=='heldout' and e['policy']==policy]
        grouped[policy]={'world_layouts':len(arr),'safe':sum(e['protected_contacts']==0 for e in arr),
            'goal_completed':sum(e['goal_completed'] for e in arr),'protected_contacts':sum(e['protected_contacts'] for e in arr),
            'forced_stops':sum(e['forced_stops'] for e in arr),
            'mean_novel_positions':sum(e['novel_positions'] for e in arr)/len(arr),
            'mean_steps':sum(e['steps'] for e in arr)/len(arr)}
    unique_traces=len({e['trace_sha256'] for e in entries})
    report={'schema':'DRIV3R-preference-P14-protected-overlay/v1','source_P11_SHA256':base.P11_SHA,
      'source_W16_SHA256':base.W16_SHA,'source_head_SHA256':cksha,
      'environment':'unchanged synthetic W0; external position overlay proxy not game protected-material painting',
      'cohorts':{'train':TRAIN,'validation':VALIDATION,'heldout':HELDOUT},
      'mask_design':'two fixed-before-inference rectangles, deterministic seed vertical jitter; no heldout trails used',
      'trajectories':len(entries),'unique_trajectory_bytes':unique_traces,'host_legal_action_decisions':forward_count,'original_png_byte_matches':original_png_matches,
      'paired_comparisons':len(pairs),'training':training,'pairwise_heldout':test_ranker(model,test),
      'summary':grouped,'human_ratings':0,'owner_ratings':0,
      'limits':['No W0 protected-cell or paint action ABI: occupancy is a research-only proxy',
         'External candidate filter is stricter than W0 action legality; not native CyberSand host verification',
         'Pairs are algorithmic priorities, not human creativity judgments',
         'Linear ranker scores completed episodes, NOT online STOP actions; do not deploy',
         'Source archive W11/W16 trained auxiliary head, not root language model']}
    (out/'P14-results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    (out/'P14-rollout-index.json').write_text(json.dumps(entries,indent=2,sort_keys=True)+'\n')
    (out/'P14-pairs.json').write_text(json.dumps(pairs,indent=2,sort_keys=True)+'\n')
    (out/'P14-SHA256SUMS.txt').write_text('\n'.join(f'{base.sha(p.read_bytes())}  {p.name}' for p in sorted(out.iterdir()) if p.is_file())+'\n')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--p11-artifact',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--images',action='store_true')
    args=ap.parse_args();r=experiment(args.p11_artifact.read_bytes(),args.out,args.images)
    print(json.dumps({'trajectories':r['trajectories'],'forward_passes':r['host_legal_action_decisions'],
        'summary':r['summary'],'ranker':r['pairwise_heldout']},indent=2))
