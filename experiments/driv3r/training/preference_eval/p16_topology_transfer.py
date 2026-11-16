"""P16: genuinely fresh-seed protected-position transfer; synthetic W0 only.

Frozen W16 auxiliary MLP and immutable original P11 producer. New seeds do not have archived original model trajectories. This tests
external hypothetical position masks, never native protected paint/write sets.
Unchanged W0 remains the final host-legal action authority. No human ratings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import itertools
from collections import defaultdict
from pathlib import Path
import torch
import p12_recovery_preference as base
import p14_protected_overlay as p14

FAMILIES=base.FAMILIES
HELDOUT=(2903,2917,2927,2939,2953,2969,2971,2987,2999,3011,3023,3037)
LAYOUTS=('alternating_teeth','two_gate_cage','island_chain')
POLICIES=('original_late','position_veto','exit_aware_veto','two_ring_veto')
GOAL=48
P14_RANKER_SHA='fece7f68288df0c1fc5777b10c8c68694206842f48d49fae0533e84bb1db91f9'
DIRECTIONS=((1,0),(0,1),(-1,0),(0,-1))


def masks(layout):
    """Predeclared, fixed topology layouts independent of all trace outcomes."""
    if layout=='alternating_teeth':
        return {(x,y) for j,x in enumerate((10,14,18,22))
                for y in (range(40,51) if j%2 else range(47,59))}
    if layout=='two_gate_cage':
        return {(x,y) for x in range(11,21) for y in range(40,57)
                if (x in (11,20) or y in (40,56)) and (x,y) not in ((11,48),(20,50))}
    if layout=='island_chain':
        return {(x+dx,y+dy) for x,y in ((10,45),(13,49),(16,44),(19,51),(22,47))
                for dx in (0,1) for dy in (0,1)}
    raise ValueError('unknown layout')


def two_ring_count(point,mask):
    x,y=point
    return sum(0<=x+dx<96 and 0<=y+dy<96 and (x+dx,y+dy) not in mask
               for dx in range(-2,3) for dy in range(-2,3)
               if abs(dx)+abs(dy) in (1,2))

def exit_count(point, mask):
    x,y=point
    return sum(0<=x+dx<96 and 0<=y+dy<96 and (x+dx,y+dy) not in mask
               for dx,dy in DIRECTIONS)


def sampled_move(logits,allow,seed,family,t,destinations,visited,mask,policy):
    if not allow: return None
    if policy not in ('exit_aware_veto','two_ring_veto'):
        return base.choice(logits,allow,.7,base.uniform(seed,family,t,'w11-CRN'))
    # Bounded, transparent option-preservation heuristic; not a trained reward.
    # Retains W16 logit structure and CRN sampling; penalises local future traps.
    augmented=list(logits)
    for a in allow:
        i=base.ACTIONS.index(a)
        pos=destinations[a]
        augmented[i]+=(.42*math.log1p(two_ring_count(pos,mask)) if policy=='two_ring_veto' else .7*math.log1p(exit_count(pos,mask)))+.55*float(pos not in visited)
    return base.choice(augmented,allow,.7,base.uniform(seed,family,t,'w11-CRN'))


def rollout(w0,head,seed,family,layout,policy,source):
    mask=masks(layout)
    history=[];visited={(7,48)};last=None;decisions=[]
    def choose(obs):
        nonlocal last
        t=obs['tick']
        if t:
            if last is None or (obs['x'],obs['y'])!=last['destination']:
                raise AssertionError('pinned-W0 source destination mismatch')
            if last['action']=='RETRACE':history.pop()
            elif last['action']!='STOP':history.append(last['prior'])
            visited.add((obs['x'],obs['y']))
        destinations=p14.legal_destinations(obs,history,visited)
        candidates=[a for a in obs['legal'] if a!='STOP' and a in destinations]
        safe=[a for a in candidates if destinations[a] not in mask]
        with torch.no_grad():
            logits=head(torch.tensor([base.model_features(obs)],dtype=torch.float32))[0].tolist()
        allow=candidates if policy=='original_late' else safe
        move=sampled_move(logits,allow,seed,family,t,destinations,visited,mask,policy)
        late=t>=124 and base.uniform(seed,family,t,'p10-independent-STOP-CRN-v1')<base.stop_prob(logits,obs['legal'])
        stop=late or (policy!='original_late' and not safe)
        action='STOP' if stop else move
        if action not in obs['legal']:raise AssertionError('host-illegal action')
        dest=(obs['x'],obs['y']) if action=='STOP' else destinations[action]
        if policy!='original_late' and dest in mask:
            raise AssertionError('masked position entered')
        decisions.append({'tick':t,'action':action,'destination':list(dest),
            'host_legal':action in obs['legal'],'safe_candidates':len(safe),
            'forced_stop':bool(policy!='original_late' and not safe),
            'visited_prefix_count':len(visited),'mask_sha256':base.sha(json.dumps(sorted(mask)).encode())})
        last={'action':action,'destination':dest,'prior':(obs['x'],obs['y'],obs['heading'])}
        return action
    rec=w0.run(seed,'callback',family,128,selector=choose)
    if len(rec['events'])!=len(decisions) or not all(d['action']==e['applied'] and e['raw_legal'] for d,e in zip(decisions,rec['events'])):
        raise AssertionError('invalid W0 receipts')
    if policy=='original_late' and source is not None and (rec['trajectory']!=source['trajectory'] or rec['metrics']!=source['metrics']):
        raise AssertionError('original P11 replay drift')
    m=p14.records_metrics(rec,mask)
    if policy!='original_late' and m['protected_contacts']:raise AssertionError('veto exposure')
    return rec,decisions,m


def ranker_topology_transfer(rows, model_bytes):
    """Frozen P14 learned ranker vs the P14 transparent comparator on unseen masks.

    Proxy-label agreement, not human interest. P16 masks are new but seeds are
    the same previously analyzed P14 held-out cohort, so not fresh-world proof.
    """
    if base.sha(model_bytes)!=P14_RANKER_SHA:raise ValueError('P14 ranker hash changed')
    state=torch.load(__import__('io').BytesIO(model_bytes),map_location='cpu',weights_only=True)
    if state['human_ratings']!=0 or state['target_type']!='synthetic-protection-completion-proxy':
        raise AssertionError('misidentified P14 ranker')
    ranker=torch.nn.Linear(7,1,bias=False)
    ranker.load_state_dict(state['weights'],strict=True)
    ranker.eval()
    group=defaultdict(dict)
    for r in rows:group[(r['family'],r['seed'],r['layout'])][r['policy']]=r
    pairs=[]
    for (family,seed,layout),g in group.items():
        for pa,pb in itertools.combinations(POLICIES,2):
            a,b=g[pa],g[pb]
            if p14.priority_score(a)==p14.priority_score(b):continue
            pairs.append({'family':family,'seed':seed,'layout':layout,'a':pa,'b':pb,
                'x':[u-v for u,v in zip(p14.rank_features(a),p14.rank_features(b))],
                'target_a_preferred':int(p14.priority_score(a)>p14.priority_score(b)),
                'source_a_trace':a['trace_sha256'],'source_b_trace':b['trace_sha256'],
                'label_type':'P14_deterministic_priority_proxy','human_owner_rating':None})
    return p14.test_ranker(ranker,pairs),pairs


def experiment(source_bytes, out, images=False, p14_ranker_bytes=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    archive,w0,head,cksha=base.open_pinned(source_bytes)
    rows=[];actions=0;exact_pngs=0
    for family in FAMILIES:
        for seed in HELDOUT:
            source,source_sha=None,None # genuinely unseen seeds absent from original source archive
            for layout in LAYOUTS:
                mask=masks(layout)
                mask_sha=base.sha(json.dumps(sorted(mask)).encode())
                for policy in POLICIES:
                    rec,decisions,metrics=rollout(w0,head,seed,family,layout,policy,source)
                    png=w0.png_bytes(rec)
                    if policy=='original_late':
                        if source is not None and png!=archive.read(f'rollouts/{family}-{seed}-intact-late.png'):
                            raise AssertionError('original PNG mismatch')
                        if source is not None: exact_pngs+=1
                    trace=(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n').encode()
                    stem=f'{family}-{seed}-{layout}-{policy}'
                    if images:
                        (out/f'{stem}.png').write_bytes(png)
                        (out/f'{stem}.json').write_bytes(trace)
                    actions+=len(decisions)
                    rows.append({'family':family,'seed':seed,'layout':layout,'policy':policy,
                        'source_trace_sha256':source_sha,'p11_sha256':base.P11_SHA,
                        'w16_head_sha256':cksha,'mask_sha256':mask_sha,
                        'trace_sha256':base.sha(trace),'png_sha256':base.sha(png),
                        'mask_cells':len(mask),'actions_host_legal':all(d['host_legal'] for d in decisions),
                        'forced_stops':sum(d['forced_stop'] for d in decisions),**metrics})
    transfer=None
    if p14_ranker_bytes is not None:
        transfer,transfer_pairs=ranker_topology_transfer(rows,p14_ranker_bytes)
        (out/'P16-proxy-pairs.json').write_text(json.dumps(transfer_pairs,indent=2,sort_keys=True)+'\n')
    group=defaultdict(dict)
    for r in rows:group[(r['family'],r['seed'],r['layout'])][r['policy']]=r
    paired=[]
    for (family,seed,layout),g in sorted(group.items()):
        if set(g)!=set(POLICIES):raise AssertionError('missing policies')
        original=g['original_late'];veto=g['position_veto'];option=g['exit_aware_veto']
        paired.append({'family':family,'seed':seed,'layout':layout,
            'novel_original':original['novel_positions'],
            'novel_veto':veto['novel_positions'],
            'novel_exit_aware':option['novel_positions'],
            'novel_two_ring':g['two_ring_veto']['novel_positions'],
            'protected_original':original['protected_contacts'],
            'protected_veto':veto['protected_contacts'],
            'protected_exit_aware':option['protected_contacts'],
            'protected_two_ring':g['two_ring_veto']['protected_contacts'],
            'branches_veto':veto['branches'],'branches_exit_aware':option['branches'],
            'branches_two_ring':g['two_ring_veto']['branches'],
            'goal_original':original['goal_completed'],'goal_veto':veto['goal_completed'],
            'goal_exit_aware':option['goal_completed'],
            'goal_two_ring':g['two_ring_veto']['goal_completed'],
            'paired_trace_sha256':{p:g[p]['trace_sha256'] for p in POLICIES},
            'human_owner_rating':None,'label_type':'measured_trajectory_outcome_not_preference'})
    def agg(policy):
        a=[r for r in rows if r['policy']==policy]
        return {'count':len(a),'safe_worlds':sum(r['protected_contacts']==0 for r in a),
            'goal_worlds':sum(r['goal_completed'] for r in a),
            'contacts':sum(r['protected_contacts'] for r in a),
            'forced_stops':sum(r['forced_stops'] for r in a),
            'mean_novel':sum(r['novel_positions'] for r in a)/len(a),
            'mean_branches':sum(r['branches'] for r in a)/len(a)}
    wins=sum(p['novel_exit_aware']>p['novel_veto'] for p in paired)
    losses=sum(p['novel_exit_aware']<p['novel_veto'] for p in paired)
    # Conservative resampling unit is the shared seed: three families and
    # three layouts within one seed are correlated matched conditions.
    by_seed=defaultdict(list)
    for p in paired:by_seed[p['seed']].append(p)
    if set(by_seed)!=set(HELDOUT) or any(len(v)!=9 for v in by_seed.values()):
        raise AssertionError('seed-block accounting drift')
    rng=random.Random(15016)
    means=[];goal_differences=[]
    for _ in range(10000):
        block=[p for s in rng.choices(HELDOUT,k=len(HELDOUT)) for p in by_seed[s]]
        means.append(sum(p['novel_exit_aware']-p['novel_veto'] for p in block)/len(block))
        goal_differences.append(sum(int(p['goal_exit_aware'])-int(p['goal_veto']) for p in block)/len(block))
    means.sort();goal_differences.sort()
    uncertainty={'unit':'source seed (all families and layouts resampled together)',
        'seed_blocks':len(HELDOUT),'bootstrap_samples':10000,'bootstrap_seed':15016,
        'exit_aware_minus_veto_mean_novel':sum(p['novel_exit_aware']-p['novel_veto'] for p in paired)/len(paired),
        'novel_95pct_percentile_ci':[means[250],means[9750]],
        'exit_aware_minus_veto_completion_fraction':sum(int(p['goal_exit_aware'])-int(p['goal_veto']) for p in paired)/len(paired),
        'completion_95pct_percentile_ci':[goal_differences[250],goal_differences[9750]]}
    report={'schema':'DRIV3R-P16-mask-topology-transfer/v1',
        'source_p11_sha256':base.P11_SHA,'w16_head_sha256':cksha,
        'environment':'frozen auxiliary W16 in synthetic W0; external position-only mask; no native painting',
        'mask_layouts':LAYOUTS,'seeds':HELDOUT,'families':FAMILIES,
        'model_trajectories':len(rows),'paired_world_layouts':len(paired),
        'head_forwards':actions,'original_png_byte_matches':exact_pngs,
        'unique_trajectories':len({r['trace_sha256'] for r in rows}),
        'by_policy':{p:agg(p) for p in POLICIES},
        'two_ring_minus_veto_mean_novel':sum(p['novel_two_ring']-p['novel_veto'] for p in paired)/len(paired),
        'exit_aware_novelty_wins':wins,'exit_aware_novelty_losses':losses,
        'exit_aware_novelty_ties':len(paired)-wins-losses,
        'seed_block_uncertainty':uncertainty,
        'human_ratings':0,'owner_ratings':0,
        'transferred_P14_ranker':transfer,
        'transferred_P14_ranker_sha256':P14_RANKER_SHA if transfer else None,
        'limitations':['No native protected-cell readback or paint write-set contract',
          'Same seeds under three masks are correlated matched comparisons, not 81 independent worlds',
          'Exit count is one-ply geometric heuristic, not a trained preference model',
          'No pretrained root-language-model inference',
          'A safe destination is not a verified safe paint command']}
    (out/'P16-results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    (out/'P16-matched.json').write_text(json.dumps(paired,indent=2,sort_keys=True)+'\n')
    (out/'P16-index.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    (out/'P16-SHA256SUMS.txt').write_text('\n'.join(f'{base.sha(p.read_bytes())}  {p.name}' for p in sorted(out.iterdir()) if p.is_file())+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--p11-artifact',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--images',action='store_true')
    parser.add_argument('--p14-ranker',type=Path,default=None)
    args=parser.parse_args()
    print(json.dumps(experiment(args.p11_artifact.read_bytes(),args.out,args.images,
        args.p14_ranker.read_bytes() if args.p14_ranker else None),indent=2))
