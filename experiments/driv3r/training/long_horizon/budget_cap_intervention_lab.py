"""BUDGET-CAP-INTERVENTION-001: causal-input audit of archived neural recoveries.

Pure read-only synthetic DEVELOPMENT.  Vary only the synthetic host's
max_recoveries argument (not the traversed state); menus must remain identical.
Historical trained features assumed the fixed 12-credit limit: the recomputed
counterfactual observations are OUT OF DISTRIBUTION and not native CyberSand.
No optimization, no invented action receipts, no sealed HELDOUT access.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, resource, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from . import budget_calibration_lab as cal
from .world import digest
from .train import sha_file

VERSION='budget-cap-intervention/1'
BUDGETS=(1,3,5)
ARMS=cal.ARMS
SEEDS=cal.SEEDS
ORIGINAL_EPISODES_SHA256='e38b26ae35394f051fdc08be547821012f89d62ce9ddc0bebc80d18588a71f33'
EXPECTED_CHECKPOINT_SHA={
 '731-budget_blind.npz':'43436dbd3dd081037e89bd63e823170fcfdc100eacef2f96e1cf60e8974f5d0d',
 '731-calibrated.npz':'466ccd8581e3fce3340d665d94fe132c2230ebd15720e2bc7e672100aa48523c',
 '731-fixed3.npz':'edb8a140a5ba5b2f8da05ef27c49f5bc26892bc39e1503f65f0b3be193c76165',
 '732-budget_blind.npz':'419088d81e7fb3fc425ea8fa2e678815c00ef8e7fcdddbf2add57bb124666921',
 '732-calibrated.npz':'97b0d66364ad4e26b3541626d0073273bd05741723ff18b9fde59582f9fb6195',
 '732-fixed3.npz':'322a0b1c6ef553f0a77301bdaa0a842550d202fadfc7b140cb96c51ba0843c30'}
# Indices pinned to the exact original source's feature layout.
RECOVERY_USED_INDEX=18
REMAINING_INDEX=30
RECOVERY_SLACK_INDEX=34
RAW_BUDGET_INDEX=cal.alias.WIDTH-3


def load_models(folder):
    paths={p.name:p for p in Path(folder).glob('*.npz')}
    if set(paths)!=set(EXPECTED_CHECKPOINT_SHA):
        raise ValueError('Missing/unexpected original checkpoints')
    for n,h in EXPECTED_CHECKPOINT_SHA.items():
        if sha_file(paths[n])!=h:raise ValueError('Original checkpoint SHA mismatch: '+n)
    return {seed:{arm:cal.alias.Ranker.load(paths[f'{seed}-{arm}.npz']) for arm in ARMS}
            for seed in SEEDS}


def intervention(w,goal,arm,remaining):
    """Vary only legal recovery cap; preserve state, history, terrain, painted cells.

    The World.menu(max_recoveries) API supports a counterfactual cap parameter.
    No claim that CyberSand exposes mutable caps at runtime. Base feature
    extractor hardcodes 12; manually recompute only four cap-related features.
    """
    if remaining not in BUDGETS:raise ValueError('invalid remaining budget')
    used=w.recoveries
    cap=used+remaining
    assert 0<used<cap
    base=w.menu(12)
    legal=w.menu(cap)
    if legal!=base:raise ValueError('counterfactual changed legal menu')
    menu,x=cal.matrix(w,goal,arm)
    assert menu==legal
    x=x.copy()
    if arm!='budget_blind':
        x[:len(menu),RECOVERY_USED_INDEX]=used/cap
        x[:len(menu),REMAINING_INDEX]=remaining/cap
        x[:len(menu),RECOVERY_SLACK_INDEX]=(remaining-(len(w.path)-1))/cap
        x[:len(menu),RAW_BUDGET_INDEX]=remaining/cap
    else:
        assert np.all(x[:,[RECOVERY_USED_INDEX,REMAINING_INDEX,RECOVERY_SLACK_INDEX,RAW_BUDGET_INDEX]]==0)
    return menu,x,cap


def select_cases():
    # Original DEVELOPMENT only; identical states are used for all cap probes.
    return tuple(cal.cases('development'))


def run(checkpoint_dir, out, cases=None, original_episodes=None):
    t=time.perf_counter();models=load_models(checkpoint_dir)
    scenarios=select_cases() if cases is None else cases
    rows=[];time_ms=[]
    for c in scenarios:
        w=c['state'];g=c['goal'];pre=w.state_sha();base_menu=w.menu(12)
        if 'BACKTRACK' not in base_menu:raise AssertionError('Missing recovery action')
        if w.logical_moves>64:raise AssertionError('Synthetic 64 logical-move ceiling exceeded')
        for seed in SEEDS:
            for arm in ARMS:
                for b in BUDGETS:
                    menu,x,cap=intervention(w,g,arm,b)
                    start=time.perf_counter_ns();p=models[seed][arm].probs(x,len(menu))
                    time_ms.append((time.perf_counter_ns()-start)/1e6)
                    if abs(p.sum()-1)>1e-8 or not np.isfinite(p).all():raise AssertionError('Bad distribution')
                    chosen=menu[int(p.argmax())]
                    oracle='BACKTRACK' if c['depth']<=b else 'STOP'
                    rows.append({'case':c['case'],'rotation':c['rot']*90,'kind':c['kind'],
                                 'original_budget':c['budget'],'budget':b,'cap':cap,
                                 'depth':c['depth'],'seed':seed,'arm':arm,'state_sha256':pre,
                                 'menu':menu,'probabilities':[round(float(v),10) for v in p],
                                 'backtrack_prob':round(float(p[menu.index('BACKTRACK')]),10),
                                 'chosen':chosen,'oracle':oracle,'correct':chosen==oracle})
        assert w.state_sha()==pre and w.menu(12)==base_menu, 'Mutated baseline state'
    keyed=defaultdict(dict)
    for r in rows:
        k=(r['case'],r['rotation'],r['kind'],r['original_budget'],r['seed'],r['arm'])
        keyed[k][r['budget']]=r
    assert all(set(v)==set(BUDGETS) and len({r['state_sha256'] for r in v.values()})==1
               for v in keyed.values())
    table={};witness=[]
    for arm in ARMS:
        triads=[(k,v) for k,v in keyed.items() if k[-1]==arm]
        inverted=[(k,v) for k,v in triads if any(v[b]['chosen']=='BACKTRACK' and v[b+2]['chosen']=='STOP' for b in (1,3))]
        probinv=[(k,v) for k,v in triads if any(v[b+2]['backtrack_prob']+1e-9 < v[b]['backtrack_prob'] for b in (1,3))]
        table[arm]={'triads':len(triads), 'choice_order_inversions':len(inverted),
                    'backtrack_probability_nonmonotone':len(probinv),
                    'correct_by_budget':{str(b):sum(v[b]['correct'] for _,v in triads) for b in BUDGETS},
                    'chosen_backtrack_by_budget':{str(b):sum(v[b]['chosen']=='BACKTRACK' for _,v in triads) for b in BUDGETS},
                    'false_retreat_by_budget':{str(b):sum(v[b]['chosen']=='BACKTRACK' and v[b]['oracle']=='STOP' for _,v in triads) for b in BUDGETS},
                    'mean_backtrack_probability_by_budget':{str(b):round(statistics.mean(v[b]['backtrack_prob'] for _,v in triads),6) for b in BUDGETS}}
        ordered=[];seen=set()
        for k,v in inverted:
            if (k[0],k[2]) not in seen:
                seen.add((k[0],k[2]));ordered.append((k,v))
            if len(ordered)>=35:break
        witness.extend({'arm':arm,'case':k[0],'rotation':k[1],'kind':k[2],
                        'original_budget':k[3],'seed':k[4],'state_sha256':v[1]['state_sha256'],
                        'depth':v[1]['depth'], 'choices':[v[b]['chosen'] for b in BUDGETS],
                        'backtrack_probs':[v[b]['backtrack_prob'] for b in BUDGETS]}
                        for k,v in ordered)
    # A no-intervention anchor must reconstruct the previous complete census.
    anchor_count=0
    if original_episodes is not None:
        if sha_file(original_episodes)!=ORIGINAL_EPISODES_SHA256:
            raise ValueError('Original DEVELOPMENT episodes SHA mismatch')
        prior=json.loads(Path(original_episodes).read_text())
        original={(r['case'],r['rotation'],r['kind'],r['budget'],r['seed'],r['arm']):r for r in prior}
        anchors=[r for r in rows if r['budget']==r['original_budget']]
        assert len(anchors)==len(prior)==len(original)==2880
        for r in anchors:
            k=(r['case'],r['rotation'],r['kind'],r['budget'],r['seed'],r['arm'])
            old=original[k]
            assert r['state_sha256']==old['state_before'] and r['chosen']==old['first']
            assert r['correct']==old['correct'] and r['oracle']==old['label']
        anchor_count=len(anchors)
    # Split by completely distinct original-state SHA, paired features are unchanged.
    result={'experiment':'BUDGET-CAP-INTERVENTION-001','version':VERSION,
            'source_commit':'cd642b7d319c80526eb97e980e5401d9fb0c74e2',
            'original_checkpoints_sha256':EXPECTED_CHECKPOINT_SHA,
            'original_case_count':len(scenarios),'independent_state_count':len({r['state_sha256'] for r in rows}),
            'probability_vectors':len(rows),'unique_triads':len(keyed),
            'original_episode_anchors_verified':anchor_count,
            'original_episodes_sha256':ORIGINAL_EPISODES_SHA256,
            'model_feature_source_sha256':{name:sha_file(Path(__file__).parent/name) for name in ('world.py','policy.py','history_alias_lab.py','budget_calibration_lab.py','goal_recovery_lab.py')},
            'results':table,'witnesses':witness,
            'latency_model_only_median_ms':round(statistics.median(time_ms),6),
            'latency_model_only_p95_ms':round(float(np.quantile(time_ms,.95)),6),
            'wall_s':round(time.perf_counter()-t,3),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'heldout_opened':False,'native_execution':False,'optimizer_updates':0,
            'game_contract_warning':'Changing max_recoveries is a synthetic World.menu parameter; the native game does not expose this intervention.',
            'OOD_warning':'Historical checkpoints trained under a fixed 12-credit feature normalization; variable caps are out of training distribution.',
            'selection_warning':'STOP denotes task-specific inability to reach a historical goal; not gameplay-wide stop advice.',
            'receipt_warning':'No actions executed in counterfactuals; output vectors are not full action-receipt trajectories.'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'probabilities.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({p.name:sha_file(p) for p in out.glob('*.json') if p.name!='SHA256SUMS.json'},indent=2,sort_keys=True)+'\n')
    return result


def verify(checkpoint_dir,out,original_episodes=None):
    out=Path(out);s=json.loads((out/'SHA256SUMS.json').read_text())
    assert s=={p.name:sha_file(p) for p in out.glob('*.json') if p.name!='SHA256SUMS.json'}
    recorded=json.loads((out/'analysis.json').read_text())
    fresh=run(checkpoint_dir,out/'independent_replay',original_episodes=original_episodes)
    for k in ('results','original_case_count','probability_vectors','unique_triads','witnesses'):
        assert recorded[k]==fresh[k],f'{k} replay divergence'
    assert json.loads((out/'probabilities.json').read_text())==json.loads((out/'independent_replay/probabilities.json').read_text())
    return {'replayed_probability_vectors':fresh['probability_vectors'],'all_source_hashes_unchanged':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--original-episodes',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    print(json.dumps(verify(a.checkpoints,a.output,a.original_episodes) if a.verify else run(a.checkpoints,a.output,original_episodes=a.original_episodes),indent=2))
