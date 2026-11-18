"""GOAL-UTILITY-001: host-legal goal reach vs brush-only counterfactuals.

Restores eight old research checkpoints; no training, game modification, or HELDOUT.
The suffix controllers receive the explicit goal as an experimental observation
feature, not a claim about CyberSand's production/native interface. Both first
choices are actual offered actions. Outcomes are paired from identical states.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time
import zipfile

from . import recovery_debt_audit as previous, retrace_choice_lab as model_input
from .world import MOVE, Spec, specs
from .train import sha_file

VERSION='goal-utility/1'
GOALS={'north_exit':(53,45),'south_spur':(54,50)}
CHOICES=('MOVE_S','BACKTRACK')
ARMS=previous.ARMS
SEEDS=previous.SEEDS
SOURCES=('goal_utility_audit.py','recovery_debt_audit.py','retrace_choice_lab.py',
         'sequential_probe_lab.py','world.py','policy.py','train.py')


def development_specs():
    """Fresh development seeds 3001/3002, 3101/3102 ...; no HELDOUT seeds."""
    return [sp for sp in specs('development',3) if sp.seed%100 in (1,2)]


def rollout(sp, seed, arm, model, goal_name, first, full=True):
    if sp not in development_specs() or seed not in SEEDS or arm not in ARMS or goal_name not in GOALS or first not in CHOICES:
        raise ValueError('unauthorized split, checkpoint, goal, or action')
    q=previous.fixture(sp,11); q['goal']=GOALS[goal_name]; w=q['world']
    if first not in w.menu(12):raise ValueError('first choice not legal')
    initial=w.state_sha(); start_ink=len(w.ink)
    rows=[];lat=[];actions=Counter();goal_step=None
    initial_probs=None; initial_greedy=None
    for i in range(128):
        if w.stopped:break
        menu,x=model_input.inputs(q,128)
        start=time.perf_counter_ns(); p=model.probs(x,len(menu))
        lat.append((time.perf_counter_ns()-start)/1e6)
        dist=[round(float(v),8) for v in p]
        greedy=menu[int(p.argmax())]
        if i==0:initial_greedy=greedy;initial_probs={a:dist[j] for j,a in enumerate(menu)}
        chosen=first if i==0 else greedy
        receipt=w.apply(chosen,12);actions[chosen]+=1
        if w.cursor==GOALS[goal_name] and goal_step is None:goal_step=i+1
        if full:rows.append({'legal':menu,'p':dist,'action':chosen,'event':receipt})
    viable_moves=[a for a in w.menu(12) if a in MOVE]
    # This is immediate actionable branching, NOT a global proof of reachability.
    return {'version':VERSION,'split':sp.split,'spec':sp.__dict__,'seed':seed,'arm':arm,
      'goal':goal_name,'target':GOALS[goal_name],'first':first,
      'initial_sha256':initial,'base_sha256':w.base_sha(),
      'prefix':q['prefix'] if full else None,
      'first_greedy':initial_greedy,'first_probs':initial_probs,
      'goal_reached':goal_step is not None,'goal_step':goal_step,
      'painted':len(w.ink)-start_ink,'stopped':w.stopped,
      'recoveries':w.recoveries,'decisions':sum(actions.values()),
      'remaining_forward_moves':len(viable_moves),
      'end_menu':w.menu(12),'final_cursor':list(w.cursor),
      'final_sha256':w.state_sha(),
      'actions':dict(sorted(actions.items())),
      'mean_latency_ms':round(statistics.mean(lat),6),
      'rows':rows if full else None}


def compact(e):
    return {k:v for k,v in e.items() if k not in ('rows','prefix')}


def compute_matrix(episodes):
    bykey=defaultdict(dict)
    for e in episodes:
        key=(e['spec']['family'],e['spec']['seed'],e['goal'],e['seed'],e['arm'])
        if e['first'] in bykey[key]:raise ValueError('duplicate counterfactual')
        bykey[key][e['first']]=e
    if len(bykey)!=len(development_specs())*len(GOALS)*len(SEEDS)*len(ARMS):raise ValueError('missing cases')
    comparisons=[]
    for key,choice in sorted(bykey.items()):
        if set(choice)!=set(CHOICES):raise ValueError('missing side of pair')
        m,b=(choice[a] for a in CHOICES)
        if m['initial_sha256']!=b['initial_sha256'] or m['first_probs']!=b['first_probs']:
            raise ValueError('counterfactual state/distribution mismatch')
        delta=int(m['goal_reached'])-int(b['goal_reached'])
        time_delta=(m['goal_step'] or 129)-(b['goal_step'] or 129)
        comparisons.append({'family':key[0],'world_seed':key[1],'goal':key[2],
           'model_seed':key[3],'arm':key[4],'goal_success_move_minus_backtrack':delta,
           'painted_move_minus_backtrack':m['painted']-b['painted'],
           'goal_time_move_minus_backtrack':time_delta,
           'greedy_first':m['first_greedy'],'first_probs':m['first_probs'],
           'move_goal_step':m['goal_step'],'backtrack_goal_step':b['goal_step'],
           'move_forward_options_end':m['remaining_forward_moves'],
           'backtrack_forward_options_end':b['remaining_forward_moves'],
           'move_stopped':m['stopped'],'backtrack_stopped':b['stopped']})
    groups={}
    for goal in GOALS:
        cases=[c for c in comparisons if c['goal']==goal]
        records=[e for e in episodes if e['goal']==goal]
        controls={}
        for arm in ARMS:
            these=[c for c in cases if c['arm']==arm]
            controls[arm]={'comparisons':len(these),
              'move_goal_wins':sum(c['goal_success_move_minus_backtrack']>0 for c in these),
              'backtrack_goal_wins':sum(c['goal_success_move_minus_backtrack']<0 for c in these),
              'no_success_difference':sum(c['goal_success_move_minus_backtrack']==0 for c in these),
              'greedy_goals':sum(next(e['goal_reached'] for e in records if
                e['spec']['seed']==c['world_seed'] and e['seed']==c['model_seed'] and
                e['arm']==arm and e['first']==c['greedy_first']) for c in these if c['greedy_first'] in CHOICES),
              'greedy_eligible':sum(c['greedy_first'] in CHOICES for c in these),
              'mean_paint_delta':round(statistics.mean(c['painted_move_minus_backtrack'] for c in these),3),
              'fast_goal_first_choice':sum(c['greedy_first']==('BACKTRACK' if goal=='north_exit' else 'MOVE_S') for c in these),
              'mean_capped_goal_time_delta':round(statistics.mean(c['goal_time_move_minus_backtrack'] for c in these),3)}
        robust=[]
        for sp in development_specs():
            c=[x for x in cases if x['world_seed']==sp.seed]
            deltas=[x['goal_success_move_minus_backtrack'] for x in c]
            times=[x['goal_time_move_minus_backtrack'] for x in c]
            robust.append({'family':sp.family,'world_seed':sp.seed,
              'all_8_move_goal_win':all(x==1 for x in deltas),
              'all_8_backtrack_goal_win':all(x==-1 for x in deltas),
              'mixed':min(deltas)<max(deltas),
              'all_8_backtrack_time_margin8':all(t>=8 for t in times),
              'all_8_move_time_margin8':all(t<=-8 for t in times),
              'goal_time_min_max':[min(times),max(times)],
              'deltas':{
                f"{x['model_seed']}/{x['arm']}":x['goal_success_move_minus_backtrack'] for x in c}})
        groups[goal]={'controls':controls,'robust_worlds':robust,
          'all_8_move_goal_wins':sum(r['all_8_move_goal_win'] for r in robust),
          'all_8_backtrack_goal_wins':sum(r['all_8_backtrack_goal_win'] for r in robust),
          'all_8_backtrack_time_margin8':sum(r['all_8_backtrack_time_margin8'] for r in robust),
          'all_8_move_time_margin8':sum(r['all_8_move_time_margin8'] for r in robust)}
    return comparisons,groups


def summarize(report,episodes):
    comp,groups=compute_matrix(episodes)
    counts=Counter(a for e in episodes for a,n in e['actions'].items() for _ in range(n))
    report['comparisons']=len(comp);report['episodes']=len(episodes)
    report['decisions']=sum(e['decisions'] for e in episodes)
    report['goal_summary']=groups
    report['actions']=dict(sorted(counts.items()))
    report['goal_counts']={g:{first:sum(e['goal_reached'] for e in episodes if e['goal']==g and e['first']==first)
                                  for first in CHOICES} for g in GOALS}
    report['early_stop']={g:{first:sum(e['stopped'] for e in episodes if e['goal']==g and e['first']==first)
                                  for first in CHOICES} for g in GOALS}
    report['mean_inference_latency_ms']=round(statistics.mean(e['mean_latency_ms'] for e in episodes),6)
    report['first_action_probs_sample']=comp[:2]
    return report


def run(archive,out):
    if out.exists():raise FileExistsError('refuse overwrite')
    out.mkdir(parents=True);(out/'episodes').mkdir();begin=time.perf_counter()
    checkpoints={};models={};episodes=[]
    for seed in SEEDS:
        models[seed],checkpoints[str(seed)]=previous.restored(archive,seed)
        for sp in development_specs():
            for g in GOALS:
                for arm in ARMS:
                    for first in CHOICES:
                        e=rollout(sp,seed,arm,models[seed][arm],g,first)
                        filename=f'{len(episodes):03d}.json'
                        (out/'episodes'/filename).write_text(json.dumps(e,sort_keys=True,separators=(',',':'))+'\n')
                        episodes.append(e)
    report={'version':VERSION,'archive_sha256':sha_file(archive),
        'source_sha256':{s:sha_file(Path(__file__).parent/s) for s in SOURCES},
        'checkpoint_sha256':checkpoints,
        'sampling':{'split':'development','fresh_per_family_seeds':[1,2],
                    'goals':{k:list(v) for k,v in GOALS.items()},
                    'policy_seeds':list(SEEDS),'arms':list(ARMS),
                    'initial_actions':list(CHOICES),'horizon':128,'spent_recovery':11},
        'optimizer_updates':0,'paid_compute':False,'heldout_opened':False,
        'native_receipts':0,'goal_claim':'synthetic target occupancy only',
        'wall_s':None,'peak_rss_kib':None}
    summarize(report,episodes)
    report['wall_s']=round(time.perf_counter()-begin,3)
    report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    manifest={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*.json') if p.name!='SHA256SUMS.json'}
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return report


def verify(archive,out):
    r=json.loads((out/'report.json').read_text());manifest=json.loads((out/'SHA256SUMS.json').read_text())
    actual={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*.json') if p.name!='SHA256SUMS.json'}
    if actual!=manifest:raise ValueError('evidence manifest mismatch')
    if r['version']!=VERSION or sha_file(archive)!=r['archive_sha256'] or r['source_sha256']!={s:sha_file(Path(__file__).parent/s) for s in SOURCES}:
        raise ValueError('source/archive drift')
    models={};checkpoints={};episodes=[]
    for seed in SEEDS:
        models[seed],checkpoints[str(seed)]=previous.restored(archive,seed)
    if checkpoints!=r['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
    for file in sorted((out/'episodes').glob('*.json')):
        e=json.loads(file.read_text());sp=Spec(**e['spec'])
        if (sp not in development_specs() or e['seed'] not in SEEDS or e['goal'] not in GOALS or
            e['arm'] not in ARMS or e['first'] not in CHOICES):raise ValueError('incorrect partition')
        q=previous.fixture(sp,11);q['goal']=GOALS[e['goal']];w=q['world'];start_ink=len(w.ink)
        if (e['initial_sha256']!=w.state_sha() or e['prefix']!=q['prefix'] or e['base_sha256']!=w.base_sha()):raise ValueError('prefix mismatch')
        if not e['rows'] or len(e['rows'])>128:raise ValueError('bad transcript length')
        for i,row in enumerate(e['rows']):
            if w.stopped:raise ValueError('actions after STOP')
            menu,x=model_input.inputs(q,128);p=models[e['seed']][e['arm']].probs(x,len(menu));dist=[round(float(v),8) for v in p]
            action=e['first'] if i==0 else menu[int(p.argmax())]
            if row['legal']!=menu or row['p']!=dist or row['action']!=action or row['event']!=w.apply(action,12):
                raise ValueError('model or legal event mismatch')
        actual_hit=next((i+1 for i,v in enumerate(e['rows']) if v['event']['cursor_after']==list(GOALS[e['goal']])),None)
        if (e['goal_step']!=actual_hit or e['goal_reached']!=(actual_hit is not None) or
            e['painted']!=len(w.ink)-start_ink or e['stopped']!=w.stopped or
            e['decisions']!=len(e['rows']) or e['recoveries']!=w.recoveries or
            e['final_sha256']!=w.state_sha() or e['final_cursor']!=list(w.cursor) or
            e['end_menu']!=w.menu(12) or e['remaining_forward_moves']!=sum(a in MOVE for a in w.menu(12))):
            raise ValueError('terminal or goal metric mismatch')
        expected_probs={a:e['rows'][0]['p'][i] for i,a in enumerate(e['rows'][0]['legal'])}
        if e['first_probs']!=expected_probs or e['first_greedy']!=e['rows'][0]['legal'][max(range(len(e['rows'][0]['p'])),key=lambda i:e['rows'][0]['p'][i])]:
            raise ValueError('initial decision mismatch')
        episodes.append(e)
    check={k:v for k,v in r.items() if k not in ('wall_s','peak_rss_kib')}
    verify_report={k:v for k,v in summarize({'version':VERSION},episodes).items() if k not in ('version',)}
    if any(check[k]!=v for k,v in verify_report.items()):raise ValueError('summary mismatch')
    return {'verified_episodes':len(episodes),'verified_decisions':sum(e['decisions'] for e in episodes),
            'manifest_files':len(manifest),'checkpoints':sum(len(x) for x in checkpoints.values())}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint-zip',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--verify',action='store_true')
    args=ap.parse_args()
    print(json.dumps(verify(args.checkpoint_zip,args.out) if args.verify else run(args.checkpoint_zip,args.out),sort_keys=True))
