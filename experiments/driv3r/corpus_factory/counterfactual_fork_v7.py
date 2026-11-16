#!/usr/bin/env python3
"""Synthetic *branch-only* history necessity benchmark; no game-side changes.

Pairs share the same 5x5, goal delta and legal menu at a recovery fork.
An executed dead-end excursion changes only the host-observed path memory.
This intentionally does NOT assess general worm navigation or native ABI parity.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from collections import Counter, deque
import numpy as np
from sklearn.linear_model import SGDClassifier

SCHEMA = 'driv3r/synthetic-counterfactual-fork-v7'
SEEDS = {'TRAIN': range(10000, 10256), 'DEV': range(20000, 20064),
         'HELDOUT': range(30000, 30256)}
HUB = (8, 12)
START = (5, 12)
GOAL = (21, 12)
ACTIONS = ('LEFT', 'RIGHT')
EPOCHS = 24


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def corridor(seed, good):
    """Two indistinguishable hub entrances, one distant dead end and one goal exit."""
    if good not in ACTIONS:
        raise ValueError('bad good arm')
    limit = 12 + seed % 3
    open_cells = {(x, 12) for x in range(5, 9)} | {(x, 12) for x in range(16, 22)}
    for arm, y in [('LEFT', 6), ('RIGHT', 18)]:
        dy = 1 if arm == 'RIGHT' else -1
        for j in range(7):
            open_cells.add((8, 12 + j * dy))
        endpoint = 16 if arm == good else limit
        for x in range(8, endpoint + 1):
            open_cells.add((x, y))
        if arm == good:
            for yy in range(min(y, 12), max(y, 12) + 1):
                open_cells.add((16, yy))
    # Protected material is a fixed, disjoint mask: no legal path may overwrite it.
    protected = {(10 + seed % 3, 10), (11 + seed % 3, 14)}
    # World-specific remote protected-material field, outside the playable fork.
    rng = random.Random(seed)
    while len(protected) < 10:
        protected.add((rng.randrange(23, 32), rng.randrange(1, 22)))
    if protected & open_cells:
        raise AssertionError('protected corridor')
    world = {'seed': seed, 'good_arm': good, 'open': open_cells,
             'protected': protected, 'bad_endpoint': limit}
    return world


def route_to_dead_end(world):
    bad = 'RIGHT' if world['good_arm'] == 'LEFT' else 'LEFT'
    yy = 18 if bad == 'RIGHT' else 6
    limit = world['bad_endpoint']
    outward = [START, (6, 12), (7, 12), HUB]
    outward += [(8, y) for y in (range(13, 19) if bad == 'RIGHT' else range(11, 5, -1))]
    outward += [(x, yy) for x in range(9, limit + 1)]
    return outward + outward[-2::-1][:-3]  # return to HUB, not past it


def validate_path(world, path):
    if not path or path[0] != START or path[-1] != HUB:
        raise AssertionError('prelude must start at START, end at HUB')
    for a, b in zip(path, path[1:]):
        if abs(a[0]-b[0])+abs(a[1]-b[1]) != 1:
            raise AssertionError('nonadjacent host move')
        if b not in world['open'] or b in world['protected']:
            raise AssertionError('illegal or protected prelude move')


def example(seed, good, split):
    world = corridor(seed, good)
    path = route_to_dead_end(world)
    validate_path(world, path)
    local = [''.join('.' if (HUB[0]+dx, HUB[1]+dy) in world['open']
                     else 'P' if (HUB[0]+dx, HUB[1]+dy) in world['protected']
                     else '#' for dx in range(-2, 3)) for dy in range(-2, 3)]
    visits = Counter(path)
    adjacent = {'LEFT': (8, 11), 'RIGHT': (8, 13)}
    memory = [visits[adjacent[a]] for a in ACTIONS]
    if sorted(memory) != [0, 2]:
        raise AssertionError(f'executed history not asymmetric: {memory}')
    geometry = {'open': sorted(map(list, world['open'])),
                'protected': sorted(map(list, world['protected']))}
    return {'split': split, 'scenario_id': f'fork:{seed}:{good}',
            'paired_group_id': f'fork_pair:{seed}', 'world_sha256': digest(geometry),
            'source': 'independent-synthetic-v7', 'difficulty': 'B2-distant-dead-end',
            'observation': {'local_5x5': local, 'goal_delta': [13, 0],
                            'heading': 'E', 'legal': ['LEFT', 'RIGHT', 'RETRACE', 'STOP'],
                            'candidate_action_subset': list(ACTIONS)},
            'host_observed_memory': {'next_cell_visits': dict(zip(ACTIONS, memory)),
                                     'prior_executed_steps': len(path)-1},
            'teacher_action': good,
            'hard_negative': ('RIGHT' if good == 'LEFT' else 'LEFT'),
            'execution': 'synthetic-host-applied-prelude', 'native_receipt': False,
            'prelude_trace_sha256': digest(path), 'prelude_positions': path}


def check_partitions(rows):
    membership = {}
    world_ids = set()
    for r in rows:
        key = r['paired_group_id'];split = r['split']
        if key in membership and membership[key] != split:
            raise ValueError('group split collision')
        membership[key] = split
        if r['world_sha256'] in world_ids:
            raise ValueError('geometry leakage')
        world_ids.add(r['world_sha256'])
        if r['teacher_action'] not in r['observation']['legal']:
            raise ValueError('illegal teacher')
    for split in SEEDS:
        ss = [r for r in rows if r['split'] == split]
        if len(ss) != len(SEEDS[split])*2:
            raise ValueError(f'wrong number of {split} rows')
        by_pair = {}
        for r in ss:
            by_pair.setdefault(r['paired_group_id'], []).append(r)
        for group, pair in by_pair.items():
            if len(pair) != 2:
                raise ValueError('unpaired worlds')
            if pair[0]['observation'] != pair[1]['observation']:
                raise ValueError(f'local alias broken {group}')
            if pair[0]['teacher_action'] == pair[1]['teacher_action']:
                raise ValueError('pair labels do not conflict')
            if pair[0]['host_observed_memory'] == pair[1]['host_observed_memory']:
                raise ValueError('no counterfactual memory difference')


def features(rows, mode):
    """Capacity-matched: two numeric history slots in both arms.

    Sham features are keyed ONLY by the paired seed, identical in both halves.
    Thus they may have variance but cannot identify the true arm in a pair.
    """
    x=[]
    for r in rows:
        obs=r['observation'];v=r['host_observed_memory']['next_cell_visits']
        seed=int(r['paired_group_id'].split(':')[-1])
        if mode=='memory':
            hist=[float(v['LEFT'])/2, float(v['RIGHT'])/2]
        elif mode=='sham':
            hist=[float((seed*17+3)%7)/6, float((seed*29+5)%11)/10]
        elif mode=='swap':
            hist=[float(v['RIGHT'])/2, float(v['LEFT'])/2]
        else:
            raise ValueError('unknown feature arm')
        flat=[float(c=='.') for line in obs['local_5x5'] for c in line]
        x.append(flat+[obs['goal_delta'][0]/32, obs['goal_delta'][1]/32]+hist)
    return np.array(x, dtype=np.float64)


def train(rows, mode):
    x=features(rows, mode); y=np.array([int(r['teacher_action']=='LEFT') for r in rows])
    model=SGDClassifier(loss='log_loss', max_iter=EPOCHS, tol=None, random_state=37,
                        alpha=0.001, shuffle=True)
    model.fit(x,y)
    return model


def branch_reaches_goal(world, arm):
    """After host applies one candidate step, is goal reachable without resetting at HUB?"""
    next_pos=(8, 11) if arm=='LEFT' else (8, 13)
    if next_pos not in world['open'] or next_pos in world['protected']:
        raise AssertionError('illegal chosen fork action')
    queue=deque([next_pos]);seen={next_pos,HUB}
    while queue:
        x,y=queue.popleft()
        if (x,y)==GOAL:return True
        for n in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if n in world['open'] and n not in world['protected'] and n not in seen:
                seen.add(n);queue.append(n)
    return False


def assess(rows, model, mode):
    x=features(rows, mode); score=model.decision_function(x)
    # >0 breaks ties toward RIGHT; equal-state pairs always return one of each.
    predictions=['LEFT' if p>0 else 'RIGHT' for p in score]
    correct=sum(r['teacher_action']==p for r,p in zip(rows,predictions))
    continuation_success=0
    for r,p in zip(rows,predictions):
        world=corridor(int(r['scenario_id'].split(':')[1]),r['teacher_action'])
        outcome=branch_reaches_goal(world,p)
        if outcome != (p==r['teacher_action']):
            raise AssertionError('counterfactual path validity failed')
        continuation_success+=int(outcome)
    pair_results={}
    for r,p in zip(rows,predictions):
        pair_results.setdefault(r['paired_group_id'],[]).append(int(r['teacher_action']==p))
    if any(len(v)!=2 for v in pair_results.values()):raise AssertionError('unpaired scoring')
    return {'examples':len(rows),'independent_pairs':len(pair_results),
            'correct':correct,'accuracy':correct/len(rows),
            'chosen_branch_has_goal_path':continuation_success,
            'both_correct_pairs':sum(sum(v)==2 for v in pair_results.values()),
            'one_correct_pairs':sum(sum(v)==1 for v in pair_results.values()),
            'neither_correct_pairs':sum(sum(v)==0 for v in pair_results.values())}


def run(output):
    rows=[example(seed, good, split) for split,seeds in SEEDS.items()
          for seed in seeds for good in ACTIONS]
    check_partitions(rows)
    train_rows=[r for r in rows if r['split']=='TRAIN']
    held_rows=[r for r in rows if r['split']=='HELDOUT']
    dev_rows=[r for r in rows if r['split']=='DEV']
    sham=train(train_rows,'sham');memory=train(train_rows,'memory')
    report={'schema':SCHEMA,'source':'novel-independent-synthetic-host',
            'data':{split:{'rows':len([r for r in rows if r['split']==split]),
                            'world_groups':len(SEEDS[split])} for split in SEEDS},
            'training':{'optimizer':'sklearn.SGDClassifier(log_loss)',
                        'epochs':EPOCHS,'seeds_train_only':True,
                        'feature_dimension':features(train_rows,'memory').shape[1]},
            'DEV':{'sham':assess(dev_rows,sham,'sham'),
                   'memory':assess(dev_rows,memory,'memory')},
            'HELDOUT':{'sham':assess(held_rows,sham,'sham'),
                       'memory':assess(held_rows,memory,'memory'),
                       'memory_permuted_at_inference':assess(held_rows,memory,'swap')},
            'safety':{'host_legal_choices':len(rows),'protected_entry':0,
                      'canonical_native_abi_verified':False,'game_files_changed':0},
            'caveats':['Branch-choice-only conditional test, not autonomous navigation',
                       'The two labels in every pair are intentionally balanced',
                       'Synthetic host and scripted executed prelude, not native receipt',
                       'Held-out seeds but same procedural generator family',
                       'Do not interpret as native CyberSand or full agent competence']}
    report['coefficients_sha256']={mode:hashlib.sha256(model.coef_.tobytes()+model.intercept_.tobytes()).hexdigest()
                                   for mode,model in [('sham',sham),('memory',memory)]}
    output=Path(output)
    if output.exists():raise FileExistsError('immutable output directory exists')
    output.mkdir(parents=True)
    train_export=[]
    for r in train_rows:
        prompt={'observation':r['observation'],'host_observed_memory':r['host_observed_memory']}
        train_export.append({'messages':[{'role':'system','content':'Select a host-legal branch action, LEFT or RIGHT, after an observed dead-end excursion in an external synthetic corridor.'},
                                          {'role':'user','content':json.dumps(prompt,sort_keys=True)},
                                          {'role':'assistant','content':r['teacher_action']}],
                             'provenance':{k:r[k] for k in ('split','scenario_id','paired_group_id','world_sha256','prelude_trace_sha256','source','difficulty','native_receipt')},
                             'hard_negative':r['hard_negative']})
    payload=''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in train_export).encode()
    (output/'train_only.jsonl').write_bytes(payload)
    report['train_sha256']=hashlib.sha256(payload).hexdigest()
    report['train_rows']=len(train_export)
    (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (output/'samples.json').write_text(json.dumps([{
        'scenario_id':r['scenario_id'], 'world_sha256':r['world_sha256'],
        'prelude_positions':r['prelude_positions'], 'memory':r['host_observed_memory'],
        'teacher_action':r['teacher_action']} for r in held_rows[:8]],sort_keys=True,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(run(args.out)['HELDOUT'],sort_keys=True,indent=2))
