"""GOAL-NATIVE-GATE-001: matched trained spatial-goal rankers under 64 synthetic logical moves.

External 64-move censor models a *historical* CyberSand limit; no native runtime
or changed World.action menu. All evaluator goals are source-route waypoints,
not the training teacher's radial targets. No sealed structural HELDOUT.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import resource
import statistics
import time

from .goal_conditioning_lab import GoalRanker, train
from .goal_observability_lab import source_route, shortest_path_action, strict_step
from .policy import random_legal_persistent
from .train import sha_file
from .world import World, MOVE, digest, replay, specs

VERSION = 'goal-native-gate/1'
SOURCE_FILES = ('world.py', 'policy.py', 'train.py',
                'goal_conditioning_lab.py', 'goal_observability_lab.py',
                'goal_native_gate_lab.py')
SEEDS = (711, 712)
ARMS = ('conditioned', 'blind', 'goal_oracle', 'random_persistent')


def episode(sp, targets, models, arm, seed, perturb):
    if sp.split != 'development' or arm not in ARMS:
        raise ValueError('Only DEVELOPMENT evaluation and known arms')
    world = World.create(sp)
    rng = random.Random(sp.seed * 481 + int(perturb) * 103)
    goal_idx = 0
    arrivals = []
    distribution = []
    latencies = []
    interventions = []
    stop = '64_logical_gate'
    for tick in range(128):
        if goal_idx == len(targets):
            stop = 'all_goals'; break
        if world.stopped:
            stop = 'policy_stop'; break
        if world.logical_moves >= 64:
            stop = '64_logical_gate'; break
        goal = targets[goal_idx]
        if world.cursor == goal:
            arrivals.append([goal_idx, tick, world.logical_moves]); goal_idx += 1
            continue
        st = time.perf_counter_ns()
        if arm in ('conditioned', 'blind'):
            proposed, probs = models[arm].choose(world, goal, 64, arm == 'blind')
        elif arm == 'goal_oracle':
            proposed, probs = shortest_path_action(world, goal), None
        else:
            proposed, probs = random_legal_persistent(world, rng), None
        latencies.append((time.perf_counter_ns() - st) / 1e6)
        actual = proposed
        if perturb and tick in (7, 22):
            alternatives = sorted(a for a in world.menu(12) if a in MOVE and a != proposed)
            if alternatives:
                actual = alternatives[-1]
            interventions.append([tick, proposed, actual])
        if probs is not None:
            distribution.append({'step': tick, 'goal_index': goal_idx,
                                 'goal': list(goal), 'pre_sha256': world.state_sha(),
                                 'legal': world.menu(12), 'proposed': proposed,
                                 'probabilities': probs})
        receipt = strict_step(world, actual)
        if receipt is None:
            stop = '64_logical_gate'; break
        if actual == 'STOP':
            stop = 'policy_stop'; break
        if world.cursor == goal:
            arrivals.append([goal_idx, tick + 1, world.logical_moves]); goal_idx += 1
    else:
        stop = '128_decision_safety_gate'
    if replay(sp, world.decisions, 12).state_sha() != world.state_sha():
        raise AssertionError('Host-legal replay mismatch')
    actions = dict(sorted(Counter(e['chosen'] for e in world.decisions).items()))
    result = {'spec': sp.ident, 'family': sp.family, 'seed': seed, 'arm': arm,
              'perturb': perturb, 'world_base_sha256': world.base_sha(),
              'targets': [list(v) for v in targets], 'arrivals': arrivals,
              'goals_reached': len(arrivals), 'goals_offered': len(targets),
              'decisions': world.steps, 'logical_moves': world.logical_moves,
              'painted': len(world.ink), 'recoveries': world.recoveries,
              'stop': stop, 'actions': actions,
              'protected_material_edits': len(world.ink & (set(world.materials) | world.protected)),
              'synthetic_relocations': actions.get('RELOCATE_START', 0),
              'trace_sha256': digest(world.decisions),
              'final_state_sha256': world.state_sha(),
              'median_choice_ms': round(statistics.median(latencies), 6) if latencies else None}
    trace = {'result': result, 'events': world.decisions,
             'distributions': distribution, 'interventions': interventions,
             'spec': {'split': sp.split, 'family': sp.family, 'seed': sp.seed,
                      'mode': sp.mode, 'challenge': sp.challenge}}
    return result, trace


def verify_trace(trace, models):
    from .world import Spec
    r = trace['result']; sp = Spec(**trace['spec'])
    w = World.create(sp)
    assert sp.ident == r['spec'] and w.base_sha() == r['world_base_sha256']
    goals = [tuple(g) for g in r['targets']]
    check, original_world = source_route(sp)
    assert check == goals and original_world.base_sha() == w.base_sha()
    rng = random.Random(sp.seed * 481 + int(r['perturb']) * 103)
    arrivals = []
    goal_idx = 0
    events = iter(trace['events']); probabilities = iter(trace['distributions'])
    interventions = []
    for tick in range(128):
        if goal_idx == len(goals): stop = 'all_goals'; break
        if w.stopped: stop = 'policy_stop'; break
        if w.logical_moves >= 64: stop = '64_logical_gate'; break
        goal = goals[goal_idx]
        if w.cursor == goal:
            arrivals.append([goal_idx, tick, w.logical_moves]); goal_idx += 1; continue
        if r['arm'] in ('conditioned', 'blind'):
            proposed, probs = models[r['arm']].choose(w, goal, 64, r['arm'] == 'blind')
        elif r['arm'] == 'goal_oracle':
            proposed, probs = shortest_path_action(w, goal), None
        else:
            proposed, probs = random_legal_persistent(w, rng), None
        actual = proposed
        if r['perturb'] and tick in (7, 22):
            alt = sorted(a for a in w.menu(12) if a in MOVE and a != proposed)
            if alt: actual = alt[-1]
            interventions.append([tick, proposed, actual])
        if probs is not None:
            p = next(probabilities)
            assert p == {'step': tick, 'goal_index': goal_idx, 'goal': list(goal),
                         'pre_sha256': w.state_sha(), 'legal': w.menu(12),
                         'proposed': proposed, 'probabilities': probs}
        cost = len(w.path) - 1 if actual == 'RELOCATE_START' else 0 if actual == 'STOP' else 1
        if w.logical_moves + cost > 64:
            stop = '64_logical_gate'; break
        ev = next(events)
        assert ev == w.apply(actual, 12)
        if actual == 'STOP': stop = 'policy_stop'; break
        if w.cursor == goal:
            arrivals.append([goal_idx, tick + 1, w.logical_moves]); goal_idx += 1
    else:
        stop = '128_decision_safety_gate'
    assert next(events, None) is None and next(probabilities, None) is None
    assert interventions == trace['interventions']
    assert arrivals == r['arrivals'] and stop == r['stop']
    assert w.steps == r['decisions'] and w.logical_moves == r['logical_moves']
    assert w.state_sha() == r['final_state_sha256'] and digest(w.decisions) == r['trace_sha256']
    assert len(w.ink & (set(w.materials) | w.protected)) == 0
    return w.steps, len(trace['distributions'])


def summarize(rows):
    out = {}
    for arm in ARMS:
        rr = [r for r in rows if r['arm'] == arm]
        count = Counter()
        for row in rr: count.update(row['actions'])
        lats = [r['median_choice_ms'] for r in rr if r['median_choice_ms'] is not None]
        out[arm] = {'episodes': len(rr),
                    'goals_reached': sum(r['goals_reached'] for r in rr),
                    'goals_offered': sum(r['goals_offered'] for r in rr),
                    'all_goals': sum(r['goals_reached'] == r['goals_offered'] for r in rr),
                    'early_stops': sum(r['stop'] == 'policy_stop' for r in rr),
                    '64_gate': sum(r['stop'] == '64_logical_gate' for r in rr),
                    'synthetic_relocations': sum(r['synthetic_relocations'] for r in rr),
                    'mean_painted': round(statistics.mean(r['painted'] for r in rr), 3),
                    'mean_decisions': round(statistics.mean(r['decisions'] for r in rr), 3),
                    'median_episode_choice_ms': round(statistics.median(lats), 6),
                    'actions': dict(sorted(count.items()))}
    return out


def sha_tree(out):
    return {str(p.relative_to(out)): sha_file(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name != 'SHA256SUMS.json'}


def run(out, updates=40):
    if out.exists(): raise FileExistsError('Immutable evidence destination exists')
    if not 1 <= updates <= 100: raise ValueError('Optimizer bound')
    out.mkdir(parents=True)
    (out/'checkpoints').mkdir(); (out/'traces').mkdir()
    started = time.perf_counter()
    checkpoint_rows = {}
    output_rows = []
    total_verified_decisions = 0
    total_verified_distributions = 0
    spec_list = specs('development', 3)  # 12 source-generated DEVELOPMENT worlds
    for seed in SEEDS:
        models, history = train(seed, updates)
        for arm in ('conditioned', 'blind'):
            cp = out/'checkpoints'/f'{seed}-{arm}.npz'
            models[arm].save(cp)
            restored = GoalRanker.load(cp)
            assert restored.updates == 3*updates
            assert all((restored.p[k] == models[arm].p[k]).all() for k in restored.p)
            checkpoint_rows[f'{seed}-{arm}'] = {
                'sha256': sha_file(cp), 'optimizer_updates': restored.updates,
                'parameters': sum(p.size for p in restored.p.values())}
            models[arm] = restored
        checkpoint_rows[f'{seed}-training_stages'] = history
        for sp in spec_list:
            targets, refworld = source_route(sp)
            for perturb in (False, True):
                for arm in ARMS:
                    result, trace = episode(sp, targets, models, arm, seed, perturb)
                    n, p = verify_trace(trace, models)
                    total_verified_decisions += n; total_verified_distributions += p
                    output_rows.append(result)
                    name = f'{seed}-{sp.family}-{sp.seed}-{int(perturb)}-{arm}.json'
                    (out/'traces'/name).write_text(json.dumps(trace, sort_keys=True, separators=(',', ':'))+'\n')
    summary = summarize(output_rows)
    evidence = {'version': VERSION, 'split': 'development',
                'remote_source_head_before_edit': '9420cc6cae9c316c6efe76fe1532817d324f2f41',
                'source_sha256': {name:sha_file(Path(__file__).parent/name) for name in SOURCE_FILES},
                'source_git_sha1': {}, 'training_seeds': list(SEEDS),
                'optimizer_updates_per_stage': updates,
                'checkpoint_identities': checkpoint_rows,
                'development_worlds': [sp.ident for sp in spec_list],
                'structural_heldout_opened': False, 'native_host_executed': False,
                'native_action_receipts': 0, 'external_64_logical_move_censor': True,
                'episodes': len(output_rows), 'independently_replayed_actions': total_verified_decisions,
                'independently_recomputed_probabilities': total_verified_distributions,
                'summary': summary, 'per_episode': output_rows,
                'wall_s': round(time.perf_counter()-started, 3),
                'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'no_paid_compute': True,
                'caveats': ['TRAIN and DEVELOPMENT share geometry families; structural HELDOUT sealed.',
                            'Source-derived goal waypoints may become inaccessible after wrong moves.',
                            '64-move gate is synthetic, not a native runtime qualification.',
                            'RELOCATE_START remains a synthetic unvalidated action in World; report separately.',
                            'Engineered Manhattan features are supplied explicitly to goal-conditioned model.']}
    (out/'evidence.json').write_text(json.dumps(evidence, sort_keys=True, indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps(sha_tree(out), sort_keys=True, indent=2)+'\n')
    return evidence


def verify(out):
    evidence = json.loads((out/'evidence.json').read_text())
    expected = json.loads((out/'SHA256SUMS.json').read_text())
    assert sha_tree(out) == expected
    assert evidence['version'] == VERSION
    assert evidence['source_sha256'] == {n: sha_file(Path(__file__).parent/n) for n in SOURCE_FILES}
    n=p=0; rows=[]
    for seed in SEEDS:
        models = {a: GoalRanker.load(out/'checkpoints'/f'{seed}-{a}.npz')
                  for a in ('conditioned','blind')}
        for arm in models:
            cp = out/'checkpoints'/f'{seed}-{arm}.npz'
            meta = evidence['checkpoint_identities'][f'{seed}-{arm}']
            assert meta['sha256'] == sha_file(cp) and meta['optimizer_updates'] == models[arm].updates
        for f in sorted((out/'traces').glob(f'{seed}-*.json')):
            trace=json.loads(f.read_text()); a,b=verify_trace(trace, models)
            n+=a; p+=b; rows.append(trace['result'])
    assert n == evidence['independently_replayed_actions']
    assert p == evidence['independently_recomputed_probabilities']
    assert summarize(rows) == evidence['summary']
    return {'episodes':len(rows),'decisions':n,'probabilities':p,'files':len(expected)}


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', type=Path)
    ap.add_argument('--verify', type=Path)
    ap.add_argument('--updates',type=int,default=40)
    args=ap.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.updates),sort_keys=True))
