"""DETOUR-RECOVERY-001: frozen goal heads after identical forced tunnel mistakes.

DEVELOPMENT-only synthetic `trap_corridor` variant; checkpoint hashes pinned in
external archive. Complete shortest path only within *monotone-prefix recovery*
class: first unwind 0..N history edges, then travel over unpainted empty cells.
This is a lower-bound/feasible comparator, NOT a general optimal native planner.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

from .goal_conditioning_lab import GoalRanker
from .goal_observability_lab import shortest_path_action, strict_step
from .world import World, MOVE, Spec, digest, replay, specs
from .train import sha_file

VERSION = 'detour-recovery/1'
SEEDS = (711, 712)
ARMS = ('conditioned', 'blind', 'legacy_bfs', 'prefix_search')
GOALS = ((60, 48), (55, 56))
DEPTHS = (3, 5)
SOURCE = ('detour_recovery_lab.py', 'world.py', 'goal_conditioning_lab.py',
          'goal_observability_lab.py', 'policy.py', 'train.py')


def fixture(spec: Spec, depth: int) -> World:
    if spec.split != 'development' or spec.challenge != 'trap_corridor' or depth not in DEPTHS:
        raise ValueError('DEVELOPMENT-only fixed corridor intervention')
    w = World.create(spec)
    for _ in range(depth):
        assert 'MOVE_E' in w.menu(12)
        assert strict_step(w, 'MOVE_E') is not None
    assert w.cursor == (48 + depth, 48) and w.logical_moves == depth
    return w


def shortest_empty_path(w: World, start: tuple[int, int], goal: tuple[int, int]):
    """Exhaustive 4-neighbour BFS through currently empty cells, no painted crossing."""
    if w.occupied(goal): return None
    q = deque([start])
    prev = {start: None}
    while q:
        pos = q.popleft()
        if pos == goal:
            path = []
            while prev[pos] is not None:
                parent, action = prev[pos]
                path.append(action)
                pos = parent
            return tuple(reversed(path))
        for action, (dx, dy) in MOVE.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt not in prev and not w.occupied(nxt):
                prev[nxt] = (pos, action)
                q.append(nxt)
    return None


def prefix_plan(w: World, goal: tuple[int, int], cap: int = 64):
    """Exact least-cost plan among backtrack-prefix then empty-cell paths.

    Does not assume that painting an already visited historical cell is legal.
    Does not inspect unknown future native state, and never proposes relocation.
    """
    legal = w.menu(12)
    if w.cursor == goal: return ()
    if w.stopped: return None
    allowed_backtracks = min(12 - w.recoveries, len(w.path) - 1)
    candidates = []
    for k in range(allowed_backtracks + 1):
        if k and 'BACKTRACK' not in legal: break
        point = w.path[-k-1]
        path = shortest_empty_path(w, point, goal)
        if path is not None and w.logical_moves + k + len(path) <= cap:
            candidates.append((k + len(path), k, path))
    if not candidates: return None
    _, k, path = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return ('BACKTRACK',) * k + path


def step_action(w: World, goal, arm, model):
    if arm in ('conditioned', 'blind'):
        return model.choose(w, goal, 64, arm == 'blind')
    if arm == 'legacy_bfs':
        return shortest_path_action(w, goal), None
    if arm == 'prefix_search':
        plan = prefix_plan(w, goal)
        return (plan[0] if plan else 'STOP'), None
    raise ValueError('Unknown arm')


def episode(sp: Spec, depth: int, goal, arm: str, seed: int, model):
    w = fixture(sp, depth)
    initial = w.state_sha()
    initial_events = list(w.decisions)
    samples, latencies = [], []
    stop = '64_logical_gate'
    for tick in range(64):
        if w.cursor == goal:
            stop = 'goal_arrived'; break
        if w.stopped:
            stop = 'policy_stop'; break
        if w.logical_moves >= 64:
            stop = '64_logical_gate'; break
        now = time.perf_counter_ns()
        proposed, probabilities = step_action(w, goal, arm, model)
        latencies.append((time.perf_counter_ns() - now) / 1e6)
        if proposed not in w.menu(12):
            raise AssertionError('Controller proposed illegal action')
        if probabilities is not None:
            samples.append({'tick': tick, 'pre_sha256': w.state_sha(), 'legal': w.menu(12),
                            'chosen': proposed, 'probabilities': probabilities})
        if strict_step(w, proposed) is None:
            stop = '64_logical_gate'; break
    else:
        stop = '64_decision_safety_gate'
    assert replay(sp, w.decisions, 12).state_sha() == w.state_sha()
    actions = dict(sorted(Counter(e['chosen'] for e in w.decisions[len(initial_events):]).items()))
    row = {'spec': sp.ident, 'family': sp.family, 'world_seed': sp.seed,
           'depth': depth, 'goal': list(goal), 'seed': seed, 'arm': arm,
           'initial_state_sha256': initial, 'base_sha256': w.base_sha(),
           'arrived': w.cursor == goal, 'stop': stop,
           'prelude_moves': len(initial_events), 'continuation_actions': len(w.decisions) - len(initial_events),
           'total_logical_moves': w.logical_moves, 'recovery_actions': actions.get('BACKTRACK', 0),
           'synthetic_relocations': actions.get('RELOCATE_START', 0),
           'protected_material_edits': len(w.ink & (set(w.materials) | w.protected)),
           'action_counts': actions, 'final_cursor': list(w.cursor),
           'final_sha256': w.state_sha(), 'trace_sha256': digest(w.decisions),
           'median_choose_ms': round(statistics.median(latencies), 6) if latencies else None}
    trace = {'row': row, 'spec': {'split': sp.split, 'family': sp.family, 'seed': sp.seed,
            'mode': sp.mode, 'challenge': sp.challenge}, 'prelude': initial_events,
            'events': w.decisions[len(initial_events):], 'probabilities': samples}
    return row, trace


def verify_trace(trace, model):
    sp = Spec(**trace['spec'])
    row = trace['row']; goal = tuple(row['goal'])
    w = fixture(sp, row['depth'])
    assert trace['prelude'] == w.decisions and row['initial_state_sha256'] == w.state_sha()
    assert w.base_sha() == row['base_sha256']
    probs = iter(trace['probabilities'])
    continuation = iter(trace['events'])
    for tick in range(64):
        if w.cursor == goal:
            stop = 'goal_arrived'; break
        if w.stopped:
            stop = 'policy_stop'; break
        if w.logical_moves >= 64:
            stop = '64_logical_gate'; break
        proposed, p = step_action(w, goal, row['arm'], model)
        assert proposed in w.menu(12)
        if p is not None:
            sample = next(probs)
            assert sample == {'tick': tick, 'pre_sha256': w.state_sha(),
                              'legal': w.menu(12), 'chosen': proposed, 'probabilities': p}
        cost = len(w.path)-1 if proposed=='RELOCATE_START' else 0 if proposed=='STOP' else 1
        if w.logical_moves + cost > 64:
            stop = '64_logical_gate'; break
        ev = next(continuation)
        assert ev == w.apply(proposed, 12)
    else:
        stop = '64_decision_safety_gate'
    assert next(probs, None) is None and next(continuation, None) is None
    assert stop == row['stop'] and (w.cursor == goal) == row['arrived']
    assert w.state_sha() == row['final_sha256'] and digest(w.decisions) == row['trace_sha256']
    assert w.logical_moves == row['total_logical_moves'] and len(w.decisions)-row['depth'] == row['continuation_actions']
    assert len(w.ink & (set(w.materials) | w.protected)) == 0
    return row['continuation_actions'], len(trace['probabilities'])


def summarize(rows):
    output = {}
    for arm in ARMS:
        rr = [r for r in rows if r['arm'] == arm]
        counts = Counter()
        for r in rr: counts.update(r['action_counts'])
        times = [r['median_choose_ms'] for r in rr if r['median_choose_ms'] is not None]
        output[arm] = {'episodes': len(rr), 'arrivals': sum(r['arrived'] for r in rr),
                       'all_replay_legal': all(not r['protected_material_edits'] for r in rr),
                       'mean_continuation_actions': round(statistics.mean(r['continuation_actions'] for r in rr), 3),
                       'mean_final_logical_moves': round(statistics.mean(r['total_logical_moves'] for r in rr), 3),
                       'median_episode_choice_ms': round(statistics.median(times), 6) if times else None,
                       'action_counts': dict(sorted(counts.items()))}
    return output


def run(out: Path, checkpoint_dir: Path):
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    (out/'traces').mkdir()
    t = time.perf_counter()
    rows = []
    sha = {}
    receipts = vectors = 0
    for seed in SEEDS:
        cp = checkpoint_dir/f'{seed}-conditioned.npz'
        cpblind = checkpoint_dir/f'{seed}-blind.npz'
        for arm, path in [('conditioned',cp),('blind',cpblind)]:
            sha[f'{seed}-{arm}'] = {'sha256': sha_file(path), 'parameters': 769}
        models = {'conditioned': GoalRanker.load(cp), 'blind': GoalRanker.load(cpblind)}
        assert models['conditioned'].updates == models['blind'].updates == 120
        for original in specs('development', 3):
            sp = Spec(original.split, original.family, original.seed, original.mode, 'trap_corridor')
            for depth in DEPTHS:
                for target_index, goal in enumerate(GOALS):
                    w = fixture(sp, depth)
                    assert not w.occupied(goal)
                    plan = prefix_plan(w, goal)
                    assert plan is not None and plan[0] == 'BACKTRACK' and len(plan) >= depth
                    for arm in ARMS:
                        row, trace = episode(sp, depth, goal, arm, seed, models.get(arm))
                        n, p = verify_trace(trace, models.get(arm))
                        receipts += n + depth; vectors += p
                        rows.append(row)
                        name = f'{seed}-{original.family}-{original.seed}-{depth}-{target_index}-{arm}.json'
                        (out/'traces'/name).write_text(json.dumps(trace, sort_keys=True, separators=(',',':'))+'\n')
    evidence = {'version': VERSION, 'split': 'development', 'structural_heldout_opened': False,
                'world_variant': 'trap_corridor', 'same_forced_prelude_across_all_arms': True,
                'goals': [list(g) for g in GOALS], 'depths': list(DEPTHS),
                'source_sha256': {n: sha_file(Path(__file__).parent/n) for n in SOURCE},
                'checkpoints': sha, 'fresh_optimizer_steps': 0,
                'episodes': len(rows), 'verified_action_receipts': receipts,
                'verified_neural_probability_vectors': vectors,
                'summary': summarize(rows), 'per_episode': rows,
                'wall_s': round(time.perf_counter()-t, 3),
                'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'native_executed': False, 'native_action_receipts': 0, 'synthetic_64_gate': True,
                'limitations': ['Frozen models originally trained on normal challenge; this corridor is out-of-distribution.',
                                'Four structural DEVELOPMENT geometry families share an identical purpose-built central corridor.',
                                'Prefix search is complete only for backtrack-prefix then empty-cell moves, not for arbitrary state-space paths.',
                                'All synthetic trajectories are not native host acceptance.',
                                'Checkpoint binaries kept outside Git; immutable Actions artifacts not yet published.']}
    (out/'evidence.json').write_text(json.dumps(evidence, sort_keys=True, indent=2)+'\n')
    sums = {str(f.relative_to(out)):sha_file(f) for f in sorted(out.rglob('*')) if f.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(sums, sort_keys=True, indent=2)+'\n')
    return evidence


def verify(out, checkpoint_dir):
    evidence = json.loads((out/'evidence.json').read_text())
    sums = json.loads((out/'SHA256SUMS.json').read_text())
    assert sums == {str(f.relative_to(out)):sha_file(f) for f in sorted(out.rglob('*'))
                    if f.is_file() and f.name != 'SHA256SUMS.json'}
    assert evidence['source_sha256'] == {n:sha_file(Path(__file__).parent/n) for n in SOURCE}
    models = {(seed, arm):GoalRanker.load(checkpoint_dir/f'{seed}-{arm}.npz')
              for seed in SEEDS for arm in ('conditioned','blind')}
    for seed in SEEDS:
        for arm in ('conditioned','blind'):
            assert evidence['checkpoints'][f'{seed}-{arm}']['sha256'] == sha_file(checkpoint_dir/f'{seed}-{arm}.npz')
    rows=[]; receipts=vectors=0
    for path in sorted((out/'traces').glob('*.json')):
        t=json.loads(path.read_text());r=t['row']
        n,p=verify_trace(t,models.get((r['seed'],r['arm'])))
        receipts+=n+r['depth'];vectors+=p;rows.append(r)
    assert len(rows)==evidence['episodes'] and summarize(rows)==evidence['summary']
    assert receipts==evidence['verified_action_receipts'] and vectors==evidence['verified_neural_probability_vectors']
    return {'episodes':len(rows), 'receipts':receipts, 'vectors':vectors, 'files':len(sums)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--checkpoints', type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.verify,args.checkpoints) if args.verify else run(args.out,args.checkpoints),sort_keys=True))
