"""SPARSE-HANDOFF-001: event-triggered, single-search planner handoff.

DEVELOPMENT-only frozen neural policy evaluation. Explicit synthetic material
screens perturb the earlier corridor fixture. This is not a native CyberSand
run or a claim about completeness of unrestricted planning. HELDOUT untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import statistics
import time

from .detour_recovery_lab import prefix_plan, GOALS, DEPTHS
from .goal_conditioning_lab import GoalRanker
from .goal_observability_lab import strict_step
from .world import World, Spec, specs, digest, MOVE, MATERIAL_Z, SOLID
from .train import sha_file

VERSION = 'sparse-handoff/1'
SEEDS = (711, 712)
LAYOUTS = ('original', 'east_sleeve')
ARMS = ('neural', 'planner_once', 'late24', 'stall4', 'stall8', 'reserve16', 'hybrid4')
CAP = 64
SOURCES = ('sparse_handoff_lab.py', 'world.py', 'goal_conditioning_lab.py',
           'goal_observability_lab.py', 'detour_recovery_lab.py', 'policy.py', 'train.py')


def overlay(w, layout):
    """Explicit research overlays, never changes to the canonical World source."""
    if layout == 'original':
        return
    if layout == 'north_ledge':
        for x in range(46, 68):
            if x not in (48, 49): w.materials[(x, 44)] = MATERIAL_Z
        for y in range(42, 47): w.materials[(61, y)] = SOLID
    elif layout == 'south_ledge':
        for x in range(46, 67):
            if x not in (46, 47): w.materials[(x, 53)] = MATERIAL_Z
        for y in range(53, 61): w.materials[(62, y)] = SOLID
    elif layout == 'east_sleeve':
        for y in range(44, 59):
            if y not in (45, 56): w.materials[(58, y)] = SOLID
    elif layout == 'staggered_piers':
        for p in ((46,55),(47,55),(48,55),(49,55),(51,54),(52,54),
                  (53,54),(54,54),(59,52),(59,53),(59,54),(59,55)):
            w.materials[p] = MATERIAL_Z
    else:
        raise ValueError(f'Unrecognized layout: {layout}')


def fixture(spec, depth, layout):
    if spec.split != 'development' or spec.challenge != 'trap_corridor' or depth not in DEPTHS:
        raise ValueError('Strict DEVELOPMENT and trap-corridor boundary')
    w = World.create(spec)
    overlay(w, layout)
    for _ in range(depth):
        assert strict_step(w, 'MOVE_E') is not None
    assert w.cursor == (48 + depth, 48)
    return w


def distance(w, goal):
    return abs(w.cursor[0]-goal[0]) + abs(w.cursor[1]-goal[1])


def decide(w, goal, model, arm, tick, stall, cached):
    """Pure decision; cached is None (not committed) or remaining tuple.

    The entire plan is computed at most once. Future actions use the original
    legal menu, not a privileged synthetic teleport or a recomputed oracle.
    """
    planner_now = (arm == 'planner_once' or
                   (arm == 'late24' and tick >= 24) or
                   (arm == 'stall4' and stall >= 4) or
                   (arm == 'stall8' and stall >= 8) or
                   (arm == 'reserve16' and CAP-w.logical_moves <= 16) or
                   (arm == 'hybrid4' and (stall >= 4 or CAP-w.logical_moves <= 16)))
    if cached is not None or planner_now:
        calls = int(cached is None)
        if cached is None: cached = prefix_plan(w, goal, CAP)
        if cached is None:
            return 'STOP', None, cached, calls, 'planner_unreachable'
        action = cached[0] if cached else 'STOP'
        if action not in w.menu(12): raise AssertionError('Cached plan lost host legality')
        return action, None, cached[1:] if cached else (), calls, 'planner_cached'
    proposed, probabilities = model.choose(w, goal, CAP, False)
    assert proposed in w.menu(12)
    # The hybrid reacts to neural policy STOP as a second, observed failure cue.
    if arm == 'hybrid4' and proposed == 'STOP':
        plan = prefix_plan(w, goal, CAP)
        if plan is not None:
            action = plan[0] if plan else 'STOP'
            assert action in w.menu(12)
            return action, probabilities, plan[1:] if plan else (), 1, 'stop_rescue'
        return proposed, probabilities, None, 1, 'stop_unreachable'
    return proposed, probabilities, None, 0, 'neural'


def episode(spec, layout, depth, goal, seed, arm, model):
    w = fixture(spec, depth, layout)
    initial_sha = w.state_sha(); initial_base = w.base_sha(); prelude = list(w.decisions)
    cached = None; best = distance(w, goal); stall = 0
    entries = []; control_ms = []; calls = 0; reason = 'decision_gate'
    for tick in range(CAP):
        if w.cursor == goal: reason = 'goal'; break
        if w.stopped: reason = 'policy_stop'; break
        if w.logical_moves >= CAP: reason = 'logical_gate'; break
        before = w.state_sha(); menu = w.menu(12)
        t = time.perf_counter_ns()
        action, prob, cached, n_calls, source = decide(w, goal, model, arm, tick, stall, cached)
        control_ms.append((time.perf_counter_ns()-t)/1e6)
        calls += n_calls
        assert action in menu
        receipt = strict_step(w, action)
        if receipt is None: reason = 'logical_gate'; break
        now = distance(w, goal)
        if now < best: best = now; stall = 0
        else: stall += 1
        entries.append({'tick': tick, 'pre_sha256': before, 'menu': menu,
                        'action': action, 'probabilities': prob, 'source': source,
                        'planner_calls': n_calls, 'receipt': receipt})
    else: reason = 'decision_gate'
    continuation = [e['action'] for e in entries]
    row = {'family': spec.family, 'seed': spec.seed, 'model_seed': seed, 'layout': layout,
           'depth': depth, 'goal': list(goal), 'arm': arm, 'arrived': w.cursor == goal,
           'reason': reason, 'initial_sha256': initial_sha, 'base_sha256': initial_base,
           'final_sha256': w.state_sha(), 'trace_sha256': digest(w.decisions),
           'decisions': len(entries), 'logical_moves': w.logical_moves, 'planner_calls': calls,
           'painted_cells': len(w.ink),
           'turns': sum(a != b for a,b in zip([a for a in continuation if a in MOVE],
                                               [a for a in continuation if a in MOVE][1:])),
           'planner_decisions': sum(e['source'] != 'neural' for e in entries),
           'neural_decisions': sum(e['source'] == 'neural' for e in entries),
           'action_counts': dict(sorted(Counter(continuation).items())),
           'protected_material_edits': len(w.ink & (set(w.materials)|w.protected)),
           'median_choose_ms': round(statistics.median(control_ms), 6) if control_ms else 0}
    trace = {'spec': dict(split=spec.split, family=spec.family, seed=spec.seed,
                          mode=spec.mode, challenge=spec.challenge),
             'row': row, 'prelude': prelude, 'entries': entries}
    return row, trace


def verify_trace(trace, model):
    row = trace['row']; spec = Spec(**trace['spec']); goal = tuple(row['goal'])
    w = fixture(spec, row['depth'], row['layout'])
    assert w.state_sha() == row['initial_sha256'] and w.base_sha() == row['base_sha256']
    assert w.decisions == trace['prelude']
    cached = None; best = distance(w, goal); stall = 0; calls = 0
    iterator = iter(trace['entries'])
    for tick in range(CAP):
        if w.cursor == goal: reason = 'goal'; break
        if w.stopped: reason = 'policy_stop'; break
        if w.logical_moves >= CAP: reason = 'logical_gate'; break
        before = w.state_sha(); menu = w.menu(12)
        action, prob, cached, n_calls, source = decide(w, goal, model, row['arm'], tick, stall, cached)
        if w.logical_moves + (len(w.path)-1 if action == 'RELOCATE_START' else
                              0 if action == 'STOP' else 1) > CAP:
            reason = 'logical_gate'; break
        e = next(iterator)
        assert e['tick'] == tick and e['pre_sha256'] == before and e['menu'] == menu
        assert e['action'] == action and e['probabilities'] == prob
        assert e['source'] == source and e['planner_calls'] == n_calls
        assert e['receipt'] == strict_step(w, action)
        calls += n_calls
        now = distance(w, goal)
        if now < best: best = now; stall = 0
        else: stall += 1
    else: reason = 'decision_gate'
    assert next(iterator, None) is None
    assert reason == row['reason'] and (w.cursor == goal) == row['arrived']
    assert w.state_sha() == row['final_sha256'] and digest(w.decisions) == row['trace_sha256']
    assert calls == row['planner_calls'] <= 1
    assert len(trace['entries']) == row['decisions'] and w.logical_moves == row['logical_moves'] <= CAP
    assert len(w.ink & (set(w.materials)|w.protected)) == row['protected_material_edits'] == 0
    return len(w.decisions), sum(e['probabilities'] is not None for e in trace['entries'])


def summarize(rows):
    summary = {}
    for arm in ARMS:
        rr = [r for r in rows if r['arm'] == arm]
        actions = Counter()
        for r in rr: actions.update(r['action_counts'])
        summary[arm] = {'episodes': len(rr), 'arrivals': sum(r['arrived'] for r in rr),
                        'planner_searches': sum(r['planner_calls'] for r in rr),
                        'planner_actions': sum(r['planner_decisions'] for r in rr),
                        'neural_actions': sum(r['neural_decisions'] for r in rr),
                        'mean_decisions': round(statistics.mean(r['decisions'] for r in rr), 3),
                        'median_episode_choose_ms': round(statistics.median(r['median_choose_ms'] for r in rr), 6),
                        'action_counts': dict(sorted(actions.items()))}
    return summary


def run(out, checkpoint_dir, per_family=1):
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); (out/'traces').mkdir()
    start = time.perf_counter(); models = {}
    checkpoints = {}
    for seed in SEEDS:
        cp = checkpoint_dir / f'{seed}-conditioned.npz'
        models[seed] = GoalRanker.load(cp)
        assert models[seed].updates == 120
        checkpoints[str(seed)] = sha_file(cp)
    rows = []; infeasible = []; receipts = vectors = 0
    for base in specs('development', per_family):
        spec = Spec(base.split, base.family, base.seed, base.mode, 'trap_corridor')
        for layout in LAYOUTS:
            for depth in (5,):
                for index, goal in enumerate(GOALS):
                    w = fixture(spec, depth, layout)
                    plan = prefix_plan(w, goal, CAP)
                    if plan is None:
                        infeasible.append({'spec': spec.ident, 'layout': layout,
                                           'depth': depth, 'goal_index': index,
                                           'base_sha256': w.base_sha()})
                        continue
                    assert plan and plan[0] == 'BACKTRACK'
                    for seed in SEEDS:
                        for arm in ARMS:
                            row, trace = episode(spec, layout, depth, goal, seed, arm, models[seed])
                            n, v = verify_trace(trace, models[seed]); receipts += n; vectors += v
                            rows.append(row)
                            filename = f'{seed}-{base.family}-{base.seed}-{layout}-{depth}-{index}-{arm}.json'
                            (out/'traces'/filename).write_text(json.dumps(trace, sort_keys=True, separators=(',', ':'))+'\n')
    evidence = {'version': VERSION, 'source_head': '728a1c6131bc8da12165c8b94080a29b28d3da64',
                'split': 'development', 'structural_heldout_opened': False,
                'synthetic_64_gate': True, 'native_receipts': 0, 'new_optimizer_updates': 0,
                'checkpoint_sha256': checkpoints,
                'source_sha256': {n: sha_file(Path(__file__).parent/n) for n in SOURCES},
                'total_candidate_states': len(specs('development', per_family))*len(LAYOUTS)*len(DEPTHS)*len(GOALS),
                'infeasible_states': infeasible, 'evaluated_state_count': len(rows)//(len(SEEDS)*len(ARMS)),
                'episodes': len(rows), 'verified_receipts': receipts,
                'verified_neural_distributions': vectors,
                'summary': summarize(rows), 'rows': rows,
                'cpu_wall_s': round(time.perf_counter()-start, 3),
                'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'limitations': ['DEVELOPMENT geometries share training structural families.',
                   'All layouts overlay the same forced eastern trap; not five independent world generators.',
                   'Planner uses privileged complete synthetic material map and history.',
                   'Prefix search is restricted to monotone retreat then empty-cell continuation.',
                   'Unsolvable states are explicitly excluded from comparative arrivals and counted separately.',
                   'Wall-clock action latency includes local Python overhead, not native host.',
                   'No structural HELDOUT or native CyberSand evaluation.']}
    (out/'evidence.json').write_text(json.dumps(evidence, sort_keys=True, indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)): sha_file(p)
           for p in sorted(out.rglob('*')) if p.is_file()}, sort_keys=True, indent=2)+'\n')
    return evidence


def verify(out, checkpoint_dir):
    ev = json.loads((out/'evidence.json').read_text())
    hashes = json.loads((out/'SHA256SUMS.json').read_text())
    assert hashes == {str(p.relative_to(out)): sha_file(p) for p in sorted(out.rglob('*'))
                      if p.is_file() and p.name != 'SHA256SUMS.json'}
    assert ev['source_sha256'] == {n: sha_file(Path(__file__).parent/n) for n in SOURCES}
    models = {s: GoalRanker.load(checkpoint_dir/f'{s}-conditioned.npz') for s in SEEDS}
    assert ev['checkpoint_sha256'] == {str(s): sha_file(checkpoint_dir/f'{s}-conditioned.npz') for s in SEEDS}
    rows = []; receipts = vectors = 0
    for path in sorted((out/'traces').glob('*.json')):
        trace = json.loads(path.read_text())
        n, v = verify_trace(trace, models[trace['row']['model_seed']])
        rows.append(trace['row']); receipts += n; vectors += v
    assert len(rows) == ev['episodes'] and summarize(rows) == ev['summary']
    assert receipts == ev['verified_receipts'] and vectors == ev['verified_neural_distributions']
    return {'episodes': len(rows), 'receipts': receipts, 'probability_vectors': vectors,
            'source_file_count': len(SOURCES), 'artifact_files': len(hashes)}


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', type=Path)
    ap.add_argument('--verify', type=Path)
    ap.add_argument('--checkpoints', required=True, type=Path)
    ap.add_argument('--per-family', type=int, default=1)
    args=ap.parse_args()
    print(json.dumps(verify(args.verify, args.checkpoints) if args.verify else
                     run(args.out, args.checkpoints, args.per_family), sort_keys=True))
