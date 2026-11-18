"""STALE-ROUTE-GATE-001: falsify synthetic cached-route invalidation strategies.

Only DEVELOPMENT families, original synthetic World/legal menus and 64-logical-move
external gate. A deterministic obstructing cell is selected from the *pre-event*
route, never from outcome data; an irrelevant material edit is the paired control.
Neither this fixture nor historical d_common is CyberSand's native host contract.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

from .world import MOVE, MATERIAL_Z, Spec, digest, specs
from .topology_adaptive_handoff_lab import overlay_world, LAYOUTS, GOALS
from .detour_recovery_lab import prefix_plan
from .goal_conditioning_lab import GoalRanker
from .goal_observability_lab import strict_step
from .train import sha_file

VERSION = 'stale-route-gate/1'
CAP = 64
EVENT_TICK = 2
GATE_INDEX = 5  # sixth initial planned waypoint; three actions after the event
HAZARDS = ('on_route', 'off_route')
ARMS = ('stale_cache', 'epoch_all', 'route_overlap', 'lazy_illegal',
        'neural_recovery')
MODEL_SEEDS = (711, 712)
SOURCE = ('stale_route_gate_lab.py', 'world.py', 'topology_adaptive_handoff_lab.py',
          'detour_recovery_lab.py', 'goal_observability_lab.py',
          'goal_conditioning_lab.py', 'train.py', 'policy.py')


def route_positions(w, actions):
    """Record cursor destinations, replaying only the original strict legal menu."""
    t = deepcopy(w)
    positions = []
    for action in actions:
        assert action in t.menu(12)
        assert strict_step(t, action) is not None
        positions.append(t.cursor)
    return tuple(positions)


def scenario(sp: Spec, layout: str, goal):
    if sp.split != 'development' or sp.challenge != 'normal' or layout not in LAYOUTS:
        raise ValueError('Only prespecified DEVELOPMENT overlays and ordinary challenge')
    w = overlay_world(sp, layout)
    plan = prefix_plan(w, goal, CAP)
    if plan is None or len(plan) <= GATE_INDEX + 2:
        raise AssertionError('No long cached plan: do not silently drop scenarios')
    assert all(a in MOVE for a in plan), 'initial plan must not require synthetic recovery'
    positions = route_positions(w, plan)
    blocker = positions[GATE_INDEX]
    assert not w.occupied(blocker) and blocker != goal
    path = set(positions) | {w.cursor, goal}
    # An unoccupied control cell far from every intended route point, within
    # the same central synthetic arena; determined before any controller runs.
    candidates = [(x, y) for x in range(41, 72) for y in range(39, 70)
                  if (x, y) not in path and not w.occupied((x, y))]
    far = sorted(candidates, key=lambda p: (-min(abs(p[0]-q[0]) + abs(p[1]-q[1])
                                              for q in path), p))[0]
    assert far not in path and far != blocker
    return plan, positions, {'on_route': blocker, 'off_route': far}


def one_episode(sp, layout, goal_idx, hazard, arm, model_seed, model):
    assert arm in ARMS and hazard in HAZARDS and goal_idx in (0, 1)
    if arm == 'neural_recovery':
        assert model is not None and model_seed in MODEL_SEEDS
    else:
        assert model is None and model_seed is None
    goal = GOALS[goal_idx]
    plan, positions, sites = scenario(sp, layout, goal)
    w = overlay_world(sp, layout)
    initial = w.state_sha()
    cache = tuple(plan)
    targets = tuple(positions)
    entries = []
    event = None
    invalidations = []
    latencies = []
    searches = 1  # initial prefix search, charged to every controller
    neural_mode = False
    reason = 'decision_gate'
    for tick in range(CAP):
        if w.cursor == goal:
            reason = 'goal_arrived'
            break
        if w.stopped:
            reason = 'policy_stop'
            break
        if w.logical_moves >= CAP:
            reason = 'logical_gate'
            break
        if tick == EVENT_TICK:
            cell = sites[hazard]
            before = w.state_sha()
            assert cell not in w.ink and cell != w.cursor and not w.occupied(cell)
            overlap = cell in targets
            w.materials[cell] = MATERIAL_Z
            event = {'tick': tick, 'cell': list(cell), 'material': MATERIAL_Z,
                     'pre_sha256': before, 'post_sha256': w.state_sha(),
                     'hazard': hazard, 'remaining_route_intersection': overlap,
                     'remaining_route_targets': [list(p) for p in targets]}
            assert overlap == (hazard == 'on_route')
            if arm == 'epoch_all' or (arm == 'route_overlap' and overlap):
                invalidations.append({'tick': tick, 'reason': 'epoch_all' if arm == 'epoch_all'
                                      else 'route_intersection', 'pre_cache_size': len(cache)})
                cache = None
                targets = ()
        start = time.perf_counter_ns()
        used = 0
        probabilities = None
        if arm == 'fresh_each':
            q = prefix_plan(w, goal, CAP)
            searches += 1
            used = 1
            action = q[0] if q else 'STOP'
            source = 'fresh_plan' if q else 'planner_no_route'
        else:
            if cache is not None and len(cache) and cache[0] not in w.menu(12):
                if arm == 'lazy_illegal':
                    invalidations.append({'tick': tick, 'reason': 'next_action_illegal',
                                          'pre_cache_size': len(cache)})
                    cache = None
                    targets = ()
                elif arm == 'neural_recovery':
                    neural_mode = True
                    invalidations.append({'tick': tick, 'reason': 'neural_on_illegal',
                                          'pre_cache_size': len(cache)})
                    cache = None
                    targets = ()
                else:
                    invalidations.append({'tick': tick, 'reason': 'stale_route_stopped',
                                          'pre_cache_size': len(cache)})
                    cache = ()
                    targets = ()
            if neural_mode:
                action, probabilities = model.choose(w, goal, CAP, False)
                source = 'neural_recovery'
            elif cache is None:
                q = prefix_plan(w, goal, CAP)
                used = 1
                searches += 1
                cache = tuple(q or ())
                targets = route_positions(w, cache) if cache else ()
                action = cache[0] if cache else 'STOP'
                source = 'replanned' if cache else 'planner_no_route'
                cache, targets = cache[1:], targets[1:]
            elif cache:
                action = cache[0]
                source = 'initial_cache' if tick < EVENT_TICK else 'cached_route'
                cache, targets = cache[1:], targets[1:]
            else:
                action = 'STOP'
                source = 'stale_stop'
        latencies.append((time.perf_counter_ns() - start) / 1e6)
        menu = w.menu(12)
        assert action in menu, (arm, action, menu)
        pre = w.state_sha()
        receipt = strict_step(w, action)
        if receipt is None:
            reason = 'logical_gate'
            break
        entries.append({'tick': tick, 'pre_sha256': pre, 'menu': menu,
                        'action': action, 'source': source, 'probabilities': probabilities,
                        'planner_calls': used, 'receipt': receipt})
    else:
        reason = 'decision_gate'
    assert event is not None
    counts = dict(sorted(Counter(x['action'] for x in entries).items()))
    sources = dict(sorted(Counter(x['source'] for x in entries).items()))
    row = {'family': sp.family, 'world_seed': sp.seed, 'layout': layout,
           'goal_idx': goal_idx, 'goal': list(goal), 'hazard': hazard, 'arm': arm,
           'model_seed': model_seed, 'arrived': w.cursor == goal,
           'reason': reason, 'decisions': len(entries), 'logical_moves': w.logical_moves,
           'planner_calls': searches, 'neural_actions': sources.get('neural_recovery', 0),
           'first_event_overlap': event['remaining_route_intersection'],
           'invalidations': invalidations, 'action_counts': counts, 'source_counts': sources,
           'initial_sha256': initial, 'final_sha256': w.state_sha(),
           'trace_sha256': digest(w.decisions), 'final_cursor': list(w.cursor),
           'protected_material_edits': len(w.ink & (w.protected | set(w.materials))),
           'median_choose_ms': round(statistics.median(latencies), 6) if latencies else 0}
    tr = {'version': VERSION, 'spec': vars(sp), 'row': row,
          'reference': {'initial_plan': list(plan), 'initial_positions': [list(x) for x in positions],
                        'candidate_cells': {k: list(v) for k, v in sites.items()}},
          'material_event': event, 'entries': entries}
    return row, tr


def verify_trace(tr, models):
    """Independent episode execution plus full receipt and neural-vector comparison."""
    row = tr['row']; sp = Spec(**tr['spec']); arm = row['arm']
    model = models[row['model_seed']] if arm == 'neural_recovery' else None
    check, original = one_episode(sp, row['layout'], row['goal_idx'], row['hazard'],
                                  arm, row['model_seed'], model)
    for key in row:
        if key != 'median_choose_ms':
            assert check[key] == row[key], key
    assert tr['spec'] == original['spec']
    assert tr['reference'] == original['reference']
    assert tr['material_event'] == original['material_event']
    assert tr['entries'] == original['entries']
    assert tr['version'] == VERSION
    return len(tr['entries']), sum(x['probabilities'] is not None for x in tr['entries'])


def summarize(rows):
    out = {}
    for arm in ARMS:
        a = [r for r in rows if r['arm'] == arm]
        actions, sources = Counter(), Counter()
        for r in a:
            actions.update(r['action_counts']); sources.update(r['source_counts'])
        out[arm] = {'episodes': len(a), 'arrivals': sum(r['arrived'] for r in a),
                    'planner_calls': sum(r['planner_calls'] for r in a),
                    'neural_actions': sum(r['neural_actions'] for r in a),
                    'median_choose_ms': round(statistics.median(r['median_choose_ms'] for r in a), 6),
                    'by_hazard': {hazard: {'episodes': sum(r['hazard'] == hazard for r in a),
                                     'arrivals': sum(r['arrived'] for r in a if r['hazard'] == hazard),
                                     'planner_calls': sum(r['planner_calls'] for r in a if r['hazard'] == hazard)}
                                  for hazard in HAZARDS},
                    'actions': dict(sorted(actions.items())),
                    'sources': dict(sorted(sources.items()))}
    return out


def campaign(out: Path, checkpoints: Path, n: int = 1):
    if n != 1:
        raise ValueError('Prespecified n=1 DEVELOPMENT only, not sealed HELDOUT')
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    models = {s: GoalRanker.load(checkpoints / f'{s}-conditioned.npz') for s in MODEL_SEEDS}
    traces, rows = [], []
    for sp, layout in zip(specs('development', n), LAYOUTS):
        for goal_idx in range(2):
            for hazard in HAZARDS:
                for arm in ARMS:
                    seeds = MODEL_SEEDS if arm == 'neural_recovery' else (None,)
                    for seed in seeds:
                        row, tr = one_episode(sp, layout, goal_idx, hazard,
                                                  arm, seed, models.get(seed))
                        rows.append(row); traces.append(tr)
    assert len(rows) == 4 * 2 * 2 * 6  # 96, four distinct mask/base combinations
    (out / 'receipts.jsonl').write_text(''.join(json.dumps(t, sort_keys=True, separators=(',', ':'))+'\n'
                                                for t in traces))
    ev = {'version': VERSION, 'date': '2026-11-18', 'split': 'development',
          'heldout_opened': False, 'native_receipts': 0, 'optimizer_updates': 0,
          'synthetic_64_move_cap': True, 'event_tick': EVENT_TICK, 'gate_index': GATE_INDEX,
          'hazards': HAZARDS, 'arms': ARMS, 'layouts': LAYOUTS,
          'checkpoint_sha256': {str(s): sha_file(checkpoints / f'{s}-conditioned.npz') for s in MODEL_SEEDS},
          'source_sha256': {f: sha_file(Path(__file__).parent / f) for f in SOURCE},
          'episodes': len(rows), 'actions': sum(r['decisions'] for r in rows),
          'neural_probability_vectors': sum(x['probabilities'] is not None
                                            for t in traces for x in t['entries']),
          'summary': summarize(rows), 'rows': rows,
          'receipts_sha256': sha_file(out / 'receipts.jsonl'),
          'execution_wall_s': round(time.perf_counter() - start, 3),
          'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'limitations': ['Deterministic future-route closure is engineered from a reference plan, not naturally occurring reaction physics.',
                          'Four topology overlays reuse the synthetic 96x96 World; four outer base families share a cleared central region.',
                          'The restricted prefix planner is not complete host-state search.',
                          'Frozen tiny rankers only run on blocked-path recovery; engineered goal features remain.',
                          'This is DEVELOPMENT, not structural HELDOUT or native CyberSand acceptance.']}
    (out / 'evidence.json').write_text(json.dumps(ev, indent=2, sort_keys=True) + '\n')
    return ev


def verify(out: Path, checkpoints: Path):
    models = {s: GoalRanker.load(checkpoints / f'{s}-conditioned.npz') for s in MODEL_SEEDS}
    ev = json.loads((out / 'evidence.json').read_text())
    assert ev['source_sha256'] == {f: sha_file(Path(__file__).parent / f) for f in SOURCE}
    assert ev['checkpoint_sha256'] == {str(s): sha_file(checkpoints / f'{s}-conditioned.npz') for s in MODEL_SEEDS}
    assert ev['receipts_sha256'] == sha_file(out / 'receipts.jsonl')
    traces = [json.loads(x) for x in (out / 'receipts.jsonl').read_text().splitlines()]
    assert len(traces) == ev['episodes']
    counts = [verify_trace(t, models) for t in traces]
    rows = [t['row'] for t in traces]
    assert rows == ev['rows'] and ev['summary'] == summarize(rows)
    assert sum(x[0] for x in counts) == ev['actions']
    assert sum(x[1] for x in counts) == ev['neural_probability_vectors']
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('run', 'verify'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--checkpoints', type=Path, required=True)
    a = ap.parse_args()
    ev = campaign(a.out, a.checkpoints) if a.cmd == 'run' else verify(a.out, a.checkpoints)
    print(json.dumps({'episodes': ev['episodes'], 'actions': ev['actions'],
                      'vectors': ev['neural_probability_vectors'], 'summary': ev['summary'],
                      'wall_s': ev['execution_wall_s'], 'peak_rss_kib': ev['peak_rss_kib']}, sort_keys=True))


if __name__ == '__main__':
    main()
