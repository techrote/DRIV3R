"""GATE-OSCILLATION-001: closed/reopened material gate on a cached synthetic path.

Development-only, deterministic event schedule; never native CyberSand physics.
The event site comes from the initial route before any outcome is observed.
History, legal menus, material occupancy and receipts are replayed on the original
read-only synthetic World. Structural heldout remains sealed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time
from .world import MATERIAL_Z, Spec, digest, specs
from .stale_route_gate_lab import scenario, route_positions, MODEL_SEEDS
from .topology_adaptive_handoff_lab import overlay_world, LAYOUTS, GOALS
from .detour_recovery_lab import prefix_plan
from .goal_conditioning_lab import GoalRanker
from .goal_observability_lab import strict_step
from .train import sha_file

VERSION = 'gate-oscillation/1'
CAP = 64
EVENT_TICK = 2
MODES = ('early_reopen', 'late_reopen', 'unrelated_reopen', 'rapid_bounce')
ARMS = ('blind', 'always', 'intersect', 'partial_two', 'lazy', 'neural')
SOURCES = ('world.py', 'stale_route_gate_lab.py', 'topology_adaptive_handoff_lab.py',
           'detour_recovery_lab.py', 'goal_conditioning_lab.py', 'goal_observability_lab.py',
           'train.py', 'policy.py', 'gate_oscillation_lab.py')


def episode(sp: Spec, layout: str, goal_idx: int, mode: str, arm: str, seed=None, model=None):
    if sp.split != 'development' or sp.challenge != 'normal' or mode not in MODES or arm not in ARMS:
        raise ValueError('Unapproved split, challenge, mode, or arm')
    if (arm == 'neural') != (model is not None):
        raise ValueError('Frozen checkpoint required only for neural recovery')
    goal = GOALS[goal_idx]
    initial_plan, _, sites = scenario(sp, layout, goal)
    w = overlay_world(sp, layout)
    cache = tuple(initial_plan)
    targets = route_positions(w, cache)
    gate_cell = sites['off_route' if mode == 'unrelated_reopen' else 'on_route']
    schedule = ((2, 'close'), (3, 'open'), (4, 'close'), (6, 'open')) if mode == 'rapid_bounce' else (
        (2, 'close'), (4 if mode == 'early_reopen' else 9, 'open'))
    initial_sha = w.state_sha()
    calls = 1  # initial search, charged to every arm
    entries, events, invalidations, latency = [], [], [], []
    neural = False
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
        if any(t == tick for t, _ in schedule):
            before = w.state_sha()
            kind = next(k for t, k in schedule if t == tick)
            if kind == 'close':
                assert gate_cell not in w.ink and not w.occupied(gate_cell)
                w.materials[gate_cell] = MATERIAL_Z
            else:
                assert w.materials.get(gate_cell) == MATERIAL_Z
                del w.materials[gate_cell]
            actual_overlap = gate_cell in targets
            observed_overlap = gate_cell in (targets[:2] if arm == 'partial_two' else targets)
            events.append({'tick': tick, 'kind': kind, 'cell': list(gate_cell),
                           'pre_sha256': before, 'post_sha256': w.state_sha(),
                           'actual_overlap': actual_overlap, 'observed_overlap': observed_overlap})
            should_invalidate = (arm == 'always' or
                                 (arm in ('intersect', 'partial_two') and observed_overlap))
            if should_invalidate and cache is not None:
                invalidations.append({'tick': tick, 'reason': kind + '_event', 'remaining': len(cache)})
                cache, targets = None, ()
        start = time.perf_counter_ns()
        chosen = 'cached'
        probs = None
        used = 0
        if cache and cache[0] not in w.menu(12):
            if arm == 'lazy':
                invalidations.append({'tick': tick, 'reason': 'next_illegal', 'remaining': len(cache)})
                cache, targets = None, ()
            elif arm == 'neural':
                invalidations.append({'tick': tick, 'reason': 'neural_fallback', 'remaining': len(cache)})
                cache, targets = None, ()
                neural = True
        if neural:
            action, probs = model.choose(w, goal, CAP, False)
            chosen = 'neural'
        elif cache is None:
            plan = prefix_plan(w, goal, CAP)
            calls += 1
            used = 1
            cache = tuple(plan or ())
            targets = route_positions(w, cache) if cache else ()
            if cache:
                action = cache[0]
                cache, targets = cache[1:], targets[1:]
                chosen = 'replanned'
            else:
                action = 'STOP'
                chosen = 'no_route'
        elif cache and cache[0] in w.menu(12):
            action = cache[0]
            cache, targets = cache[1:], targets[1:]
        else:
            action = 'STOP'
            chosen = 'stale_stop'
        latency.append((time.perf_counter_ns()-start)/1e6)
        menu = w.menu(12)
        assert action in menu, (arm, action, menu)
        before = w.state_sha()
        receipt = strict_step(w, action)
        if receipt is None:
            reason = 'logical_gate'
            break
        entries.append({'tick': tick, 'pre_sha256': before, 'menu': menu,
                        'action': action, 'source': chosen, 'probabilities': probs,
                        'planner_calls': used, 'receipt': receipt,
                        'cursor': list(w.cursor)})
    assert events and events[0]['kind'] == 'close'
    actions = dict(sorted(Counter(x['action'] for x in entries).items()))
    sources = dict(sorted(Counter(x['source'] for x in entries).items()))
    row = {'spec': vars(sp), 'layout': layout, 'goal_idx': goal_idx, 'mode': mode,
           'arm': arm, 'seed': seed, 'arrived': w.cursor == goal, 'reason': reason,
           'decisions': len(entries), 'logical_moves': w.logical_moves,
           'planner_calls': calls, 'neural_actions': sources.get('neural', 0),
           'action_counts': actions, 'source_counts': sources, 'events_observed': len(events),
           'invalidations': invalidations, 'initial_sha256': initial_sha,
           'final_sha256': w.state_sha(), 'trace_sha256': digest(w.decisions),
           'final_cursor': list(w.cursor),
           'illegal_edits': len(w.ink & (w.protected | set(w.materials))),
           'median_choose_ms': round(statistics.median(latency), 6) if latency else 0}
    tr = {'version': VERSION, 'row': row, 'gate': list(gate_cell),
          'initial_plan': list(initial_plan), 'events': events, 'entries': entries}
    return row, tr


def verify_trace(tr, models):
    row = tr['row']
    sp = Spec(**row['spec'])
    model = models.get(row['seed']) if row['arm'] == 'neural' else None
    check, original = episode(sp, row['layout'], row['goal_idx'], row['mode'],
                              row['arm'], row['seed'], model)
    for k, v in row.items():
        if k != 'median_choose_ms':
            assert check[k] == v, k
    for k in ('version', 'gate', 'initial_plan', 'events', 'entries'):
        assert tr[k] == original[k], k
    return len(tr['entries']), sum(x['probabilities'] is not None for x in tr['entries'])


def summarize(rows):
    summary = {}
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        actions = Counter()
        for r in group:
            actions.update(r['action_counts'])
        summary[arm] = {'episodes': len(group), 'arrivals': sum(r['arrived'] for r in group),
                        'searches': sum(r['planner_calls'] for r in group),
                        'neural_actions': sum(r['neural_actions'] for r in group),
                        'median_choose_ms': round(statistics.median(r['median_choose_ms'] for r in group),6),
                        'actions': dict(sorted(actions.items())),
                        'by_mode': {m: {'arrivals': sum(r['arrived'] for r in group if r['mode']==m),
                                       'episodes': sum(r['mode']==m for r in group),
                                       'searches': sum(r['planner_calls'] for r in group if r['mode']==m)}
                                    for m in MODES}}
    return summary


def campaign(out: Path, checkpoint_dir: Path):
    out.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    models = {s: GoalRanker.load(checkpoint_dir / f'{s}-conditioned.npz') for s in MODEL_SEEDS}
    rows, traces = [], []
    for sp, layout in zip(specs('development', 1), LAYOUTS):
        for goal_idx in (0,1):
            for mode in MODES:
                for arm in ARMS:
                    seeds = MODEL_SEEDS if arm == 'neural' else (None,)
                    for seed in seeds:
                        row, trace = episode(sp, layout, goal_idx, mode, arm, seed, models.get(seed))
                        rows.append(row); traces.append(trace)
    assert len(traces) == 4 * 2 * len(MODES) * 7 == 224
    receipts = out / 'receipts.jsonl.gz'
    with gzip.open(receipts, 'wt', encoding='utf-8', compresslevel=9) as f:
        for t in traces:
            f.write(json.dumps(t, sort_keys=True, separators=(',', ':'))+'\n')
    ev = {'version': VERSION, 'split': 'development', 'heldout_opened': False,
          'native_receipts': 0, 'optimizer_updates': 0, 'cap': CAP,
          'modes': MODES, 'arms': ARMS, 'layouts': LAYOUTS, 'episodes': len(rows),
          'action_receipts': sum(r['decisions'] for r in rows),
          'neural_probability_vectors': sum(e['probabilities'] is not None for t in traces for e in t['entries']),
          'summary': summarize(rows), 'rows': rows,
          'checkpoints_sha256': {str(s): sha_file(checkpoint_dir / f'{s}-conditioned.npz') for s in MODEL_SEEDS},
          'source_sha256': {src: sha_file(Path(__file__).parent / src) for src in SOURCES},
          'receipts_sha256': sha_file(receipts),
          'wall_s': round(time.perf_counter()-begin, 3),
          'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'limitations': ['Engineered gates selected from a pre-event route, not native materials or reactions',
                          'Four shared central development masks; no structural HELDOUT opened',
                          'Restricted path search is not complete host-state planning',
                          'Synthetic 64-move limit; no native CyberSand actor or Workbench execution',
                          'Frozen tiny ranker checkpoints; zero optimizer updates']}
    (out/'evidence.json').write_text(json.dumps(ev, indent=2, sort_keys=True)+'\n')
    return ev


def verify(out: Path, checkpoint_dir: Path):
    ev = json.loads((out/'evidence.json').read_text())
    assert ev['source_sha256'] == {src: sha_file(Path(__file__).parent/src) for src in SOURCES}
    assert ev['checkpoints_sha256'] == {str(s): sha_file(checkpoint_dir / f'{s}-conditioned.npz') for s in MODEL_SEEDS}
    assert ev['receipts_sha256'] == sha_file(out/'receipts.jsonl.gz')
    models = {s: GoalRanker.load(checkpoint_dir / f'{s}-conditioned.npz') for s in MODEL_SEEDS}
    with gzip.open(out/'receipts.jsonl.gz','rt',encoding='utf-8') as f:
        traces = [json.loads(x) for x in f]
    checks = [verify_trace(tr, models) for tr in traces]
    rows = [tr['row'] for tr in traces]
    assert rows == ev['rows'] and ev['summary'] == summarize(rows)
    assert sum(t[0] for t in checks) == ev['action_receipts']
    assert sum(t[1] for t in checks) == ev['neural_probability_vectors']
    return ev


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('run','verify'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True)
    a = p.parse_args()
    e = campaign(a.out, a.checkpoints) if a.command == 'run' else verify(a.out, a.checkpoints)
    print(json.dumps({'episodes': e['episodes'], 'action_receipts': e['action_receipts'],
                      'neural_probability_vectors': e['neural_probability_vectors'],
                      'summary': e['summary'], 'wall_s': e['wall_s'],
                      'max_rss_kib': e['max_rss_kib']}, sort_keys=True))
