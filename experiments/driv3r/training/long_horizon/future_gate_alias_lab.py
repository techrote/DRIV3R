"""FUTURE-GATE-ALIAS-001: learn planner handoff under unobservable gate futures.

Synthetic DEVELOPMENT-only study, not CyberSand native input/output. Host-like
legal one-cell painting, history BACKTRACK, explicit STOP, material events.
Use native-like 64 painted-move censor; 128 shadow cap is research-only.
Structural HELDOUT is never opened. No external checkpoint download.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

VERSION = 'future-gate-alias/1'
HORIZONS = (36, 72, 128)
TRAIN_SEEDS = tuple(range(1000, 1048))
DEV_SEEDS = tuple(range(3000, 3016))
MODES = ('early', 'late', 'irrelevant')
ARMS = ('blind', 'lazy', 'overlap', 'epoch', 'learned', 'privileged_oracle')
MOVES = (('MOVE_E', (1, 0)), ('MOVE_N', (0, -1)),
         ('MOVE_S', (0, 1)), ('MOVE_W', (-1, 0)))
DELTAS = dict(MOVES)
UPDATES_PER_STAGE = 180
NATIVE_LOGICAL_CAP = 64


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Spec:
    split: str
    seed: int
    horizon: int
    mode: str

    def __post_init__(self):
        if self.split not in ('train', 'development') or self.horizon not in HORIZONS or self.mode not in MODES:
            raise ValueError('Only enumerated train/development scenarios')
        if self.seed not in (TRAIN_SEEDS if self.split == 'train' else DEV_SEEDS):
            raise ValueError('Seed is outside its explicit partition')

    @property
    def key(self):
        return f'{self.split}:{self.seed}:{self.horizon}:{self.mode}'


class World:
    """Synthetic paint world. History BACKTRACK does not erase existing ink."""
    def __init__(self, sp):
        self.sp = sp
        self.goal_x = {36: 24, 72: 50, 128: 88}[sp.horizon] + (sp.seed % 3 - 1)
        self.gate_x = {36: 12, 72: 24, 128: 43}[sp.horizon] + ((sp.seed // 3) % 3 - 1)
        self.width = self.goal_x + 3
        self.start = (2, 8)
        self.goal = (self.goal_x, 8)
        self.gap_y = (3, 4, 12, 13)[sp.seed % 4]
        self.gate = (self.gate_x, 8)
        self.offroute = (self.goal_x - 2, 15)
        self.materials = {(self.gate_x, y) for y in range(17)
                          if y not in (8, self.gap_y)}
        self.materials.update((x, y) for x in range(7, self.gate_x)
                              for y in (7, 9))
        self.ink = {self.start}
        self.trail = [self.start]
        self.cursor = self.start
        self.decisions = 0
        self.logical_moves = 0
        self.stopped = False
        self.material_epoch = 0

    def occupied(self, p):
        x, y = p
        return not (0 <= x < self.width and 0 <= y < 17) or p in self.ink or p in self.materials

    def menu(self):
        if self.stopped:
            return ()
        legal = [a for a, (dx, dy) in MOVES
                 if not self.occupied((self.cursor[0]+dx, self.cursor[1]+dy))]
        if len(self.trail) > 1:
            legal.append('BACKTRACK')
        legal.append('STOP')
        return tuple(legal)

    def step(self, action):
        if action not in self.menu():
            raise ValueError(f'Illegal synthetic action: {action}')
        if action in DELTAS:
            dx, dy = DELTAS[action]
            self.cursor = (self.cursor[0]+dx, self.cursor[1]+dy)
            self.trail.append(self.cursor)
            self.ink.add(self.cursor)
            self.logical_moves += 1
        elif action == 'BACKTRACK':
            self.trail.pop()
            self.cursor = self.trail[-1]
        else:
            self.stopped = True
        self.decisions += 1

    def event(self, tick, kind):
        cell = self.offroute if self.sp.mode == 'irrelevant' else self.gate
        pre = self.state_sha()
        if kind == 'close':
            assert tick == 5 and cell not in self.ink and cell not in self.materials
            self.materials.add(cell)
        else:
            assert kind == 'open' and cell in self.materials
            self.materials.remove(cell)
        self.material_epoch += 1
        return {'tick': tick, 'kind': kind, 'cell': list(cell),
                'pre_sha256': pre, 'post_sha256': self.state_sha(),
                'epoch': self.material_epoch}

    def state_sha(self):
        return digest({'spec': self.sp.key, 'cursor': self.cursor,
                       'trail': self.trail, 'ink': sorted(self.ink),
                       'materials': sorted(self.materials), 'epoch': self.material_epoch,
                       'decisions': self.decisions, 'logical_moves': self.logical_moves,
                       'stopped': self.stopped})


def bfs_plan(w):
    """Reverse shortest-route search followed by legal history-return search."""
    if w.cursor == w.goal:
        return ()
    q = deque([w.goal])
    dist = {w.goal: 0}
    while q:
        p = q.popleft()
        for _, (dx, dy) in MOVES:
            n = (p[0]+dx, p[1]+dy)
            if n not in dist and not w.occupied(n):
                dist[n] = dist[p] + 1
                q.append(n)
    best = None
    for k, point in enumerate(reversed(w.trail)):
        for action, (dx, dy) in MOVES:
            n = (point[0]+dx, point[1]+dy)
            if n not in dist:
                continue
            candidate = (k+1+dist[n], k, action, n)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    if best is None:
        return None
    _, k, action, p = best
    path = ['BACKTRACK']*k + [action]
    while p != w.goal:
        options = [(a, (p[0]+dx, p[1]+dy)) for a, (dx, dy) in MOVES
                   if dist.get((p[0]+dx, p[1]+dy)) == dist[p]-1]
        assert options
        action, p = options[0]
        path.append(action)
    return tuple(path)


def cache_targets(w, cache):
    pos = w.cursor
    ancestors = list(w.trail)
    points = []
    for action in cache:
        if action == 'BACKTRACK':
            assert len(ancestors) > 1
            ancestors.pop()
            pos = ancestors[-1]
        else:
            dx, dy = DELTAS[action]
            pos = (pos[0]+dx, pos[1]+dy)
            ancestors.append(pos)
        points.append(pos)
    return tuple(points)


def observed_features(w, cache):
    cell = w.offroute if w.sp.mode == 'irrelevant' else w.gate
    targets = cache_targets(w, cache)
    return np.asarray((1., float(cell in targets), w.sp.horizon/128.,
                       min(1., (abs(cell[0]-w.cursor[0])+abs(cell[1]-w.cursor[1]))/w.sp.horizon),
                       len(cache)/w.sp.horizon, w.gate_x/w.width,
                       w.cursor[0]/w.width, w.material_epoch/2.), dtype=np.float64)


class GateModel:
    """161 real trainable parameters: eight observation features -> 16 -> one."""
    def __init__(self, seed=941):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, .15, (8, 16))
        self.b1 = np.zeros(16)
        self.W2 = r.normal(0, .15, (16, 1))
        self.b2 = np.zeros(1)

    def weights(self):
        return self.W1, self.b1, self.W2, self.b2

    def probability(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        z = (h @ self.W2 + self.b2).reshape(-1)
        return 1/(1+np.exp(-np.clip(z, -40, 40)))

    def checkpoint_sha(self):
        return hashlib.sha256(b''.join(x.tobytes() for x in self.weights())).hexdigest()


def episode(sp, arm, model=None, record=False, paint_cap=NATIVE_LOGICAL_CAP):
    if arm not in ARMS or paint_cap not in (64, 128) or (paint_cap == 128 and sp.horizon != 128):
        raise ValueError('Unapproved arm or native/shadow paint cap')
    if arm == 'learned' and model is None:
        raise ValueError('Learned arm requires optimized weights')
    w = World(sp)
    first = bfs_plan(w)
    assert first and all(a.startswith('MOVE_') for a in first)
    cache = tuple(first)
    calls = 1
    invalidations = 0
    illegal_prevented = 0
    counts = Counter()
    records = []
    features = None
    prediction = None
    for tick in range(sp.horizon):
        if w.cursor == w.goal or w.stopped or w.logical_moves >= paint_cap:
            break
        event = None
        if tick == 5:
            event = w.event(tick, 'close')
            features = observed_features(w, cache)
            on_route = bool(features[1])
            assert on_route == (sp.mode != 'irrelevant')
            if arm == 'epoch':
                invalid = True
            elif arm == 'overlap':
                invalid = on_route
            elif arm == 'privileged_oracle':
                invalid = on_route and sp.mode == 'late'
            elif arm == 'learned':
                prediction = float(model.probability(features.reshape(1,-1))[0])
                invalid = prediction >= .5
            else:
                invalid = False
            if invalid:
                cache = ()
                invalidations += 1
        if tick == 8 and sp.mode in ('early', 'irrelevant'):
            event = w.event(tick, 'open')
        if tick == sp.horizon-1 and sp.mode == 'late' and w.gate in w.materials:
            event = w.event(tick, 'open')
        if not cache:
            plan = bfs_plan(w)
            calls += 1
            cache = plan or ()
        next_action = cache[0] if cache else 'STOP'
        if next_action not in w.menu():
            illegal_prevented += 1
            if arm == 'blind':
                next_action = 'STOP'
            else:
                plan = bfs_plan(w)
                calls += 1
                cache = plan or ()
                next_action = cache[0] if cache else 'STOP'
        if next_action in cache[:1]:
            cache = cache[1:]
        pre = w.state_sha() if record else None
        menu = w.menu() if record else None
        w.step(next_action)
        counts[next_action] += 1
        if record:
            records.append({'tick': tick, 'event': event, 'pre_sha256': pre,
                            'menu': menu, 'action': next_action,
                            'post_sha256': w.state_sha(), 'cursor': list(w.cursor)})
    row = {'spec': sp.key, 'arm': arm, 'paint_cap': paint_cap,
           'cohort': 'native64' if paint_cap == 64 else 'shadow128',
           'arrived': w.cursor == w.goal, 'decisions': w.decisions,
           'paint_moves': w.logical_moves, 'backtracks': counts['BACKTRACK'],
           'planner_calls': calls, 'invalidations': invalidations,
           'illegal_prevented': illegal_prevented,
           'reason': ('goal_arrived' if w.cursor == w.goal else
                      'paint_censor' if w.logical_moves >= paint_cap else
                      'policy_stop' if w.stopped else 'decision_budget'),
           'action_counts': dict(sorted(counts.items())),
           'state_sha256': w.state_sha(),
           'trace_sha256': digest(records) if record else None,
           'event_feature_sha256': digest(features.tolist()) if features is not None else None,
           'prediction': prediction}
    if record:
        return row, {'spec': sp.key, 'arm': arm, 'paint_cap': paint_cap,
                     'initial_sha256': World(sp).state_sha(), 'entries': records,
                     'final_sha256': w.state_sha(),
                     'features': features.tolist() if features is not None else None,
                     'prediction': prediction}
    return row


def utility(row):
    return (1000 if row['arrived'] else 0) - 2*row['decisions'] - 3*row['planner_calls'] - row['backtracks']


def corpus():
    xs, ys, conflicts, labels = [], [], [], Counter()
    for horizon in HORIZONS:
        for seed in TRAIN_SEEDS:
            for mode in MODES:
                sp = Spec('train', seed, horizon, mode)
                w = World(sp)
                for _ in range(5):
                    w.step('MOVE_E')
                w.event(5, 'close')
                x = observed_features(w, bfs_plan(World(sp))[5:])
                cap = 128 if horizon == 128 else 64
                lazy = episode(sp, 'lazy', paint_cap=cap)
                overlap = episode(sp, 'overlap', paint_cap=cap)
                y = int(utility(overlap) > utility(lazy))
                xs.append(x)
                ys.append(y)
                labels[(horizon, mode, y)] += 1
        for seed in TRAIN_SEEDS:
            ia = (seed-1000)*len(MODES)+HORIZONS.index(horizon)*len(TRAIN_SEEDS)*len(MODES)
            ib = ia+1
            assert np.array_equal(xs[ia], xs[ib]), 'Future schedule leaked into visible features'
            if ys[ia] != ys[ib]:
                conflicts.append((horizon, seed))
    return np.stack(xs), np.array(ys, dtype=np.float64), conflicts, labels


def train(xs, ys):
    model = GateModel()
    params = model.weights()
    moments = [np.zeros_like(p) for p in params]
    squares = [np.zeros_like(p) for p in params]
    rng = np.random.default_rng(1901)
    losses = []
    step = 0
    for horizon in HORIZONS:
        idx = np.arange(HORIZONS.index(horizon)*len(TRAIN_SEEDS)*len(MODES),
                        (HORIZONS.index(horizon)+1)*len(TRAIN_SEEDS)*len(MODES))
        for _ in range(UPDATES_PER_STAGE):
            sample = rng.choice(idx, size=48, replace=True)
            x, y = xs[sample], ys[sample]
            h = np.tanh(x @ model.W1 + model.b1)
            p = model.probability(x)
            dz = ((p-y)/len(y)).reshape(-1, 1)
            dh = (dz @ model.W2.T)*(1-h*h)
            grads = [x.T @ dh, dh.sum(axis=0), h.T @ dz, dz.sum(axis=0)]
            step += 1
            for i, (param, g) in enumerate(zip(params, grads)):
                moments[i] = .9*moments[i] + .1*g
                squares[i] = .999*squares[i] + .001*g*g
                mh = moments[i]/(1-.9**step)
                vh = squares[i]/(1-.999**step)
                param -= .025*mh/(np.sqrt(vh)+1e-8)
        q = np.clip(model.probability(xs[idx]), 1e-10, 1-1e-10)
        loss = float(np.mean(-ys[idx]*np.log(q)-(1-ys[idx])*np.log(1-q)))
        losses.append({'horizon': horizon, 'steps': step, 'cross_entropy': loss,
                       'accuracy': float(np.mean((q>=.5)==ys[idx]))})
    return model, losses


def replay(trace):
    split, sid, horizon, mode = trace['spec'].split(':')
    sp = Spec(split, int(sid), int(horizon), mode)
    w = World(sp)
    if w.state_sha() != trace['initial_sha256']:
        raise AssertionError('Initial world hash mismatch')
    for entry in trace['entries']:
        tick = entry['tick']
        if entry['event'] is not None:
            ev = w.event(tick, entry['event']['kind'])
            if ev != entry['event']:
                raise AssertionError('Event receipt mismatch')
        if w.state_sha() != entry['pre_sha256'] or list(w.menu()) != list(entry['menu']):
            raise AssertionError('State/menu mismatch before action')
        w.step(entry['action'])
        if w.state_sha() != entry['post_sha256'] or list(w.cursor) != entry['cursor']:
            raise AssertionError('Action/state mismatch')
    if w.state_sha() != trace['final_sha256']:
        raise AssertionError('Final state mismatch')
    return len(trace['entries'])


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        _, _, horizon, mode = row['spec'].split(':')
        groups[(row['cohort'], horizon, mode, row['arm'])].append(row)
    out = []
    for (cohort, horizon, mode, arm), part in sorted(
            groups.items(), key=lambda k: (int(k[0][1]), k[0][0], k[0][2], k[0][3])):
        actions = Counter()
        for x in part:
            actions.update(x['action_counts'])
        out.append({'cohort': cohort, 'horizon': int(horizon), 'mode': mode,
                    'arm': arm, 'n': len(part), 'goals': sum(x['arrived'] for x in part),
                    'mean_decisions': round(sum(x['decisions'] for x in part)/len(part), 3),
                    'mean_backtracks': round(sum(x['backtracks'] for x in part)/len(part), 3),
                    'mean_planner_calls': round(sum(x['planner_calls'] for x in part)/len(part), 3),
                    'paint_censored': sum(x['reason']=='paint_censor' for x in part),
                    'action_counts': dict(sorted(actions.items()))})
    return out


def campaign(output, n_dev=16):
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    xs, ys, conflicts, labels = corpus()
    ttrain = time.perf_counter()
    model, losses = train(xs, ys)
    training_s = round(time.perf_counter()-ttrain, 3)
    checkpoint = output/'future_gate_alias_checkpoint.npz'
    np.savez_compressed(checkpoint, W1=model.W1, b1=model.b1,
                        W2=model.W2, b2=model.b2)
    checkpoint_file_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = []
    galleries = []
    verified = 0
    times = []
    for horizon in HORIZONS:
        for seed in DEV_SEEDS[:n_dev]:
            for mode in MODES:
                sp = Spec('development', seed, horizon, mode)
                caps = (64, 128) if horizon == 128 else (64,)
                for paint_cap in caps:
                    for arm in ARMS:
                        start = time.perf_counter_ns()
                        row, trace = episode(sp, arm, model, record=True, paint_cap=paint_cap)
                        times.append(time.perf_counter_ns()-start)
                        verified += replay(trace)
                        if arm == 'learned':
                            prob = float(model.probability(np.asarray(trace['features']).reshape(1,-1))[0])
                            if prob != trace['prediction']:
                                raise AssertionError('Neural probability failed deterministic reproduction')
                        rows.append(row)
                        if seed == DEV_SEEDS[0] and mode in ('early', 'late') and arm in ('lazy','overlap','learned'):
                            galleries.append(trace)
    preds = defaultdict(list)
    correct = 0
    for row in rows:
        if row['arm']=='learned' and row['cohort']=='native64':
            mode = row['spec'].split(':')[-1]
            preds[mode].append(row['prediction'])
            correct += int((row['prediction'] >= .5) == (mode=='late'))
    n_predictions = sum(map(len, preds.values()))
    audit = {mode:{'n':len(ps),'mean':round(sum(ps)/len(ps),6),
                   'min':round(min(ps),6),'max':round(max(ps),6),
                   'eager_decisions':sum(p>=.5 for p in ps)}
             for mode,ps in sorted(preds.items())}
    result = {'version': VERSION, 'horizons': HORIZONS,
              'train_seeds': len(TRAIN_SEEDS), 'dev_seeds': n_dev,
              'train_examples': len(xs), 'train_positive_labels': int(ys.sum()),
              'matched_early_late_alias_pairs': len(TRAIN_SEEDS)*len(HORIZONS),
              'conflicting_training_alias_pairs': len(conflicts),
              'label_by_horizon_mode': {':'.join(map(str,k)):v for k,v in sorted(labels.items())},
              'optimizer_updates': 3*UPDATES_PER_STAGE,
              'model_parameters': sum(x.size for x in model.weights()),
              'training_losses': losses,
              'weights_sha256': model.checkpoint_sha(),
              'external_checkpoint_sha256': checkpoint_file_sha,
              'training_s': training_s, 'evaluation_episodes': len(rows),
              'replayed_actions': verified, 'native_gate_unchanged': 64,
              'shadow_research_only_cap': 128,
              'dev_handoff_accuracy': round(correct/n_predictions,6),
              'dev_handoff_examples': n_predictions,
              'learned_predicted_replanning': audit,
              'evaluation_mean_ms': round(sum(times)/len(times)/1e6, 3),
              'evaluation_p95_ms': round(sorted(times)[int(.95*(len(times)-1))]/1e6, 3),
              'rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'wall_s': round(time.perf_counter()-t0, 3),
              'summary': aggregate(rows),
              'native_receipts': 0, 'structural_heldout_opened': False,
              'model_weights_in_git': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'episode_results_sha256': digest(rows)}
    (output/'future_gate_alias_evidence.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    (output/'future_gate_alias_traces.json').write_text(json.dumps(galleries, indent=1, sort_keys=True)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dev', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.dev <= len(DEV_SEEDS):
        parser.error('--dev must be 1..16')
    result = campaign(args.out, args.dev)
    print(json.dumps({k:v for k,v in result.items() if k!='summary'}, indent=2))


if __name__ == '__main__':
    main()
