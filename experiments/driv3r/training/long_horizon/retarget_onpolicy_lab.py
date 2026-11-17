"""RETARGET-DAGGER-001: matched student-visited imitation for surprise goals.

Research-only synthetic CyberSand-style host menus. No native ABI claims; no
structural held-out access. Branch artifacts hold trained weights, never Git.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import goal_retarget_lab as fixture, goal_recovery_lab as prior
from .world import World, MOVE, specs
from . import policy
from .train import sha_file

VERSION = 'retarget-dagger/1'
STAGES = (36, 72, 128)
MODES = ('static', 'retarget')
ARMS = ('frozen', 'static_adapt', 'retarget_adapt')
SOURCES = ('retarget_onpolicy_lab.py', 'goal_retarget_lab.py',
           'goal_recovery_lab.py', 'goal_conditioning_lab.py',
           'world.py', 'policy.py', 'train.py')


def student_corpus(model, seed, stage, mode, sample_n=768):
    """Student drives every legal decision; teacher labels visited TRAIN states.

    Equal-size uniform resampling avoids giving either arm more optimizer examples
    merely because one survived longer. Interventions are seeded, never illegal.
    """
    if mode not in MODES or stage not in STAGES or sample_n < 32:
        raise ValueError('invalid bounded training conditions')
    X, masks, labels, counts = [], [], [], Counter()
    visits, forced, premature = 0, 0, 0
    for sp in specs('train', 2):
        for variant in (0, 1):
            w = World.create(sp)
            targets = fixture.targets_for(sp, seed+variant*41, mode)
            rng = random.Random(seed*811+sp.seed*43+variant*19+stage)
            for t in range(stage):
                if w.stopped:
                    break
                g = targets[fixture.active_phase(t, mode)]
                menu, features = prior.matrix(w, g, stage, False)
                label = prior.label(w, g, stage, rng)
                if label not in menu:
                    raise ValueError('illegal teacher label')
                mask = np.zeros(policy.ACTION_MAX, bool)
                mask[:len(menu)] = True
                X.append(features); masks.append(mask)
                labels.append(menu.index(label)); counts[label] += 1
                action, _ = model.choose(w, g, stage, False)
                # Matched perturbation opportunities, not synthetic free moves.
                if t in (19, 57, 91, 119):
                    alternative = [a for a in menu if a in MOVE and a != action]
                    if alternative:
                        action = rng.choice(alternative)
                        forced += 1
                w.apply(action, 12); visits += 1
            premature += int(w.stopped and w.steps < stage)
    if not X:
        raise ValueError('empty student-visited corpus')
    rng = np.random.default_rng(seed*997+stage*47)
    index = rng.integers(0, len(X), size=sample_n)
    return (np.stack(X)[index], np.stack(masks)[index],
            np.asarray(labels, np.int64)[index], {
                'split': 'train', 'mode': mode, 'visited_states': visits,
                'unique_training_states': len(X), 'samples': sample_n,
                'student_early_stops': premature, 'forced_legal_deviations': forced,
                'teacher_action_counts': dict(sorted(counts.items()))})


def train(seed, base_updates, adapt_updates, sample_n=768):
    if not (1 <= base_updates <= 100 and 1 <= adapt_updates <= 60
            and 32 <= sample_n <= 4096):
        raise ValueError('bounded optimizer limits')
    originals, history = prior.train(seed, base_updates)
    base = originals['goal_recovery']
    models = {'frozen': base, 'static_adapt': copy.deepcopy(base),
              'retarget_adapt': copy.deepcopy(base)}
    adaptation = []
    for stage in STAGES:
        record = {'stage': stage}
        for arm, mode in (('static_adapt', 'static'), ('retarget_adapt', 'retarget')):
            x, mask, y, meta = student_corpus(models[arm], seed, stage, mode, sample_n)
            outcome = models[arm].fit(x, mask, y, adapt_updates,
                                      seed*313+stage*7)
            record[arm] = {'corpus': meta, 'optimizer': outcome}
        adaptation.append(record)
    return models, {'initial_training': history, 'on_policy_stages': adaptation}


def cases():
    return fixture.cases()  # DEVELOPMENT only, earlier published fixture frozen.


def measure(models, seed, quick=False, image_dir=None):
    eps = []
    for sp in (cases()[:1] if quick else cases()):
        for horizon in (128,) if quick else (128, 160):
            for mode in (('retarget',) if quick else MODES):
                for arm in ARMS:
                    png = (image_dir / f'{sp.ident}-{seed}-{arm}.png'
                           if image_dir is not None and horizon==160
                           and mode=='retarget' and sp==cases()[0] else None)
                    ep = fixture.run_episode(sp, {'goal_recovery': models[arm]},
                                             'goal_recovery', seed, horizon,
                                             mode, png)
                    ep['trained_arm'] = arm
                    fixture.verify_episode(ep, sp, {'goal_recovery': models[arm]})
                    eps.append(ep)
    return eps


def summarize(eps):
    result = {}
    for mode in MODES:
        result[mode] = {}
        for arm in ARMS:
            rows = [e for e in eps if e['mode']==mode and e['trained_arm']==arm]
            if not rows:
                continue
            counts = Counter()
            for e in rows:
                counts.update(e['action_counts'])
            windows = [w for e in rows for w in e['switch_windows'].values()]
            windows = [w for w in windows if w['steps'] and e_is_retarget(mode)]
            result[mode][arm] = {
                'episodes': len(rows),
                'mean_painted': round(statistics.mean(e['metrics']['unique_painted_cells'] for e in rows), 4),
                'early_stops': sum(bool(e['metrics']['stopped_early']) for e in rows),
                'targets_during_instruction': sum(len(e['reached_epochs']) for e in rows),
                'mean_switch_gain': round(statistics.mean(w['net_distance_gain'] for w in windows), 4) if windows else None,
                'mean_latency_ms': round(statistics.mean(e['mean_latency_ms'] for e in rows), 6),
                'actions': dict(sorted(counts.items()))}
    return result


def e_is_retarget(mode):
    return mode == 'retarget'


def provenance():
    directory = Path(__file__).parent
    return {n: sha_file(directory/n) for n in SOURCES}


def run(out, seed=619, base_updates=60, adapt_updates=20,
        quick=False, sample_n=768):
    if out.exists():
        raise FileExistsError('immutable evidence directory already exists')
    start = time.perf_counter()
    out.mkdir(parents=True)
    models, histories = train(seed, base_updates, adapt_updates, sample_n)
    checkpoints = {}
    for arm, model in models.items():
        cp = out/(arm+'.npz')
        model.save(cp)
        loaded = prior.Ranker.load(cp)
        if any(not np.array_equal(model.p[k], loaded.p[k]) for k in model.p):
            raise ValueError('checkpoint restore mismatch')
        checkpoints[arm] = {'sha256': sha_file(cp),
            'parameters': int(sum(v.size for v in model.p.values())),
            'optimizer_steps': model.updates}
    (out/'traces').mkdir()
    (out/'png').mkdir()
    episodes = measure(models, seed, quick, out/'png')
    for i, e in enumerate(episodes):
        (out/'traces'/f'{i:03d}-{e["trained_arm"]}.json').write_text(
            json.dumps(e, sort_keys=True, separators=(',', ':'))+'\n')
    report = {'version': VERSION, 'seed': seed,
              'base_updates_per_stage': base_updates,
              'adapt_updates_per_stage': adapt_updates,
              'matched_adaptation_samples_per_stage': sample_n,
              'histories': histories, 'checkpoints': checkpoints,
              'evaluation': summarize(episodes), 'episodes':len(episodes),
              'decisions': sum(len(e['rows']) for e in episodes),
              'source_sha256': provenance(), 'evaluation_split':'development',
              'structural_heldout_opened': False, 'native_receipts':0,
              'paid_compute':False,
              'elapsed_s': round(time.perf_counter()-start,3),
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({
        str(f.relative_to(out)):sha_file(f) for f in out.rglob('*') if f.is_file()
        }, indent=2, sort_keys=True)+'\n')
    return report


def verify(out):
    report=json.loads((out/'report.json').read_text())
    if report['version']!=VERSION or report['source_sha256']!=provenance():
        raise ValueError('source hashes changed')
    actual={str(f.relative_to(out)):sha_file(f) for f in out.rglob('*')
            if f.is_file() and f.name!='SHA256SUMS.json'}
    if actual != json.loads((out/'SHA256SUMS.json').read_text()):
        raise ValueError('artifact hashes changed')
    models={arm:prior.Ranker.load(out/(arm+'.npz')) for arm in ARMS}
    for arm in ARMS:
        if sha_file(out/(arm+'.npz'))!=report['checkpoints'][arm]['sha256'] or \
           models[arm].updates!=report['checkpoints'][arm]['optimizer_steps']:
            raise ValueError('checkpoint identity differs')
    case_map={sp.ident:sp for sp in cases()}
    eps=[]
    for path in sorted((out/'traces').glob('*.json')):
        ep=json.loads(path.read_text())
        arm=ep['trained_arm']
        fixture.verify_episode(ep, case_map[ep['spec']],
                               {'goal_recovery': models[arm]})
        eps.append(ep)
    if len(eps)!=report['episodes'] or sum(len(e['rows']) for e in eps)!=report['decisions']:
        raise ValueError('episode/decision count differs')
    s=summarize(eps)
    for mode in s:
        for arm in s[mode]:
            a=dict(s[mode][arm]);b=dict(report['evaluation'][mode][arm])
            a.pop('mean_latency_ms');b.pop('mean_latency_ms')
            if a!=b:
                raise ValueError('summary differs')
    return {'episodes':len(eps),'decisions':report['decisions'],
            'checkpoints':len(models),'verified_hashes':len(actual)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--seed',type=int,default=619)
    parser.add_argument('--base-updates',type=int,default=60)
    parser.add_argument('--adapt-updates',type=int,default=20)
    parser.add_argument('--sample-n',type=int,default=768)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(
        args.out,args.seed,args.base_updates,args.adapt_updates,args.quick,args.sample_n),sort_keys=True))
