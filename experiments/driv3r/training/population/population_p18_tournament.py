"""P18: novelty-versus-hardness selection in 48-turn synthetic game populations on synthetic CyberSand-like maps.

Research only; never imports CyberSand or edits runtime. Parent selection, curriculum,
and synthetic preferences remain handcrafted; legal actions are host-filtered.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
from population_lab import (DIRECTIONS, NICHES, Policy, World, draw_svg,
                            softmax, stable_seed, utility)

ARMS = ('reward_pbt', 'hardness_pbt', 'novelty_pbt', 'random_parent')
TRAIN_CARDS = (0, 1, 2, 3)
HOLDOUT_CARDS = (4, 5)
ACTIONS = tuple(DIRECTIONS) + ('RETRACE', 'STOP')


def scenario(seed: int, card: int, horizon: int) -> World:
    """Six fixed scenario *families*, split by family before model selection.

    All are external modifications to a synthetic World; never native evidence.
    Each family preserves source-host menu legality, original paint tree and
    changing target. Material hazards are a protected-cell surrogate.
    """
    if card not in range(6) or not 24 <= horizon <= 80:
        raise ValueError('invalid scenario card or horizon')
    w = World.generate(seed, max_steps=horizon)
    rng = np.random.default_rng(stable_seed(seed, 'p18-geometry', card))
    def add_obstacle(points):
        for p in points:
            if 1 <= p[0] < 21 and 1 <= p[1] < 21:
                w.obstacles.add(p)
    def add_protected(points):
        for p in points:
            if 1 <= p[0] < 21 and 1 <= p[1] < 21:
                w.protected.add(p)
    if card == 1:  # dead-end comb and a recovery corridor
        for x in (5, 10, 15):
            gap = 3 + (x * 3 + seed) % 12
            add_obstacle((x, y) for y in range(3, 19) if abs(y - gap) > 1)
    elif card == 2:  # protected-material obstacle channel
        add_protected((x, y) for x in range(5, 18) for y in (6, 7, 15) if (x + seed) % 5 != 0)
    elif card == 3:  # staggered long detours and goal reversal
        for y in (5, 11, 17):
            gap = 4 + (y + seed) % 13
            add_obstacle((x, y) for x in range(4, 20) if abs(x - gap) > 1)
    elif card == 4:  # heldout: protected diagonal and diagonal cul-de-sacs
        offset = int(rng.integers(0, 4))
        add_protected((x, 4 + (x + offset) % 11) for x in range(5, 19))
        add_obstacle((x, 5 + (x + 2 * offset) % 10) for x in range(6, 19))
    elif card == 5:  # heldout: material teeth interrupted by narrow exits
        offset = int(rng.integers(0, 3))
        for x in (6 + offset, 10 + offset, 14 + offset, 18 + offset):
            add_protected((x, y) for y in range(4, 19) if (y + x + seed) % 7 not in (0, 1))
    # The initial cursor is always legal. Protected and obstacle sets disjoint.
    reserved = {w.cursor, w.goal, w.second_goal}
    w.obstacles.difference_update(reserved)
    w.protected.difference_update(reserved)
    w.obstacles.difference_update(w.protected)
    # Guarantee at least one exploratory choice without hiding later traps.
    for dx, dy in DIRECTIONS.values():
        p = (w.cursor[0] + dx, w.cursor[1] + dy)
        w.obstacles.discard(p)
        w.protected.discard(p)
    assert not w.painted & (w.protected | w.obstacles)
    return w


def trajectory(policy, niche, seed, card, horizon, *, teacher_mix=0., train_mode=False):
    """Actually evolve state using the selected legal action each decision."""
    rng = np.random.default_rng(stable_seed('p18-policy', seed, card, niche))
    w = scenario(seed, card, horizon)
    total = 0.; losses = []
    chosen = []; progress_before = []; progress_after = []
    for _ in range(horizon):
        legal = w.legal()
        X = np.stack([w.feature(a) for a in legal])
        scores = np.asarray([utility(niche, x, w) for x in X])
        if train_mode:
            target = softmax(scores / .42)
            losses.append(policy.teach(X, target))
            idx = int(rng.choice(len(legal), p=target)) if rng.random() < teacher_mix else policy.choice(X, rng)
        elif policy is None:
            valid = [i for i, a in enumerate(legal) if a != 'STOP']
            idx = int(rng.choice(valid)) if valid else legal.index('STOP')
        elif policy == 'oracle':
            idx = int(scores.argmax())
        elif policy == 'persistent':
            idx = next((i for i, a in enumerate(legal) if a == w.heading), None)
            if idx is None:
                idx = next((i for i, a in enumerate(legal) if a in DIRECTIONS), None)
            if idx is None:
                idx = legal.index('RETRACE') if 'RETRACE' in legal else legal.index('STOP')
        else:
            idx = policy.choice(X)
        action = legal[idx]
        assert action in w.legal()
        value = float(scores[idx]); total += value
        (progress_after if w.switched else progress_before).append(float(X[idx, 4]))
        w.step(action)
        chosen.append(action)
        if action == 'STOP':
            break
    assert all(h['action'] in ACTIONS for h in w.history)
    assert not (w.painted & (w.obstacles | w.protected))
    junctions = len({p for p in w.parent.values() if list(w.parent.values()).count(p) > 1})
    stats = dict(seed=seed, card=card, reward=round(total, 5),
        steps=w.steps, painted=len(w.painted) - 1, retrace=chosen.count('RETRACE'),
        branch_junctions=junctions, early_stop=int(w.stopped and w.steps < horizon),
        changed_goal=int(w.switched), pre_switch_progress=round(sum(progress_before), 4),
        post_switch_progress=round(sum(progress_after), 4),
        legal_rate=1., illegal_paint=0, descriptor=w.descriptor(),
        actions=chosen, mean_train_ce=(round(float(np.mean(losses)), 5) if losses else None))
    return stats, w


def summary(rows):
    d = {key: round(float(np.mean([r[key] for r in rows])), 5)
         for key in ('reward','steps','painted','retrace','branch_junctions',
                     'early_stop','changed_goal','pre_switch_progress','post_switch_progress','legal_rate','illegal_paint')}
    d['mean_descriptor'] = np.mean([r['descriptor'] for r in rows], axis=0).round(5).tolist()
    d['by_card'] = {str(c): round(float(np.mean([r['reward'] for r in rows if r['card']==c])), 5)
                    for c in sorted(set(r['card'] for r in rows))}
    return d


def challenge_cards(policy, niche, train_seed, horizon):
    """Training-only frozen-parent probe. No selection or holdout peeking."""
    scores = []
    for card in TRAIN_CARDS:
        outcome, _ = trajectory(policy, niche, stable_seed(train_seed, 'probe', card), card, horizon)
        oracle, _ = trajectory('oracle', niche, stable_seed(train_seed, 'probe', card), card, horizon)
        scores.append(oracle['reward'] - outcome['reward'] + 3 * outcome['early_stop'])
    return sorted(range(4), key=lambda j: (-scores[j], j)), scores


def teach_offspring(policy, niche, seeds, horizon, arm):
    update_before = policy.updates
    chosen_cards = []; hardness_calls = 0; losses = []
    for i, s in enumerate(seeds):
        if arm == 'hardness_pbt':
            ranked, _ = challenge_cards(policy, niche, s, horizon)
            card = ranked[0]
            hardness_calls += 8  # four policy and four oracle training-only probes
        else:
            card = TRAIN_CARDS[i % len(TRAIN_CARDS)]
        out, _ = trajectory(policy, niche, s, card, horizon, teacher_mix=.25, train_mode=True)
        chosen_cards.append(card)
        if out['mean_train_ce'] is not None:
            losses.append(out['mean_train_ce'])
    return dict(updates=policy.updates-update_before, probe_rollouts=hardness_calls,
                training_episodes=len(seeds), curriculum_cards=chosen_cards,
                mean_train_ce=round(float(np.mean(losses)),5))


def dist(a,b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))


def lineage_ranking(records, arm):
    """On selection-only descriptors; no holdout information accessible."""
    if arm != 'novelty_pbt':
        return sorted(records, key=lambda r:(-r['selection']['reward'], r['id']))
    scores = [r['selection']['reward'] for r in records]
    low, high = min(scores), max(scores)
    raw_novelty = {}
    for r in records:
        others = [x for x in records if x is not r]
        nearest = sorted(dist(r['selection']['mean_descriptor'],x['selection']['mean_descriptor'])
                         for x in others)
        raw_novelty[r['id']] = float(np.mean(nearest[:3])) if nearest else 0.
    nmin,nmax = min(raw_novelty.values()),max(raw_novelty.values())
    ranked = []
    for r in records:
        # Normalize BOTH signals within the selection-only candidate pool;
        # otherwise the reward range silently dominates the novelty term.
        reward = (r['selection']['reward']-low)/(high-low+1e-8)
        novelty = (raw_novelty[r['id']]-nmin)/(nmax-nmin+1e-8)
        ranked.append((.35*reward + .65*novelty,r))
    return [r for _,r in sorted(ranked,key=lambda pair:(-pair[0],pair[1]['id']))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(folder):
    files = sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (Path(folder)/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(folder).as_posix()}\n' for p in files))


def run(output, cohorts=(676,677), initial=2, generations=2, offspring=2,
        training_worlds=6, selection_worlds=3, heldout_worlds=4, horizon=48,
        arms=ARMS):
    if (not 1<=len(cohorts)<=5 or len(set(cohorts))!=len(cohorts) or
        not 2<=initial<=4 or not 1<=generations<=3 or not 1<=offspring<=4 or
        not 2<=training_worlds<=16 or not 2<=selection_worlds<=8 or
        not 2<=heldout_worlds<=12 or not 24<=horizon<=80 or
        not arms or any(a not in ARMS for a in arms) or len(set(arms))!=len(arms)):
        raise ValueError('P18 bounded CPU envelope')
    out = Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'checkpoints').mkdir();(out/'gallery').mkdir()
    started=time.monotonic(); summaries=[]; ledger=[]; seed_records=[]; holdout_rows=[]
    for seed in cohorts:
        splits = {name:[stable_seed('p18',seed,name,i) for i in range(count)]
                  for name,count in (('train',training_worlds),('selection',selection_worlds),('holdout',heldout_worlds))}
        flat = sum(splits.values(),[])
        assert len(flat)==len(set(flat))
        seed_records.append(dict(cohort=seed, splits={k:dict(seeds=v,sha256=hashlib.sha256(json.dumps(v).encode()).hexdigest()) for k,v in splits.items()}))
        for niche in NICHES:
            for arm in arms:
                members=[]; parents=[]; current=[]
                for gen in range(generations+1):
                    n = initial if gen == 0 else offspring
                    nextgen=[]
                    for j in range(n):
                        identity=f'{seed}-{niche}-{arm}-g{gen}-{j}'
                        if gen==0:
                            pseed=stable_seed('p18-initial',seed,niche,j)
                            p=Policy(pseed,lr=.004*(1+j*.15),temperature=.9+j*.1)
                            parent_id=None
                        else:
                            pool=lineage_ranking(current,arm)
                            if arm=='random_parent':
                                parent= current[stable_seed('p18-random-parent',seed,niche,gen,j)%len(current)]
                            else:
                                parent=pool[j%min(2,len(pool))]
                            parent_id=parent['id']
                            pseed=stable_seed('p18-mutate',seed,niche,gen,j)
                            p=parent['policy'].mutate(pseed)
                        before=hashlib.sha256(json.dumps(p.serialized(),sort_keys=True).encode()).hexdigest()
                        trseeds=[stable_seed(s,'epoch',gen,j) for s in splits['train']]
                        fit=teach_offspring(p,niche,trseeds,horizon,arm)
                        after=hashlib.sha256(json.dumps(p.serialized(),sort_keys=True).encode()).hexdigest()
                        assert before!=after and fit['updates']>0
                        select=[trajectory(p,niche,s,c,horizon)[0] for s,c in itertools.product(splits['selection'],TRAIN_CARDS)]
                        record=dict(id=identity,cohort=seed,niche=niche,arm=arm,generation=gen,
                            parent=parent_id,initial_or_mutation_seed=pseed,checkpoint_sha256=None,
                            initial_model_sha256=before,final_model_sha256=after,
                            training=fit,selection=summary(select),policy=p)
                        ck=out/'checkpoints'/f'{identity}.json'
                        ck.write_text(json.dumps(p.serialized(),separators=(',',':'))+'\n')
                        record['checkpoint_sha256']=sha(ck)
                        nextgen.append(record);members.append(record)
                    current=nextgen
                selected=lineage_ranking(members,arm)[:2]
                assert len(selected)==2
                for rank,r in enumerate(selected):
                    tests=[trajectory(r['policy'],niche,s,c,horizon)[0]
                           for s,c in itertools.product(splits['holdout'],HOLDOUT_CARDS)]
                    holdout_rows += [dict(id=r['id'],**item) for item in tests]
                    stat=summary(tests)
                    if rank == 0:
                        gseed=splits['holdout'][0]
                        tr,w=trajectory(r['policy'],niche,gseed,HOLDOUT_CARDS[0],horizon)
                        gallery=out/'gallery'/f'{seed}-{arm}-{niche}'
                        gallery.with_suffix('.svg').write_text(draw_svg(w))
                        gallery.with_suffix('.json').write_text(json.dumps(dict(
                            model=r['id'],cohort=seed,niche=niche,arm=arm,native=False,
                            scenario_card=HOLDOUT_CARDS[0],metrics=tr,obstacles=sorted(w.obstacles),
                            protected=sorted(w.protected),painted=sorted(w.painted),
                            parent_edges=[[list(k),list(v)] for k,v in w.parent.items()],
                            goal_switch_turn=horizon//2),indent=2)+'\n')
                    summaries.append(dict(cohort=seed,arm=arm,niche=niche,rank=rank,
                                          selected=r['id'],selection_reward=r['selection']['reward'],heldout=stat))
                for r in members:
                    ledger.append({k:v for k,v in r.items() if k!='policy'})
    agg={}
    for arm in arms:
        xs=[r for r in summaries if r['arm']==arm and r['rank']==0]
        agg[arm]={k:round(float(np.mean([r['heldout'][k] for r in xs])),5)
                  for k in ('reward','painted','retrace','branch_junctions','early_stop','changed_goal','post_switch_progress','legal_rate')}
        desc=[r['heldout']['mean_descriptor'] for r in xs]
        agg[arm]['across_niche_descriptor_diversity']=round(float(np.mean([dist(a,b) for i,a in enumerate(desc) for b in desc[i+1:]])),5)
        agg[arm]['evaluated_champions']=len(xs)
    controls={}
    # Independent synthetic controls receive exactly the same untouched heldout maps.
    for niche in NICHES:
        for kind,controller in (('nonstop_random',None),('persistent','persistent'),('oracle','oracle')):
            rows=[trajectory(controller,niche,s,c,horizon)[0] for seed in cohorts
                  for s in next(d for d in seed_records if d['cohort']==seed)['splits']['holdout']['seeds']
                  for c in HOLDOUT_CARDS]
            controls[f'{niche}:{kind}']=summary(rows)
    report=dict(format='population-p18-autonomous-tournament',native=False,pretrained_slm=False,
        source='synthetic-P0',cohorts=list(cohorts),niches=list(NICHES),arms=list(arms),
        parameters=dict(initial=initial,generations=generations,offspring=offspring,
          training_worlds=training_worlds,selection_worlds=selection_worlds,
          heldout_worlds=heldout_worlds,horizon=horizon),
        split_manifest=seed_records,selection=summaries,aggregate=agg,controls=controls,
        candidates=len(ledger),optimizer_updates=sum(r['training']['updates'] for r in ledger),
        train_probe_rollouts=sum(r['training']['probe_rollouts'] for r in ledger),
        elapsed_s=round(time.monotonic()-started,3),python=platform.python_version(),numpy=np.__version__,
        limitations=['Synthetic 22x22 World, no CyberSand native/WB validation',
         'Handcrafted niche utility and scenario-hardness are not human preferences',
         'Host masking enforces legality by construction; cannot claim learned safety',
         'Six niche labels are separately trained, not verified artistic skill',
         'PBT offspring are teacher-augmented supervised Adam, not policy-gradient RL',
         'Hardness arm spends additional TRAIN-only environment probes; unequal wall-clock cost',
         'Changed goals occur mid-episode only if the policy stays active long enough',
         'No pretrained SLM or actual human/game self-play'])
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'lineage.json').write_text(json.dumps(ledger,indent=2)+'\n')
    # Useful small evidence export. Full original trajectories are retained as JSON in gallery.
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['cohort','arm','niche','rank','selected','selection_reward','heldout_reward','heldout_painted','heldout_retrace','heldout_post_switch'])
        writer.writeheader()
        for r in summaries:
            writer.writerow(dict(cohort=r['cohort'],arm=r['arm'],niche=r['niche'],rank=r['rank'],
                selected=r['selected'],selection_reward=r['selection_reward'],
                heldout_reward=r['heldout']['reward'],heldout_painted=r['heldout']['painted'],
                heldout_retrace=r['heldout']['retrace'],heldout_post_switch=r['heldout']['post_switch_progress']))
    manifest(out)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--cohorts',type=int,nargs='+',default=[676,677])
    p.add_argument('--initial',type=int,default=2)
    p.add_argument('--generations',type=int,default=2)
    p.add_argument('--offspring',type=int,default=2)
    p.add_argument('--training-worlds',type=int,default=6)
    p.add_argument('--selection-worlds',type=int,default=3)
    p.add_argument('--heldout-worlds',type=int,default=4)
    p.add_argument('--horizon',type=int,default=48)
    a=p.parse_args()
    r=run(**vars(a))
    print(json.dumps(dict(cohorts=r['cohorts'],candidates=r['candidates'],optimizer_updates=r['optimizer_updates'],aggregate=r['aggregate']),sort_keys=True))
if __name__=='__main__':main()
