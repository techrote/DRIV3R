"""C42 synthetic maze, legality, disjoint-graph and evidence regression tests."""
from __future__ import annotations
import collections, json
from pathlib import Path
import numpy as np
import c42_wilson_transfer as c
import c34_independent_maze as m34
from experiment import DIRECTIONS,sha


def test_independent_generators_connected_and_acyclic():
    for family in c.FAMILIES:
        for seed in (31,127,1999):
            a=c.generate(seed,family)
            assert a==c.generate(seed,family)
            assert len(a)==97
            seen={(1,1)};frontier=[(1,1)]
            while frontier:
                p=frontier.pop()
                for dx,dy in DIRECTIONS:
                    q=(p[0]+dx,p[1]+dy)
                    if q in a and q not in seen:seen.add(q);frontier.append(q)
            assert seen==a
            assert sum(sum((x+dx,y+dy) in a for dx,dy in DIRECTIONS) for x,y in a)//2==96
        assert len({sha(sorted(c.generate(s,family))) for s in range(12)})==12
    assert c.generate(31,'wilson')!=c.generate(31,'kruskal')


def test_host_prefix_and_protected_contract():
    for family in c.FAMILIES:
        for case in ('detour','forward'):
            w,info=c.world(4891,family,case,-1)
            assert len(w.history)==24
            assert w.invalid_attempts==0
            assert not any(p in w.protected for p in w.history)
            assert w.goal not in w.protected
            assert info['initial_retreat']==(case=='detour')
            assert info['graph_sha']==sha(sorted((x,y) for x in range(15) for y in range(15) if (x,y) not in w.protected))
            menu=w.candidates(.35)
            legal=[i for i,t in enumerate(menu) if t.legal]
            assert len(legal)>=1
            for j in legal:
                from c34_independent_maze import executed_suffix
                assert np.isfinite(executed_suffix(w,j,steps=2))


def test_training_and_holdout_graphs_disjoint():
    train,receipts=c.collect(per_group=2)
    blind=c.offline({},per_group=2)
    t={x[0] for x in receipts};b={x['graph_sha'] for x in blind}
    assert len(t)==16 and len(b)==16 and not t.intersection(b)
    assert all(np.isfinite(y).all() for _,y in train)


def test_evidence_invariants():
    fp=Path(__file__).parent.parent/'output'/'C42-summary.json'
    if not fp.exists():return
    report=json.loads(fp.read_text())
    assert report['host_rejections']==0
    assert report['blind_graphs']==64 and report['train_graphs']==72 and report['online_graphs']==32
    assert report['online_episodes']==1024
    assert all(r['unguided_exact']==32 for r in report['models'].values())
    assert report['models']['C42-PAIR']['guided_goals']==18
    assert report['models']['C34-RESET']['guided_goals']==19
    assert report['models']['C42-SHUFFLED']['guided_goals']==18
