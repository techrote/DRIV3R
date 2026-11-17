"""Targeted synthetic-fixture regression checks. No native CyberSand claims."""
import json
from pathlib import Path
import numpy as np
import c41_topology_transfer as c
from c34_independent_maze import shortest

def test_prim_connectivity_and_different_from_dfs():
    for s in (101,203,307,409):
        a=c.independent_prim(s,False);b=c.independent_prim(s,True)
        assert len(a)==97 and len(b)==107 and a<=b
        assert all(shortest(a,(1,1),x) for x in a)
        assert a!=c._ORIGINAL_MAZE(s,'tree')

def test_world_host_legal_prefix_and_protected():
    for fam in c.FAMILIES:
        w,info=c.world(410701+sum(map(ord,fam)),fam,'detour',1)
        assert len(w.history)==24 and w.invalid_attempts==0
        assert w.pos not in w.protected and w.goal not in w.protected
        assert info['graph_sha']==c.sha(sorted((x,y) for x in range(15) for y in range(15) if (x,y) not in w.protected))

def test_blind_oracle_labels_and_policy_separation():
    assert c.ARMS[0]==('none',0.)
    assert set(('C40-PAIR','C41-MIX','C41-SHUFFLED','C41-RANDOM'))<=set(c.MODELS)
    assert all(f in c.FAMILIES for f in ('prim','prim_braid'))
    assert c.collect(per_group=1)[0]

def test_summary_integrity():
    f=Path(__file__).parent/'C41_out/C41-summary.json'
    if not f.exists():return
    s=json.loads(f.read_text())
    assert s['online_episodes']==2304 and s['graphs_online']==96 and s['graphs_offline']==144
    assert s['host_rejections']==0
    for m in c.MODELS:
        assert s['models'][m]['unguided_exact']==96
        assert s['models'][m]['unguided_n']==96
