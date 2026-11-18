"""C60 synthetic-host legality and evidence regression tests."""
import ast, hashlib, inspect, json, pathlib
import numpy as np
from types import SimpleNamespace
import c60_remote_frontier_rl as c60

ROOT=pathlib.Path(__file__).resolve().parents[1]


def test_no_hidden_map_access_in_policy_features():
    source=inspect.getsource(c60.action_features)+inspect.getsource(c60.explored_route)+inspect.getsource(c60.exploration_delta)
    tree=ast.parse(source)
    attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
    assert 'protected' not in attrs and 'allowed' not in attrs and 'material_at' not in attrs
    class W:
        pos=(3,4);goal=(9,9);last_dir=(1,0);personality=-1
        def distance(self,p=None):return 11 if p is None else 8
    ledger=SimpleNamespace(visits=__import__('collections').Counter(),actions=[],frontiers=lambda :set())
    menu=[SimpleNamespace(action=('INK',1,0),target=(4,4),legal=True)]
    x=c60.action_features(W(),ledger,menu)
    assert x.shape==(1,12) and np.isfinite(x).all()


def test_probabilities_and_bias_bounded():
    w=np.ones(c60.NFEAT,dtype=np.float32)*100
    fs=np.random.default_rng(7).normal(size=(9,c60.NFEAT)).astype('float32')
    d=c60.exploration_delta(None,None,[None]*9,fs,'C60-ALIGNED',w,1.)
    assert np.max(np.abs(d))<=c60.BONUS+1e-5
    p=c60.softmax([-1e9,0,2,3])
    assert np.isfinite(p).all() and np.isclose(p.sum(),1) and p[0]>=0


def test_executable_heldout_graph_disjointness():
    s=json.loads((ROOT/'results/C60-summary.json').read_text())
    tr={x['graph'] for x in s['cohorts']['train']}
    on={x['graph'] for x in s['cohorts']['online']}
    assert tr and on and tr.isdisjoint(on)
    assert len(tr)==s['training_graphs'] and len(on)==s['online_graphs']
    assert s['eligible_train_graphs']>0 and s['eligible_online_graphs']>0


def test_real_optimizer_and_shuffled_control():
    s=json.loads((ROOT/'results/C60-summary.json').read_text())
    assert s['training']['updates']>0
    assert s['shuffled_control_training']['updates']==s['training']['updates']
    assert np.linalg.norm(np.asarray(s['weights']['C60-ALIGNED'])-s['weights']['C60-SHUFFLED'])>0.01
    for m in ('ALIGNED','SHUFFLED'):
        p=ROOT/f'results/C60-{m}.safetensors'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==s['checkpoint_sha256'][m]


def test_full_ledger_host_legality_and_no_guidance_parity():
    rows=json.loads((ROOT/'results/C60-online.json').read_text())
    s=json.loads((ROOT/'results/C60-summary.json').read_text())
    assert len(rows)==s['records']
    assert len(rows)*len(c60.POLICIES)==s['policy_suffixes']
    unguided=0
    for r in rows:
        assert r['outcomes'] and set(r['outcomes'])==set(c60.POLICIES)
        for z in r['outcomes'].values():
            assert z['host_rejections']==0
            assert z['protected_sha'] == r['outcomes']['C34-RESET']['protected_sha']
        if r['arm'][0]=='none':
            unguided+=1
            assert len({(z['action_sha'],z['ink_sha']) for z in r['outcomes'].values()})==1
    assert unguided==s['unguided_parity_conditions']


def test_trajectory_receipts_match_online_outcomes():
    rows=json.loads((ROOT/'results/C60-online.json').read_text())
    traces=json.loads((ROOT/'results/C60-traces.json').read_text())
    assert len(traces)==json.loads((ROOT/'results/C60-summary.json').read_text())['replayed_traces']
    for tr in traces:
        found=[r for r in rows if (r['graph'],r['drop'],tr['policy'],r['arm'])==(tr['graph'],tr['drop'],tr['policy'],['backtrack',1.])]
        assert len(found)==1
        z=found[0]['outcomes'][tr['policy']]
        assert tr['outcome']['action_sha']==z['action_sha'] and tr['outcome']['ink_sha']==z['ink_sha']
