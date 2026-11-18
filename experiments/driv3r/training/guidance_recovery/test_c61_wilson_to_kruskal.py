"""C61 source-matched regression and evidence consistency (synthetic host)."""
import ast,collections,hashlib,json,sys,zlib,base64
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'lab'))
import c61_wilson_to_kruskal as c61


def data():
    summary=json.loads((BASE/'results/C61-summary.json').read_text())
    ledger=json.loads(zlib.decompress(base64.b64decode((BASE/'results/C61-executed-ledger.zlib.b64').read_text())))
    return summary,ledger


def test_cross_family_disjoint_graphs():
    s,_=data()
    training=s['train_graphs'];test=s['test_graphs']
    allids=[x['graph'] for x in training+test]
    assert len(allids)==len(set(allids))
    assert {x['family'] for x in training}=={'tree','braid','prim','prim_braid','wilson'}
    assert {x['family'] for x in test}=={'tree','braid','prim','prim_braid','wilson','kruskal'}
    assert not any(x['family']=='kruskal' for x in training)
    assert len([r for r in test if r['family']=='kruskal'])==24


def test_full_ledger_and_safe_actions():
    s,l=data();rows=l['results']
    assert len(rows)==s['matched_conditions'] and len(rows)*len(c61.POLICIES)==s['suffixes']
    assert all(set(r['outcomes'])==set(c61.POLICIES) for r in rows)
    for r in rows:
        sha={v['protected_sha'] for v in r['outcomes'].values()}
        assert len(sha)==1 and all(v['host_rejections']==0 for v in r['outcomes'].values())
        if r['arm'][0]=='none':
            assert len({(v['action_sha'],v['ink_sha']) for v in r['outcomes'].values()})==1


def test_actual_optimizer_and_shuffled_control():
    s,_=data()
    assert s['training']['updates']>0 and s['shuffled_training']['updates']>0
    # Unequal number of minibatches (12 vs 9) is disclosed, not concealed.
    for key in c61.CHECKPOINTS:
        assert hashlib.sha256((BASE/'results'/(key+'.safetensors')).read_bytes()).hexdigest()==s['checkpoint_sha256'][key]
    assert np.linalg.norm(np.array(s['weights']['C61-WILSON'])-s['weights']['C61-SHUFFLED'])>0


def test_frozen_C60_identity_and_no_hidden_policy_map():
    s,_=data();p=BASE/'publication/C60-summary.json'
    assert s['frozen_C60_checkpoint_sha256']==json.loads(p.read_text())['checkpoint_sha256']['ALIGNED']
    for n,digest in s['policy_source_hashes'].items():
        assert hashlib.sha256((BASE/'frozen/research-source'/n).read_bytes()).hexdigest()==digest
    src=(BASE/'frozen/research-source/c60_remote_frontier_rl.py').read_text()
    t=ast.parse(src)
    for name in ('action_features','explored_route','exploration_delta'):
        fn=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
        attrs={v.attr for v in ast.walk(fn) if isinstance(v,ast.Attribute)}
        assert attrs.isdisjoint({'protected','allowed','material_at'})


def test_gallery_is_executed_ledger_only():
    s,l=data();traces=l['traces']
    assert len(traces)==s['replayed_traces']
    for t in traces:
        matches=[r for r in l['results'] if (r['graph'],r['drop'],r['arm'])==(t['graph'],t['drop'],['backtrack',1.])]
        assert len(matches)==1
        out=matches[0]['outcomes'][t['policy']]
        assert (t['action_sha'],t['ink_sha'])==(out['action_sha'],out['ink_sha'])
        assert len(t['path'])==out['steps']


def test_stochastic_not_forced():
    weights=np.full(c61.c60.NFEAT,100,dtype=np.float32)
    features=np.array([[1]*c61.c60.NFEAT,[-1]*c61.c60.NFEAT],np.float32)
    bias=c61.c60.exploration_delta(None,None,[None,None],features,'C60-ALIGNED',weights,1.)
    assert np.all(abs(bias)<=c61.c60.BONUS+1e-5)
    z=c61.c60.softmax([0.,100.]);assert np.isfinite(z).all() and z.sum()>.999
