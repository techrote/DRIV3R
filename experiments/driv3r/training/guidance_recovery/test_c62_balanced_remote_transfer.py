"""C62 evidence integrity / scientific control tests. No native game mutation."""
import ast,base64,collections,hashlib,json,sys,zlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'lab'))
import c62_balanced_remote_transfer as c62

def data():
    out=ROOT/'results62'
    summary=json.loads((out/'C62-summary.json').read_text())
    ledger=json.loads(zlib.decompress(base64.b64decode((out/'C62-executed-ledger.zlib.b64').read_text())))
    return summary,ledger

def test_cohort_separation():
    s,_=data()
    tr=s['graph_receipts'][0]+s['graph_receipts'][1]
    te=s['graph_receipts'][2]+s['graph_receipts'][3]
    ids=[r['graph'] for r in tr+te]
    assert len(ids)==len(set(ids))
    assert all(r['family']!='kruskal' for r in tr)
    assert len([x for x in te if x['family']=='kruskal'])==8
    assert s['past_graph_identities_excluded']>=80
    assert s['eligible_train_prefixes']>20 and s['eligible_test_prefixes']>=8

def test_source_and_checkpoint_custody():
    s,_=data()
    assert hashlib.sha256((ROOT/'lab/c62_balanced_remote_transfer.py').read_bytes()).hexdigest()==s['experiment_source_sha256']
    for n,sha in s['policy_source_sha256'].items():
        assert hashlib.sha256((ROOT/'frozen/research-source'/n).read_bytes()).hexdigest()==sha
    for name,digest in s['checkpoint_sha256'].items():
        assert hashlib.sha256((ROOT/'results62'/(name+'.safetensors')).read_bytes()).hexdigest()==digest
    assert s['training']['updates']==s['control']['updates']==c62.EPOCHS*c62.UPDATES_PER_EPOCH
    assert s['training']['graphs']==s['control']['graphs']
    assert s['training']['episodes_per_epoch']==s['control']['episodes_per_epoch']
    assert np.linalg.norm(np.array(s['weights']['C62-ALIGNED'])-s['weights']['C62-SHUFFLED'])>0

def test_all_host_actions_and_unguided_parity():
    s,l=data();rows=l['records']
    assert s['executed_suffixes']==len(rows)*len(c62.POLICIES)
    assert s['matched_conditions']==len(rows)
    assert len({(r['graph'],r['drop'],tuple(r['arm'])) for r in rows})==len(rows)
    for r in rows:
        assert set(r['outcomes'])==set(c62.POLICIES)
        assert all(v['host_rejections']==0 for v in r['outcomes'].values())
        assert len({v['protected_sha'] for v in r['outcomes'].values()})==1
        if r['arm'][0]=='none':
            assert len({(v['action_sha'],v['ink_sha']) for v in r['outcomes'].values()})==1
    assert s['unguided_parity_conditions']==sum(r['arm'][0]=='none' for r in rows)

def test_replayed_gallery_are_actual_paths():
    s,l=data();tr=l['gallery_traces']
    assert len(tr)==s['replayed_gallery_traces']==18
    for t in tr:
        match=[r for r in l['records'] if (r['graph'],r['drop'],r['arm'])==(t['graph'],t['drop'],['backtrack',1.])]
        assert len(match)==1
        out=match[0]['outcomes'][t['policy']]
        assert (t['action_sha'],t['ink_sha'])==(out['action_sha'],out['ink_sha'])
        assert len(t['path'])==out['steps']
        assert t['goal']==out['goal']

def test_no_hidden_graph_in_policy_features():
    src=ast.parse((ROOT/'frozen/research-source/c60_remote_frontier_rl.py').read_text())
    for name in ('action_features','explored_route','exploration_delta'):
        f=next(node for node in src.body if isinstance(node,ast.FunctionDef) and node.name==name)
        refs={a.attr for a in ast.walk(f) if isinstance(a,ast.Attribute)}
        assert refs.isdisjoint({'protected','allowed','material_at'})

def test_research_disposition_guard():
    s,_=data()
    # The eligible Kruskal result is not above frozen frontier: no promotion.
    kr=s['aggregate']['kruskal|0.0']['policies']
    assert kr['C62-ALIGNED']['goals']<=kr['C60-FRONTIER']['goals']
    assert all(isinstance(x['eligible_pairs'],int) for x in s['personality_full'].values())
