"""C63 provenance, counterfactual-control and fully executed host-legality audit."""
import ast,base64,collections,hashlib,json,sys,zlib
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lab'))
import c63_counterfactual_gate as c63


def read():
    s=json.loads((ROOT/'results63/C63-summary.json').read_text())
    ledger=json.loads(zlib.decompress(base64.b64decode((ROOT/'results63/C63-executed-ledger.zlib.b64').read_text())))
    assert s==ledger['summary']
    return s,ledger


def test_custody_and_optimizer_budget():
    s,_=read()
    assert c63.sha256_bytes((ROOT/'lab/c63_counterfactual_gate.py').read_bytes())==s['source_sha256']
    assert s['sources']==c63.source_hashes()
    assert s['optimizer']['aligned']['updates']==s['optimizer']['shuffled']['updates']==c63.UPDATES
    assert s['optimizer']['aligned']['rows']==s['optimizer']['shuffled']['rows']==s['training_pairs']
    for name,digest in s['checkpoints'].items():
        file=ROOT/'results63'/(name+'.safetensors')
        assert hashlib.sha256(file.read_bytes()).hexdigest()==digest
        assert np.asarray(load_file(str(file))['weight']).shape==(1,c63.FEATURES)
    assert s['checkpoints']['C63-ALIGNED']!=s['checkpoints']['C63-SHUFFLED']


def test_graph_exclusion_and_sample_floor():
    s,l=read()
    rs=s['graph_receipts']
    allids=[r['graph'] for group in rs for r in group]
    assert len(allids)==len(set(allids))
    assert not ({r['graph'] for g in rs[:2] for r in g}&{r['graph'] for g in rs[2:] for r in g})
    assert s['exclusions']['C60_C61_C62_excluded']>180
    assert s['eligible_training']>=30 and s['eligible_test']>=30
    assert all(r['family']!='kruskal' for g in rs[:2] for r in g)
    assert {r['family'] for g in rs[2:] for r in g}>={'kruskal','wilson','tree'}
    assert not ({r['graph'] for r in l['records']} & {r['graph'] for r in l['training_counterfactuals']})


def test_host_legality_and_parity():
    s,l=read();records=l['records']
    assert s['matched_condition_rows']==len(records)
    assert s['policy_outcomes']==len(records)*len(c63.POLICIES)
    assert s['independent_host_suffixes']==len(records)*4
    assert s['unguided_parity']==sum(r['arm'][0]=='none' for r in records)
    assert len({(r['graph'],r['drop'],tuple(r['arm'])) for r in records})==len(records)
    for r in records:
        assert set(r['outcomes'])==set(c63.POLICIES)
        assert all(v['host_rejections']==0 for v in r['outcomes'].values())
        assert len({v['protected_sha'] for v in r['outcomes'].values()})==1
        if r['arm'][0]=='none':
            assert len({(v['action_sha'],v['ink_sha']) for v in r['outcomes'].values()})==1
        for pol in ('C63-ALIGNED','C63-SHUFFLED','C63-COIN'):
            assert r['choices'][pol] in ('C34-RESET','C60-FRONTIER')
            assert r['outcomes'][pol]==r['outcomes'][r['choices'][pol]]
            q=r['gate_probabilities'][pol]
            assert .1<=q<=.9 or (pol=='C63-COIN' and q==.5)
            if r['arm'][0]=='none':assert r['choices'][pol]=='C34-RESET'
            else:assert (r['choices'][pol]=='C60-FRONTIER')==(c63.choice_uniform(r['graph'],r['arm'][1],r['drop'])<q)


def test_actual_training_counterfactuals():
    s,l=read();rows=l['training_counterfactuals']
    assert len(rows)==s['training_pairs']
    assert all(r['protected_sha'] and len(r['frontier_action_sha'])==64 and len(r['reset_action_sha'])==64 for r in rows)
    assert len({r['graph'] for r in rows})==s['eligible_training']
    assert sum(r['adv']>0 for r in rows)==s['train_advantages']['positive']
    assert sum(r['adv']<0 for r in rows)==s['train_advantages']['negative']
    assert sum(r['adv']==0 for r in rows)==s['train_advantages']['zero']


def test_gallery_replayed_paths_are_executed():
    s,l=read();tr=l['gallery_traces']
    assert s['replayed_traces']==len(tr) and len(tr)>=12
    for t in tr:
        matches=[r for r in l['records'] if r['graph']==t['graph'] and r['drop']==0 and r['arm']==['backtrack',1.]]
        assert len(matches)==1
        v=matches[0]['outcomes'][t['policy']]
        assert t['action_sha']==v['action_sha'] and t['ink_sha']==v['ink_sha']
        assert t['goal']==v['goal'] and len(t['path'])==v['steps']
        assert t['actual']==matches[0]['choices'][t['policy']]


def test_no_hidden_geometry_in_inference():
    mod=ast.parse((ROOT/'lab/c63_counterfactual_gate.py').read_text())
    f=next(x for x in mod.body if isinstance(x,ast.FunctionDef) and x.name=='observe_features')
    src=ast.get_source_segment((ROOT/'lab/c63_counterfactual_gate.py').read_text(),f)
    forbidden={'protected','allowed','graph_edges','adjacency','shortest_path'}
    attrs={x.attr for x in ast.walk(f) if isinstance(x,ast.Attribute)}
    assert forbidden.isdisjoint(attrs)
    assert 'actual' not in src and 'outcome' not in src
    assert 'led.see(w)' in src
