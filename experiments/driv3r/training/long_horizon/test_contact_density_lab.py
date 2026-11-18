"""CONTROL and host-boundary tests for CONTACT-DENSITY-001."""
from __future__ import annotations
import json
import numpy as np
import pytest
from . import contact_density_lab as lab
from .world import World, TRAIN_FAMILIES, specs


def test_factorial_holds_teacher_and_only_changes_reward_coefficients():
    w=World.create(specs('train',1)[0]); start=w.state_sha(); target=(49,48)
    from .delayed_target_lab import delayed_value
    d={a:delayed_value(w,a,target,36,173) for a in w.menu(12)}
    assert w.state_sha()==start
    for v in d.values():
        assert lab.score(v,22,1.5)==pytest.approx(v['value'])
        assert lab.score(v,0,0)==pytest.approx(.25*(v['gain']+v['painted'])-12*v['stopped'])
    assert len(lab.FACTORS)==5
    assert lab.FACTORS['contact_only']==(22,0)
    assert lab.FACTORS['recovery_only']==(0,1.5)


def test_balanced_indices_deterministic_bounded_and_no_leak():
    positive=np.array([True,False,False,False,True,False,False,False])
    i=lab.balance_indices(positive,17)
    assert len(i)==8 and len(set(i))<=8 and int(positive[i].sum())==4
    assert np.array_equal(i,lab.balance_indices(positive,17))
    with pytest.raises(ValueError):lab.balance_indices(np.ones(8,bool),17)
    assert all(s.split=='development' and 3000<=s.seed<4000 for s in lab.cases())
    assert set(s.family for s in lab.DEV_SPECS)==set(TRAIN_FAMILIES)
    assert {s.seed for s in lab.DEV_SPECS}.isdisjoint({s.seed for s in lab.fixture.cases()})


def test_label_menu_determinism_and_distribution():
    fake={'menu':['MOVE_N','MOVE_E','STOP'],'imitation':'MOVE_N',
      'delayed_scores':{
        'MOVE_N':{'reached_new_target':False,'gain':1,'painted':1,'recovery_spent':0,'stopped':False},
        'MOVE_E':{'reached_new_target':True,'gain':-2,'painted':1,'recovery_spent':2,'stopped':False},
        'STOP':{'reached_new_target':False,'gain':0,'painted':0,'recovery_spent':0,'stopped':True}}}
    labels, eligible, meta=lab.labels_from_audit([fake],np.array([[True,True,True,False,False,False,False]]))
    assert eligible.tolist()==[True]
    assert labels['imitation'].tolist()==[0]
    assert labels['proxy_only'].tolist()==[0]
    assert labels['full'].tolist()==[1]
    assert meta['contact_positive']==1


def test_evidence_replay_tamper_and_exact_source(tmp_path):
    out=tmp_path/'evidence'
    report=lab.run(out,641,base_updates=1,adapt_updates=1,samples=32,quick=True)
    assert report['episodes']==len(lab.ARMS)
    assert lab.verify(out)['episodes']==len(lab.ARMS)
    trace=next((out/'traces').glob('*.json'))
    ep=json.loads(trace.read_text())
    ep['rows'][0]['probabilities']={}
    trace.write_text(json.dumps(ep)+'\n')
    with pytest.raises(ValueError):lab.verify(out)


def test_provenance_and_no_heldout():
    assert lab.STAGES==(36,72,128)
    assert set(lab.SOURCE_FILES)==set(lab.provenance())
    assert all(len(x)==64 for x in lab.provenance().values())
    assert all(s.family not in ('archipelago','staggered_vault') for s in lab.cases())
