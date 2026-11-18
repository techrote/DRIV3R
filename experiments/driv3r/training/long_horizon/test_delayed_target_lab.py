"""DELAYED-TARGET-001 host/legal, source, and checkpoint replay guardrails."""
from __future__ import annotations
import json
import numpy as np
import pytest
from . import delayed_target_lab as lab
from . import goal_recovery_lab as host
from .world import World, specs, MOVE


def state():
    return World.create(specs('train',1)[0])


def test_delayed_value_is_deterministic_host_legal_and_nonmutating():
    w=state(); before=w.state_sha(); target=(49,48)
    legal=w.menu(12)
    values={a:lab.delayed_value(w,a,target,36,17,12) for a in legal}
    assert values=={a:lab.delayed_value(w,a,target,36,17,12) for a in legal}
    assert w.state_sha()==before and all(np.isfinite(v['value']) for v in values.values())
    assert all(v['contact_step'] is None or 1<=v['contact_step']<=12 for v in values.values())
    assert 'MOVE_E' in legal and values['MOVE_E']['contact_step']==1
    assert values['MOVE_E']['reached_new_target']
    with pytest.raises(ValueError):lab.delayed_value(w,'ILLEGAL',target,36,17,12)
    with pytest.raises(ValueError):lab.delayed_value(w,legal[0],target,36,17,19)
    with pytest.raises(ValueError):lab.delayed_value(w,legal[0],target,36,17,5)


def test_student_visited_corpus_and_label_controls_are_matched():
    model=host.Ranker(41)
    X,M,labels,audit,meta=lab.corpus(model,631,36,32)
    assert X.shape==(32,7,host.WIDTH) and M.shape==(32,7)
    assert all(M[i,labels[a][i]] for a in labels for i in range(32))
    assert meta['split']=='train' and len(audit)==32
    assert all(item['imitation'] in item['menu'] and item['short_cost'] in item['menu']
               and item['delayed_target'] in item['menu'] for item in audit)
    assert all(set(item['menu'])==set(item['delayed_scores']) for item in audit)


def test_partition_and_source_provenance():
    assert lab.STAGES==(36,72,128) and lab.ARMS==(
        'frozen','imitation','short_cost','delayed_target')
    assert set(lab.SOURCE_FILES)==set(lab.provenance())
    assert all(len(x)==64 for x in lab.provenance().values())
    assert all(s.split=='train' for s in specs('train',2))
    assert all(s.split=='development' for s in lab.fixture.cases())


def test_evidence_replay_and_probability_tamper_rejected(tmp_path):
    out=tmp_path/'quick'
    r=lab.run(out,631,base_steps=1,adapt_steps=1,n=32,quick=True)
    assert r['episodes']==4
    assert lab.verify(out)['verified_episodes']==4
    trace=next((out/'traces').glob('*.json'))
    ep=json.loads(trace.read_text())
    ep['rows'][0]['probabilities']={}
    trace.write_text(json.dumps(ep)+'\n')
    with pytest.raises(ValueError):lab.verify(out)


def test_synthetic_material_boundaries():
    w=state()
    assert w.menu(12) and all(a in MOVE or a=='STOP' for a in w.menu(12))
    for a in ('MOVE_N','MOVE_E','MOVE_S','MOVE_W'):
        assert a in w.menu(12)
    assert not w.ink.intersection(w.protected)
