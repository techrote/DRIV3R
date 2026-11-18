"""COST-RETARGET-001 bounded synthetic training and evidence guardrails."""
from __future__ import annotations
import json
import numpy as np
import pytest
from . import cost_retarget_lab as lab
from .world import World, specs
from . import goal_recovery_lab as prior


def _state():
    return World.create(specs('train',1)[0])


def test_candidate_costs_only_use_host_menu_and_do_not_mutate_state():
    w=_state(); target=(56,53)
    before=w.state_sha(); menu=w.menu(12)
    costs=[lab.future_score(w,a,target,0,36,7,3) for a in menu]
    assert before==w.state_sha()
    assert len(costs)==len(menu)
    assert all(np.isfinite(x['score']) for x in costs)
    assert all(len(x['continuation'])<=3 for x in costs)
    assert all(c['score']<0 for a,c in zip(menu,costs) if a=='STOP')
    with pytest.raises(ValueError):lab.future_score(w,'INVALID',target,0,36,7)
    with pytest.raises(ValueError):lab.future_score(w,menu[0],target,0,36,7,9)


def test_counterfactual_labels_are_host_legal_and_deterministic():
    w=_state(); goal=(57,48)
    x=lab.labels_at_state(w,goal,36,0,1001,3)
    y=lab.labels_at_state(w,goal,36,0,1001,3)
    assert x==y
    menu,im,cost,scores=x
    assert im in menu and cost in menu and set(scores)==set(menu)
    assert cost==max(menu,key=lambda a:(scores[a]['score'],a==im,a)) or \
        scores[cost]['score']==max(v['score'] for v in scores.values())


def test_train_corpus_is_train_partition_and_matched():
    model=prior.Ranker(7)
    X,M,yi,yc,audit,meta=lab.training_corpus(model,1001,36,32,2)
    assert X.shape==(32,7,prior.WIDTH)
    assert M.shape==(32,7)
    assert all(M[i, yi[i]] and M[i, yc[i]] for i in range(len(yi)))
    assert meta['split']=='train' and meta['sample_n']==32
    assert len(audit)==32
    assert all(entry['state_sha256'] and len(entry['scores'])==len(entry['menu'])
               for entry in audit)


def test_provenance_and_heldout_isolation():
    assert lab.STAGES==(36,72,128)
    assert lab.EVAL_MODES==('retarget','static')
    assert set(lab.SOURCE_FILES)==set(lab.provenance())
    assert all(len(d)==64 for d in lab.provenance().values())
    assert all(s.split=='development' for s in lab.fixture.cases())
    assert all(s.split=='train' for s in specs('train',2))


def test_counterfactual_replay_audit_rejects_tampering(tmp_path):
    out=tmp_path/'quick'
    report=lab.run(out,seed=621,base_updates=1,adapt_updates=1,
                   sample_n=32,lookahead=2,quick=True)
    assert report['episodes']==3
    assert lab.verify(out)['episodes']==3
    trace=next((out/'traces').glob('*.json'))
    ep=json.loads(trace.read_text());ep['rows'][0]['probabilities']={}
    trace.write_text(json.dumps(ep)+'\n')
    with pytest.raises(ValueError):lab.verify(out)
