"""Focused safety and empirical-label guardrails for FAILURE-OUTCOME-001."""
from pathlib import Path
import copy
import json
import random
import pytest
from experiments.driv3r.training.long_horizon import failure_outcome_lab as lab
from experiments.driv3r.training.long_horizon import goal_recovery_lab as host
from experiments.driv3r.training.long_horizon.world import World, specs


def test_partition_and_owner_scope():
    assert all(s.split=='development' and 3000<=s.seed<4000 for s in lab.cases())
    assert not any('heldout' in s.ident for s in lab.cases())
    assert lab.STAGES==(36,72,128)
    assert set(lab.ARMS)=={'frozen','imitation','survival_guard','outcome_aggressive'}


def test_lexicographic_outcome_guard():
    row={'completed':24,'goals':0,'paint':10,'recovery':0}
    better={'completed':24,'goals':1,'paint':10,'recovery':0}
    unsafe={'completed':24,'goals':2,'paint':9,'recovery':0}
    menu=['MOVE_E','MOVE_N','STOP']
    outcome={menu[0]:row,menu[1]:better,menu[2]:unsafe}
    assert lab.outcome_labels(menu,'MOVE_E',outcome)==('MOVE_N','STOP')
    with pytest.raises(ValueError):lab.outcome_labels(menu,'MOVE_W',outcome)


def test_suffix_is_host_legal_and_replayable():
    sp=specs('train',1)[0]
    w=World.create(sp)
    models,_=host.train(701,1)
    ts=lab.fixture.targets_for(sp,703,'retarget')
    start=w.state_sha()
    for first in w.menu(12):
        result=lab.student_suffix(w,first,ts,0,36,models['goal_recovery'],8)
        assert 1<=result['completed']<=8
        sim=copy.deepcopy(w)
        for action in result['actions']:
            assert action in sim.menu(12)
            sim.apply(action,12)
        assert sim.state_sha()==result['last_sha256']
    assert w.state_sha()==start


def test_checkpoint_and_trace_tampering(tmp_path):
    out=tmp_path/'original'
    lab.run(out,seed=701,base_updates=1,adapt_updates=1,n=24,depth=4,quick=True)
    verified=lab.verify(out)
    assert verified['checkpoints']==4 and verified['episodes']==4
    trace=next((out/'traces').glob('*.json'))
    j=json.loads(trace.read_text());j['rows'][0]['requested']='STOP'
    trace.write_text(json.dumps(j)+'\n')
    with pytest.raises(ValueError,match='evidence hashes'):lab.verify(out)
