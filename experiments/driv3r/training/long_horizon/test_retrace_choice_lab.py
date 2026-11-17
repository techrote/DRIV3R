"""RETRACE-CHOICE-001: legal fixture, split, evidence, and optimizer controls."""
import copy
import json
import numpy as np
import pytest
from . import retrace_choice_lab as lab
from .world import Spec, specs


def test_menus_are_identical_but_outcomes_are_not():
    for sp in specs('train',1):
        for turn in range(4):
            trap,open_=lab.paired(sp,turn)
            assert trap['world'].menu(12)==open_['world'].menu(12)
            assert trap['world'].recoveries==open_['world'].recoveries==0
            assert lab.teacher(trap,'contrastive')=='BACKTRACK'
            assert lab.teacher(open_,'contrastive')==open_['side']
            ma,xa=lab.inputs(trap,128);mb,xb=lab.inputs(open_,128)
            assert ma==mb and not np.array_equal(xa,xb)
            assert lab.teacher(trap,'always_move') in ma
            assert lab.teacher(open_,'always_retrace') in mb
    h=Spec('heldout','archipelago',9000,'wall')
    with pytest.raises(ValueError):lab.fork(h,0,'blocked_spur')


def test_training_labels_replay_and_tamper():
    xx,mm,yy,rows,meta=lab.examples(811,36,natural=8,repeats=2)
    assert len(rows)==4*4*2*3
    assert lab.verify_train(rows)==len(rows)
    for arm in xx:
        assert xx[arm].shape==(len(yy[arm]),7,lab.seq.WIDTH)
        assert np.all(mm[arm][np.arange(len(yy[arm])),yy[arm]])
    bad=copy.deepcopy(rows);bad[0]['features_sha256']='bad'
    with pytest.raises(ValueError):lab.verify_train(bad)
    bad=copy.deepcopy(rows);bad[0]['spec']['split']='heldout'
    with pytest.raises(ValueError):lab.verify_train(bad)


def test_neural_rollout_and_tamper():
    sp=specs('development',1)[0]
    model=lab.seq.ProbeRanker(811)
    ep=lab.episode(sp,0,'blocked_spur','contrastive',model,36)
    assert lab.verify_episode(ep,model)==len(ep['rows'])
    bad=copy.deepcopy(ep)
    label=bad['rows'][0]['action']
    bad['rows'][0]['probabilities'][label]=0.123
    with pytest.raises(ValueError):lab.verify_episode(bad,model)
