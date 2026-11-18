"""Bounded source-contract tests for GOAL-TIME-TRANSFER-001."""
import copy
import pytest
from . import goal_time_transfer_lab as lab
from .world import Spec, specs, MOVE


def test_rotated_prehistory_and_legal_fork():
    for sp in specs('train',1):
        for turns in range(4):
            for lengths in lab.TRAIN_LENGTHS+lab.DEV_LENGTHS:
                q=lab.fixture(sp,turns,*lengths)
                w=q['world']
                assert len(q['prefix'])==28 and w.recoveries==11
                assert 'BACKTRACK' in w.menu(12) and q['side'] in w.menu(12)
                assert all(r['chosen'] in r['legal'] for r in q['prefix'])
                assert not (w.ink & set(w.materials)) and not (w.ink & w.protected)
                for g in lab.GOAL:
                    assert not w.occupied(q['targets'][g])


def test_partition_and_geometry_guards():
    train=specs('train',1)[0]
    held=Spec('heldout','archipelago',9000,'wall')
    with pytest.raises(ValueError,match='HELDOUT'):lab.fixture(held,0,2,2)
    with pytest.raises(ValueError,match='unregistered'):lab.fixture(train,0,8,8)
    with pytest.raises(ValueError,match='rotation'):lab.fixture(train,4,2,2)


def test_observation_receipts_and_counterfactual_start():
    sp=specs('train',1)[0]
    q1=lab.fixture(sp,2,2,2);q2=lab.fixture(sp,2,2,2)
    q1['goal']=q1['targets']['north'];q2['goal']=q2['targets']['south']
    menu1,x1=lab.observation(q1,128);menu2,x2=lab.observation(q2,128)
    assert q1['world'].state_sha()==q2['world'].state_sha()
    assert menu1==menu2 and len(menu1)<=7
    assert not (x1==x2).all(), 'goal must be observable without changing legal menu'


def test_invalid_episode_partition():
    sp=specs('train',1)[0]
    with pytest.raises(ValueError,match='DEVELOPMENT'):
        lab.episode(sp,811,'frozen',None,0,lab.DEV_LENGTHS[0],'north')


def test_strict_supervision_thresholds():
    # The two suffixes must agree, not merely one catastrophic failure.
    assert lab.GOAL==('north','south')
    assert lab.SUFFIX==('frozen','contrastive')
    assert set(lab.ARMS)=={'frozen','arrival','paint'}
