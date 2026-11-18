"""Source-contract checks for the independently verified two-seed replication."""
from __future__ import annotations
import json
import numpy as np
import pytest
from . import goal_time_replication_audit as lab
from . import goal_time_transfer_lab as trial, goal_conditioning_lab as goal, policy
from .world import specs, Spec


def test_goal_swap_is_host_state_invariant():
    for sp in specs('development',1):
        for turns in range(4):
            q=trial.fixture(sp,turns,*trial.DEV_LENGTHS[0]);w=q['world']
            sha=w.state_sha();menu=w.menu(12)
            q['goal']=q['targets']['north'];m1,x1=trial.observation(q,128)
            q['goal']=q['targets']['south'];m2,x2=trial.observation(q,128)
            assert m1==m2==menu and w.state_sha()==sha
            assert not np.array_equal(x1,x2)
            x1[:,len(policy.FEATURES):goal.FEATURE_DIM]=0
            x2[:,len(policy.FEATURES):goal.FEATURE_DIM]=0
            assert np.array_equal(x1,x2)


def test_restoration_rejects_checkpoint_corruption(tmp_path):
    (tmp_path/'checkpoints').mkdir()
    (tmp_path/'report.json').write_text(json.dumps({'checkpoint_sha256':{'811':{'frozen':'0'*64,'arrival':'0'*64,'paint':'0'*64}}}))
    (tmp_path/'checkpoints'/'811-frozen.npz').write_bytes(b'fake model')
    with pytest.raises(ValueError,match='identity'):lab.restore(tmp_path,811)


def test_sealed_structural_heldout_rejected():
    with pytest.raises(ValueError,match='HELDOUT'):
        trial.fixture(Spec('heldout','archipelago',9000,'wall'),0,*trial.DEV_LENGTHS[0])


def test_balanced_pair_count_contract():
    assert lab.SEEDS==(811,814)
    assert lab.ARMS==('frozen','arrival','paint')
    assert len(trial.DEV_LENGTHS)==2
