"""Direct state-counterfactual and independent replay acceptance tests."""
from pathlib import Path
import copy
import json
import tempfile
import pytest
from . import outcome_fork_lab as lab
from .world import Spec, specs


def test_matched_host_legal_counterfactual_forks():
    sp=specs('train',1)[0]
    model=lab.seq.ProbeRanker(811)
    for t in range(4):
        for kind in lab.prior.KINDS:
            q=lab.prior.fork(sp,t,kind)
            a=lab.suffix(sp,t,kind,q['side'],model,36,trace=True)
            b=lab.suffix(sp,t,kind,'BACKTRACK',model,36,trace=True)
            assert a['initial_sha256']==b['initial_sha256']
            assert a['events'][0]['chosen']==q['side']
            assert b['events'][0]['chosen']=='BACKTRACK'
            assert a['events'][0]['native_receipt'] is None
            assert b['events'][0]['native_receipt'] is None
            assert all(row['event']['chosen'] in row['event']['legal'] for row in a['events'][1:])


def test_split_seal_and_replay_source():
    model=lab.seq.ProbeRanker(811)
    bad=Spec('heldout','archipelago',9000,'wall')
    with pytest.raises(ValueError):lab.suffix(bad,0,'blocked_spur','BACKTRACK',model,128)
    sp=specs('development',1)[0]
    ep=lab.episode(sp,0,'open_spur','outcome',model,36)
    assert lab.prior.verify_episode(ep,model)==len(ep['rows'])
    ep['rows'][0]['event']['chosen']='STOP'
    with pytest.raises(ValueError):lab.prior.verify_episode(ep,model)


def test_source_integrity_and_checkpoint_rebuild():
    assert all((Path(lab.__file__).parent/n).exists() for n in lab.SOURCE)
    assert lab.STAGES==(36,72,128)
    assert lab.ARMS==('frozen','always_move','contrastive','outcome')
