"""Targeted checks for C38 return-sign research (no native/production calls)."""
import json
import os
import pytest
from pathlib import Path
import numpy as np
import torch
from c38_return_sign import SignHead, train, heldout_labels
from c34_independent_maze import make_world


def test_output_is_bounded_for_every_state():
    h=SignHead()
    x=torch.randn(100,9)
    y=h(x)
    assert y.shape==(100,)
    assert torch.isfinite(y).all() and (torch.abs(y)<1.500001).all()


def test_training_actual_labels_and_unaligned_control():
    h,s,r=train(seed=381013,epochs=2)
    assert s['train_graphs']==80 and s['train_samples']==320
    assert s['nonzero_targets']>250
    assert s['shuffled_label_sha256']!=s['dataset_sha256']
    assert s['fit']['sign']['updates']==s['fit']['permuted']['updates']
    assert all(x['back_return'] is not None and x['forward_return'] is not None for x in r)


def test_training_replay_is_bitwise_reproducible():
    a,s,_=train(seed=381013,epochs=2)
    b,t,_=train(seed=381013,epochs=2)
    assert s==t
    for h,g in zip(a,b):
        assert all(torch.equal(h.state_dict()[k],g.state_dict()[k]) for k in h.state_dict())


def test_withheld_world_is_executed_and_legal():
    w,info=make_world(890011,'braid','detour',1)
    assert len(w.history)==24 and w.invalid_attempts==0
    assert all(w.allowed(p) for p in w.history)
    assert info['graph_nodes']>97 and w.goal not in w.history


def test_committed_evidence_invariants():
    location=os.getenv('C38_EVIDENCE_JSON')
    if not location:pytest.skip('Set C38_EVIDENCE_JSON to an independently produced summary')
    p=Path(location)
    assert p.is_file(), 'Requested evidence must exist' 
    q=json.loads(p.read_text())
    assert q['training_reproduction_identical']
    for cohort,models in q['policy'].items():
        for model,score in models.items():
            assert score['host_rejections']==0
            assert score['unguided_n']==score['unguided_exact']
            assert score['guided_n']==16
