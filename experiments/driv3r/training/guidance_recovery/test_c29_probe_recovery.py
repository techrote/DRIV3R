"""C29 synthetic host safety, provenance, negative controls and optimizer guards."""
from __future__ import annotations
import os
from pathlib import Path
import pytest
import numpy as np
import torch
from c28_temporal_option import executed_world, OptionHead
from c29_probe_recovery import ProbeHead, Recoverer, fresh_indices,train_probe,custom_rollout,PREREQS
from c29_deepdecoy_test import deep_world,shortest
from personality_retention_core import load_c12
from c19_return_anchored import a0
from signed_feedback import HostAdvice

ROOT=Path(os.environ.get('DRIV3R_C29_ARCHIVE_ROOT',Path(__file__).resolve().parents[1]))
PATHS=[ROOT/'prerequisites'/f'guidance_recovery_{x}.safetensors' for x in PREREQS]

def require():
    if not all(p.is_file() for p in PATHS):pytest.skip('Historical checkpoint prerequisites absent; external bundle required')

def test_deepdecoy_legality_and_local_observational_ambiguity():
    for i in range(4):
        for person in (-1,1):
            deep=deep_world(i,person,333000+i)
            assert shortest(deep)==28
            assert len(deep.events)==len(deep.history) and len(deep.history)>=28
            assert deep.invalid_attempts==0
            m=deep.candidates(.35);inds=fresh_indices(deep,m)
            assert inds and all(m[j].features[22]==.5 for j in inds)

def test_one_step_decoy_separable_by_host_visible_degree():
    for i in range(8):
        d=executed_world(i,'decoy',1,333000+i,False)
        t=executed_world(i,'tunnel',1,333000+i,False)
        assert all(d.candidates(.35)[j].features[22]==.75 for j in fresh_indices(d,d.candidates(.35)))
        assert all(t.candidates(.35)[j].features[22]==.5 for j in fresh_indices(t,t.candidates(.35)))

def test_train_real_gradients_and_repeatability():
    a,metrics,_=train_probe(seed=29111,worlds=4,epochs=6)
    b,metrics2,_=train_probe(seed=29111,worlds=4,epochs=6)
    assert metrics['updates']>0 and metrics['epoch_losses'][-1]<metrics['epoch_losses'][0]
    assert metrics['train_sha256']==metrics2['train_sha256']
    for key,value in a.state_dict().items():
        assert torch.equal(value,b.state_dict()[key])
        assert torch.isfinite(value).all()

def test_no_guidance_exactness_and_host_authority():
    require();base=a0(load_c12(PATHS))
    for outcome in ('reverse','tunnel','decoy'):
        for person in (-1,1):
            seed=288000+person*101
            old=custom_rollout(base,OptionHead(),ProbeHead(),0,outcome,person,.35,'none',0.,seed,30,baseline_variant='C18-A0')
            for guarded in (False,True):
                r=custom_rollout(base,OptionHead(),ProbeHead(),0,outcome,person,.35,'none',0.,seed,30,safe_start=guarded)
                assert r['action_sha256']==old['action_sha256'] and r['ink_sha256']==old['ink_sha256']
                assert not r['host_rejected'] and not r['raw_illegal_top1']

def test_cancellation_aborts_live_probe():
    w=executed_world(1,'decoy',1,3891,False)
    a=HostAdvice(w);a.issue_backtrack(w,1.,ttl=9)
    rec=Recoverer(OptionHead(),ProbeHead(),safe_start=True)
    rec.modify(a0_noop(),w,a,.35)
    assert rec.phase=='probe'
    a.cancel(w);rec.modify(a0_noop(),w,a,.35)
    assert rec.phase=='released'

# Small adapter with the expected (10,29) logits interface; only used in state test.
class a0_noop(torch.nn.Module):
    def forward(self,x):return torch.zeros(x.shape[0],dtype=x.dtype)
