"""Deterministic C28 guard, host authority and scientific caveat regression tests."""
from collections import deque
import os
import pytest
import numpy as np
import torch
from c28_temporal_option import (executed_world,history_evidence,TemporalOption,OptionHead,
  scores,probs,train,rollout)
from signed_feedback import HostAdvice
from personality_retention_core import load_c12
from c19_return_anchored import a0
from pathlib import Path

ROOT=Path(os.environ.get("DRIV3R_C28_ARCHIVE_ROOT",Path(__file__).resolve().parents[1]))
PATHS=[ROOT/'prerequisites'/f'guidance_recovery_{s}.safetensors' for s in
       ('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
        'persistent_feedback','signed_feedback','selective_continue')]


def require_prerequisites():
    if not all(p.exists() for p in PATHS):
        pytest.skip("historical C12 checkpoint prerequisites supplied only in external evidence bundle")


def base():
    require_prerequisites()
    torch.set_num_threads(1)
    return a0(load_c12(PATHS))


def shortest(w):
    q=deque([(w.pos,0)]);seen={w.pos}
    while q:
        p,d=q.popleft()
        if p==w.goal:return d
        for a,b in ((1,0),(-1,0),(0,1),(0,-1)):
            t=(p[0]+a,p[1]+b)
            if t not in seen and w.allowed(t):seen.add(t);q.append((t,d+1))
    return None


def test_host_executes_every_prefix_step():
    for trained in (True,False):
        w=executed_world(0,'reverse',1,1,trained)
        assert len(w.events)==len(w.history)
        assert all(k=='INK' for k,_,_,_ in w.events)
        assert len(w.inked)==len(w.history)+1
        assert w.invalid_attempts==0
        assert shortest(w)==(26 if trained else 28)


def test_false_local_geometry_is_not_oracle():
    for i in range(8):
        a=executed_world(i,'reverse',1,2,False)
        b=executed_world(i,'tunnel',1,2,False)
        c=executed_world(i,'decoy',1,2,False)
        assert shortest(a)==28 and shortest(c)==28 and shortest(b)==8
        for w in (a,b,c):
            assert history_evidence(w)['stack_moves']==28
            assert history_evidence(w)['detour_ratio']==3.5


def test_no_cue_does_not_activate_option():
    w=executed_world(0,'reverse',1,3,False);a=HostAdvice(w)
    menu=w.candidates(.35)
    o=TemporalOption(OptionHead(),True,veto=True)
    assert not o.observe(w,a,menu)


def test_guard_accepts_recovery_and_abstains_when_uncertain():
    for outcome in ('reverse','tunnel','decoy'):
        w=executed_world(0,outcome,1,4,False)
        a=HostAdvice(w);a.issue_backtrack(w,strength=.35,ttl=9)
        o=TemporalOption(OptionHead(),True,veto=True)
        yes=o.observe(w,a,w.candidates(.35))
        assert yes==(outcome=='reverse')
        assert o.veto_latched==(outcome!='reverse')


def test_option_expires_when_goal_changes_or_host_cancels():
    for change in ('cancel','goal'):
        w=executed_world(0,'reverse',1,5,False)
        a=HostAdvice(w);a.issue_continue(w,strength=1.,ttl=9)
        o=TemporalOption(OptionHead(),True,veto=True)
        assert o.observe(w,a,w.candidates(1.))
        if change=='cancel':a.cancel(w)
        else:w.goal=(13,13)
        assert not o.observe(w,a,w.candidates(1.))


def test_all_actions_stay_host_legal_and_stochastic():
    prior=base();w=executed_world(0,'reverse',1,6,False)
    a=HostAdvice(w);a.issue_continue(w,strength=.35,ttl=9)
    option=TemporalOption(OptionHead(),True,veto=True)
    menu,logits,raw,bonus=scores(prior,w,a,.35,option)
    p=probs(logits)
    assert raw==0 and bonus>0
    assert all(p[j]==0 for j,c in enumerate(menu) if not c.legal)
    assert 0<p[8]<=.9850000001
    assert 1-p[8]>=.0149999999
    assert abs(sum(p)-1)<1e-9


def test_safety_and_no_host_cue_equivalence():
    require_prerequisites()
    prior=load_c12(PATHS)
    head=OptionHead()
    for outcome in ('reverse','tunnel','decoy'):
        for person in (-1,1):
            seed=277700+person*9
            base=rollout(a0(prior),head,0,outcome,person,0.,'none',0.,seed,False,35,'C18-A0')
            option=rollout(a0(prior),head,0,outcome,person,0.,'none',0.,seed,False,35,'VETO')
            assert base['action_sha256']==option['action_sha256']
            assert base['ink_sha256']==option['ink_sha256']
            assert option['host_rejected']==0
            assert option['raw_illegal_top1']==0


def test_training_updates_real_parameters():
    head,info,_=train(base(),n=1,epochs=2,seed=28111)
    assert info['updates']>0 and info['losses'][-1]<info['losses'][0]
    assert info['mean_p_backtrack']<1
    assert all(torch.isfinite(x).all() for x in head.parameters())
