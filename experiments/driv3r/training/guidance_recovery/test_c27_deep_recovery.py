"""Bounded research-contract tests for C27; no native game modifications."""
import numpy as np
import torch
import c27_deep_recovery as c27
from c26_stuck_interventions import observe40
from signed_feedback import HostAdvice
from personality_retention_core import load_c12
from attested_stack_cue import KEYS
from pathlib import Path

ARCHIVE=Path(__file__).parent.parent

def prior():
    names=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
           'persistent_feedback','signed_feedback','selective_continue')
    return load_c12([ARCHIVE/'prerequisites'/f'guidance_recovery_{n}.safetensors' for n in names])

def test_deep_legal_detour():
    for train in (True,False):
        z=c27.oracle_geometry_audit(train)
        assert z['detour_ratio']>2 and z['first_shortest_step_increases_manhattan']
        assert z['shortest_legal_moves']>z['manhattan']+10
        oracle=c27.oracle_retrace(train)
        assert oracle['success'] and not oracle['host_rejected']
        assert oracle['steps']>=z['shortest_legal_moves']

def test_distinct_detector_and_cue_boundary():
    torch.set_num_threads(1)
    m=c27.DualHead(prior(),mode='detector')
    w=c27.long_world(0,'reverse',1,27001,True)
    a=HostAdvice(w)
    cs,x=observe40(w,1.,a)
    with torch.no_grad():
        previous=m.base(torch.from_numpy(x[:,:32]))
        new=m(torch.from_numpy(x))
    assert torch.equal(previous,new)
    assert cs[8].legal
    a.issue_backtrack(w,strength=.35,ttl=9)
    _,cue=observe40(w,1.,a)
    with torch.no_grad():after=m(torch.from_numpy(cue))
    assert torch.equal(after[:8],m.base(torch.from_numpy(cue[:,:32]))[:8])
    assert torch.equal(after[9:],m.base(torch.from_numpy(cue[:,:32]))[9:])

if __name__=='__main__':
    test_deep_legal_detour();test_distinct_detector_and_cue_boundary();print('C27 targeted tests PASS')