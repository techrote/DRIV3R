"""C33 regression suite: independent graph structure and host-safety controls."""
import hashlib
import torch
from safetensors.torch import save
from c33_irregular_recovery import (SCENARIOS,FAMILIES,graft_world,train,ReturnHead,
  PREREQS,load_c12,OptionHead,load_file,a0,play,summary)
from pathlib import Path


def test_structural_graph_diversity():
    fingerprints=set()
    for family in FAMILIES:
        for scene in SCENARIOS:
            w,info=graft_world(9,scene,family,1,1337)
            assert w.goal not in w.history and w.allowed(w.goal)
            assert info['added']>0 and info['branch_sha256'] not in fingerprints
            assert not w.invalid_attempts and len(w.history)==27
            fingerprints.add(info['branch_sha256'])
    assert len(fingerprints)==8


def test_graft_and_host_permissions():
    for family in FAMILIES:
        w,info=graft_world(8,'lure4',family,-1,10001)
        protected=frozenset(w.protected)
        for _ in range(15):
            choices=[c for c in w.candidates(.35) if c.legal and c.action[0]=='INK']
            if not choices:break
            c=choices[0]
            assert w.apply(c)
            assert frozenset(w.protected)==protected
        assert w.invalid_attempts==0 and not (w.inked&w.protected)


def test_no_guidance_identity_and_cue_safety():
    root=Path(__file__).resolve().parents[1]
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]);base=a0(prior)
    option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    for family in FAMILIES:
        for directive,strength in (('none',0.),('continue',1.),('backtrack',1.)):
            w,_=graft_world(8,'off-early',family,1,3388)
            a=play(base,option,ReturnHead(),w,3388,.35,directive,strength,'C18-A0',45)
            v,_=graft_world(8,'off-early',family,1,3388)
            b=play(base,option,ReturnHead(),v,3388,.35,directive,strength,'C33-FIXED',45)
            assert b['host_rejected']==b['raw_illegal_top1']==0
            if directive=='none':
                assert a['action_sha256']==b['actions_sha']
                assert a['ink_sha256']==b['ink_sha']


def test_optimizer_reproducibility():
    a,fit,_=train(330071,12);b,fit2,_=train(330071,12)
    assert fit==fit2 and fit['updates']>0 and fit['examples']>20
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
