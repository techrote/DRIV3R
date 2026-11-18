"""C40 synthetic research safety and optimizer determinism tests."""
import copy
import numpy as np
import torch
from c40_pairwise_rank import train, ActionHead, collect
from c39_action_value import action_features
from c34_independent_maze import make_world


def test_host_legal_candidate_features():
    world,_=make_world(400117,'tree','detour',1)
    menu=world.candidates(.35)
    legal=[i for i in range(9) if menu[i].legal]
    assert len(legal)>=2
    x=action_features(world,menu)[legal]
    assert x.shape==(len(legal),34) and np.isfinite(x).all()
    assert all(menu[i].legal for i in legal)


def test_pairwise_optimizer_replica_and_shuffled_control():
    rng=np.random.default_rng(40019)
    states=[(rng.normal(size=(5,34)).astype('float32'),rng.normal(size=5).astype('float32')) for _ in range(12)]
    a,s=train(states,seed=400117,epochs=2)
    b,t=train(states,seed=400117,epochs=2)
    assert s==t
    for x,y in zip(a,b):
        assert all(torch.equal(x.state_dict()[k],y.state_dict()[k]) for k in x.state_dict())
    assert any(not torch.equal(a[0].state_dict()[k],a[1].state_dict()[k]) for k in a[0].state_dict())
    assert s['C40-PAIR']['optimizer_updates']==24
    assert s['C40-SHUFFLED']['optimizer_updates']==24


def test_frozen_action_head_parameter_count():
    h=ActionHead()
    assert sum(p.numel() for p in h.parameters())<5000
    assert h(torch.zeros(3,34)).shape==(3,)
