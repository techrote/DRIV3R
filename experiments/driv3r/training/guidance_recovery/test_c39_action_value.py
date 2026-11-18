"""C39 bounded offline and host-legal regression tests; no paid resources."""
import copy, json
import numpy as np
import torch
from c34_independent_maze import make_world
from c39_action_value import ActionHead, action_features, collect, heldout, ConditionalPolicy
from c39_audit import replay, bootstrap_online, bootstrap_offline


def test_action_features_observation_only():
    w,_=make_world(91173,'tree','detour',-1)
    menu=w.candidates(.35)
    x=action_features(w,menu)
    assert x.shape==(9,34) and np.isfinite(x).all()
    assert np.allclose(x[:,9:],np.asarray([c.features for c in menu[:9]]))


def test_label_generation_uses_legally_executed_counterfactuals():
    _,fit,receipt=collect(seed=390107,graphs=56,steps=3,epochs=1)
    assert fit['graphs']==56
    assert fit['candidate_examples']>=40
    assert all(len(r['legal'])==len(r['returns']) and len(r['legal'])>=2 for r in receipt)
    assert all(0<=j<9 for r in receipt for j in r['legal'])


def test_checkpoint_seed_replication():
    a,stats,rec=collect(seed=390107,graphs=56,steps=3,epochs=2)
    b,stats2,rec2=collect(seed=390107,graphs=56,steps=3,epochs=2)
    assert stats==stats2 and rec==rec2
    for x,y in zip(a,b):
        assert all(torch.equal(x.state_dict()[k],y.state_dict()[k]) for k in x.state_dict())


def test_legal_mask_never_modified():
    from c28_temporal_option import OptionHead
    from signed_feedback import HostAdvice
    w,_=make_world(91173,'tree','detour',1)
    advice=HostAdvice(w);advice.issue_backtrack(w,strength=1.,ttl=9)
    # Compare against a stand-alone policy's legal menu; no safety contract change.
    from c28_temporal_option import OptionHead
    option=OptionHead().eval();policy=ConditionalPolicy(option,ActionHead().eval())
    from c29_probe_recovery import PREREQS
    from personality_retention_core import load_c12
    from c19_return_anchored import a0
    from safetensors.torch import load_file
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'root'
    if not root.exists():return
    base=a0(load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]))
    menu,logits,_,_=policy.modify(base,w,advice,1.)
    assert [c.legal for c in menu]==[c.legal for c in w.candidates(1.)]
    assert not np.any(np.isnan(logits))
    for j,c in enumerate(menu):
        if not c.legal:assert logits[j] < -10



def test_complete_host_replay_and_bootstrap(tmp_path):
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'out'
    if not (root/'C39-cohort-A.json').exists():return
    rows=json.loads((root/'C39-cohort-A.json').read_text())
    matched=[r for r in rows if r['trajectory'] is not None]
    assert matched and all(replay(r)==len(r['trajectory']) for r in matched)
    groups={'A':rows}
    assert bootstrap_online(groups,reps=50)['C34-RESET']['graph_clusters']>0
    held=json.loads((root/'C39-heldout.json').read_text())
    assert bootstrap_offline(held,reps=50)['C39-SHUFFLED']['graph_clusters']>=10
