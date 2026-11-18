"""C37 targeted deterministic training, cohort pairing and host legality tests."""
from pathlib import Path
import copy, numpy as np
from c37_shuffle_replication import (collect,train_controls,run_policy,
       evaluate,summarize,ARMS,make_world,PREREQS,load_c12,a0,OptionHead,load_file)
ROOT=Path(__file__).resolve().parent.parent


def subjects():
    prior=load_c12([ROOT/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS])
    base=a0(prior);option=OptionHead()
    option.load_state_dict(load_file(str(ROOT/'results/C28-TEMPORAL.safetensors')))
    heads,summary,_=train_controls(epochs=2)
    return base,option,heads


def test_train_controls_executed_targets_and_nulls():
    heads,fit,receipts=train_controls(epochs=2)
    assert len(receipts)==fit['samples']>=250
    assert fit['updates_per_trained_arm']>0
    assert fit['permuted_targets_sha256']!=fit['dataset_sha256']
    assert fit['head_difference_mae']>0.
    for r in receipts:
        assert -1.5<=r['target']<=1.5
        assert r['back_return'] is not None and r['forward_return'] is not None
    assert all(np.isfinite(x) for x in fit['actual_target_mae'].values())


def test_train_repeat_byte_identical():
    a,ma,_=train_controls(epochs=2);b,mb,_=train_controls(epochs=2)
    assert ma==mb
    for h1,h2 in zip(a,b):
        assert all(np.array_equal(h1.state_dict()[k].numpy(),h2.state_dict()[k].numpy())
                   for k in h1.state_dict())


def test_matched_arms_legal_and_unguided_invariant():
    base,opt,heads=subjects();w,info=make_world(380233,'braid','detour',1)
    for directive,strength in (('none',0.),('backtrack',1.),('continue',.35)):
        rr={name:run_policy(base,opt,heads,copy.deepcopy(w),info,380233,1.,
                      directive,strength,name,True,30,False) for name in ARMS}
        assert all(x['host_rejections']==0 for x in rr.values())
        if directive=='none':
            assert len({r['action_sha'] for r in rr.values()})==1
            assert len({r['ink_sha'] for r in rr.values()})==1


def test_summarize_and_withheld_geometry():
    base,opt,heads=subjects()
    rows,graphs=evaluate(base,opt,heads,380239,replicates=1,horizon=14)
    result=summarize(rows)
    assert set(result)==set(ARMS)
    assert len(graphs)==32
    assert all(x['unguided_exact']==32 and x['unguided_n']==32 for x in result.values())
    assert all(x['host_rejections']==0 for x in result.values())
    assert all(len(x['condition_goals'])==32 for x in result.values())
    assert {x['graph_nodes'] for x in graphs.values()}=={97,111}
