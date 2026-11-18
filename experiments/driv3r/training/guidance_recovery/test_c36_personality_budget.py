"""C36 targeted synthetic-host and style-budget regression checks."""
import copy
import hashlib
import numpy as np
from pathlib import Path
from c36_personality_budget import (
    ARMS, style_project, persona_score, make_world, PREREQS, load_c12, a0,
    OptionHead, load_file, RetreatHead, run_one, eval_cohort, summarize,
    HostAdvice, scores, probs)

ROOT=Path(__file__).resolve().parent.parent

def _subjects():
    prior=load_c12([ROOT/'prerequisites'/f'guidance_recovery_{x}.safetensors' for x in PREREQS]);base=a0(prior)
    opt=OptionHead();opt.load_state_dict(load_file(str(ROOT/'results/C28-TEMPORAL.safetensors')))
    head=RetreatHead()
    return base,opt,head

def test_projection_turn_bound_and_class_mass():
    w,_=make_world(342233,'braid','forward',1);base,_,_=_subjects();advice=HostAdvice(w);advice.issue_backtrack(w,strength=1.,ttl=9)
    menu,ref,_,_=scores(base,w,advice,1.)
    prop=np.asarray(ref,dtype=np.float64).copy();prop[0]+=7.
    out,before,after,alpha=style_project(menu,ref,prop,w.personality,max_shift=.045)
    assert after<=.04501 and after<=before+1e-6
    assert 0.<=alpha<=1. and np.allclose(out[8:],prop[8:])
    assert np.isclose(probs(out)[:8].sum(),probs(prop)[:8].sum(),atol=1e-6)

def test_no_guidance_exact_all_variants():
    base,opt,head=_subjects();w,info=make_world(342233,'tree','detour',-1)
    vals={}
    for model in ARMS:
        vals[model]=run_one(base,opt,head,head,copy.deepcopy(w),info,342233,1.,'none',0.,model,False,55)
        assert not vals[model]['host_rejections']
    assert len({x['action_sha'] for x in vals.values()})==1
    assert len({x['ink_sha'] for x in vals.values()})==1

def test_guidance_legal_and_prerequisites_untouched():
    base,opt,head=_subjects();w,info=make_world(342377,'braid','detour',1)
    material=hashlib.sha256(repr(sorted(w.protected)).encode()).hexdigest()
    for model in ARMS:
        v=run_one(base,opt,head,head,copy.deepcopy(w),info,342377,.35,'backtrack',1.,model,True,62)
        assert v['host_rejections']==0 and v['goals'] in (0,1,2)
        assert v['style_post_sum']<=v['style_pre_sum']+1e-7
    assert material==hashlib.sha256(repr(sorted(w.protected)).encode()).hexdigest()

def test_matched_condition_summary():
    base,opt,head=_subjects();rows,graphs=eval_cohort(base,opt,head,head,342397,n=1,horizon=20)
    r=summarize(rows)
    assert len(graphs)>0
    assert all(x['unguided_exact']==32 and x['unguided_n']==32 for x in r.values())
    assert all(x['safety_host_rejections']==0 for x in r.values())
    assert all(len(x['groups'])==32 for x in r.values())
