"""D022: second on-policy pass and trap-hard-negative study, research-only.

This is *synthetic* D017-teacher distillation, NOT fresh Frankenbrain inference
or native CyberSand gameplay. Training/checkpoints are external artifacts.
No dependencies, game ABI or shared host source are modified.
"""
from __future__ import annotations
import argparse, collections
from pathlib import Path
import numpy as np
from .lab import ACTIONS, STYLES, Game, feats
from .repair import adverse, load
from .source_parent_branch_gate import Controller, Gate, parents, sha, js
from .branch_retention_020 import CASES, fit, rollout, summary, bootstrap, admissibility
from .onpolicy_repair_021 import D007, D017, Q8, jsonl

SCHEMA='driv3r.distillation.d022-iterated-onpolicy/1'
TRAIN=range(61025,61041)
VALID=range(66001,66009)
TEST=range(67001,67017)
CONFIRM=range(68001,68017)
STRESS=range(69001,69017)

def check_splits():
    groups=[set(g) for g in (TRAIN,VALID,TEST,CONFIRM,STRESS)]
    if any(groups[i]&groups[j] for i in range(5) for j in range(i+1,5)):
        raise AssertionError('seed overlap')
    assert [len(g) for g in groups]==[16,8,16,16,16]

def collect(student, teacher):
    """Every label comes from the frozen D017 synthetic controller.

    The parent Q8 source contributes only style identity. Case labels and
    teacher/student mismatch are captured before applying the student action.
    """
    records=[];counts=collections.Counter()
    for seed in TRAIN:
        for style in STYLES:
            for case in CASES:
                g=Game(seed,style)
                for tick in range(80):
                    adverse(g,case);o=g.obs();x=feats(o)
                    act=student.predict(o);teacher_act=teacher.predict(o)
                    legal=np.asarray([a in o['legal'] for a in ACTIONS],np.uint8)
                    if act not in o['legal'] or teacher_act not in o['legal']:
                        raise AssertionError('illegal prediction')
                    scores=teacher.base.scores(x).copy()
                    scores[3]+=teacher.bonus_for(o)
                    scores=np.where(legal,scores,-1e5)
                    mismatch=act!=teacher_act
                    # A trap disagreement is a hard negative for the standalone
                    # student's current ranking; do not relabel parent actions.
                    records.append((x.astype(np.float32),scores.astype(np.float32),legal,
                                    ACTIONS.index(teacher_act),mismatch,case=='trap'))
                    counts[f'{case}/{"mismatch" if mismatch else "match"}']+=1
                    g.step(act)
                    if g.stopped:break
    output={key:np.asarray([r[j] for r in records],dtype=dtype) for j,(key,dtype)
            in enumerate((('x',np.float32),('t',np.float32),('m',np.uint8),
                          ('y',np.int64),('d',np.bool_),('trap',np.bool_)))}
    if output['x'].shape[1:]!=(100,) or output['t'].shape[1:]!=(6,):
        raise ValueError('bad feature shape')
    if not output['m'][np.arange(len(output['y'])),output['y']].all():
        raise AssertionError('illegal label')
    return output,dict(counts)

def mix(offline, earlier, newest, mode):
    """Predeclared curricula, no validation/test row can enter training."""
    if mode not in ('uniform','trap_hard'):raise ValueError(mode)
    d=np.flatnonzero(newest['d'])
    td=np.flatnonzero(newest['d'] & newest['trap'])
    cases=[]
    for key in ('x','t','m','y'):
        z=[offline[key],earlier[key],newest[key],newest[key][d]]
        if mode=='trap_hard':
            z.extend((newest[key][td],newest[key][td],newest[key][newest['trap']]))
        cases.append(np.concatenate(z))
    return dict(zip(('x','t','m','y'),cases))

def tail(rows, model, comparator):
    """Paired, seed-block tail analysis with zero cherry-picked seed removal."""
    lookup={(r['seed'],r['style'],r['case'],r['model']):r for r in rows}
    ss=sorted({r['seed'] for r in rows})
    d=np.asarray([np.mean([lookup[s,sty,'trap',model]['cells']-
                          lookup[s,sty,'trap',comparator]['cells'] for sty in STYLES]) for s in ss])
    return {'mean':round(float(np.mean(d)),4),'min_seed_delta':round(float(np.min(d)),4),
            'p10_seed_delta':round(float(np.quantile(d,.10)),4),
            'seeds_loss_ge_5':int((d<=-5).sum()),'seed_count':len(ss)}

def evaluate(rows,name,seeds):
    s=summary(rows)
    return {'summary':s,'guard_vs_d017':admissibility(s,name),
            'ci_vs_d017':bootstrap(rows,name,seeds),
            'trap_tail_vs_d017':tail(rows,name,'d017'),
            'trap_tail_vs_d021':tail(rows,name,'d021')}

def main(argv=None):
    ap=argparse.ArgumentParser()
    for key in ('d007','d017','q8','d021','offline','earlier'):
        ap.add_argument('--'+key,required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--steps',type=int,default=120)
    a=ap.parse_args(argv)
    check_splits()
    if not 1<=a.steps<=180:raise ValueError('max 180 optimizer steps per candidate')
    src={k:sha(getattr(a,k)) for k in ('d007','d017','q8','d021','offline','earlier')}
    if [src[k] for k in ('d007','d017','q8')]!=[D007,D017,Q8]:
        raise ValueError('frozen source SHA mismatch')
    for k in ('offline','earlier'):
        with np.load(getattr(a,k),allow_pickle=False) as z:
            ds={n:z[n] for n in ('x','t','m','y')}
        if ds['x'].shape[1]!=100 or ds['t'].shape[1]!=6:
            raise ValueError('invalid '+k+' schema')
        if not ds['m'][np.arange(len(ds['y'])),ds['y']].all():
            raise AssertionError('illegal '+k+' teacher target')
        if k=='offline':offline=ds
        else:earlier=ds
    teacher=Controller(load(a.d007),parents(a.q8),Gate.read(a.d017),.5,.58)
    baseline=load(a.d021)
    newest,counts=collect(baseline,teacher)
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'d022_onpolicy.npz',**newest)
    variants={};valid={};updates=0
    # Only the curriculum differs; the exact same mild D021 source checkpoint
    # and optimizer budget anchor both arms.
    for idx,mode in enumerate(('uniform','trap_hard')):
        data=mix(offline,earlier,newest,mode)
        obj=load(a.d021)
        info=fit(obj,data,a.steps,220221+idx,.12,anchor_strength=.085,lr=.00065)
        updates+=info['updates']
        obj.save(out/('d022_'+mode+'.npz'))
        variants[mode]=obj
        v,_,_=rollout(VALID,{'d017':teacher,'d021':baseline,'d022':obj})
        scores=evaluate(v,'d022',VALID)
        valid[mode]={'fit':info,'corpus_rows':len(data['y']),
                     'checkpoint_sha256':sha(out/('d022_'+mode+'.npz')),
                     'guard':scores['guard_vs_d017'],
                     'trap_tail_vs_d021':scores['trap_tail_vs_d021']}
    # A candidate cannot compensate for failed branch/coverage retention by
    # maximizing coverage elsewhere; reject both if none meets hard guards.
    eligible=[m for m in variants if valid[m]['guard']['admissible']]
    selected=max(eligible or variants,key=lambda m:valid[m]['guard']['score'])
    controls={'d017':teacher,'d021':baseline,'d022':variants[selected]}
    result={'schema':SCHEMA,'environment':'external_synthetic_host',
            'native_cybersand_executed':False,'fresh_parent_forwards':0,
            'source_q8_seed':570,'source_sha256':src,
            'splits':{k:[v.start,v.stop-1] for k,v in
                      [('train',TRAIN),('validation',VALID),('test',TEST),
                       ('confirmation',CONFIRM),('stress',STRESS)]},
            'new_onpolicy_rows':len(newest['y']),
            'onpolicy_disagreements':int(newest['d'].sum()),
            'trap_disagreements':int((newest['d']&newest['trap']).sum()),
            'case_counts':counts,'old_corpus_rows':len(earlier['y']),
            'offline_rows':len(offline['y']),
            'optimizer_updates':updates,'validation':valid,
            'selected_for_diagnostic':selected,'validation_eligible':eligible,
            'new_onpolicy_sha256':sha(out/'d022_onpolicy.npz')}
    for split,seeds in [('test',TEST),('confirmation',CONFIRM),('stress',STRESS)]:
        pics=out/'pngs' if split=='test' else None
        r,receipts,_=rollout(seeds,controls,draw=pics,
             receipt_seeds=(67001,67005,67009,67013) if split=='test' else ())
        jsonl(out/(f'd022_{split}.jsonl'),r)
        if split=='test':jsonl(out/'d022_action_receipts.jsonl',receipts)
        result[split]=evaluate(r,'d022',seeds)
    result['candidate_status']='RESEARCH_ONLY' if eligible and all(
        result[k]['guard_vs_d017']['admissible'] for k in ('test','confirmation','stress')) else 'REJECTED'
    (out/'d022_metrics.json').write_text(js(result)+'\n')
    (out/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n'
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt'))
    print(js({'selected':selected,'status':result['candidate_status'],
              'training_rows':len(newest['y']),'updates':updates,
              'eval':{k:result[k]['guard_vs_d017']['delta'] for k in ('test','confirmation','stress')}}),flush=True)
    return result

if __name__=='__main__':main()
