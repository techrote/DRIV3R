"""Independent frozen-checkpoint validation for C55 (no retraining)."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from safetensors.torch import load_file
import c55_observed_waypoint_distillation as c55
import c54_delayed_branch_option as c54

VALIDATION_SEED=550118+2_600_013

def run(prior,train_output,out,unseen_reps=8,familiar_reps=4):
    prior=Path(prior);train_output=Path(train_output);out=Path(out);out.mkdir(exist_ok=True,parents=True)
    summary=json.loads((train_output/'C55-summary.json').read_text())
    historic=json.loads((prior/'C54-summary.json').read_text())
    exclusions={r['graph'] for values in historic['worlds'].values() for r in values}
    exclusions.update(r['graph'] for values in summary['receipts'].values() for r in values)
    base,option,_=c55.c54.c52.c50.c47.c44.prerequisites(prior/'predecessor_C42')
    heads={}
    for name,expected in summary['checkpoint_sha256'].items():
       file=train_output/(name+'.safetensors')
       assert hashlib.sha256(file.read_bytes()).hexdigest()==expected
       head=c55.Head();head.load_state_dict(load_file(str(file)));head.eval();heads[name]=head
    unseen,ur=c54.c52.worlds(c54.c52.ONLINE_FAMILIES,c54.c52.c50.c47.c42.world,
                         VALIDATION_SEED,unseen_reps,exclude=exclusions)
    fam,fr=c54.c52.worlds(c54.c52.TRAIN_FAMILIES,c54.c52.c50.c47.c41.world,
                         VALIDATION_SEED+100009,familiar_reps,
                         exclude=exclusions|{r['graph'] for r in ur})
    rows,traces,skips=c55.evaluate(base,option,unseen+fam,heads,trace_limit=27)
    # Cohort graph identity must be unique and disjoint from ALL prior tests.
    s=c55.summary(rows,summary['training'],summary['blind'],rows,
                  summary['checkpoint_sha256'],{'external':skips},
                  {'train':summary['receipts']['train'],
                   'blind':summary['receipts']['blind'],
                   'online':ur+fr})
    assert not ({r['graph'] for r in ur+fr}&exclusions)
    s['validation']={'independent_unseen_graphs':len(ur),'independent_familiar_graphs':len(fr),
       'excluded_previous_graphs':len(exclusions),'seed':VALIDATION_SEED,
       'frozen_checkpoint_no_retraining':True,
       'no_parameters_tuned_using_external_outcomes':True,
       'post_preview_exploratory':True}
    for fname,obj in [('C55-external-online.json',rows),('C55-external-traces.json',traces),('C55-external-summary.json',s)]:
        (out/fname).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'graphs':s['validation'],'eligible_episodes':s['guided_per_policy'],
       'goals':{k:v['goal_events'] for k,v in s['models'].items()},
       'paired_vs_RESET':s['paired']['C55-ALIGNED__C34-RESET'],
       'route_vs_RESET':s['paired']['C55-ROUTE__C34-RESET'],
       'gates':s['promotion_gates']},indent=2))
    return s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prior');p.add_argument('train_output');p.add_argument('out')
    p.add_argument('--unseen-reps',type=int,default=8);p.add_argument('--familiar-reps',type=int,default=4)
    a=p.parse_args();run(a.prior,a.train_output,a.out,a.unseen_reps,a.familiar_reps)
