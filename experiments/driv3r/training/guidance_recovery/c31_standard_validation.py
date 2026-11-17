import argparse,json
from pathlib import Path
from safetensors.torch import load_file
from c31_observed_progress import *
args=argparse.ArgumentParser();args.add_argument('root');args.add_argument('out');args.add_argument('--checkpoint',default='out_revised/C31-PROGRESS.safetensors');opt=args.parse_args()
root=Path(opt.root)
prior=load_c12([root/'prerequisites'/f'guidance_recovery_{name}.safetensors' for name in PREREQS]);base=a0(prior)
option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
h=ProgressHead();h.load_state_dict(load_file(str(root/opt.checkpoint)))
rows=[]
for cohort,cohortseed in [('a',481000),('b',581000)]:
 for i in range(2):
  for personality in (-1,1):
   for outcome in ('reverse','short','near','tunnel','decoy'):
    for arm in (.35,1.):
     for directive in ('none','backtrack','continue'):
      for strength in ( (0.,) if directive=='none' else (.35,1.) ):
       ss=cohortseed+i*8017+personality*313+('reverse','short','near','tunnel','decoy').index(outcome)*3319+int(arm*100)*127
       for name in ('C18-A0','C31-DIRECT','C31-LEARNED','C31-RULE'):
        w=executed_world(i,outcome,personality,ss,False)
        if name=='C18-A0':
         cp=ProbeHead();cp.load_state_dict(load_file(str(root/'results/C29-PROBE.safetensors')))
         r=custom_rollout(base,option,cp,i,outcome,personality,arm,directive,strength,ss,horizon=90,world_override=w,baseline_variant='C18-A0')
        else:r=rollout(base,option,h,w,ss,arm,directive,strength,'LEARNED' if name.endswith('LEARNED') else ('DIRECT' if name.endswith('DIRECT') else 'RULE'),90)
        rows.append(dict(r,model=name,cohort=cohort,index=i,personality=personality,outcome=outcome,arm=arm,directive=directive,strength=strength))
print('NUMBER',len(rows))
for cohort in ('a','b'):
 sub=[r for r in rows if r['cohort']==cohort]
 baseline={(r['index'],r['personality'],r['outcome'],r['arm'],r['directive'],r['strength']):r for r in sub if r['model']=='C18-A0'}
 for name in ('C18-A0','C31-DIRECT','C31-LEARNED','C31-RULE'):
  s=[r for r in sub if r['model']==name];g=[r for r in s if r['directive']!='none'];off=[r for r in s if r['directive']=='none']
  groups={ (o,d,st) for o in ('reverse','short','near','tunnel','decoy') for d in ('backtrack','continue') for st in (.35,1.)};reg=[]
  for o,d,st in groups:
   baseline_total=sum(r['success'] for r in sub if r['model']=='C18-A0' and (r['outcome'],r['directive'],r['strength'])==(o,d,st))
   model_total=sum(r['success'] for r in g if (r['outcome'],r['directive'],r['strength'])==(o,d,st))
   if model_total<baseline_total:reg.append((o,d,st,baseline_total,model_total))
  exact=sum(r['action_sha256']==baseline[(r['index'],r['personality'],r['outcome'],r['arm'],r['directive'],r['strength'])]['action_sha256'] and r['ink_sha256']==baseline[(r['index'],r['personality'],r['outcome'],r['arm'],r['directive'],r['strength'])]['ink_sha256'] for r in off)
  print(cohort,name,'guided',sum(r['success'] for r in g),'/',len(g),'regress',reg,'off',exact,len(off))
Path(opt.out).write_text(json.dumps(rows,sort_keys=True,indent=2))
