"""C35 bounded, independent source-matched per-family evaluation partition.

No optimizer or network/API calls. Resumes without recomputing other families.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from safetensors.torch import load_file
import c35_cross_generator_calibration as c

def run(root,out,family,seed=351701,n=1,horizon=70,checkpoint_dir=None):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert family in c.TRAIN_FAMILIES+c.TEST_FAMILIES
    p=c.load_c12([root/'prerequisites'/f'guidance_recovery_{k}.safetensors' for k in c.PREREQS]);base=c.a0(p)
    opt=c.OptionHead();opt.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else root/'results'
    heads=[]
    for i in range(3):
        h=c.CalHead();h.load_state_dict(load_file(str(checkpoint_dir/f'C35-CAL-{i}.safetensors')));h.eval();heads.append(h)
    rows=[];graphs={}
    for case in c.CASES:
     for switch in (False,True):
      for i in range(n):
       for person in (-1,1):
        for arm in (.35,1.):
         for directive,strength in c.DIRECTIVES:
          s=(seed+10007*i+379*person+int(arm*100)*199+sum(map(ord,case))*411+
             (c.TRAIN_FAMILIES+c.TEST_FAMILIES).index(family)*17291+int(switch)*12373)
          w,info=c.make_world(s,family,case,person);graphs[info['graph_sha']]=info
          for model in ('C18-A0','C34-RESET','C35-SIGNED','C35-GUARD'):
           rr=c.play(base,opt,heads,copy.deepcopy(w),info,s,arm,directive,strength,model,switch,horizon,
               trace=i==0 and person==1 and arm==1. and directive=='backtrack' and strength==1.)
           rows.append({'model':model,'family':family,'case':case,'switch':switch,'i':i,
                'personality':person,'arm':arm,'directive':directive,'strength':strength,
                'seed':s,'graph_sha':info['graph_sha'],**rr})
    target=out/f'{family}-{seed}.json';target.write_text(json.dumps({'rows':rows,'graphs':graphs},sort_keys=True)+'\n')
    print(family,seed,len(rows),{m:sum(x['total_success'] for x in rows if x['model']==m and x['directive']!='none') for m in ('C18-A0','C34-RESET','C35-SIGNED','C35-GUARD')}, flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('family');p.add_argument('--seed',type=int,default=351701);p.add_argument('--n',type=int,default=1);p.add_argument('--horizon',type=int,default=70);p.add_argument('--checkpoint-dir');a=p.parse_args()
    run(a.root,a.out,a.family,a.seed,a.n,a.horizon,a.checkpoint_dir)
