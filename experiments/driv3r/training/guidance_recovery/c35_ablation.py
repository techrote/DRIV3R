"""C35 matched zero-training and sign-inversion controls, no extra optimizer."""
import argparse,copy,json
from pathlib import Path
from safetensors.torch import load_file
import torch
import c35_cross_generator_calibration as c

def run(root,out,family,seed=351701,horizon=70,checkpoint_dir=None):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert family in c.TRAIN_FAMILIES+c.TEST_FAMILIES
    p=c.load_c12([root/'prerequisites'/f'guidance_recovery_{k}.safetensors' for k in c.PREREQS]);base=c.a0(p)
    opt=c.OptionHead();opt.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else root/'results'
    heads={}
    for name in ('C35-RANDOM','C35-INVERT'):
        ensemble=[]
        for i in range(3):
            torch.manual_seed(351011+i*557)
            h=c.CalHead()
            if name=='C35-INVERT':
                h.load_state_dict(load_file(str(checkpoint_dir/f'C35-CAL-{i}.safetensors')))
                with torch.no_grad():
                    h.net[-1].weight.mul_(-1);h.net[-1].bias.mul_(-1)
            h.eval();ensemble.append(h)
        heads[name]=ensemble
    rows=[]
    for case in c.CASES:
     for switch in (False,True):
      for person in (-1,1):
       for arm in (.35,1.):
        for directive,strength in c.DIRECTIVES:
         s=(seed+379*person+int(arm*100)*199+sum(map(ord,case))*411+
             (c.TRAIN_FAMILIES+c.TEST_FAMILIES).index(family)*17291+int(switch)*12373)
         w,info=c.make_world(s,family,case,person)
         for name in heads:
            result=c.play(base,opt,heads[name],copy.deepcopy(w),info,s,arm,directive,strength,name,switch,horizon)
            rows.append({'model':name,'cohort_seed':seed,'family':family,'case':case,'switch':switch,
                'i':0,'personality':person,'arm':arm,'directive':directive,'strength':strength,'seed':s,'graph_sha':info['graph_sha'],**result})
    (out/f'{family}-{seed}-ablation.json').write_text(json.dumps(rows,sort_keys=True)+'\n')
    print(family,seed,{name:sum(r['total_success'] for r in rows if r['model']==name and r['directive']!='none') for name in heads})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('family');p.add_argument('--seed',type=int,default=351701);p.add_argument('--checkpoint-dir');a=p.parse_args();run(a.root,a.out,a.family,a.seed,checkpoint_dir=a.checkpoint_dir)
