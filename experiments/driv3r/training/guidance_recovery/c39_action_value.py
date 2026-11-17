"""C39: action-conditional, host-executed return-value research (synthetic only).

The label oracle runs independent legal candidate suffixes OFFLINE, including
repeated goal changes and false branches. Deployed inference sees host-observed
features + the host legal menu; never the graph or future outcome. No CyberSand
production change, native-ABI validation, or model promotion is implied.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from experiment import sha
from signed_feedback import HostAdvice
from c28_temporal_option import OptionHead, scores, probs
from c29_probe_recovery import PREREQS
from c34_independent_maze import make_world, executed_suffix, obs, ReArmedWatch
from c36_personality_budget import load_c12, a0, style_project

SCHEMA='driv3r-c39-action-conditional/1'
MODELS=('C18-A0','C34-RESET','C39-ALIGNED','C39-SHUFFLED','C39-RANDOM')
DIRECTIVES=(('none',0.),('backtrack',1.),('continue',1.))

class ActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(34,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def action_features(w,menu):
    shared=obs(w,menu)
    return np.asarray([np.concatenate((shared,np.asarray(c.features,np.float32))) for c in menu[:9]],np.float32)


def collect(seed=390107,graphs=56,steps=3,epochs=10):
    """Train only on tree mazes; held-out tree and braid are new graph seeds."""
    rng=np.random.default_rng(seed+551)
    X=[];Y=[];groups=[];receipts=[];invalid=0
    for case in ('detour','forward'):
      for personality in (-1,1):
       for i in range(graphs//4):
        s=seed+17117*i+107*personality+(case=='forward')*991
        w,info=make_world(s,'tree',case,personality)
        for t in range(steps):
            menu=w.candidates(.35);F=action_features(w,menu)
            legal=[j for j in range(9) if menu[j].legal]
            if len(legal)<2:break
            # Execute each alternative in an independent host copy. Labels only.
            returns=np.asarray([executed_suffix(w,j,steps=10) for j in legal],np.float32)
            mean=float(returns.mean())
            ids=[]
            for j,r in zip(legal,returns):
                X.append(F[j]);Y.append(float(np.clip(r-mean,-2.,2.)))
                ids.append(len(Y)-1)
            groups.append(ids)
            receipts.append({'graph':info['graph_sha'],'state':sha([w.pos,w.history,w.goal]),
                'legal':legal,'returns':[float(x) for x in returns],'menu':sha([(c.action,c.target,c.legal) for c in menu])})
            # Host-executed advancing prefix; no teleport or illegal action.
            forward=[j for j in legal if j<8 and menu[j].target not in w.history]
            j=max(forward,key=lambda x:w.distance()-w.distance(menu[x].target)) if forward else 8
            assert menu[j].legal and w.apply(menu[j]) and not w.invalid_attempts
    X=np.asarray(X,np.float32);Y=np.asarray(Y,np.float32)
    permuted=Y.copy()
    for ids in groups:
        shuffled=rng.permutation(ids)
        permuted[ids]=Y[shuffled]
    assert len({r['graph'] for r in receipts})>=12 and len(X)>250
    assert not np.array_equal(Y,permuted) and np.isfinite(X).all()
    torch.set_num_threads(1);torch.manual_seed(seed+999)
    original=ActionHead().eval();models=[copy.deepcopy(original) for _ in range(2)]+[original]
    orders=[rng.permutation(len(Y)) for _ in range(epochs)]
    stats={}
    for label,head,labels in zip(('ALIGNED','SHUFFLED'),models,(Y,permuted)):
        opt=torch.optim.AdamW(head.parameters(),lr=.008,weight_decay=.01)
        curves=[];updates=0
        for order in orders:
            total=0.
            for start in range(0,len(order),64):
                ids=order[start:start+64]
                target=torch.from_numpy(labels[ids]);inp=torch.from_numpy(X[ids]);loss=nn.functional.smooth_l1_loss(head(inp),target)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
                total+=float(loss.detach());updates+=1
            curves.append(round(total,7))
        head.eval()
        with torch.no_grad():mae=float(torch.mean(torch.abs(head(torch.from_numpy(X))-torch.from_numpy(Y))))
        stats[label]={'updates':updates,'losses':curves,'train_mae_vs_true':mae}
    stats.update({'graphs':len({r['graph'] for r in receipts}),'states':len(groups),'candidate_examples':len(Y),
      'target_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
      'shuffle_sha256':hashlib.sha256(permuted.tobytes()).hexdigest(),'receipts_sha256':sha(receipts)})
    return models,stats,receipts


def heldout(models,seed=930501,graphs_per_family=12):
    """Blind action ranking with labels from separately executed host futures."""
    rows=[]
    for family in ('tree','braid'):
      for case in ('detour','forward'):
       for personality in (-1,1):
        for i in range(graphs_per_family):
            s=seed+i*19079+personality*187+(case=='forward')*509+(family=='braid')*37091
            w,info=make_world(s,family,case,personality)
            for t in range(3):
                menu=w.candidates(.35);legal=[j for j in range(9) if menu[j].legal]
                if len(legal)<2:break
                target=[executed_suffix(w,j,steps=10) for j in legal]
                inputs=torch.from_numpy(action_features(w,menu)[legal])
                with torch.no_grad():values=[h(inputs).detach().numpy().tolist() for h in models]
                rows.append({'graph':info['graph_sha'],'family':family,'case':case,'personality':personality,
                             'state':sha([w.pos,w.history,w.goal]),'legal':legal,'returns':target,'values':values,
                             'oracle_best':max(target),'menu_sha':sha([(c.action,c.target,c.legal) for c in menu])})
                forward=[j for j in legal if j<8 and menu[j].target not in w.history]
                j=max(forward,key=lambda x:w.distance()-w.distance(menu[x].target)) if forward else 8
                assert w.apply(menu[j]) and w.invalid_attempts==0
    totals={}
    for k,name in enumerate(('C39-ALIGNED','C39-SHUFFLED','C39-RANDOM')):
        preds=[r['returns'][int(np.argmax(r['values'][k]))] for r in rows]
        regret=[r['oracle_best']-p for p,r in zip(preds,rows)]
        totals[name]={'mean_oracle_regret':round(float(np.mean(regret)),6),
            'oracle_top1':sum(abs(v-r['oracle_best'])<1e-8 for v,r in zip(preds,rows)),
            'states':len(rows),'by_family':{fam:round(float(np.mean([d for d,r in zip(regret,rows) if r['family']==fam])),6)
              for fam in ('tree','braid')}}
    return rows,{'graphs':len({r['graph'] for r in rows}),'states':len(rows),'models':totals}


class ConditionalPolicy(ReArmedWatch):
    def __init__(self,option,head):
        super().__init__(option,None);self.action_head=head;self.action_states=0;self.action_changed=0;self.personality_shifts=[]
    def modify(self,base,w,advice,arm):
        menu,logits,bad,target=super().modify(base,w,advice,arm)
        if advice.directive==1 and arm>0 and self.phase in ('watch','probe','retrace'):
            legal=[j for j in range(9) if menu[j].legal]
            if len(legal)>1:
                with torch.no_grad():pred=self.action_head(torch.from_numpy(action_features(w,menu)[legal])).numpy()
                before=np.asarray(logits).copy()
                centered=pred-np.mean(pred)
                for j,v in zip(legal,centered):logits[j]+=float(.8*np.tanh(v))
                # Preserve baseline directional personality preference within 0.045.
                logits,pre,post,alpha=style_project(menu,before,logits,w.personality,max_shift=.045)
                self.personality_shifts.append((pre,post))
                self.action_changed+=int(np.max(np.abs(logits-before))>1e-6)
                self.action_states+=1
        return menu,logits,bad,target


def play(base,option,models,w,info,seed,person,model,directive,strength,switch,horizon=78,trace=False):
    advice=HostAdvice(w)
    if directive!='none':getattr(advice,'issue_'+directive)(w,strength=strength,ttl=9)
    policy=(None if model=='C18-A0' else
            ConditionalPolicy(option,models[{'C39-ALIGNED':0,'C39-SHUFFLED':1,'C39-RANDOM':2}[model]])
            if model.startswith('C39-') else ReArmedWatch(option,None))
    rng=np.random.default_rng(seed+int(strength*100)*31+{'none':0,'backtrack':411,'continue':977}[directive])
    actions=[];path=[];goals=[0,0];illegal=0;inkpref=0;inkopp=0
    protected=sha(sorted(w.protected))
    for t in range(horizon):
        if switch and t==50:
            goals[0]=int(goals[0] or w.distance()==0)
            w.goal=info['second_goal'];w.goal_change=True;w.goals_seen+=1;w.stuck=2
            if directive!='none':getattr(advice,'issue_'+directive)(w,strength=strength,ttl=9)
        if w.stopped:break
        if w.distance()==0:
            goals[int(switch and t>=50)]=1
            if not switch or t>=50:break
            continue
        if policy is None:menu,logits,bad,_=scores(base,w,advice,1.)
        else:menu,logits,bad,_=policy.modify(base,w,advice,1.)
        illegal+=bad
        pick=int(rng.choice(len(menu),p=probs(logits)))
        chosen=menu[pick];assert chosen.legal and w.apply(chosen) and w.invalid_attempts==0
        if policy is not None:policy.after(chosen)
        advice.advance(w);actions.append((chosen.action,chosen.target))
        if chosen.action[0] in ('INK','LIFT'):
            turn=float(chosen.features[13]-chosen.features[14])*person
            inkpref+=turn>0;inkopp+=turn<0
        if trace:path.append({'t':t,'position':w.pos,'goal':w.goal,'action':chosen.action[0],'phase':policy.phase if policy else 'baseline'})
        if w.distance()==0:goals[int(switch and t>=50)]=1
    assert protected==sha(sorted(w.protected))
    return {'goals':sum(goals),'first':goals[0],'second':goals[1],'steps':len(actions),
        'actions_sha':sha(actions),'ink_sha':sha(sorted(w.inked)),
        'host_rejections':w.invalid_attempts,'raw_illegal_top1':illegal,
        'preferred':int(inkpref),'opposed':int(inkopp),
        'action_states':getattr(policy,'action_states',0),
        'action_changed':getattr(policy,'action_changed',0),
        'personality_shift_pre':sum(v[0] for v in getattr(policy,'personality_shifts',[])),
        'personality_shift_post':sum(v[1] for v in getattr(policy,'personality_shifts',[])),
        'trajectory':path if trace else None}


def evaluate(base,option,heads,seed,episodes_per_case=1,horizon=78,directives=None):
    rows=[]
    for family in ('tree','braid'):
      for case in ('detour','forward'):
       for i in range(episodes_per_case):
        for personality in (-1,1):
         for switch in (False,True):
          for directive,strength in (DIRECTIVES if directives is None else directives):
            s=seed+17117*i+371*personality+(family=='braid')*14497+(case=='forward')*673+(switch)*2039
            world,info=make_world(s,family,case,personality)
            for model in MODELS:
                r=play(base,option,heads,copy.deepcopy(world),info,s,personality,model,directive,strength,
                       switch,horizon,trace=i==0 and personality==1 and directive=='backtrack' and strength==1.)
                rows.append({'model':model,'family':family,'case':case,'i':i,'switch':switch,
                             'personality':personality,'directive':directive,'strength':strength,
                             'seed':s,'graph_sha':info['graph_sha'],'prefix_sha':info['prefix_sha'],**r})
    return rows


def summarize(rows):
    conditions=lambda r:(r['family'],r['case'],r['switch'],r['directive'])
    key=lambda r:(r['family'],r['case'],r['i'],r['switch'],r['personality'],r['directive'])
    base={key(r):r for r in rows if r['model']=='C18-A0'}
    summaries={}
    for name in MODELS:
        rr=[r for r in rows if r['model']==name]
        guided=[r for r in rr if r['directive']!='none'];idle=[r for r in rr if r['directive']=='none']
        cond={str(c):sum(r['goals'] for r in guided if conditions(r)==c) for c in sorted(set(conditions(r) for r in guided))}
        summaries[name]={'goal_events':sum(r['goals'] for r in guided),'guided_n':len(guided),
            'by_family':{fam:sum(r['goals'] for r in guided if r['family']==fam) for fam in ('tree','braid')},
            'conditions':cond,'unguided_exact':sum((r['actions_sha'],r['ink_sha'])==(base[key(r)]['actions_sha'],base[key(r)]['ink_sha']) for r in idle),
            'unguided_n':len(idle),'host_rejections':sum(r['host_rejections'] for r in rr),
            'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in rr),
            'action_changed':sum(r['action_changed'] for r in rr),
            'personality':{str(p):{'preferred':sum(r['preferred'] for r in guided if r['personality']==p),
                            'opposed':sum(r['opposed'] for r in guided if r['personality']==p)} for p in (-1,1)}}
    for name in MODELS:
        for ref in ('C34-RESET','C39-SHUFFLED','C39-RANDOM'):
            summaries[name]['regressions_vs_'+ref]=[k for k,v in summaries[ref]['conditions'].items() if summaries[name]['conditions'][k]<v]
    return summaries


def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    heads,stats,receipts=collect()
    repeat,repstats,_=collect();assert stats==repstats
    checks={}
    for name,head,copyhead in zip(('C39-ALIGNED','C39-SHUFFLED','C39-RANDOM'),heads,repeat):
        for suffix,h in (('',head),('-REPLICA',copyhead)):
            save_file({k:v.detach().cpu().contiguous() for k,v in h.state_dict().items()},str(out/(name+suffix+'.safetensors')))
        assert (out/(name+'.safetensors')).read_bytes()==(out/(name+'-REPLICA.safetensors')).read_bytes()
        checks[name]=hashlib.sha256((out/(name+'.safetensors')).read_bytes()).hexdigest()
    held,scored=heldout(heads)
    (out/'C39-heldout.json').write_text(json.dumps(held,sort_keys=True)+'\n')
    (out/'C39-train-receipts.json').write_text(json.dumps(receipts,sort_keys=True)+'\n')
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]);base=a0(prior)
    option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    cohorts={};graph_ids={}
    for name,seed in (('A',391101),('B',719013),('C',937071),('D',1049003),('E',1217077),('F',1432901)):
        print('C39 cohort',name,flush=True,file=sys.stderr)
        cohort_path=out/f'C39-cohort-{name}.json'
        if cohort_path.exists():
            rows=json.loads(cohort_path.read_text())
        else:
            rows=evaluate(base,option,heads,seed,episodes_per_case=1,horizon=78)
            cohort_path.write_text(json.dumps(rows,sort_keys=True)+'\n')
        cohorts[name]=summarize(rows);graph_ids[name]=sha(sorted({r['graph_sha'] for r in rows}))
    report={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'train':stats,'checkpoint_sha256':checks,'optimizer_replica_byte_identical':True,
        'heldout':scored,'cohorts':cohorts,'graph_sets_sha256':graph_ids}
    (out/'C39-summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

def run_weak(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    heads=[]
    for label in ('C39-ALIGNED','C39-SHUFFLED','C39-RANDOM'):
        h=ActionHead();h.load_state_dict(load_file(str(out/(label+'.safetensors'))));h.eval();heads.append(h)
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]);base=a0(prior)
    option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    result={}
    for label,seed in (('A',391101),('B',719013)):
        fp=out/f'C39-weak-{label}.json'
        if fp.exists():rows=json.loads(fp.read_text())
        else:
            rows=evaluate(base,option,heads,seed,episodes_per_case=1,horizon=78,
                          directives=(('backtrack',.35),('continue',.35)))
            fp.write_text(json.dumps(rows,sort_keys=True)+'\n')
        result[label]=summarize(rows)
    (out/'C39-weak-summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('--weak-only',action='store_true')
    a=p.parse_args()
    if a.weak_only:
        r=run_weak(a.root,a.out)
        print(json.dumps({c:{m:v['goal_events'] for m,v in x.items()} for c,x in r.items()},indent=2))
        sys.exit(0)
    r=run(a.root,a.out)
    print(json.dumps({'training':r['train'],'heldout':r['heldout'],
         'cohorts':{n:{k:(v['goal_events'],len(v['regressions_vs_C34-RESET'])) for k,v in s.items()} for n,s in r['cohorts'].items()}},indent=2))
