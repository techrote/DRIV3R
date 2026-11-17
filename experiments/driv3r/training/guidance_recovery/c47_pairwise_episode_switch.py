"""C47: differential-goal supervision for cautious neural guidance intervention.

Synthetic host research ONLY. All labels come from legal executed trajectories on
matched host copies; no future result/graph oracle enters deployed features.
This experiment does not modify or validate the native CyberSand contract.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import sha
from c34_independent_maze import obs
from c39_action_value import play, action_features
import c41_topology_transfer as c41
import c42_wilson_transfer as c42
import c44_onpolicy_pairwise as c44

SEED=471731
BASES=('C18-A0','C34-RESET','C40-PAIR')
MODELS=BASES+('C47-ALIGNED','C47-SHUFFLED','C47-RANDOM','C47-RAW-RULE')
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.),('continue',1.))
TRAIN_ARMS=ARMS[1:]
COHORTS=(('A',477011),('B',578013),('C',679017))

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(19,12),nn.Tanh(),nn.Linear(12,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def context(w,prior,directive,strength,switch):
    """Host-observed starting state, advice and frozen-head dispersion only."""
    menu=w.candidates(strength if directive!='none' else 0.)
    o=obs(w,menu)
    legal=[i for i,c in enumerate(menu[:9]) if c.legal]
    with torch.no_grad():
        scores=prior(torch.from_numpy(action_features(w,menu)[legal])).numpy()
    spread=float(np.std(scores)) if len(scores)>1 else 0.
    # 9 obs dimensions + five cue/context + five bounded action-score features.
    meta=[float(w.personality),float(switch),float(strength),
          float(directive=='backtrack'),float(directive=='continue')]
    stats=[spread,float(np.ptp(scores)),float(np.mean(np.abs(scores))),
           len(legal)/9.,float(np.mean(scores>0))]
    x=np.concatenate((o,np.asarray(meta+stats,np.float32))).astype(np.float32)
    assert x.shape==(19,) and np.isfinite(x).all()
    return x


def world_rows(families,maker,seed,rep):
    arr=[]
    for fi,fam in enumerate(families):
      for case in ('detour','forward'):
       for person in (-1,1):
        for switch in (False,True):
         for i in range(rep):
          s=seed+fi*170003+(case=='forward')*19013+(person+1)*3737+switch*2971+i*21041
          w,info=maker(s,fam,case,person)
          arr.append((w,info,s,fam,case,switch))
    assert len({r[1]['graph_sha'] for r in arr})==len(arr)
    return arr


def execute(base,option,prior,w,info,seed,directive,strength,switch,name,trace=False):
    return play(base,option,([prior]*3 if name=='C40-PAIR' else []),
      copy.deepcopy(w),info,seed,w.personality,
      ('C39-ALIGNED' if name=='C40-PAIR' else name),
      directive,strength,switch,78,trace)


def paired_outcomes(base,option,prior,worlds,arms,retain=False):
    rows=[];traces=[]
    for w,info,seed,fam,case,switch in worlds:
      for directive,strength in arms:
        r={name:execute(base,option,prior,w,info,seed,directive,strength,switch,name) for name in BASES}
        assert all(z['host_rejections']==0 for z in r.values())
        assert all(z['actions_sha'] and z['ink_sha'] for z in r.values())
        rows.append({'graph':info['graph_sha'],'family':fam,'case':case,'personality':w.personality,
          'switch':switch,'directive':directive,'strength':strength,'seed':seed,
          'protected_sha':sha(sorted(w.protected)),
          'x':context(w,prior,directive,strength,switch).tolist(),
          'delta':int(r['C40-PAIR']['goals']-r['C34-RESET']['goals']),
          'results':r})
        if retain and directive=='backtrack' and strength==1. and fam=='wilson' and case=='detour' and switch and w.personality==1:
          for name in BASES:
            if sum(t['model']==name for t in traces)>=2:continue
            replay=execute(base,option,prior,w,info,seed,directive,strength,switch,name,True)
            assert (replay['actions_sha'],replay['ink_sha'])==(r[name]['actions_sha'],r[name]['ink_sha'])
            traces.append({'graph':info['graph_sha'],'family':fam,'model':name,'protected':sorted(w.protected),
              'initial':w.pos,'goal_events':r[name]['goals'],'actions_sha':r[name]['actions_sha'],
              'ink_sha':r[name]['ink_sha'],'trajectory':replay['trajectory']})
    return rows,{r[1]['graph_sha'] for r in worlds},traces


def shuffled_deltas(rows,seed):
    """Group-level shuffling keeps paired advice-arm vectors and strata prevalence."""
    rng=np.random.default_rng(seed+179)
    result=np.array([r['delta'] for r in rows],dtype=np.float32)
    groups=collections.defaultdict(list)
    for ix,r in enumerate(rows):
        groups[(r['family'],r['case'],r['personality'],r['switch'],r['graph'])].append(ix)
    strata=collections.defaultdict(list)
    for key in groups:strata[key[:4]].append(key)
    for keys in strata.values():
        keys=sorted(keys)
        # Mandatory nonidentity graph-level permutation, including 2-graph strata.
        offset=int(rng.integers(1,len(keys))) if len(keys)>1 else 0
        perm=np.roll(np.arange(len(keys)),offset)
        for dst,src in enumerate(perm):
            ai=groups[keys[dst]];bi=groups[keys[int(src)]]
            assert len(ai)==len(bi)
            result[ai]=[rows[j]['delta'] for j in bi]
    assert np.sort(result).tolist()==sorted(r['delta'] for r in rows)
    return result


def train(rows,seed=SEED,epochs=32):
    torch.set_num_threads(1);torch.manual_seed(seed)
    x=np.asarray([r['x'] for r in rows],np.float32)
    labels=np.asarray([r['delta'] for r in rows],np.float32)
    shifted=shuffled_deltas(rows,seed)
    assert not np.array_equal(labels,shifted),'shuffled control inactive'
    mean=x.mean(0);scale=np.maximum(x.std(0),.05)
    xx=torch.from_numpy(np.clip((x-mean)/scale,-6,6).astype(np.float32))
    torch.manual_seed(seed+1);init=Head();models=[copy.deepcopy(init),copy.deepcopy(init),init]
    rng=np.random.default_rng(seed+27);orders=[rng.permutation(len(x)) for _ in range(epochs)]
    stats={}
    for name,net,y in zip(('aligned','shuffled'),models,(labels,shifted)):
        truth=torch.from_numpy(y)
        # Weight from each arm's OWN labels, never the aligned labels. Otherwise
        # the shuffled control would inadvertently receive oracle information.
        weights=torch.from_numpy(np.where(y!=0,4.,.35).astype(np.float32))
        optimizer=torch.optim.AdamW(net.parameters(),lr=.004,weight_decay=.02)
        losses=[];updates=0
        for order in orders:
            total=0.
            for start in range(0,len(order),24):
                ids=order[start:start+24]
                pred=net(xx[ids]);loss=torch.sum(weights[ids]*nn.functional.smooth_l1_loss(pred,truth[ids],reduction='none'))/torch.sum(weights[ids]);
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
                updates+=1;total+=float(loss.detach())
            losses.append(round(total,6))
        net.eval();stats[name]={'updates':updates,'epoch_losses':losses}
    stats['train_graphs']=len({r['graph'] for r in rows});stats['records']=len(rows)
    stats['differential_outcomes']=collections.Counter(map(str,labels.tolist()))
    stats['targets_sha256']=hashlib.sha256(labels.tobytes()).hexdigest()
    stats['shuffled_sha256']=hashlib.sha256(shifted.tobytes()).hexdigest()
    stats['normalization_sha256']=hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()
    return models,stats,mean,scale


def select(row,models,mean,scale,model):
    if model in BASES:return model
    if model=='C47-RAW-RULE':
        # Predefined non-neural baseline: high ranker dispersion yields C40.
        return 'C40-PAIR' if row['x'][14]>.4 else 'C34-RESET'
    ix=MODELS.index(model)-3
    x=torch.from_numpy(np.clip((np.asarray(row['x'],np.float32)-mean)/scale,-6,6))
    with torch.no_grad():adv=float(models[ix](x).item())
    return 'C40-PAIR' if adv>.15 else 'C34-RESET'


def assess(rows,models,mean,scale,retain=False):
    output=[];decisions=collections.Counter();newtraces=[]
    for row in rows:
      for model in MODELS:
        chosen=select(row,models,mean,scale,model)
        outcome=row['results'][chosen]
        decisions[(model,chosen)]+=1
        output.append({k:row[k] for k in ('graph','family','case','personality','switch','directive','strength','protected_sha')}
         |{'model':model,'selected':chosen,'delta':row['delta'],
          **{k:v for k,v in outcome.items() if k!='trajectory'}})
    return output,{m:{name:int(decisions[(m,name)]) for name in BASES} for m in MODELS}


def summarize(trainrows,blindrows,online,graph_sets,models,mean,scale,stats,checks,selections):
    tr,bl,on=graph_sets
    assert tr.isdisjoint(bl) and tr.isdisjoint(on) and bl.isdisjoint(on)
    ref={(r['graph'],r['directive'],r['strength']):r for r in online if r['model']=='C18-A0'}
    grouped=collections.defaultdict(collections.Counter);bygraph=collections.defaultdict(collections.Counter)
    for r in online:
        assert r['host_rejections']==0
        if r['directive']=='none':
            base=ref[r['graph'],r['directive'],r['strength']]
            assert (r['actions_sha'],r['ink_sha'])==(base['actions_sha'],base['ink_sha'])
        else:
            grouped[(r['family'],r['case'],r['switch'],r['directive'],r['strength'])][r['model']]+=r['goals']
            bygraph[r['graph']][r['model']]+=r['goals']
    summary={'schema':'driv3r-c47-differential-goal-guidance/1',
      'graph_counts':{'train':len(tr),'blind':len(bl),'online':len(on)},
      'optimizer':stats,'checkpoints_sha256':checks,'total_labeled_episodes':len(online),
      'guided_episodes_per_policy':sum(r['directive']!='none' for r in online)//len(MODELS),
      'models':{},'paired':{},'blind':{},'selection':selections,'limitations':[
        'synthetic CyberSand-like host, not CyberSand native ABI or physical acceptance',
        'episode-start fixed controller selector, no mid-trajectory feedback switching',
        'label oracle uses matched full outcomes offline only; results rare and stochastic',
        'starting-state features do not expose distant maze structure or true future utility',
        'class weighting, 0.15 intervention threshold and 0.4 raw rule predetermined, no held-out tuning',
        'graph-cluster bootstrap descriptive; not proof of neural superiority',
        'trajectory-level personality preservation unproven; no native replay',
        'trained checkpoints remain local; GitHub Actions immutable artifact custody pending']}
    for name in MODELS:
        rr=[r for r in online if r['model']==name];guided=[r for r in rr if r['directive']!='none']
        summary['models'][name]={'guided_goal_events':sum(r['goals'] for r in guided),
         'unguided_hash_equal':sum(r['directive']=='none' for r in rr),
         'condition_regressions_vs_RESET':sum(v[name]<v['C34-RESET'] for v in grouped.values()),
         'by_family':{f:sum(r['goals'] for r in guided if r['family']==f) for f in c42.FAMILIES},
         'selected_pair':sum(r['selected']=='C40-PAIR' for r in guided),
         'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in rr),
         'preferred':sum(r['preferred'] for r in guided),'opposed':sum(r['opposed'] for r in guided)}
    for model in MODELS[3:]:
        for against in ('C34-RESET','C40-PAIR','C47-SHUFFLED'):
            if model==against:continue
            d=np.asarray([v[model]-v[against] for v in bygraph.values()],np.float64)
            rng=np.random.default_rng(SEED+MODELS.index(model)*49+MODELS.index(against))
            draws=rng.integers(len(d),size=(4000,len(d)))
            summary['paired'][model+'__'+against]={'delta':int(d.sum()),
             'positive_graphs':int((d>0).sum()),'negative_graphs':int((d<0).sum()),
             '95pct_graph_cluster_bootstrap':np.quantile(d[draws].sum(1),[.025,.975]).tolist()}
    for name in MODELS[3:]:
        chosen=[select(r,models,mean,scale,name) for r in blindrows]
        d=np.asarray([r['delta'] for r in blindrows],np.float64)
        true=(d>0);pred=np.array([z=='C40-PAIR' for z in chosen])
        summary['blind'][name]={'records':len(d),'positive_diff':int(true.sum()),'negative_diff':int((d<0).sum()),
          'ties':int((d==0).sum()),'selected_pair':int(pred.sum()),
          'strict_win_precision':float(np.mean(true[pred])) if pred.sum() else None,
          'incremental_goal_events_vs_RESET':int(d[pred].sum()),
          'false_positive_interventions':int(np.logical_and(pred,d<0).sum())}
    return summary


def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1)
    base,option,prior=c44.prerequisites(root)
    tw=world_rows(c41.FAMILIES,c41.world,SEED,2)
    tr,gt,_=paired_outcomes(base,option,prior,tw,TRAIN_ARMS)
    heads,stats,mean,scale=train(tr)
    repeat,repeatstats,rmean,rscale=train(tr)
    assert stats==repeatstats and np.array_equal(mean,rmean) and np.array_equal(scale,rscale)
    assert all(all(torch.equal(m.state_dict()[k],r.state_dict()[k]) for k in m.state_dict()) for m,r in zip(heads,repeat))
    checks={}
    for name,net in zip(MODELS[3:5],heads[:2]):
        path=out/(name+'.safetensors')
        save_file({k:v.contiguous() for k,v in net.state_dict().items()},str(path))
        checks[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    bws=world_rows(c42.FAMILIES,c42.world,471937,1)
    blind,gb,_=paired_outcomes(base,option,prior,bws,TRAIN_ARMS)
    rows=[];go=set();traces=[];selections=collections.defaultdict(collections.Counter)
    for cname,cseed in COHORTS:
        ws=world_rows(c42.FAMILIES,c42.world,cseed,1)
        block,g,paths=paired_outcomes(base,option,prior,ws,ARMS,retain=True)
        assert not(go&g);go|=g
        mapped,sel=assess(block,heads,mean,scale)
        rows+=mapped;traces+=paths
        for m,v in sel.items():selections[m].update(v)
        print('C47 online',cname,len(mapped),flush=True)
    summary=summarize(tr,blind,rows,(gt,gb,go),heads,mean,scale,stats,checks,dict(selections))
    summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['root_hashes']={'C40-prior':hashlib.sha256((root/'out_c40/C40-PAIR.safetensors').read_bytes()).hexdigest()}
    summary['graph_sets_sha256']={n:sha(sorted(g)) for n,g in (('train',gt),('blind',gb),('online',go))}
    for filename,content in [('C47-summary.json',summary),('C47-train.json',tr),('C47-blind.json',blind),('C47-online.json',rows),('C47-curated-paths.json',traces)]:
        (out/filename).write_text(json.dumps(content,sort_keys=True,indent=2 if filename=='C47-summary.json' else None)+'\n')
    print(json.dumps({'graph_sets':summary['graph_counts'],'train_diffs':stats['differential_outcomes'],
      'goals':{k:v['guided_goal_events'] for k,v in summary['models'].items()},
      'regressions':{k:v['condition_regressions_vs_RESET'] for k,v in summary['models'].items()},
      'selections':{k:v['selected_pair'] for k,v in summary['models'].items()},
      'blind':summary['blind'],'paired':summary['paired']},indent=2))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('out')
    a=ap.parse_args();run(a.root,a.out)
