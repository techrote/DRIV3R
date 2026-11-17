"""C48: braid-enriched, counterfactually supervised selective recovery.

Research-only synthetic host. Episode-start advice/candidate observations only at
inference. Full executed-goal differences supervise training; never used during
inference. This is *not* validation of native CyberSand actions or materials.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
import c47_pairwise_episode_switch as c47

NAME='C48'
SEED=480117
ARM_TRAIN=(('backtrack',.35),('backtrack',1.))
ARMS=c47.ARMS
MODELS=('C18-A0','C34-RESET','C40-PAIR','C47-ALIGNED','C48-HARD','C48-SHUFFLED','C48-RANDOM')
COHORTS=(('ID-A',481911,c47.c41.FAMILIES,c47.c41.world,1),
         ('ID-B',482921,c47.c41.FAMILIES,c47.c41.world,1),
         ('OOD-A',483931,c47.c42.FAMILIES,c47.c42.world,1),
         ('OOD-B',484941,c47.c42.FAMILIES,c47.c42.world,1))

class BinaryHead(c47.Head):pass

def train(rows,seed=SEED,epochs=36):
    """Weighted, FULL-label training, not train-subset cherry-picking.

    Graph-stratum permutation makes a source-matched label-shuffle control.
    None of the labels or test graph IDs reach online inference.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    labels=np.asarray([r['delta'] for r in rows],np.float32)
    shifted=c47.shuffled_deltas(rows,seed)
    assert not np.array_equal(labels,shifted)
    x=np.asarray([r['x'] for r in rows],np.float32)
    mean=x.mean(0);scale=np.maximum(x.std(0),.05)
    xx=torch.from_numpy(np.clip((x-mean)/scale,-6,6).astype(np.float32))
    torch.manual_seed(seed+11)
    initialized=BinaryHead()
    heads=[copy.deepcopy(initialized),copy.deepcopy(initialized)]
    rng=np.random.default_rng(seed+27)
    orders=[rng.permutation(len(rows)) for _ in range(epochs)]
    stats={}
    for name,net,raw in zip(('hard','shuffled'),heads,(labels,shifted)):
        optimizer=torch.optim.AdamW(net.parameters(),lr=.0025,weight_decay=.02)
        yy=torch.from_numpy((raw>0).astype(np.float32))
        # Win/loss receive equal magnitude, ties weakly discourage intervention.
        weights=torch.from_numpy(np.where(raw>0,12.,np.where(raw<0,12.,.20)).astype(np.float32))
        losses=[];steps=0
        for order in orders:
            epoch=0.
            for off in range(0,len(order),32):
                ix=order[off:off+32]
                pred=net(xx[ix]);loss=(nn.functional.binary_cross_entropy_with_logits(pred,yy[ix],reduction='none')*weights[ix]).sum()/weights[ix].sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step();steps+=1;epoch+=float(loss.detach())
            losses.append(round(epoch,6))
        net.eval()
        stats[name]={'adamw_updates':steps,'epoch_losses':losses,
                     'target_count':collections.Counter(map(str,raw.tolist()))}
    stats['train_graphs']=len({r['graph'] for r in rows})
    stats['train_rows']=len(rows)
    stats['targets_sha256']=hashlib.sha256(labels.tobytes()).hexdigest()
    stats['shuffled_sha256']=hashlib.sha256(shifted.tobytes()).hexdigest()
    stats['x_mean_scale_sha256']=hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()
    return heads,mean,scale,stats


def old_c47(root):
    """Immutable C47 pretrained control; normalization recovered from its source cohort."""
    tr=json.loads((Path(root)/'output/C47-train.json').read_text())
    x=np.asarray([r['x'] for r in tr],np.float32)
    net=c47.Head()
    net.load_state_dict(load_file(str(Path(root)/'output/C47-ALIGNED.safetensors')))
    net.eval()
    return net,x.mean(0),np.maximum(x.std(0),.05)


def choose(row,models,mean,scale,old,model):
    if model in c47.BASES:return model
    # No intervention outside host backtrack cues. Host legality is unchanged.
    if row['directive']!='backtrack':return 'C34-RESET'
    if model=='C48-RANDOM':
        # Graph/arm-seeded, deterministic 20% random intervention control.
        receipt=f"{row['graph']}|{row['directive']}|{row['strength']}|{SEED}"
        return 'C40-PAIR' if int(hashlib.sha256(receipt.encode()).hexdigest()[:8],16)/2**32<.20 else 'C34-RESET'
    if model=='C47-ALIGNED':
        net,m,sc=old;threshold=.15
    else:
        net=models[0 if model=='C48-HARD' else 1]
        m,sc=mean,scale;threshold=1.0  # sigmoid(1.0) = 0.731; pre-specified
    x=torch.from_numpy(np.clip((np.asarray(row['x'],np.float32)-m)/sc,-6,6))
    with torch.no_grad(): score=float(net(x).item())
    return 'C40-PAIR' if score>threshold else 'C34-RESET'


def enrich(base,opt,prior):
    # Train on braid-only graphs; retain all matched outcome examples in train.
    tw=c47.world_rows(('braid','prim_braid'),c47.c41.world,480137,8)
    trainrows,train_graphs,_=c47.paired_outcomes(base,opt,prior,tw,ARM_TRAIN)
    return trainrows,train_graphs


def eval_worlds(base,opt,prior,worlds,heads,mean,scale,old,retain=False):
    all_rows=[];selected=collections.Counter();paths=[]
    for w,info,seed,family,case,switch in worlds:
      for directive,strength in ARMS:
        ref={k:c47.execute(base,opt,prior,w,info,seed,directive,strength,switch,k) for k in c47.BASES}
        x=c47.context(w,prior,directive,strength,switch).tolist()
        row={'graph':info['graph_sha'],'family':family,'case':case,'personality':w.personality,
             'switch':switch,'directive':directive,'strength':strength,'seed':seed,
             'x':x,'delta':ref['C40-PAIR']['goals']-ref['C34-RESET']['goals']}
        for model in MODELS:
            selected_name=choose(row,heads,mean,scale,old,model)
            selected[(model,selected_name)]+=1
            rr=ref[selected_name]
            assert rr['host_rejections']==0
            all_rows.append({k:row[k] for k in ('graph','family','case','personality','switch','directive','strength','seed','delta')}
                            |{'model':model,'selected':selected_name,
                              **{k:v for k,v in rr.items() if k!='trajectory'}})
        if retain and directive=='backtrack' and strength==1. and (row['delta']!=0 or len(paths)<6):
            if sum(k['graph']==row['graph'] for k in paths):continue
            if len({k['graph'] for k in paths})>=10:continue
            for model in ('C34-RESET','C40-PAIR','C48-HARD','C48-SHUFFLED'):
                chosen=choose(row,heads,mean,scale,old,model)
                traced=c47.execute(base,opt,prior,w,info,seed,directive,strength,switch,chosen,True)
                assert (traced['actions_sha'],traced['ink_sha'])==(ref[chosen]['actions_sha'],ref[chosen]['ink_sha'])
                paths.append({'graph':row['graph'],'family':family,'case':case,'personality':w.personality,
                              'model':model,'selected':chosen,'goal_events':traced['goals'],
                              'initial':w.pos,'protected':sorted(w.protected),'trajectory':traced['trajectory'],
                              'actions_sha':traced['actions_sha'],'ink_sha':traced['ink_sha']})
    return all_rows,selected,paths


def summarize(trainrows,blindrows,online,gt,gb,go,head_stats,checks,source_sha):
    assert gt.isdisjoint(gb) and gt.isdisjoint(go) and gb.isdisjoint(go)
    assert len(go)==len({r['graph'] for r in online})
    baseline={(r['graph'],r['directive'],r['strength']):r for r in online if r['model']=='C18-A0'}
    groups=collections.defaultdict(collections.Counter)
    clustered=collections.defaultdict(collections.Counter)
    for r in online:
        assert r['host_rejections']==0
        if r['directive']=='none':
            b=baseline[(r['graph'],r['directive'],r['strength'])]
            assert (r['actions_sha'],r['ink_sha'])==(b['actions_sha'],b['ink_sha'])
        else:
            groups[(r['family'],r['case'],r['switch'],r['directive'],r['strength'])][r['model']]+=r['goals']
            clustered[r['graph']][r['model']]+=r['goals']
    summary={'schema':'driv3r-c48-hard-case-guidance/1','source_sha256':source_sha,
     'graph_counts':{'train':len(gt),'blind':len(gb),'online':len(go)},
     'train_differences':dict(collections.Counter(map(str,(r['delta'] for r in trainrows)))),
     'training':head_stats,'checkpoints_sha256':checks,
     'graph_sha256':{n:c47.sha(sorted(s)) for n,s in [('train',gt),('blind',gb),('online',go)]},
     'models':{},'paired':{},'blind':{},'limitations':[
       'synthetic CyberSand-like host only; not native or physical CyberSand acceptance',
       'intentional braid-enriched training distribution, weak generalization to tree topologies possible',
       'no mid-trajectory controller change; decisions at episode start only',
       'full-episode outcome labels have high tie rates and low discriminative power',
       'trained head uses starting-observation features; cannot infer hidden maze topology',
       'bootstraps descriptive and conditions multiple-comparison exploratory, not confirmation',
       'unguided parity is insufficient to establish whole-trajectory personality equivalence',
       'checkpoints remain local (not immutable GitHub Actions custody)']}
    for model in MODELS:
        z=[r for r in online if r['model']==model]
        guided=[r for r in z if r['directive']!='none']
        counts=collections.Counter(r['selected'] for r in guided)
        pref=sum(r['preferred'] for r in guided);opp=sum(r['opposed'] for r in guided)
        summary['models'][model]={
            'guided_goals':int(sum(r['goals'] for r in guided)),
            'guided_episodes':len(guided),'unguided_exact':sum(r['directive']=='none' for r in z),
            'regressed_condition_groups_vs_RESET':int(sum(v[model]<v['C34-RESET'] for v in groups.values())),
            'selected_pair':int(counts['C40-PAIR']),
            'preferred_ink':pref,'opposed_ink':opp,
            'whole_path_preferred_ink_fraction':pref/max(1,pref+opp),
            'raw_illegal_top1':int(sum(r['raw_illegal_top1'] for r in z)),
            'by_family':dict(collections.Counter({f:sum(r['goals'] for r in guided if r['family']==f) for f in sorted({r['family'] for r in online})})),
            'mean_personality_shift_within_state':sum(r['personality_shift_post'] for r in guided)/max(1,sum(r['action_states'] for r in guided))}
    for model in MODELS[3:]:
        for against in ('C34-RESET','C47-ALIGNED','C48-SHUFFLED'):
            if model==against:continue
            d=np.array([v[model]-v[against] for v in clustered.values()],np.int32)
            rng=np.random.default_rng(SEED+37*MODELS.index(model)+MODELS.index(against))
            idx=rng.integers(len(d),size=(4000,len(d)))
            summary['paired'][f'{model}__{against}']={
                'delta':int(d.sum()), 'positive_graphs':int((d>0).sum()),'negative_graphs':int((d<0).sum()),
                'bootstrap_95_descriptive':np.quantile(d[idx].sum(1),[.025,.975]).tolist()}
    for model in MODELS[3:]:
        br=[r for r in blindrows if r['model']==model and r['directive']!='none']
        picks=[r for r in br if r['selected']=='C40-PAIR']
        summary['blind'][model]={'guided_n':len(br),'pair_picks':len(picks),
            'genuine_wins':sum(r['delta']>0 for r in picks),'genuine_losses':sum(r['delta']<0 for r in picks),
            'excess_goals_vs_RESET':sum(r['delta'] for r in picks)}
    return summary


def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    base,opt,prior=c47.c44.prerequisites(root/'previous/predecessor_C42')
    old=old_c47(root)
    tr,gt=enrich(base,opt,prior)
    print('C48 training generated',len(tr),len(gt),flush=True)
    heads,mean,scale,stats=train(tr)
    again,m2,s2,st2=train(tr)
    assert stats==st2 and np.array_equal(mean,m2) and np.array_equal(scale,s2)
    assert all(all(torch.equal(x.state_dict()[k],y.state_dict()[k]) for k in x.state_dict()) for x,y in zip(heads,again))
    checks={}
    for name,net in zip(('C48-HARD','C48-SHUFFLED'),heads):
        p=out/(name+'.safetensors');save_file({k:v.contiguous() for k,v in net.state_dict().items()},str(p))
        checks[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    print('C48 training replicated',flush=True)
    bw=c47.world_rows(c47.c42.FAMILIES,c47.c42.world,480717,1)
    blind,_,_=eval_worlds(base,opt,prior,bw,heads,mean,scale,old)
    gb={r[1]['graph_sha'] for r in bw}
    print('C48 blind complete',len(blind),flush=True)
    allrows=[];allpaths=[];go=set();cohorts=[]
    for cname,seed,families,worldfn,rep in COHORTS:
        ws=c47.world_rows(families,worldfn,seed,rep)
        chunk,_,paths=eval_worlds(base,opt,prior,ws,heads,mean,scale,old,retain=(cname in ('ID-A','OOD-A')))
        assert not(go&{w[1]['graph_sha'] for w in ws});go|={w[1]['graph_sha'] for w in ws}
        allrows+=chunk;allpaths+=paths
        cohorts.append({'name':cname,'graphs':len(ws),'records':len(chunk)})
        print('C48 online',cname,len(chunk),flush=True)
    srcsha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary=summarize(tr,blind,allrows,gt,gb,go,stats,checks,srcsha)
    summary['cohorts']=cohorts
    summary['predecessor']={'C47_source_git_blob':'a89d852a858de828415686098b74efe205e9567c',
       'C47_checkpoint_sha256':hashlib.sha256((root/'output/C47-ALIGNED.safetensors').read_bytes()).hexdigest(),
       'C40_checkpoint_sha256':hashlib.sha256((root/'previous/predecessor_C42/out_c40/C40-PAIR.safetensors').read_bytes()).hexdigest()}
    for fname,obj in [('C48-summary.json',summary),('C48-training.json',tr),('C48-blind.json',blind),
                      ('C48-online.json',allrows),('C48-curated-paths.json',allpaths)]:
        (out/fname).write_text(json.dumps(obj,sort_keys=True,indent=2 if fname=='C48-summary.json' else None)+'\n')
    print(json.dumps({'training_deltas':summary['train_differences'],'goals':{m:d['guided_goals'] for m,d in summary['models'].items()},
           'regressions':{m:d['regressed_condition_groups_vs_RESET'] for m,d in summary['models'].items()},
           'paired':summary['paired'],'blind':summary['blind']},indent=2),flush=True)
    return summary

def staged(root,out,stage):
    """Bounded sequential entry points; no hidden automatic training on imports."""
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if stage=='train':
        base,opt,prior=c47.c44.prerequisites(root/'previous/predecessor_C42')
        tr,gt=enrich(base,opt,prior)
        heads,mean,scale,stats=train(tr)
        again,m2,s2,st2=train(tr)
        assert stats==st2 and np.array_equal(mean,m2) and np.array_equal(scale,s2)
        assert all(all(torch.equal(x.state_dict()[k],y.state_dict()[k]) for k in x.state_dict()) for x,y in zip(heads,again))
        checks={}
        for name,net in zip(('C48-HARD','C48-SHUFFLED'),heads):
            f=out/(name+'.safetensors')
            save_file({k:v.contiguous() for k,v in net.state_dict().items()},str(f))
            checks[name]=hashlib.sha256(f.read_bytes()).hexdigest()
        ledger={'optimizer':stats,'checks':checks,'mean':mean.tolist(),'scale':scale.tolist(),
                'train_graphs_sha256':c47.sha(sorted(gt))}
        for name,value in [('C48-training.json',tr),('C48-training-ledger.json',ledger)]:
            (out/name).write_text(json.dumps(value,sort_keys=True)+'\n')
        print(json.dumps({'stage':stage,'graphs':len(gt),'examples':len(tr),'labels':dict(collections.Counter(str(r['delta']) for r in tr)),
                          'checks':checks},indent=2));return
    ledger=json.loads((out/'C48-training-ledger.json').read_text())
    heads=[]
    for name in ('C48-HARD','C48-SHUFFLED'):
        f=out/(name+'.safetensors')
        assert hashlib.sha256(f.read_bytes()).hexdigest()==ledger['checks'][name]
        net=BinaryHead();net.load_state_dict(load_file(str(f)));net.eval();heads.append(net)
    mean=np.array(ledger['mean'],dtype=np.float32);scale=np.array(ledger['scale'],dtype=np.float32)
    old=old_c47(root)
    if stage=='report':
        tr=json.loads((out/'C48-training.json').read_text())
        blind=json.loads((out/'C48-blind.json').read_text())
        online=[];paths=[];go=set();cohorts=[]
        for cname,seed,families,worldfn,rep in COHORTS:
            rows=json.loads((out/f'C48-cohort-{cname}.json').read_text())
            selected=json.loads((out/f'C48-paths-{cname}.json').read_text())
            graphs={r['graph'] for r in rows}
            assert len(graphs)==len(families)*2*2*2*rep and go.isdisjoint(graphs)
            go.update(graphs);online.extend(rows);paths.extend(selected)
            cohorts.append({'name':cname,'graphs':len(graphs),'records':len(rows)})
        gt={r['graph'] for r in tr};gb={r['graph'] for r in blind}
        assert c47.sha(sorted(gt))==ledger['train_graphs_sha256']
        source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        summary=summarize(tr,blind,online,gt,gb,go,ledger['optimizer'],ledger['checks'],source_sha)
        summary['cohorts']=cohorts
        summary['predecessor']={'C47_source_git_blob':'a89d852a858de828415686098b74efe205e9567c',
            'C47_checkpoint_sha256':hashlib.sha256((root/'output/C47-ALIGNED.safetensors').read_bytes()).hexdigest(),
            'C40_checkpoint_sha256':hashlib.sha256((root/'previous/predecessor_C42/out_c40/C40-PAIR.safetensors').read_bytes()).hexdigest()}
        for name,value in [('C48-summary.json',summary),('C48-online.json',online),('C48-curated-paths.json',paths)]:
            (out/name).write_text(json.dumps(value,sort_keys=True,indent=2 if 'summary' in name else None)+'\n')
        print(json.dumps({'stage':stage,'training_deltas':summary['train_differences'],
            'goals':{m:d['guided_goals'] for m,d in summary['models'].items()},
            'regressions':{m:d['regressed_condition_groups_vs_RESET'] for m,d in summary['models'].items()},
            'paired':summary['paired'],'blind':summary['blind']},indent=2));return
    base,opt,prior=c47.c44.prerequisites(root/'previous/predecessor_C42')
    if stage=='blind':
        worlds=c47.world_rows(c47.c42.FAMILIES,c47.c42.world,480717,1)
        rows,_,_=eval_worlds(base,opt,prior,worlds,heads,mean,scale,old)
        (out/'C48-blind.json').write_text(json.dumps(rows,sort_keys=True)+'\n')
        print('blind',len(worlds),'graphs',len(rows),'records');return
    if stage.startswith('cohort-'):
        cname=stage[len('cohort-'):]
        _,seed,families,worldfn,rep=next(x for x in COHORTS if x[0]==cname)
        ws=c47.world_rows(families,worldfn,seed,rep)
        rows,_,paths=eval_worlds(base,opt,prior,ws,heads,mean,scale,old,retain=(cname in ('ID-A','OOD-A')))
        for name,value in [(f'C48-cohort-{cname}.json',rows),(f'C48-paths-{cname}.json',paths)]:
            (out/name).write_text(json.dumps(value,sort_keys=True)+'\n')
        print(cname,len(ws),'graphs',len(rows),'records',len(paths),'path checks');return
    raise ValueError(stage)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('out')
    parser.add_argument('--stage',default='all')
    a=parser.parse_args()
    if a.stage=='all':run(a.root,a.out)
    else:staged(a.root,a.out,a.stage)
