"""C49: source-matched synthetic-host mid-trajectory correction.

The action prefix is EXECUTED by the synthetic host under its own legal menu.
Recovery advice is issued only after the resulting host state is captured. A
learned head chooses whether to adopt the C34-RESET recovery controller
or retain the frozen C18 baseline controller. Frozen C40-PAIR is evaluated
separately as a comparator; future outcome labels never enter inference.
No CyberSand native/production changes; no claims of native ABI validation.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from experiment import sha
from signed_feedback import HostAdvice
from c28_temporal_option import scores, probs
import c47_pairwise_episode_switch as c47

SEED=491117
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.),('continue',1.))
TRAIN_ARMS=ARMS[1:3]
MODELS=('C18-A0','C34-RESET','C40-PAIR','C49-ALIGNED','C49-SHUFFLED','C49-RANDOM','C49-CONSERVATIVE')
TRAIN_STEPS=(9,19)
BASES=('C18-A0','C34-RESET','C40-PAIR')

class Head(c47.Head):
    pass


def prefix(base,option,w,seed,stop):
    """Real legal-action rollout, without guidance and without teleportation.

    Forced erroneous action is *not* inserted: natural mistakes are induced by
    the current stochastic game policy. This limits how many true rescue cases
    appear and is explicitly recorded as a scientific limitation.
    """
    w=copy.deepcopy(w)
    advice=HostAdvice(w);rng=np.random.default_rng(seed+99203)
    origin=sha(sorted(w.protected));acts=[];positions=[w.pos]
    for t in range(stop):
        if w.stopped or w.distance()==0:break
        menu,logits,bad,_=scores(base,w,advice,1.)
        assert bad>=0
        j=int(rng.choice(len(menu),p=probs(logits)))
        action=menu[j]
        assert action.legal and w.apply(action) and w.invalid_attempts==0
        advice.advance(w)
        acts.append((action.action,action.target));positions.append(w.pos)
    assert origin==sha(sorted(w.protected))
    return w,{'steps':len(acts),'actions_sha':sha(acts),'prefix_positions':positions,
              'prefix_actions':acts,'protected_sha':origin,'initial':positions[0],
              'reached_before_cue':w.distance()==0,'stopped_before_cue':w.stopped}


def scenarios(base,option,seed,families,worldfn,per_group=2,steps=TRAIN_STEPS):
    out=[];graph_ids=set();skipped=collections.Counter()
    for i in range(per_group):
        worlds=c47.world_rows(families,worldfn,seed+i*520021,1)
        for w,info,s,fam,case,switch in worlds:
            if switch: # score a single fixed goal for clearer counterfactual attribution
                continue
            if info['graph_sha'] in graph_ids:raise AssertionError('duplicate graph identity')
            graph_ids.add(info['graph_sha'])
            for t in steps:
                snap,receipt=prefix(base,option,w,s,t)
                if receipt['steps']<t or snap.distance()==0 or snap.stopped:
                    skipped['early_terminal']+=1;continue
                assert snap.invalid_attempts==0
                # The host, not the policy, now changes the goal to a cell
                # previously reached through legal actions. This creates a
                # genuine retracing/correction challenge rather than an
                # arbitrary pre-filled history or teleport.
                target=tuple(info['second_goal'])
                if target in snap.history or target==snap.pos:
                    target=tuple(snap.goal)
                if target in snap.history or target==snap.pos:
                    skipped['goal_on_history']+=1;continue
                receipt['old_goal']=tuple(snap.goal)
                snap.goal=target;snap.goal_change=True;snap.goals_seen+=1;snap.stuck=2
                assert target not in snap.history and snap.invalid_attempts==0
                receipt['host_goal_change_to']=target
                receipt['goal_changed']=target!=receipt['old_goal']
                out.append((snap,info,s,fam,case,True,t,receipt))
    return out,graph_ids,dict(skipped)


def counterfactuals(base,option,prior,states,arms,retain=False):
    data=[];traces=[]
    for w,info,seed,fam,case,switch,t,prefix_receipt in states:
        for directive,strength in arms:
            if directive!='backtrack':
                # All three controllers have identical unguided/CONTINUE paths:
                # the extra recovery heads are exclusively backtrack-cue gated.
                one=c47.execute(base,option,prior,w,info,seed,directive,strength,False,'C18-A0')
                results={m:one for m in BASES}
            else:
                results={m:c47.execute(base,option,prior,w,info,seed,directive,strength,False,m) for m in BASES}
            assert all(r['host_rejections']==0 for r in results.values())
            x=c47.context(w,prior,directive,strength,True)
            row={'graph':info['graph_sha'],'family':fam,'case':case,'personality':w.personality,
                'seed':seed,'prefix_t':t,'directive':directive,'strength':strength,
                'prefix_sha':prefix_receipt['actions_sha'],
                'prefix_steps':prefix_receipt['steps'],
                'protected_sha':prefix_receipt['protected_sha'],
                'x':x.tolist(),
                'delta':int(results['C34-RESET']['goals']-results['C18-A0']['goals']),
                'outcomes':results}
            data.append(row)
            if retain and directive=='backtrack' and strength==1.:
                if len({v['graph'] for v in traces})>=8:continue
                for m in BASES:
                    r=c47.execute(base,option,prior,w,info,seed,directive,strength,False,m,True)
                    assert (r['actions_sha'],r['ink_sha'])==(results[m]['actions_sha'],results[m]['ink_sha'])
                    traces.append({'graph':row['graph'],'family':fam,'case':case,'model':m,
                        'prefix_t':t,'prefix_actions_sha':prefix_receipt['actions_sha'],
                        'prefix_positions':prefix_receipt['prefix_positions'],
                        'initial':prefix_receipt['initial'],'start':w.pos,
                        'goal':w.goal,'protected':sorted(w.protected),
                        'goal_events':r['goals'],'trajectory':r['trajectory'],
                        'actions_sha':r['actions_sha'],'ink_sha':r['ink_sha']})
    return data,traces


def permuted_labels(rows,seed):
    """Source-matched stratum permutations: no direct oracle access at inference."""
    true=np.asarray([r['delta'] for r in rows],np.float32)
    y=true.copy();by=collections.defaultdict(list)
    for ix,r in enumerate(rows):
        by[(r['family'],r['case'],r['personality'],r['prefix_t'],r['directive'],r['strength'])].append(ix)
    rng=np.random.default_rng(seed+43)
    for group in by.values():
        if len(group)<2:continue
        old=true[group]
        roll=int(rng.integers(1,len(group)))
        y[group]=np.roll(old,roll)
    assert np.array_equal(np.sort(y),np.sort(true))
    assert np.any(y!=true),'inactive shuffled control'
    return y


def train(rows,seed=SEED,epochs=24):
    torch.set_num_threads(1);torch.manual_seed(seed)
    x=np.asarray([r['x'] for r in rows],np.float32)
    y=np.asarray([r['delta'] for r in rows],np.float32)
    shuffled=permuted_labels(rows,seed)
    mean=x.mean(0);scale=np.maximum(x.std(0),.05)
    xx=torch.from_numpy(np.clip((x-mean)/scale,-6,6).astype(np.float32))
    torch.manual_seed(seed+11);initial=Head().eval()
    models=[copy.deepcopy(initial),copy.deepcopy(initial)]
    rng=np.random.default_rng(seed+17)
    batches=[rng.permutation(len(rows)) for _ in range(epochs)]
    stats={}
    for name,head,yy in zip(('aligned','shuffled'),models,(y,shuffled)):
        optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.02)
        labels=torch.from_numpy((yy>0).astype(np.float32))
        weights=torch.from_numpy(np.where(yy>0,10.,np.where(yy<0,10.,.18)).astype(np.float32))
        losses=[];updates=0
        for order in batches:
            running=0.
            for start in range(0,len(order),32):
                ix=order[start:start+32]
                logits=head(xx[ix]);loss=(nn.functional.binary_cross_entropy_with_logits(logits,labels[ix],reduction='none')*weights[ix]).sum()/weights[ix].sum()
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
                running+=float(loss.detach());updates+=1
            losses.append(round(running,6))
        head.eval();stats[name]={'optimizer_updates':updates,'epoch_losses':losses}
    stats.update({'examples':len(rows),'graphs':len({r['graph'] for r in rows}),
        'deltas':dict(collections.Counter(str(int(v)) for v in y)),
        'targets_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
        'shuffled_targets_sha256':hashlib.sha256(shuffled.tobytes()).hexdigest(),
        'normalization_sha256':hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()})
    return models,mean,scale,stats


def select(row,models,mean,scale,mode):
    if mode in BASES:return mode
    if row['directive']!='backtrack':return 'C18-A0'
    if mode=='C49-CONSERVATIVE':return 'C18-A0'
    if mode=='C49-RANDOM':
        # Deliberately cannot inspect actual future goal outcomes.
        key=f"{row['graph']}|{row['prefix_t']}|{row['strength']}|{SEED}"
        return 'C34-RESET' if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)/2**32<.25 else 'C18-A0'
    ix=0 if mode=='C49-ALIGNED' else 1
    x=torch.from_numpy(np.clip((np.asarray(row['x'],np.float32)-mean)/scale,-6,6))
    with torch.no_grad():z=float(models[ix](x))
    return 'C34-RESET' if z>.75 else 'C18-A0'


def assess(rows,models,mean,scale):
    evaluated=[]
    for row in rows:
        for mode in MODELS:
            selected=select(row,models,mean,scale,mode)
            rr=row['outcomes'][selected]
            assert rr['host_rejections']==0
            evaluated.append({k:row[k] for k in ('graph','family','case','personality','prefix_t','directive','strength','prefix_sha','prefix_steps','protected_sha','delta')}
                 |{'model':mode,'selected':selected,
                   **{k:v for k,v in rr.items() if k!='trajectory'}})
    return evaluated


def report(trainrows,blindrows,online,train_graphs,blind_graphs,online_graphs,heads,stats,checks,skips):
    assert train_graphs.isdisjoint(blind_graphs) and train_graphs.isdisjoint(online_graphs) and blind_graphs.isdisjoint(online_graphs)
    grouped=collections.defaultdict(collections.Counter)
    bygraph=collections.defaultdict(collections.Counter)
    for r in online:
        if r['directive']!='none':
            grouped[(r['family'],r['case'],r['prefix_t'],r['directive'],r['strength'])][r['model']]+=r['goals']
            bygraph[r['graph']][r['model']]+=r['goals']
        assert r['host_rejections']==0
    baseline={(r['graph'],r['prefix_t'],r['directive'],r['strength']):r for r in online if r['model']=='C18-A0'}
    for r in online:
        if r['directive']=='none':
            ref=baseline[(r['graph'],r['prefix_t'],r['directive'],r['strength'])]
            assert (r['actions_sha'],r['ink_sha'])==(ref['actions_sha'],ref['ink_sha'])
    result={'schema':'driv3r-c49-midtrajectory-correction/1',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'training':stats,'checkpoints_sha256':checks,
      'graphs':{'train':len(train_graphs),'blind':len(blind_graphs),'online':len(online_graphs)},
      'graph_set_sha256':{k:sha(sorted(v)) for k,v in [('train',train_graphs),('blind',blind_graphs),('online',online_graphs)]},
      'skipped_prefixes':skips,'models':{},'paired_differences':{},'blind':{},
      'limitations':[
          'Synthetic host only: no native CyberSand or physical actor acceptance',
          'Counterfactual advice begins after host-legal prefixes and explicit host goal-change/correction events, but suffixes reset RNG stream and clock to zero; not a continuous 78-tick live host intervention',
          'Goal switching occurs between prefix and suffix; the suffix contains one fixed terminal goal',
          'Episode suffix performance differs from full long-horizon worm gameplay',
          'Recovery corpus samples stochastic natural host-executed prefixes; population is enriched by an explicit host goal-change/cue event',
          'Bootstrap intervals are descriptive and not multiplicity-adjusted',
          'Non-guided parity does not prove active-guidance whole-path personality retention',
          'Historical d_common fixture is not native canonical ABI',
          'No immutable GitHub Actions checkpoint custody']}
    for m in MODELS:
        z=[r for r in online if r['model']==m]
        guided=[r for r in z if r['directive']!='none']
        result['models'][m]={'guided_goal_events':sum(r['goals'] for r in guided),
             'guided_episodes':len(guided),
             'unguided_exact':sum(r['directive']=='none' for r in z),
             'condition_regressions_vs_RESET':sum(group[m]<group['C34-RESET'] for group in grouped.values()),
             'selected_RESET':sum(r['selected']=='C34-RESET' for r in guided),
             'by_family':dict(collections.Counter({f:sum(r['goals'] for r in guided if r['family']==f) for f in sorted({r['family'] for r in guided})})),
             'preferred_ink':sum(r['preferred'] for r in guided),
             'opposed_ink':sum(r['opposed'] for r in guided),
             'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in z),
             'mean_personality_shift':sum(r['personality_shift_post'] for r in guided)/max(1,sum(r['action_states'] for r in guided))}
    for m in ('C49-ALIGNED','C49-SHUFFLED','C49-RANDOM'):
        for against in ('C34-RESET','C49-SHUFFLED'):
            if m==against:continue
            dif=np.asarray([v[m]-v[against] for v in bygraph.values()],np.int32)
            rng=np.random.default_rng(SEED+MODELS.index(m)*41+MODELS.index(against))
            draws=dif[rng.integers(len(dif),size=(3000,len(dif)))].sum(1)
            result['paired_differences'][m+'__'+against]={
                 'goal_delta':int(dif.sum()), 'positive_graphs':int((dif>0).sum()),
                 'negative_graphs':int((dif<0).sum()),
                 'bootstrap_95_descriptive':np.quantile(draws,[.025,.975]).tolist()}
    for m in ('C49-ALIGNED','C49-SHUFFLED','C49-RANDOM'):
        picks=[r for r in blindrows if r['model']==m and r['selected']=='C34-RESET']
        result['blind'][m]={'n':sum(r['model']==m for r in blindrows),
               'selected_RESET':len(picks),
               'actual_advantage_over_A0':sum(r['delta'] for r in picks),
               'selected_wins':sum(r['delta']>0 for r in picks),'selected_losses':sum(r['delta']<0 for r in picks)}
    return result


def run(root,out,train_worlds=4,online_worlds=3):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    base,opt,prior=c47.c44.prerequisites(root/'previous/predecessor_C42')
    ledger_file=out/'C49-training-ledger.json'
    if ledger_file.exists() and (out/'C49-training.json').exists():
        ledger=json.loads(ledger_file.read_text())
        trainrows=json.loads((out/'C49-training.json').read_text())
        gt={r['graph'] for r in trainrows}
        skip_train={'early_terminal':'not retained during completed training'}
        stats=ledger['optimizer'];checks=ledger['checkpoints_sha256']
        mean=np.asarray(ledger['mean'],np.float32);scale=np.asarray(ledger['scale'],np.float32)
        models=[]
        for name in ('C49-ALIGNED','C49-SHUFFLED'):
            f=out/(name+'.safetensors')
            assert hashlib.sha256(f.read_bytes()).hexdigest()==checks[name]
            head=Head();head.load_state_dict(load_file(str(f)));head.eval();models.append(head)
        assert len(gt)==stats['graphs'] and len(trainrows)==stats['examples']
        print('C49 reused source-matched completed training/checkpoint SHA receipts',len(gt),flush=True)
    else:
        train_states,gt,skip_train=scenarios(base,opt,490117,('tree','braid','prim','prim_braid'),c47.c41.world,train_worlds)
        trainrows,_=counterfactuals(base,opt,prior,train_states,TRAIN_ARMS)
        print('C49 train: graphs',len(gt),'prefix states',len(train_states),'delta counts',dict(collections.Counter(r['delta'] for r in trainrows)),flush=True)
        if not trainrows:raise RuntimeError('no training states')
        models,mean,scale,stats=train(trainrows)
        again,m2,s2,st2=train(trainrows)
        assert stats==st2 and np.array_equal(mean,m2) and np.array_equal(scale,s2)
        assert all(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()) for a,b in zip(models,again))
        checks={}
        for name,head in zip(('C49-ALIGNED','C49-SHUFFLED'),models):
            f=out/(name+'.safetensors')
            save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(f))
            checks[name]=hashlib.sha256(f.read_bytes()).hexdigest()
        (out/'C49-training-ledger.json').write_text(json.dumps({'optimizer':stats,'checkpoints_sha256':checks,
           'mean':mean.tolist(),'scale':scale.tolist()},sort_keys=True)+'\n')
        (out/'C49-training.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='outcomes'} for r in trainrows],sort_keys=True)+'\n')
    print('C49 AdamW replicated; checkpoints',checks,flush=True)
    if (out/'C49-blind.json').exists():
        blind=json.loads((out/'C49-blind.json').read_text())
        gb={r['graph'] for r in blind};skip_blind={}
    else:
        blind_states,gb,skip_blind=scenarios(base,opt,499117,c47.c42.FAMILIES,c47.c42.world,1)
        blindrows,_=counterfactuals(base,opt,prior,blind_states,ARMS)
        blind=assess(blindrows,models,mean,scale)
        (out/'C49-blind.json').write_text(json.dumps(blind,sort_keys=True)+'\n')
    print('C49 blind',len(gb),'graphs',len(blind),flush=True)
    all_online=[];paths=[];go=set();cohorts=[];skips={'train':skip_train,'blind':skip_blind}
    for cohort,seed,families,make,rep in (('ID',498117,('tree','braid','prim','prim_braid'),c47.c41.world,online_worlds),
                                          ('OOD',497117,c47.c42.FAMILIES,c47.c42.world,online_worlds)):
        saved=out/f'C49-cohort-{cohort}.json';saved_paths=out/f'C49-paths-{cohort}.json'
        if saved.exists() and saved_paths.exists():
            evaluated=json.loads(saved.read_text());found=json.loads(saved_paths.read_text())
            graphs={r['graph'] for r in evaluated};n=len({(r['graph'],r['prefix_t']) for r in evaluated})
            skipped={}
        else:
            states,graphs,skipped=scenarios(base,opt,seed,families,make,rep)
            rows,found=counterfactuals(base,opt,prior,states,ARMS,retain=(cohort=='OOD'))
            evaluated=assess(rows,models,mean,scale)
            saved.write_text(json.dumps(evaluated,sort_keys=True)+'\n')
            saved_paths.write_text(json.dumps(found,sort_keys=True)+'\n')
            n=len(states)
        assert go.isdisjoint(graphs);go.update(graphs)
        all_online+=evaluated;paths+=found
        cohorts.append({'name':cohort,'graphs':len(graphs),'prefix_states':n,'records':len(evaluated)})
        skips[cohort]=skipped
        print('C49 cohort',cohort,cohorts[-1],flush=True)
    summary=report(trainrows,blind,all_online,gt,gb,go,models,stats,checks,skips)
    summary['cohorts']=cohorts
    summary['predecessor']={'C40_checkpoint_sha256':hashlib.sha256((root/'previous/predecessor_C42/out_c40/C40-PAIR.safetensors').read_bytes()).hexdigest()}
    for name,obj in [('C49-summary.json',summary),('C49-blind.json',blind),('C49-online.json',all_online),('C49-curated-paths.json',paths)]:
        (out/name).write_text(json.dumps(obj,sort_keys=True,indent=2 if name=='C49-summary.json' else None)+'\n')
    print(json.dumps({'training_deltas':stats['deltas'],'models':{k:{'goals':v['guided_goal_events'],'regressions':v['condition_regressions_vs_RESET'],'selected_RESET':v['selected_RESET']} for k,v in summary['models'].items()},'paired':summary['paired_differences'],'blind':summary['blind']},indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('--train-worlds',type=int,default=4);p.add_argument('--online-worlds',type=int,default=3)
    a=p.parse_args();run(a.root,a.out,a.train_worlds,a.online_worlds)
