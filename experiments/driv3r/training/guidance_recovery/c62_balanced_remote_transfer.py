"""C62: fixed-optimizer-budget remote-goal guidance transfer + movement personality.

Independent fresh topology cohort, matched on-policy training and shuffled-credit
control. Synthetic game-like host only. No game or production ABI modifications.
Policies receive current host menu, action history, observed frontiers, current
position/goal and recovery cue; no hidden connectivity, protected cells or oracle.
"""
from __future__ import annotations
import base64, collections, copy, hashlib, json, math, sys, zlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file, load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frozen/research-source'))
import c60_remote_frontier_rl as c60

SEED=620118
HORIZON=76
TRAIN_WILSON_REPS=5
TRAIN_FAMILIAR_REPS=2
TEST_OOD_REPS=2
TEST_FAMILIAR_REPS=1
EPOCHS=5
UPDATES_PER_EPOCH=8
BATCH=256
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
DROPS=(0.,.5)
POLICIES=('C18-A0','C34-RESET','C60-FRONTIER','C60-FROZEN','C62-ALIGNED','C62-SHUFFLED')


def source_hashes():
    files=['c60_remote_frontier_rl.py','c52_shortest_step_cue.py','c34_independent_maze.py',
           'c56_memory_recovery.py','c59_eligibility_personality.py','experiment.py']
    return {x:hashlib.sha256((ROOT/'frozen/research-source'/x).read_bytes()).hexdigest() for x in files}


def cohorts():
    previous=json.loads((ROOT/'publication/C60-summary.json').read_text())
    history=json.loads((ROOT/'publication/C60-historical-graph-exclusions.json').read_text())
    c61=json.loads((ROOT/'results/C61-summary.json').read_text())
    excluded=set(history['excluded_graphs'])
    excluded.update(r['graph'] for phase in ('train','online') for r in previous['cohorts'][phase])
    excluded.update(r['graph'] for phase in ('train_graphs','test_graphs') for r in c61[phase])
    gen=c60.c59.c58.c55.c54.c52.worlds
    fam=c60.c59.c58.c55.c54.c52.TRAIN_FAMILIES
    familiar=c60.c59.c58.c55.c54.c52.c50.c47.c41.world
    ood=c60.c59.c58.c55.c54.c52.c50.c47.c42.world
    allreceipts=[];sets=[];used=set(excluded)
    for families,worldfn,seed,reps in [
        (('wilson',),ood,SEED+1000,TRAIN_WILSON_REPS),
        (fam,familiar,SEED+2000,TRAIN_FAMILIAR_REPS),
        (('kruskal','wilson'),ood,SEED+3000,TEST_OOD_REPS),
        (('tree','braid'),familiar,SEED+4000,TEST_FAMILIAR_REPS)]:
        worlds,rec=gen(families,worldfn,seed,reps,exclude=used)
        ids=[r['graph'] for r in rec]
        assert not used.intersection(ids) and len(set(ids))==len(ids)
        used.update(ids);sets.append(worlds);allreceipts.append(rec)
    tr=sets[0]+sets[1];ev=sets[2]+sets[3]
    assert not {r['graph'] for r in allreceipts[0]+allreceipts[1]}&{r['graph'] for r in allreceipts[2]+allreceipts[3]}
    assert len(set(r['graph'] for rs in allreceipts for r in rs))==sum(map(len,allreceipts))
    return tr,ev,allreceipts,len(excluded)


def fair_train(episodes,base,opt,mode):
    """Identical optimizer count, minibatch size, episode cohort and epochs.

    Only training-epoch on-policy rewards enter AdamW. The negative control
    permutes the delayed credit among visited actions; no heldout feedback.
    Distinct trajectories are an inherent on-policy difference and reported.
    """
    assert mode in ('aligned','shuffled')
    torch.manual_seed(SEED+301)
    model=nn.Linear(c60.NFEAT,1,bias=False)
    nn.init.zeros_(model.weight)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.027,weight_decay=.025)
    train_order=np.random.default_rng(SEED+11)
    orders=[train_order.permutation(len(episodes)) for _ in range(EPOCHS)]
    sampling=np.random.default_rng(SEED+401)
    per_epoch=[];updates=0
    for epoch,order in enumerate(orders):
        flat=[];completed=0;total_acts=0
        for i in order:
            state,arm=episodes[int(i)]
            weights=model.weight.detach().numpy().reshape(-1).copy()
            out,dec=c60.execute(base,opt,state,arm,'C60-ALIGNED',weights,0.,collect=True)
            if not dec:continue
            rewards=np.asarray([d['reward'] for d in dec],np.float32)
            ret=np.empty(len(rewards),np.float32);carry=0.
            for k in range(len(rewards)-1,-1,-1):
                carry=float(rewards[k])+.92*carry;ret[k]=carry
            flat.extend(zip(dec,ret));completed+=int(out['goal']);total_acts+=len(dec)
        assert len(flat)>=BATCH,'training exhausted before optimizer updates'
        returns=np.asarray([float(v) for _,v in flat],np.float32)
        if mode=='shuffled':
            rng=np.random.default_rng(SEED+9000+epoch)
            rng.shuffle(returns)
        mean=float(returns.mean());std=max(float(returns.std()),1e-4)
        sampled_indices=sampling.integers(0,len(flat),size=(UPDATES_PER_EPOCH,BATCH))
        for ids in sampled_indices:
            ds=[flat[int(i)][0] for i in ids]
            feats=torch.from_numpy(np.stack([d['features'] for d in ds]))
            old_logits=torch.from_numpy(np.stack([d['logits'] for d in ds]))
            chosen=torch.tensor([d['action'] for d in ds],dtype=torch.long)
            strength=torch.tensor([d['strength'] for d in ds],dtype=torch.float32)
            adv=torch.from_numpy(((returns[ids]-mean)/std).astype(np.float32))
            boost=c60.BONUS*strength[:,None]*torch.clamp(model(feats).squeeze(-1),-1.,1.)
            scores=old_logits+boost
            logp=torch.log_softmax(scores,dim=1)
            probs=logp.exp()
            kl=(probs*(logp-torch.log_softmax(old_logits,dim=1))).sum(1).mean()
            entropy=-(probs*logp).sum(1).mean()
            loss=-(logp.gather(1,chosen[:,None]).squeeze(1)*adv).mean()+.04*kl-.008*entropy
            optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),.8);optimizer.step()
            updates+=1
        per_epoch.append({'epoch':epoch,'onpolicy_episodes':len(episodes),'actual_rollout_actions':total_acts,
                          'training_goals':completed,'minibatches':UPDATES_PER_EPOCH,
                          'samples_per_minibatch':BATCH,'mean_discounted_return':mean,
                          'weight_l2':float(torch.linalg.vector_norm(model.weight.detach()))})
    assert updates==EPOCHS*UPDATES_PER_EPOCH
    return model,{'mode':mode,'updates':updates,'epochs':per_epoch,'graphs':len({x[0][6]['graph_sha'] for x in episodes}),
                  'episodes_per_epoch':len(episodes)}


def weightmap():
    c60s=json.loads((ROOT/'publication/C60-summary.json').read_text())
    c61s=json.loads((ROOT/'results/C61-summary.json').read_text())
    return {'C60-FROZEN':np.asarray(c60s['weights']['C60-ALIGNED'],np.float32),
            }, {'C60-ALIGNED':c60s['checkpoint_sha256']['ALIGNED'],
            'C61-WILSON':c61s['checkpoint_sha256']['C61-WILSON']}


def execute(base,opt,state,arm,policy,weight,drop,trace=False):
    map_name={'C60-FROZEN':'C60-ALIGNED','C62-ALIGNED':'C60-ALIGNED','C62-SHUFFLED':'C60-ALIGNED'}.get(policy,policy)
    return c60.execute(base,opt,state,arm,map_name,weight,drop,trace=trace)[0]


def full_personality(out):
    path=out['path']
    pts=[tuple(out['start_pos'])]+[tuple(x['p']) for x in path]
    dirs=[]
    for a,b in zip(pts,pts[1:]):
        d=(b[0]-a[0],b[1]-a[1])
        if abs(d[0])+abs(d[1])==1:dirs.append(d)
    if len(dirs)<5:return None
    counts=collections.Counter(dirs)
    return {'turn_rate':sum(a!=b for a,b in zip(dirs,dirs[1:]))/max(1,len(dirs)-1),
        'reverse_rate':sum(a==(-b[0],-b[1]) for a,b in zip(dirs[1:],dirs[:-1]))/max(1,len(dirs)-1),
        'direction_entropy':-sum((n/len(dirs))*math.log2(n/len(dirs)) for n in counts.values())/2.,
        'unique_ratio':len(set(pts))/len(pts),
        'paint_fraction':sum(x['a']=='INK' for x in path)/max(1,len(path)),
        'executed_turns':len(path)}


def cluster_delta(rows,pol,cohort,drop):
    clusters=collections.defaultdict(int)
    for r in rows:
        if r['family'] not in cohort or r['drop']!=drop or r['arm'][0]=='none':continue
        clusters[r['graph']]+=r['outcomes'][pol]['goal']-r['outcomes']['C34-RESET']['goal']
    values=np.asarray(list(clusters.values()),np.int32)
    if not len(values):return None
    rng=np.random.default_rng(SEED+len(pol)+int(drop*10)+sum(len(f) for f in cohort))
    boot=values[rng.integers(0,len(values),size=(2500,len(values)))].sum(1)
    return {'net_goals':int(values.sum()),'graphs':len(values),
        'positive_graphs':int(sum(values>0)),'negative_graphs':int(sum(values<0)),
        'exploratory_cluster95':[float(x) for x in np.quantile(boot,[.025,.975])]}


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c60.SEED=SEED;c60.HORIZON=HORIZON
    c60.c59.c58.c55.TICKS=(9,)
    base,opt,_=c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(ROOT/'frozen/predecessor_C42')
    tr,ev,receipts,excluded=cohorts()
    states=c60.c59.c58.c56.states_for_goal
    trst,skiptrain=states(base,tr,'remote');evst,skiptest=states(base,ev,'remote')
    assert len(trst)>=14 and len(evst)>=8
    assert {s[6]['graph_sha'] for s in trst}.isdisjoint({s[6]['graph_sha'] for s in evst})
    episodes=[(s,a) for s in trst for a in ARMS[1:]]
    cache=out/'C62-train-cache.json'
    if cache.exists():
        cached=json.loads(cache.read_text())
        assert cached['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'source changed: invalidate training cache explicitly'
        assert cached['train_graphs']==sorted({s[6]['graph_sha'] for s in trst}), 'training graphs changed'
        fitstat=cached['training'];negstat=cached['control'];digest=cached['checkpoint_sha256']
        from types import SimpleNamespace
        fit=SimpleNamespace(weight=load_file(str(out/'C62-ALIGNED.safetensors'))['weight'])
        neg=SimpleNamespace(weight=load_file(str(out/'C62-SHUFFLED.safetensors'))['weight'])
        assert all(hashlib.sha256((out/(k+'.safetensors')).read_bytes()).hexdigest()==v for k,v in digest.items())
    else:
        fit,fitstat=fair_train(episodes,base,opt,'aligned')
        neg,negstat=fair_train(episodes,base,opt,'shuffled')
        assert fitstat['updates']==negstat['updates']==EPOCHS*UPDATES_PER_EPOCH
        digest={}
        for label,model in [('C62-ALIGNED',fit),('C62-SHUFFLED',neg)]:
            p=out/(label+'.safetensors');save_file(model.state_dict(),str(p));digest[label]=hashlib.sha256(p.read_bytes()).hexdigest()
        cache.write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'train_graphs':sorted({s[6]['graph_sha'] for s in trst}),
            'training':fitstat,'control':negstat,'checkpoint_sha256':digest},indent=2,sort_keys=True)+'\n')
    weights,oldhash=weightmap()
    weights['C62-ALIGNED']=fit.weight.detach().numpy().reshape(-1).copy()
    weights['C62-SHUFFLED']=neg.weight.detach().numpy().reshape(-1).copy()
    record_progress=out/'C62-progress.ndjson'
    trace_progress=out/'C62-trace-progress.ndjson'
    saved_rows=[json.loads(line) for line in record_progress.read_text().splitlines()] if record_progress.exists() else []
    saved_traces=[json.loads(line) for line in trace_progress.read_text().splitlines()] if trace_progress.exists() else []
    done={(r['graph'],r['drop'],tuple(r['arm'])) for r in saved_rows}
    assert len(done)==len(saved_rows), 'duplicate progressed condition'
    records=list(saved_rows);traces=list(saved_traces);noguidance=0;full=collections.defaultdict(list)
    gallery_selected={}
    for s in evst:
        fam=s[8]
        if fam not in gallery_selected and fam in ('wilson','kruskal','braid'):gallery_selected[fam]=s[6]['graph_sha']
        for drop in DROPS:
            for arm in ARMS:
                if (s[6]['graph_sha'],drop,tuple(arm)) in done:continue
                outcomes={};before_traces=len(traces)
                for pol in POLICIES:
                    w=weights.get(pol,np.zeros(c60.NFEAT,np.float32))
                    result=execute(base,opt,s,arm,pol,w,drop,trace=(arm[0]!='none'))
                    assert result['host_rejections']==0 and result['protected_sha']==s[3]['protected_sha']
                    if arm[0]!='none':
                        ff=full_personality(result)
                        if ff:full[(fam,drop,arm[1],pol)].append(ff)
                        result['personality_full']=ff
                    # Gallery: selection fixed by earliest eligible graph for each named family
                    # and both strengths for a fixed controller subset; verify hashes from same run.
                    if s[6]['graph_sha']==gallery_selected.get(fam) and arm==ARMS[-1] and pol in ('C34-RESET','C60-FRONTIER','C62-ALIGNED'):
                        traces.append({'graph':s[6]['graph_sha'],'family':fam,'drop':drop,'policy':pol,
                                       'start_pos':result['start_pos'],'target':result['target'],
                                       'goal':result['goal'],'path':result['path'],
                                       'action_sha':result['action_sha'],'ink_sha':result['ink_sha']})
                    result['path']=None
                    outcomes[pol]=result
                record={'graph':s[6]['graph_sha'],'seed':s[7],
                        'family':fam,'case':s[9],'personality':s[0].personality,
                        'drop':drop,'arm':list(arm),'prefix_sha':s[3]['prefix_sha'],
                        'target':list(s[5]),'outcomes':outcomes}
                if arm[0]=='none':
                    assert len({(z['action_sha'],z['ink_sha']) for z in outcomes.values()})==1
                records.append(record)
                with record_progress.open('a') as fh:fh.write(json.dumps(record,separators=(',',':'),sort_keys=True)+'\n')
                with trace_progress.open('a') as fh:
                    for t in traces[before_traces:]:fh.write(json.dumps(t,separators=(',',':'),sort_keys=True)+'\n')
    assert len(records)==len(evst)*len(DROPS)*len(ARMS)
    noguidance=sum(r['arm'][0]=='none' for r in records)
    agg={}
    for cohort,fams in [('kruskal',('kruskal',)),('wilson',('wilson',)),
                        ('familiar',('tree','braid','prim','prim_braid'))]:
        for drop in DROPS:
            rs=[r for r in records if r['family'] in fams and r['drop']==drop and r['arm'][0]!='none']
            agg[f'{cohort}|{drop}']={'episodes':len(rs),'policies':{p:{'goals':sum(r['outcomes'][p]['goal'] for r in rs),
                'new_sites':sum(r['outcomes'][p]['new_sites'] for r in rs),
                'wins_vs_RESET':sum(r['outcomes'][p]['goal']>r['outcomes']['C34-RESET']['goal'] for r in rs),
                'losses_vs_RESET':sum(r['outcomes'][p]['goal']<r['outcomes']['C34-RESET']['goal'] for r in rs),
                'cluster':cluster_delta(rs,p,fams,drop) if p!='C34-RESET' else None}
                for p in POLICIES}}
    psy={}
    for cohort,fams in [('kruskal',('kruskal',)),('wilson',('wilson',)),('familiar',('tree','braid','prim','prim_braid'))]:
        for drop in DROPS:
            for pol in POLICIES[2:]:
                paired=[r for r in records if r['family'] in fams and r['drop']==drop and r['arm'][0]!='none'
                    and r['outcomes'][pol].get('personality_full') and r['outcomes']['C34-RESET'].get('personality_full')]
                metrics=('turn_rate','reverse_rate','direction_entropy','unique_ratio','paint_fraction')
                psy[f'{cohort}|{drop}|{pol}']={'eligible_pairs':len(paired),
                    'mean_full_trajectory_delta_vs_RESET':{k:float(np.mean([r['outcomes'][pol]['personality_full'][k]-
                        r['outcomes']['C34-RESET']['personality_full'][k] for r in paired])) if paired else None for k in metrics},
                    'mean_executed_turn_delta':float(np.mean([r['outcomes'][pol]['steps']-r['outcomes']['C34-RESET']['steps'] for r in paired])) if paired else None}
    summary={'schema':'driv3r-c62-equal-budget-remote-transfer/1','seed':SEED,'horizon':HORIZON,
             'policy_source_sha256':source_hashes(),'experiment_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'past_graph_identities_excluded':excluded,'graph_receipts':receipts,
             'eligible_train_prefixes':len(trst),'eligible_test_prefixes':len(evst),
             'training_skips':skiptrain,'test_skips':skiptest,'training':fitstat,'control':negstat,
             'weights':{k:v.tolist() for k,v in weights.items()},'frozen_checkpoint_sha256':oldhash,
             'checkpoint_sha256':digest,'matched_conditions':len(records),
             'executed_suffixes':len(records)*len(POLICIES),'unguided_parity_conditions':noguidance,
             'replayed_gallery_traces':len(traces),'aggregate':agg,'personality_full':psy,
             'limitations':['External synthetic host and frozen historical fixture; not native CyberSand ABI',
               'Kruskal tests never trained, but Wilson and familiar training families remain in-domain',
               'Same SGD updates and sampled minibatches, but policy-dependent on-policy actions differ',
               'Full trajectory personality means depend on varying episode termination/horizon',
               'Descriptive graph-cluster bootstrap, not confirmatory hypothesis testing',
               'No paid training; CPU-only. Checkpoints not in Git history or Actions immutable artifact custody.']}
    ledger={'schema':'driv3r-c62-executed-ledger/1','summary':summary,'records':records,'gallery_traces':traces}
    raw=json.dumps(ledger,separators=(',',':'),sort_keys=True).encode()
    (out/'C62-executed-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(raw,9)).decode()+'\n')
    (out/'C62-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'C62-gallery-traces.json').write_text(json.dumps(traces,separators=(',',':'),sort_keys=True)+'\n')
    print(json.dumps({'train_states':len(trst),'test_states':len(evst),'matched':len(records),
        'suffixes':len(records)*len(POLICIES),'updates':(fitstat['updates'],negstat['updates']),
        'gallery':len(traces),'kruskal_0':agg.get('kruskal|0.0')},indent=2))
    return summary

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output');a=parser.parse_args();run(a.output)
